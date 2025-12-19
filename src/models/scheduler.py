"""
Diffusion noise scheduler for DDPM and DDIM.

Implements the forward diffusion process and reverse sampling
with support for both DDPM (stochastic) and DDIM (deterministic) methods.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class DiffusionScheduler(nn.Module):
    """
    Noise scheduler for diffusion models.
    
    Supports both DDPM and DDIM sampling methods with configurable
    noise schedules (linear, cosine).
    
    Args:
        num_steps: Number of diffusion timesteps
        schedule_type: Type of noise schedule ("linear", "cosine")
        beta_start: Starting beta value (for linear schedule)
        beta_end: Ending beta value (for linear schedule)
    """
    
    def __init__(
        self,
        num_steps: int = 1000,
        schedule_type: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.num_steps = max(1, num_steps)
        self.schedule_type = schedule_type
        
        # Compute betas based on schedule type
        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, steps=self.num_steps)
        elif schedule_type == "cosine":
            betas = self._cosine_beta_schedule()
        else:
            betas = torch.linspace(beta_start, beta_end, steps=self.num_steps)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([
            torch.ones(1), alphas_cumprod[:-1]
        ])
        
        # Register buffers (not persistent to avoid checkpoint bloat)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev, persistent=False)
        
        # Precomputed values for efficiency
        self.register_buffer(
            "sqrt_alphas_cumprod", 
            torch.sqrt(alphas_cumprod), 
            persistent=False
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
            persistent=False,
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod),
            persistent=False,
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod - 1.0),
            persistent=False,
        )
        
        # For posterior variance in DDPM
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_variance", 
            posterior_variance, 
            persistent=False
        )
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
            persistent=False,
        )

    def _cosine_beta_schedule(self, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in Improved DDPM."""
        steps = self.num_steps + 1
        t = torch.linspace(0, self.num_steps, steps)
        alphas_cumprod = torch.cos(((t / self.num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, min=0.0001, max=0.9999)

    def _extract(
        self, tensor: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size
    ) -> torch.Tensor:
        """Extract values from tensor at given timesteps."""
        max_idx = tensor.size(0) - 1
        timesteps = torch.clamp(timesteps, min=0, max=max_idx)
        out = tensor.to(timesteps.device)[timesteps]
        return out.view(-1, *([1] * (len(shape) - 1)))

    def q_sample(
        self, 
        x_start: torch.Tensor, 
        noise: torch.Tensor, 
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward diffusion: add noise to data.
        
        Args:
            x_start: Clean data [B, C, H, W]
            noise: Noise tensor [B, C, H, W]
            timesteps: Timesteps [B]
            
        Returns:
            Noisy data at given timesteps
        """
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape
        )
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise."""
        sqrt_recip = self._extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape)
        sqrt_recipm1 = self._extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape)
        return sqrt_recip * x_t - sqrt_recipm1 * noise

    def ddpm_step(
        self,
        x_t: torch.Tensor,
        predicted_noise: torch.Tensor,
        timestep: torch.Tensor,
        add_noise: bool = True,
    ) -> torch.Tensor:
        """
        Single DDPM reverse step.
        
        Args:
            x_t: Current noisy sample
            predicted_noise: Model's noise prediction
            timestep: Current timestep
            add_noise: Whether to add noise (False for t=0)
            
        Returns:
            x_{t-1}: Less noisy sample
        """
        beta = self._extract(self.betas, timestep, x_t.shape)
        alpha = self._extract(self.alphas, timestep, x_t.shape)
        alpha_bar = self._extract(self.alphas_cumprod, timestep, x_t.shape)
        
        sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1 - alpha_bar, min=1e-5))
        x_prev = (1 / torch.sqrt(alpha)) * (
            x_t - (beta / sqrt_one_minus_alpha_bar) * predicted_noise
        )
        
        if add_noise:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + torch.sqrt(beta) * noise
            
        return x_prev

    def ddim_step(
        self,
        x_t: torch.Tensor,
        predicted_noise: torch.Tensor,
        timestep: torch.Tensor,
        prev_timestep: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Single DDIM reverse step.
        
        Args:
            x_t: Current noisy sample
            predicted_noise: Model's noise prediction
            timestep: Current timestep
            prev_timestep: Previous timestep
            eta: Noise level (0 = deterministic, 1 = DDPM)
            
        Returns:
            x_{t-1}: Less noisy sample
        """
        alpha_bar_t = self._extract(self.alphas_cumprod, timestep, x_t.shape)
        alpha_bar_t_prev = self._extract(self.alphas_cumprod, prev_timestep, x_t.shape)
        
        # Predict x_0
        pred_x0 = (x_t - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        # Direction pointing to x_t
        sigma = eta * torch.sqrt(
            (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)
        )
        dir_xt = torch.sqrt(1.0 - alpha_bar_t_prev - sigma ** 2) * predicted_noise
        
        # Add noise if eta > 0
        if eta > 0:
            noise = torch.randn_like(x_t)
            x_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + sigma * noise
        else:
            x_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt
            
        return x_prev

    def get_snr(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get signal-to-noise ratio for given timesteps."""
        alpha_bar = self._extract(self.alphas_cumprod, timesteps, timesteps.shape)
        snr = alpha_bar / (1 - alpha_bar)
        return snr

    def get_min_snr_weight(
        self, 
        timesteps: torch.Tensor, 
        gamma: float = 5.0
    ) -> torch.Tensor:
        """
        Min-SNR weighting for improved training.
        
        From "Efficient Diffusion Training via Min-SNR Weighting Strategy"
        """
        snr = self.get_snr(timesteps)
        weight = torch.clamp(snr, max=gamma) / snr
        return weight.view(-1, 1, 1, 1)


__all__ = ["DiffusionScheduler"]

