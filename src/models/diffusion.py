"""
EMVA 1288 Physics-Guided Diffusion Model.

Main diffusion model that combines the U-Net backbone with physics-based
conditioning and noise generation following the EMVA 1288 standard.
"""

from typing import Dict, Iterable, Optional, Tuple, Union, Sequence

import torch
import torch.nn as nn

from models.unet import SlimUNet, SinusoidalTimeEmbedding
from models.physics_encoder import EMVA1288PhysicsEncoder
from models.scheduler import DiffusionScheduler
from models.noise_model import EMVA1288NoiseModel


class EMVA1288Diffusion(nn.Module):
    """
    Diffusion model using EMVA 1288 physics-based noise generation.
    
    The forward diffusion process can use actual CMOS noise 
    (shot + read + row + quantization) instead of simple Gaussian scaling.
    
    Args:
        in_channels: Number of input channels (4 for RGGB)
        out_channels: Number of output channels
        base_channels: Base channel count for U-Net
        channel_mults: Channel multipliers for each U-Net level
        num_steps: Number of diffusion timesteps
        time_embed_dim: Dimension of time embedding
        cond_embed_dim: Dimension of physics conditioning
        attn_type: Attention type ("linear", "channel")
        scheduler: Scheduler type ("ddpm", "ddim")
        schedule_type: Noise schedule ("linear", "cosine")
        camera_type: Camera type for EMVA 1288 noise model
        noise_code: Noise components to include (p=Poisson, r=row, q=quantization)
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 32,
        channel_mults: Iterable[int] = (1, 2, 4),
        num_steps: int = 1000,
        time_embed_dim: int = 64,
        cond_embed_dim: int = 64,
        attn_type: str = "linear",
        scheduler: str = "ddpm",
        schedule_type: str = "linear",
        camera_type: str = "SonyA7S2",
        noise_code: str = "prq",
        use_measurement_cond: bool = False,
    ):
        super().__init__()
        self.num_steps = max(1, num_steps)
        self.scheduler_type = scheduler
        self.camera_type = camera_type
        self.noise_code = noise_code
        self.use_measurement_cond = use_measurement_cond
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )
        
        # Physics conditioning
        self.physics_encoder = EMVA1288PhysicsEncoder(cond_embed_dim)
        emb_dim = time_embed_dim + cond_embed_dim
        
        # U-Net backbone
        self.unet = SlimUNet(
            in_ch=in_channels,
            out_ch=out_channels,
            base_ch=base_channels,
            channel_mults=channel_mults,
            emb_dim=emb_dim,
            attn_type=attn_type,
        )
        
        # Diffusion scheduler
        self.diffusion_scheduler = DiffusionScheduler(
            num_steps=self.num_steps,
            schedule_type=schedule_type,
        )
        
        # EMVA 1288 noise model for physics-based forward diffusion
        self.noise_model = EMVA1288NoiseModel(
            camera_type=camera_type,
            noise_code=noise_code,
        )
        
        # Register scheduler buffers for backward compatibility
        self._register_scheduler_buffers()
    
    def _register_scheduler_buffers(self):
        """Register scheduler buffers on this module for compatibility."""
        # Copy references from scheduler (they're already buffers there)
        self.register_buffer(
            "betas", 
            self.diffusion_scheduler.betas, 
            persistent=False
        )
        self.register_buffer(
            "alphas", 
            self.diffusion_scheduler.alphas, 
            persistent=False
        )
        self.register_buffer(
            "alphas_cumprod", 
            self.diffusion_scheduler.alphas_cumprod, 
            persistent=False
        )
        self.register_buffer(
            "sqrt_alphas_cumprod", 
            self.diffusion_scheduler.sqrt_alphas_cumprod, 
            persistent=False
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            self.diffusion_scheduler.sqrt_one_minus_alphas_cumprod,
            persistent=False,
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            self.diffusion_scheduler.sqrt_recip_alphas_cumprod,
            persistent=False,
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            self.diffusion_scheduler.sqrt_recipm1_alphas_cumprod,
            persistent=False,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        iso: Optional[torch.Tensor] = None,
        ratio: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        predict_noise: bool = False,
        camera_params: Optional[Dict] = None,
        cond_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            iso: ISO sensitivity [B] or [B, 1]
            ratio: Exposure ratio [B] or [B, 1]
            timesteps: Diffusion timesteps [B] (for training)
            num_steps: Number of sampling steps (for inference)
            predict_noise: If True, predict noise; else, sample denoised image
            camera_params: Optional camera parameters for noise model
            
        Returns:
            Predicted noise or denoised image
        """
        if predict_noise:
            if timesteps is None:
                raise ValueError("Timesteps required for noise prediction.")
            return self._predict_noise(x, timesteps, iso, ratio, camera_params, cond_image=cond_image)
        return self.sample(
            x,
            iso=iso,
            ratio=ratio,
            num_steps=num_steps,
            camera_params=camera_params,
            cond_image=cond_image,
        )
    
    def _prepare_iso_ratio(
        self, 
        iso: Optional[torch.Tensor], 
        ratio: Optional[torch.Tensor], 
        batch: int, 
        device: torch.device, 
        dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare ISO and ratio tensors."""
        if iso is None:
            iso = torch.zeros(batch, 1, device=device, dtype=dtype)
        else:
            iso = iso.view(batch, -1)
        if ratio is None:
            ratio = torch.ones(batch, 1, device=device, dtype=dtype)
        else:
            ratio = ratio.view(batch, -1)
        return iso, ratio
    
    def _extract(
        self, tensor: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size
    ) -> torch.Tensor:
        """Extract values from tensor at given timesteps."""
        return self.diffusion_scheduler._extract(tensor, timesteps, shape)
    
    def _time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute time embedding."""
        steps = max(self.num_steps - 1, 1)
        t = timesteps.float() / float(steps)
        return self.time_embed(t)
    
    def _predict_noise(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        iso: Optional[torch.Tensor],
        ratio: Optional[torch.Tensor],
        camera_params: Optional[Dict] = None,
        cond_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict noise for given noisy input."""
        b = x.size(0)
        iso, ratio = self._prepare_iso_ratio(iso, ratio, b, x.device, x.dtype)
        t_emb = self._time_embedding(timesteps)
        physics_emb = self.physics_encoder(iso, ratio, camera_params)
        emb = torch.cat([t_emb, physics_emb], dim=-1)
        
        if self.use_measurement_cond:
            # When measurement conditioning is enabled, U-Net expects doubled channels
            # Use x as fallback if cond_image is not provided (same as sample method)
            if cond_image is None:
                cond_image = x
            if cond_image.shape[0] != b:
                raise ValueError(f"cond_image batch mismatch: {cond_image.shape[0]} vs {b}")
            if cond_image.shape[2:] != x.shape[2:]:
                raise ValueError(f"cond_image spatial mismatch: {cond_image.shape[2:]} vs {x.shape[2:]}")
            x = torch.cat([x, cond_image], dim=1)
        
        return self.unet(x, emb)
    
    def generate_cmos_noise(
        self, 
        clean_image: torch.Tensor, 
        iso: torch.Tensor,
        ratio: torch.Tensor,
        camera_params: Optional[Union[Dict, Sequence[Dict]]] = None,
    ) -> torch.Tensor:
        """
        Generate CMOS noise using EMVA 1288 model.
        
        Args:
            clean_image: Clean image [B, C, H, W]
            iso: ISO values [B]
            ratio: Ratio values [B]
            camera_params: Optional camera parameters
            
        Returns:
            Noise tensor [B, C, H, W]
        """
        batch_size = clean_image.size(0)
        device = clean_image.device
        
        if camera_params is None:
            iso_np = iso.cpu().numpy().flatten()
            ratio_np = ratio.cpu().numpy().flatten()
            iso_val = int(iso_np[0]) if len(iso_np) > 0 else 6400
            ratio_val = float(ratio_np[0]) if len(ratio_np) > 0 else 200.0
            camera_params = self.noise_model.get_params(iso_val, ratio_val)
        
        # Generate noise for batch
        noisy_images = []
        for i in range(batch_size):
            img = clean_image[i:i+1]
            if isinstance(camera_params, (list, tuple)):
                params_i = camera_params[i]
            else:
                params_i = camera_params
            noisy = self.noise_model.generate_noise_torch(img, params_i)
            noisy_images.append(noisy)
        
        noisy_batch = torch.cat(noisy_images, dim=0)
        noise = noisy_batch - clean_image
        
        # Clean up NaN/inf
        noise = torch.where(torch.isfinite(noise), noise, torch.zeros_like(noise))
        noise = torch.clamp(noise, min=-10.0, max=10.0)
        
        return noise
    
    def q_sample(
        self, 
        x_start: torch.Tensor, 
        noise: torch.Tensor, 
        timesteps: torch.Tensor,
        iso: Optional[torch.Tensor] = None,
        ratio: Optional[torch.Tensor] = None,
        camera_params: Optional[Union[Dict, Sequence[Dict]]] = None,
        use_physics_noise: bool = True,
    ) -> torch.Tensor:
        """
        Forward diffusion with optional EMVA 1288 physics noise.
        
        Args:
            x_start: Clean images [B, C, H, W]
            noise: Base Gaussian noise [B, C, H, W]
            timesteps: Diffusion timesteps [B]
            iso: ISO sensitivity [B]
            ratio: Exposure ratio [B]
            camera_params: Camera noise parameters
            use_physics_noise: Whether to blend with CMOS physics noise
            
        Returns:
            Noisy images at given timesteps
        """
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape
        )
        
        if use_physics_noise and iso is not None and ratio is not None:
            cmos_noise = self.generate_cmos_noise(x_start, iso, ratio, camera_params)
            
            # Clean up base noise
            noise = torch.where(torch.isfinite(noise), noise, torch.zeros_like(noise))
            
            # Blend physics noise with standard noise
            blend_weight = 0.7
            blended_noise = blend_weight * cmos_noise + (1 - blend_weight) * noise
        else:
            noise = torch.where(torch.isfinite(noise), noise, torch.zeros_like(noise))
            blended_noise = noise
        
        sqrt_one_minus_alpha = torch.clamp(sqrt_one_minus_alpha, min=1e-6)
        result = sqrt_alpha * x_start + sqrt_one_minus_alpha * blended_noise
        
        # Final NaN/inf check
        result = torch.where(torch.isfinite(result), result, x_start)
        
        return result
    
    def sample(
        self,
        measurement: torch.Tensor,
        iso: Optional[torch.Tensor] = None,
        ratio: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        eta: float = 0.0,
        camera_params: Optional[Dict] = None,
        cond_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample using DDPM or DDIM.
        
        Args:
            measurement: Noisy input [B, C, H, W]
            iso: ISO sensitivity
            ratio: Exposure ratio
            num_steps: Number of sampling steps
            eta: DDIM noise level (0=deterministic, 1=DDPM)
            camera_params: Camera parameters
            
        Returns:
            Denoised image [B, C, H, W]
        """
        steps = min(num_steps or self.num_steps, self.num_steps)
        x = measurement
        if cond_image is None:
            cond_image = measurement
        
        use_ddim = self.scheduler_type == "ddim" or eta == 0.0
        
        for step in reversed(range(steps)):
            t = torch.full((x.size(0),), step, device=x.device, dtype=torch.long)
            eps = self._predict_noise(x, t, iso, ratio, camera_params, cond_image=cond_image)
            
            if use_ddim:
                prev_t = torch.clamp(t - 1, min=0)
                x = self.diffusion_scheduler.ddim_step(x, eps, t, prev_t, eta)
            else:
                x = self.diffusion_scheduler.ddpm_step(x, eps, t, add_noise=(step > 0))
        
        return torch.clamp(x, 0.0, 1.0)
    
    def physics_noise_scale(
        self, iso: torch.Tensor, ratio: torch.Tensor
    ) -> torch.Tensor:
        """Compute physics-based noise scale for conditioning."""
        iso = torch.where(torch.isfinite(iso), iso, torch.ones_like(iso) * 6400.0)
        ratio = torch.where(torch.isfinite(ratio), ratio, torch.ones_like(ratio) * 200.0)
        iso = torch.clamp(iso.view(iso.size(0), -1), min=1.0, max=1e6)
        ratio = torch.clamp(ratio.view(ratio.size(0), -1), min=1.0, max=1e6)
        
        iso_norm = iso / 6400.0
        ratio_norm = ratio / 300.0
        shot = torch.sqrt(torch.clamp(iso_norm * ratio_norm, min=1e-6))
        read = torch.sqrt(torch.clamp(iso_norm, min=1e-6)) * 0.1
        scale = torch.clamp(shot + read, min=1e-3)
        scale = torch.where(torch.isfinite(scale), scale, torch.ones_like(scale) * 1e-3)
        
        return scale.view(-1, 1, 1, 1)


__all__ = ["EMVA1288Diffusion"]

