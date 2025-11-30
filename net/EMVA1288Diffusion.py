"""
EMVA 1288 Physics-Guided Diffusion Model for CMOS Noise Denoising

This module implements a diffusion model that uses the EMVA 1288 standard
to model CMOS camera noise accurately. The forward diffusion process uses
the actual noise distribution (shot + read + row + quantization) instead
of simple Gaussian scaling.

Based on: https://kmdouglass.github.io/posts/modeling-noise-for-image-simulations/
"""

import math
from typing import Iterable, List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist

# Import noise generation from data_process
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.process import generate_noisy_torch, sample_params_max, get_camera_noisy_params_max


def _init_conv(layer: nn.Module) -> None:
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, device=device) * -(math.log(10000.0) / (half_dim - 1))
        )
        args = timesteps[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class EMVA1288PhysicsEncoder(nn.Module):
    """
    Physics encoder based on EMVA 1288 standard.
    Encodes camera parameters (ISO, ratio, exposure) into features that
    represent the noise characteristics.
    """
    def __init__(self, cond_dim: int):
        super().__init__()
        # Input features: [iso_norm, ratio_norm, log_K, log_sigGs, log_sigR, 
        #                  shot_variance, read_variance, snr]
        self.embed = nn.Sequential(
            nn.Linear(8, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.apply(_init_conv)

    def forward(self, iso: torch.Tensor, ratio: torch.Tensor, 
                camera_params: Optional[Dict] = None) -> torch.Tensor:
        """
        Encode ISO and ratio into physics-based features.
        
        Args:
            iso: ISO sensitivity [B]
            ratio: Exposure ratio [B]
            camera_params: Optional dict with K, sigGs, sigR, etc.
        """
        batch_size = iso.size(0)
        device = iso.device
        
        iso = torch.clamp(iso.view(batch_size, -1), min=1.0)
        ratio = torch.clamp(ratio.view(batch_size, -1), min=1.0)
        
        # Normalize
        iso_norm = iso / 6400.0
        ratio_norm = ratio / 300.0
        
        if camera_params is not None:
            # Use provided camera parameters
            K = torch.tensor(camera_params['K'], device=device, dtype=iso.dtype).view(-1, 1)
            sigGs = torch.tensor(camera_params['sigGs'], device=device, dtype=iso.dtype).view(-1, 1)
            sigR = torch.tensor(camera_params.get('sigR', 0.0), device=device, dtype=iso.dtype).view(-1, 1)
        else:
            # Estimate from ISO (fallback - should use actual calibration)
            log_K = torch.log(iso_norm * 8.0 + 0.1)
            K = torch.exp(log_K)
            sigGs = torch.exp(log_K * 0.85 - 0.18)
            sigR = torch.exp(log_K * 0.88 - 2.11)
        
        # EMVA 1288 calculations
        # Shot noise variance is proportional to signal: var_shot = signal / K
        # For normalized images, approximate signal as iso_norm * ratio_norm
        shot_variance = torch.clamp(iso_norm * ratio_norm / (K + 1e-8), min=1e-6)
        read_variance = sigGs ** 2
        
        # SNR approximation: SNR ≈ signal / sqrt(shot_var + read_var)
        snr = torch.clamp(
            (iso_norm * ratio_norm) / torch.sqrt(shot_variance + read_variance + 1e-8),
            min=1e-6
        )
        
        # Feature vector
        features = torch.cat([
            iso_norm,
            ratio_norm,
            torch.log1p(K),
            torch.log1p(sigGs),
            torch.log1p(sigR + 1e-6),
            torch.log1p(shot_variance),
            torch.log1p(read_variance),
            torch.log1p(snr),
        ], dim=-1)
        
        return self.embed(features)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, groups: int = 8):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

        self.apply(_init_conv)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        emb_out = self.emb_proj(F.silu(emb))[:, :, None, None]
        h = h + emb_out

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class LinearAttentionBlock(nn.Module):
    """Linear Attention: O(n) memory complexity"""
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.apply(_init_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        
        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)
        
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        
        q = q.reshape(b, c, h * w)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)
        
        kv = torch.bmm(k, v.transpose(1, 2))
        k_sum = k.sum(dim=2, keepdim=True)
        kv = kv / (k_sum + 1e-6)
        
        out = torch.bmm(kv, q)
        out = out.reshape(b, c, h, w)
        out = self.proj(out)
        return out + x


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, use_attn: bool, attn_type: str = "linear"):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch, emb_dim)
        if use_attn:
            if attn_type == "linear":
                self.attn = LinearAttentionBlock(out_ch)
            else:
                self.attn = nn.Identity()
        else:
            self.attn = nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
        self.apply(_init_conv)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.res(x, emb)
        x = self.attn(x)
        skip = x
        x = self.down(x)
        return x, skip


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, use_attn: bool, attn_type: str = "linear"):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch, emb_dim)
        if use_attn:
            if attn_type == "linear":
                self.attn = LinearAttentionBlock(out_ch)
            else:
                self.attn = nn.Identity()
        else:
            self.attn = nn.Identity()
        self.apply(_init_conv)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        target_size = skip.shape[2:]
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, emb)
        x = self.attn(x)
        return x


class SlimUNet(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        base_ch: int,
        channel_mults: Iterable[int],
        emb_dim: int,
        attn_type: str = "linear",
    ):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, emb_dim))

        self.downs = nn.ModuleList()
        self.skip_channels: List[int] = []
        curr_ch = base_ch
        mults = tuple(channel_mults)
        for mult in mults:
            out_ch_stage = base_ch * mult
            use_attn = mult == mults[-1]
            block = DownsampleBlock(curr_ch, out_ch_stage, emb_dim, use_attn, attn_type)
            self.downs.append(block)
            self.skip_channels.append(out_ch_stage)
            curr_ch = out_ch_stage

        self.mid = ResidualBlock(curr_ch, curr_ch, emb_dim)

        self.ups = nn.ModuleList()
        for skip_ch in reversed(self.skip_channels):
            use_attn = skip_ch == self.skip_channels[0]
            block = UpsampleBlock(curr_ch + skip_ch, skip_ch, emb_dim, use_attn, attn_type)
            self.ups.append(block)
            curr_ch = skip_ch

        self.out_norm = nn.GroupNorm(8, curr_ch)
        self.out_conv = nn.Conv2d(curr_ch, out_ch, 3, padding=1)
        self.apply(_init_conv)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        emb = self.emb_proj(emb)
        x = self.in_conv(x)
        skips: List[torch.Tensor] = []
        for block in self.downs:
            x, skip = block(x, emb)
            skips.append(skip)

        x = self.mid(x, emb)

        for block in self.ups:
            skip = skips.pop()
            x = block(x, skip, emb)

        x = self.out_norm(x)
        x = F.silu(x)
        return self.out_conv(x)


class EMVA1288Diffusion(nn.Module):
    """
    Diffusion model using EMVA 1288 physics-based noise generation.
    The forward diffusion process uses actual CMOS noise (shot + read + row + quant)
    instead of simple Gaussian scaling.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 32,
        channel_mults: Iterable[int] = (1, 2, 4),
        num_steps: int = 4,
        time_embed_dim: int = 64,
        cond_embed_dim: int = 64,
        attn_type: str = "linear",
        scheduler: str = "ddpm",
        camera_type: str = "SonyA7S2",
        noise_code: str = "prq",  # p=Poisson shot, r=row, q=quantization
    ):
        super().__init__()
        self.num_steps = max(1, num_steps)
        self.scheduler = scheduler
        self.camera_type = camera_type
        self.noise_code = noise_code
        
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )
        self.physics_encoder = EMVA1288PhysicsEncoder(cond_embed_dim)
        emb_dim = time_embed_dim + cond_embed_dim

        self.unet = SlimUNet(
            in_ch=in_channels,
            out_ch=out_channels,
            base_ch=base_channels,
            channel_mults=channel_mults,
            emb_dim=emb_dim,
            attn_type=attn_type,
        )
        
        # Noise schedule
        if scheduler == "ddpm":
            beta_start, beta_end = 0.0001, 0.02
        elif scheduler == "ddim":
            beta_start, beta_end = 0.0001, 0.02
        else:
            beta_start, beta_end = 0.0001, 0.02
            
        betas = torch.linspace(beta_start, beta_end, steps=self.num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer(
            "sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=False
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

    def forward(
        self,
        x: torch.Tensor,
        iso: Optional[torch.Tensor] = None,
        ratio: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        predict_noise: bool = False,
        camera_params: Optional[Dict] = None,
    ) -> torch.Tensor:
        if predict_noise:
            if timesteps is None:
                raise ValueError("Timesteps required for noise prediction.")
            return self._predict_eps(x, timesteps, iso, ratio, camera_params)
        return self.sample(x, iso=iso, ratio=ratio, num_steps=num_steps, camera_params=camera_params)

    def _prepare_iso_ratio(
        self, iso: Optional[torch.Tensor], ratio: Optional[torch.Tensor], 
        batch: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if iso is None:
            iso = torch.zeros(batch, 1, device=device, dtype=dtype)
        else:
            iso = iso.view(batch, -1)
        if ratio is None:
            ratio = torch.ones(batch, 1, device=device, dtype=dtype)
        else:
            ratio = ratio.view(batch, -1)
        return iso, ratio

    def _extract(self, tensor: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        out = tensor.to(timesteps.device)[timesteps]
        return out.view(-1, *([1] * (len(shape) - 1)))

    def _time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        steps = max(self.num_steps - 1, 1)
        t = timesteps.float() / float(steps)
        return self.time_embed(t)

    def _predict_eps(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        iso: Optional[torch.Tensor],
        ratio: Optional[torch.Tensor],
        camera_params: Optional[Dict] = None,
    ) -> torch.Tensor:
        b = x.size(0)
        iso, ratio = self._prepare_iso_ratio(iso, ratio, b, x.device, x.dtype)
        t_emb = self._time_embedding(timesteps)
        physics_emb = self.physics_encoder(iso, ratio, camera_params)
        emb = torch.cat([t_emb, physics_emb], dim=-1)
        return self.unet(x, emb)

    def generate_cmos_noise(
        self, 
        clean_image: torch.Tensor, 
        iso: torch.Tensor,
        ratio: torch.Tensor,
        camera_params: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Generate CMOS noise using EMVA 1288 model.
        Uses generate_noisy_torch from data_process for accurate physics modeling.
        """
        batch_size = clean_image.size(0)
        device = clean_image.device
        
        # Get camera parameters
        if camera_params is None:
            # Sample parameters based on ISO
            iso_np = iso.cpu().numpy().flatten()
            ratio_np = ratio.cpu().numpy().flatten()
            
            # Use first ISO value to get params (batch should have same camera)
            iso_val = int(iso_np[0]) if len(iso_np) > 0 else 6400
            camera_params = sample_params_max(
                camera_type=self.camera_type,
                iso=iso_val,
                ratio=float(ratio_np[0]) if len(ratio_np) > 0 else None
            )
        
        # Temporarily set DEVICE in process module to match our device
        import data_process.process as process_module
        original_device = getattr(process_module, 'DEVICE', None)
        process_module.DEVICE = device
        
        try:
            # Generate noise for each sample in batch
            noisy_images = []
            for i in range(batch_size):
                img = clean_image[i:i+1]  # [1, C, H, W]
                
                # Use generate_noisy_torch with physics parameters
                noisy = generate_noisy_torch(
                    img,
                    camera_type=self.camera_type,
                    noise_code=self.noise_code,
                    param=camera_params,
                    MultiFrameMean=1,
                    ori=False,
                    clip=False
                )
                noisy_images.append(noisy)
            
            noisy_batch = torch.cat(noisy_images, dim=0)
            
            # Extract just the noise (difference from clean)
            noise = noisy_batch - clean_image
            return noise
        finally:
            # Restore original device
            if original_device is not None:
                process_module.DEVICE = original_device

    def q_sample(
        self, 
        x_start: torch.Tensor, 
        noise: torch.Tensor, 
        timesteps: torch.Tensor,
        iso: Optional[torch.Tensor] = None,
        ratio: Optional[torch.Tensor] = None,
        camera_params: Optional[Dict] = None,
        use_physics_noise: bool = True,
    ) -> torch.Tensor:
        """
        Forward diffusion process using EMVA 1288 CMOS noise.
        
        Args:
            x_start: Clean images
            noise: Base Gaussian noise (for blending)
            timesteps: Diffusion timesteps
            iso: ISO sensitivity
            ratio: Exposure ratio
            camera_params: Camera noise parameters
            use_physics_noise: If True, use CMOS noise; if False, use standard Gaussian
        """
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape
        )
        
        if use_physics_noise and iso is not None and ratio is not None:
            # Generate CMOS noise matching physics model
            cmos_noise = self.generate_cmos_noise(x_start, iso, ratio, camera_params)
            
            # Blend physics noise with standard noise based on timestep
            # Early timesteps: more physics noise (realistic)
            # Later timesteps: more standard noise (for diffusion schedule)
            # IMPORTANT: Blend unscaled noise first, then apply sqrt_one_minus_alpha once
            # Using a fixed blending weight to avoid double-scaling
            # (If we used sqrt_one_minus_alpha as weight, it would be scaled twice)
            blend_weight = 0.7  # 70% physics noise, 30% standard noise
            blended_noise = blend_weight * cmos_noise + (1 - blend_weight) * noise
        else:
            blended_noise = noise
        
        # Apply scaling once to the final blended noise (no double-scaling)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * blended_noise

    def sample(
        self,
        measurement: torch.Tensor,
        iso: Optional[torch.Tensor] = None,
        ratio: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        eta: float = 0.0,
        camera_params: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Sample using DDPM (eta=1.0) or DDIM (eta=0.0) scheduler.
        """
        steps = num_steps or self.num_steps
        x = measurement
        
        if self.scheduler == "ddim" or eta == 0.0:
            # DDIM sampling
            for step in reversed(range(steps)):
                t = torch.full(
                    (x.size(0),),
                    step,
                    device=x.device,
                    dtype=torch.long,
                )
                eps = self._predict_eps(x, t, iso, ratio, camera_params)
                
                alpha_bar_t = self._extract(self.alphas_cumprod, t, x.shape)
                alpha_bar_t_prev = self._extract(
                    self.alphas_cumprod,
                    torch.clamp(t - 1, min=0),
                    x.shape
                )
                
                pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
                
                dir_xt = torch.sqrt(1.0 - alpha_bar_t_prev - eta ** 2 * (1.0 - alpha_bar_t)) * eps
                random_noise = eta * torch.sqrt(1.0 - alpha_bar_t) * torch.randn_like(x) if step > 0 else torch.zeros_like(x)
                
                x = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + random_noise
        else:
            # DDPM sampling
            for step in reversed(range(steps)):
                t = torch.full(
                    (x.size(0),),
                    step,
                    device=x.device,
                    dtype=torch.long,
                )
                eps = self._predict_eps(x, t, iso, ratio, camera_params)
                
                beta = self._extract(self.betas, t, x.shape)
                alpha = self._extract(self.alphas, t, x.shape)
                alpha_bar = self._extract(self.alphas_cumprod, t, x.shape)
                sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1 - alpha_bar, min=1e-5))
                x = (1 / torch.sqrt(alpha)) * (x - (beta / sqrt_one_minus_alpha_bar) * eps)
                if step > 0:
                    noise = torch.randn_like(x)
                    x = x + torch.sqrt(beta) * noise
        
        return torch.clamp(x, 0.0, 1.0)


__all__ = ["EMVA1288Diffusion"]

