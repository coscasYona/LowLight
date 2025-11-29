import math
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class PhysicsEncoder(nn.Module):
    def __init__(self, cond_dim: int):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(5, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )
        self.apply(_init_conv)

    def forward(self, iso: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
        iso = torch.log1p(torch.clamp(iso, min=0.0))
        ratio = torch.log1p(torch.clamp(ratio, min=0.0))
        iso_norm = iso / 10.0
        ratio_norm = ratio / 10.0
        shot = torch.sqrt(torch.clamp(iso_norm * ratio_norm, min=1e-4))
        read = torch.log1p(iso_norm + 1e-4)
        inverse_ratio = torch.reciprocal(torch.clamp(ratio + 1e-4, min=1e-4))
        feats = torch.cat([iso_norm, ratio_norm, shot, read, inverse_ratio], dim=-1)
        return self.embed(feats)


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
    """
    Linear Attention: O(n) memory complexity instead of O(n²)
    Based on "Efficient Attention: Attention with Linear Complexities"
    Uses feature map phi(x) = elu(x) + 1 to ensure positive attention weights
    """
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
        
        # Linear attention: use feature map phi(x) = elu(x) + 1
        # This ensures positive values and allows us to compute attention in O(n)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        
        # Reshape for efficient computation
        q = q.reshape(b, c, h * w)  # [B, C, N]
        k = k.reshape(b, c, h * w)  # [B, C, N]
        v = v.reshape(b, c, h * w)  # [B, C, N]
        
        # Linear attention: (Q @ K^T) @ V = Q @ (K^T @ V)
        # Compute K^T @ V first: [B, C, N] @ [B, N, C] = [B, C, C]
        kv = torch.bmm(k, v.transpose(1, 2))  # [B, C, C]
        
        # Normalize by sum of keys
        k_sum = k.sum(dim=2, keepdim=True)  # [B, C, 1]
        kv = kv / (k_sum + 1e-6)
        
        # Compute Q @ (K^T @ V): [B, C, N] @ [B, C, C] = [B, C, N]
        out = torch.bmm(kv, q)  # [B, C, N]
        out = out.reshape(b, c, h, w)
        out = self.proj(out)
        return out + x


class ChannelAttentionBlock(nn.Module):
    """
    Channel Attention: Very memory efficient, O(C) complexity
    Based on SE-Net (Squeeze-and-Excitation)
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        self.apply(_init_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        
        # Channel attention
        y = self.avg_pool(x_norm).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        return x + x_norm * y


class AttentionBlock(nn.Module):
    """
    Standard self-attention (O(n²) memory) - kept for compatibility
    Can optionally use Flash Attention if available
    """
    def __init__(self, channels: int, use_flash: bool = False):
        super().__init__()
        self.use_flash = use_flash
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.apply(_init_conv)
        
        # Try to import flash attention
        self.has_flash = False
        if use_flash:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
                self.has_flash = True
            except ImportError:
                self.has_flash = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        
        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)
        
        if self.has_flash and self.use_flash:
            # Flash Attention path (if available)
            # Reshape for flash attention: [B, C, H, W] -> [B, H*W, C]
            q = q.reshape(b, c, h * w).permute(0, 2, 1).contiguous()
            k = k.reshape(b, c, h * w).permute(0, 2, 1).contiguous()
            v = v.reshape(b, c, h * w).permute(0, 2, 1).contiguous()
            
            out = self.flash_attn_func(q, k, v)
            out = out.permute(0, 2, 1).reshape(b, c, h, w)
        else:
            # Standard attention path
            q = q.reshape(b, c, h * w).permute(0, 2, 1)
            k = k.reshape(b, c, h * w)
            v = v.reshape(b, c, h * w).permute(0, 2, 1)

            attn = torch.softmax(torch.bmm(q, k) / math.sqrt(c), dim=-1)
            out = torch.bmm(attn, v).permute(0, 2, 1).reshape(b, c, h, w)
        
        out = self.proj(out)
        return out + x


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, use_attn: bool, attn_type: str = "linear"):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch, emb_dim)
        if use_attn:
            if attn_type == "linear":
                self.attn = LinearAttentionBlock(out_ch)
            elif attn_type == "channel":
                self.attn = ChannelAttentionBlock(out_ch)
            elif attn_type == "standard":
                self.attn = AttentionBlock(out_ch, use_flash=False)
            elif attn_type == "flash":
                self.attn = AttentionBlock(out_ch, use_flash=True)
            else:
                raise ValueError(f"Unknown attention type: {attn_type}")
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
            elif attn_type == "channel":
                self.attn = ChannelAttentionBlock(out_ch)
            elif attn_type == "standard":
                self.attn = AttentionBlock(out_ch, use_flash=False)
            elif attn_type == "flash":
                self.attn = AttentionBlock(out_ch, use_flash=True)
            else:
                raise ValueError(f"Unknown attention type: {attn_type}")
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


class PhysicsGuidedStableDiffusion(nn.Module):
    """
    Improved diffusion-inspired denoiser with memory-efficient attention.
    Supports multiple attention types:
    - "linear": O(n) memory complexity (default, recommended)
    - "channel": O(C) memory complexity (most efficient)
    - "standard": O(n²) memory complexity (original)
    - "flash": Flash Attention if available (fastest, requires flash-attn package)
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
    ):
        super().__init__()
        self.num_steps = max(1, num_steps)
        self.scheduler = scheduler
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )
        self.physics_encoder = PhysicsEncoder(cond_embed_dim)
        emb_dim = time_embed_dim + cond_embed_dim

        self.unet = SlimUNet(
            in_ch=in_channels,
            out_ch=out_channels,
            base_ch=base_channels,
            channel_mults=channel_mults,
            emb_dim=emb_dim,
            attn_type=attn_type,
        )
        
        # Improved noise schedule
        if scheduler == "ddpm":
            beta_start, beta_end = 0.0001, 0.02
        elif scheduler == "ddim":
            # DDIM uses same schedule but different sampling
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
        # For DDIM
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
    ) -> torch.Tensor:
        if predict_noise:
            if timesteps is None:
                raise ValueError("Timesteps required for noise prediction.")
            return self._predict_eps(x, timesteps, iso, ratio)
        return self.sample(x, iso=iso, ratio=ratio, num_steps=num_steps)

    def _prepare_iso_ratio(
        self, iso: Optional[torch.Tensor], ratio: Optional[torch.Tensor], batch: int, device: torch.device, dtype: torch.dtype
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
    ) -> torch.Tensor:
        b = x.size(0)
        iso, ratio = self._prepare_iso_ratio(iso, ratio, b, x.device, x.dtype)
        t_emb = self._time_embedding(timesteps)
        physics_emb = self.physics_encoder(iso, ratio)
        emb = torch.cat([t_emb, physics_emb], dim=-1)
        return self.unet(x, emb)

    def sample(
        self,
        measurement: torch.Tensor,
        iso: Optional[torch.Tensor] = None,
        ratio: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Sample using DDPM (eta=1.0) or DDIM (eta=0.0) scheduler
        eta=0.0: DDIM (deterministic, faster)
        eta=1.0: DDPM (stochastic, original)
        """
        steps = num_steps or self.num_steps
        x = measurement
        
        if self.scheduler == "ddim" or eta == 0.0:
            # DDIM sampling (deterministic, faster)
            for step in reversed(range(steps)):
                t = torch.full(
                    (x.size(0),),
                    step,
                    device=x.device,
                    dtype=torch.long,
                )
                eps = self._predict_eps(x, t, iso, ratio)
                
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
            # DDPM sampling (original)
            for step in reversed(range(steps)):
                t = torch.full(
                    (x.size(0),),
                    step,
                    device=x.device,
                    dtype=torch.long,
                )
                eps = self._predict_eps(x, t, iso, ratio)
                beta = self._extract(self.betas, t, x.shape)
                alpha = self._extract(self.alphas, t, x.shape)
                alpha_bar = self._extract(self.alphas_cumprod, t, x.shape)
                sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1 - alpha_bar, min=1e-5))
                x = (1 / torch.sqrt(alpha)) * (x - (beta / sqrt_one_minus_alpha_bar) * eps)
                if step > 0:
                    noise = torch.randn_like(x)
                    x = x + torch.sqrt(beta) * noise
        
        return torch.clamp(x, 0.0, 1.0)

    def q_sample(self, x_start: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape
        )
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def physics_noise_scale(self, iso: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
        iso = torch.clamp(iso.view(iso.size(0), -1), min=1.0)
        ratio = torch.clamp(ratio.view(ratio.size(0), -1), min=1.0)
        iso_norm = iso / 6400.0
        ratio_norm = ratio / 300.0
        shot = torch.sqrt(torch.clamp(iso_norm * ratio_norm, min=1e-6))
        read = torch.sqrt(torch.clamp(iso_norm, min=1e-6)) * 0.1
        scale = torch.clamp(shot + read, min=1e-3)
        return scale.view(-1, 1, 1, 1)


# Backwards compatibility
class Net(PhysicsGuidedStableDiffusion):
    pass


__all__ = ["PhysicsGuidedStableDiffusion", "Net"]
