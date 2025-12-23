"""
SlimUNet architecture for EMVA 1288 diffusion model.

A memory-efficient U-Net with linear attention blocks for
denoising diffusion probabilistic models.

Supports optional edge conditioning via ControlNet-style integration.
"""

import math
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_conv(layer: nn.Module) -> None:
    """Xavier initialization for convolutional layers."""
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps."""
    
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


class ResidualBlock(nn.Module):
    """Residual block with embedding injection."""
    
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
    Linear Attention: O(n) memory complexity instead of O(n²).
    
    Based on "Efficient Attention: Attention with Linear Complexities"
    Uses feature map phi(x) = elu(x) + 1 to ensure positive attention weights.
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
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        
        # Reshape for efficient computation
        q = q.reshape(b, c, h * w)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)
        
        # Linear attention: Q @ (K^T @ V)
        kv = torch.bmm(k, v.transpose(1, 2))
        k_sum = k.sum(dim=2, keepdim=True)
        kv = kv / (k_sum + 1e-6)
        
        out = torch.bmm(kv, q)
        out = out.reshape(b, c, h, w)
        out = self.proj(out)
        return out + x


class ChannelAttentionBlock(nn.Module):
    """
    Channel Attention: Very memory efficient, O(C) complexity.
    Based on SE-Net (Squeeze-and-Excitation).
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
        
        y = self.avg_pool(x_norm).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        return x + x_norm * y


class DownsampleBlock(nn.Module):
    """Downsample block with residual connection and optional attention."""
    
    def __init__(
        self, 
        in_ch: int, 
        out_ch: int, 
        emb_dim: int, 
        use_attn: bool, 
        attn_type: str = "linear"
    ):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch, emb_dim)
        
        if use_attn:
            if attn_type == "linear":
                self.attn = LinearAttentionBlock(out_ch)
            elif attn_type == "channel":
                self.attn = ChannelAttentionBlock(out_ch)
            else:
                self.attn = nn.Identity()
        else:
            self.attn = nn.Identity()
            
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
        self.apply(_init_conv)

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.res(x, emb)
        x = self.attn(x)
        skip = x
        x = self.down(x)
        return x, skip


class UpsampleBlock(nn.Module):
    """Upsample block with skip connection and optional attention."""
    
    def __init__(
        self, 
        in_ch: int, 
        out_ch: int, 
        emb_dim: int, 
        use_attn: bool, 
        attn_type: str = "linear"
    ):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch, emb_dim)
        
        if use_attn:
            if attn_type == "linear":
                self.attn = LinearAttentionBlock(out_ch)
            elif attn_type == "channel":
                self.attn = ChannelAttentionBlock(out_ch)
            else:
                self.attn = nn.Identity()
        else:
            self.attn = nn.Identity()
            
        self.apply(_init_conv)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:
        target_size = skip.shape[2:]
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, emb)
        x = self.attn(x)
        return x


class SlimUNet(nn.Module):
    """
    Memory-efficient U-Net architecture for diffusion models.
    
    Args:
        in_ch: Number of input channels
        out_ch: Number of output channels
        base_ch: Base channel count
        channel_mults: Channel multipliers for each level
        emb_dim: Embedding dimension for time/conditioning
        attn_type: Type of attention ("linear", "channel", or None)
        use_edge_cond: Whether to use edge conditioning (default: False)
        edge_channels: Number of edge feature channels (default: 64)
    """
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        base_ch: int,
        channel_mults: Iterable[int],
        emb_dim: int,
        attn_type: str = "linear",
        use_edge_cond: bool = False,
        edge_channels: int = 64,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.use_edge_cond = use_edge_cond
        self.edge_channels = edge_channels
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
        
        # Edge conditioning blocks (lazy import to avoid circular dependencies)
        if use_edge_cond:
            self._init_edge_conditioning(base_ch, mults, edge_channels)
    
    def _init_edge_conditioning(
        self, 
        base_ch: int, 
        mults: Tuple[int, ...], 
        edge_channels: int
    ):
        """Initialize edge conditioning blocks."""
        from models.edge_conditioning import EdgeConditioningBlock
        
        # Edge conditioning at each encoder level
        self.edge_cond_downs = nn.ModuleList()
        for mult in mults:
            ch = base_ch * mult
            self.edge_cond_downs.append(
                EdgeConditioningBlock(ch, edge_channels)
            )
        
        # Edge conditioning at decoder levels
        self.edge_cond_ups = nn.ModuleList()
        for mult in reversed(mults):
            ch = base_ch * mult
            self.edge_cond_ups.append(
                EdgeConditioningBlock(ch, edge_channels)
            )

    def forward(
        self, 
        x: torch.Tensor, 
        emb: torch.Tensor,
        edge_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional edge conditioning.
        
        Args:
            x: Input tensor [B, C, H, W]
            emb: Conditioning embedding [B, emb_dim]
            edge_feat: Optional edge features [B, edge_channels, H, W]
            
        Returns:
            Output tensor [B, out_ch, H, W]
        """
        emb = self.emb_proj(emb)
        x = self.in_conv(x)
        skips: List[torch.Tensor] = []
        
        for i, block in enumerate(self.downs):
            x, skip = block(x, emb)
            
            # Apply edge conditioning at encoder
            if self.use_edge_cond and edge_feat is not None:
                skip = self.edge_cond_downs[i](skip, edge_feat)
            
            skips.append(skip)

        x = self.mid(x, emb)

        for i, block in enumerate(self.ups):
            skip = skips.pop()
            x = block(x, skip, emb)
            
            # Apply edge conditioning at decoder
            if self.use_edge_cond and edge_feat is not None:
                x = self.edge_cond_ups[i](x, edge_feat)

        x = self.out_norm(x)
        x = F.silu(x)
        return self.out_conv(x)


__all__ = [
    "SlimUNet",
    "SinusoidalTimeEmbedding",
    "ResidualBlock",
    "LinearAttentionBlock",
    "ChannelAttentionBlock",
]

