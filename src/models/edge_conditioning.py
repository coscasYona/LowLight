"""
Edge Conditioning Module for ControlNet-style conditioning.

Implements edge conditioning blocks that integrate edge maps
into the U-Net architecture using cross-attention and zero-convolution.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_conv(layer: nn.Module) -> None:
    """Xavier initialization for convolutional layers."""
    if isinstance(layer, nn.Conv2d):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def _zero_init_conv(layer: nn.Module) -> None:
    """Zero initialization for ControlNet-style stable training."""
    if isinstance(layer, nn.Conv2d):
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class ZeroConv2d(nn.Module):
    """
    Zero-initialized convolution for stable ControlNet training.
    
    Starts with zero weights so conditioning has no effect initially,
    then gradually learns to incorporate edge information.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            padding=kernel_size // 2
        )
        _zero_init_conv(self.conv)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EdgeCrossAttention(nn.Module):
    """
    Cross-attention between image features and edge features.
    
    Image features attend to edge features to incorporate
    structural information for edge-aware processing.
    """
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm_img = nn.GroupNorm(8, channels)
        self.norm_edge = nn.GroupNorm(8, channels)
        
        # Query from image features
        self.q = nn.Conv2d(channels, channels, 1)
        # Key and value from edge features
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        
        self.proj = nn.Conv2d(channels, channels, 1)
        
        self.apply(_init_conv)
    
    def forward(
        self, 
        img_feat: torch.Tensor, 
        edge_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-attention from image to edge features.
        
        Args:
            img_feat: Image features [B, C, H, W]
            edge_feat: Edge features [B, C, H, W]
            
        Returns:
            Attended features [B, C, H, W]
        """
        b, c, h, w = img_feat.shape
        
        # Normalize
        img_norm = self.norm_img(img_feat)
        edge_norm = self.norm_edge(edge_feat)
        
        # Compute Q, K, V
        q = self.q(img_norm)  # [B, C, H, W]
        k = self.k(edge_norm)
        v = self.v(edge_norm)
        
        # Reshape for multi-head attention
        q = q.view(b, self.num_heads, self.head_dim, h * w)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w)
        
        # Efficient linear attention using kernel trick
        # Use ELU + 1 feature map for positive attention
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        
        # Linear attention: Q @ (K^T @ V) instead of (Q @ K^T) @ V
        kv = torch.einsum('bhdn,bhdm->bhnm', k, v)  # [B, H, N, N]
        k_sum = k.sum(dim=-1, keepdim=True)  # [B, H, D, 1]
        
        # Normalize
        out = torch.einsum('bhdn,bhnm->bhdm', q, kv)
        out = out / (torch.einsum('bhdn,bhd->bhn', q, k_sum.squeeze(-1)).unsqueeze(2) + 1e-6)
        
        # Reshape back
        out = out.view(b, c, h, w)
        out = self.proj(out)
        
        return img_feat + out


class EdgeConditioningBlock(nn.Module):
    """
    Edge conditioning block for U-Net integration.
    
    Uses cross-attention to incorporate edge information into
    image features, with zero-conv for stable training.
    """
    
    def __init__(
        self, 
        channels: int, 
        edge_channels: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        
        # Project edge features to match channels
        self.edge_proj = nn.Sequential(
            nn.Conv2d(edge_channels, channels, 1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        
        # Cross-attention
        self.cross_attn = EdgeCrossAttention(channels, num_heads)
        
        # Zero-conv for stable training (ControlNet technique)
        self.zero_conv = ZeroConv2d(channels, channels)
        
        self.apply(_init_conv)
        # Re-apply zero init to zero_conv after general init
        _zero_init_conv(self.zero_conv.conv)
    
    def forward(
        self, 
        img_feat: torch.Tensor, 
        edge_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply edge conditioning to image features.
        
        Args:
            img_feat: Image features [B, C, H, W]
            edge_feat: Edge features [B, edge_channels, H, W]
            
        Returns:
            Conditioned features [B, C, H, W]
        """
        # Match spatial dimensions if needed
        if edge_feat.shape[2:] != img_feat.shape[2:]:
            edge_feat = F.interpolate(
                edge_feat, 
                size=img_feat.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Project edge features
        edge_proj = self.edge_proj(edge_feat)
        
        # Cross-attention
        attended = self.cross_attn(img_feat, edge_proj)
        
        # Apply zero-conv and add residual
        out = img_feat + self.zero_conv(attended - img_feat)
        
        return out


class EdgeConditioningEncoder(nn.Module):
    """
    Multi-scale edge feature encoder for U-Net conditioning.
    
    Produces edge features at multiple scales to match
    U-Net encoder/decoder levels.
    """
    
    def __init__(
        self, 
        base_channels: int = 64,
        num_levels: int = 4,
    ):
        super().__init__()
        self.num_levels = num_levels
        
        # Initial edge feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, base_channels, 3, padding=1),
            nn.SiLU(),
        )
        
        # Downsampling to create multi-scale features
        self.downs = nn.ModuleList()
        curr_ch = base_channels
        for i in range(num_levels - 1):
            self.downs.append(nn.Sequential(
                nn.Conv2d(curr_ch, curr_ch * 2, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(curr_ch * 2, curr_ch * 2, 3, padding=1),
                nn.SiLU(),
            ))
            curr_ch = curr_ch * 2
        
        self.apply(_init_conv)
    
    def forward(self, edge_map: torch.Tensor) -> list:
        """
        Encode edge map to multi-scale features.
        
        Args:
            edge_map: Edge map [B, 1, H, W]
            
        Returns:
            List of edge features at each scale
        """
        features = []
        
        x = self.stem(edge_map)
        features.append(x)
        
        for down in self.downs:
            x = down(x)
            features.append(x)
        
        return features


class SimpleEdgeConditioning(nn.Module):
    """
    Simple edge conditioning via channel concatenation.
    
    A lightweight alternative to cross-attention that simply
    concatenates edge features with image features.
    """
    
    def __init__(self, img_channels: int, edge_channels: int = 64):
        super().__init__()
        
        # Project concatenated features back to original channels
        self.proj = nn.Sequential(
            nn.Conv2d(img_channels + edge_channels, img_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(img_channels, img_channels, 3, padding=1),
        )
        
        # Zero-conv for gradual learning
        self.zero_conv = ZeroConv2d(img_channels, img_channels)
        
        self.apply(_init_conv)
        _zero_init_conv(self.zero_conv.conv)
    
    def forward(
        self, 
        img_feat: torch.Tensor, 
        edge_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply simple edge conditioning.
        
        Args:
            img_feat: Image features [B, C, H, W]
            edge_feat: Edge features [B, edge_channels, H, W]
            
        Returns:
            Conditioned features [B, C, H, W]
        """
        # Match spatial dimensions
        if edge_feat.shape[2:] != img_feat.shape[2:]:
            edge_feat = F.interpolate(
                edge_feat, 
                size=img_feat.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Concatenate and project
        combined = torch.cat([img_feat, edge_feat], dim=1)
        projected = self.proj(combined)
        
        # Zero-conv residual
        out = img_feat + self.zero_conv(projected - img_feat)
        
        return out


__all__ = [
    "ZeroConv2d",
    "EdgeCrossAttention",
    "EdgeConditioningBlock",
    "EdgeConditioningEncoder",
    "SimpleEdgeConditioning",
]

