"""
Edge Detection Module for ControlNet-style conditioning.

Implements differentiable edge detection methods:
- Canny edge detector (using Sobel + non-maximum suppression)
- HED-like lightweight edge detector
- Multi-scale edge detection
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_conv(layer: nn.Module) -> None:
    """Xavier initialization for convolutional layers."""
    if isinstance(layer, nn.Conv2d):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class SobelEdgeDetector(nn.Module):
    """
    Differentiable Sobel edge detector.
    
    Computes gradient magnitude using Sobel filters.
    """
    
    def __init__(self):
        super().__init__()
        # Sobel kernels for x and y gradients
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x, persistent=False)
        self.register_buffer('sobel_y', sobel_y, persistent=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute edge magnitude.
        
        Args:
            x: Input tensor [B, C, H, W] or [B, 1, H, W]
            
        Returns:
            Edge magnitude [B, 1, H, W]
        """
        # Convert to grayscale if multi-channel
        if x.size(1) > 1:
            # Simple luminance conversion
            x = x.mean(dim=1, keepdim=True)
        
        # Apply Sobel filters
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        
        # Compute gradient magnitude
        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        
        return magnitude


class CannyEdgeDetector(nn.Module):
    """
    Differentiable Canny-like edge detector.
    
    Uses Gaussian smoothing, Sobel gradients, and soft non-maximum suppression.
    """
    
    def __init__(
        self, 
        low_threshold: float = 0.1,
        high_threshold: float = 0.3,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        
        # Create Gaussian kernel for smoothing
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        gaussian = self._create_gaussian_kernel(kernel_size, sigma)
        self.register_buffer('gaussian', gaussian, persistent=False)
        self.kernel_size = kernel_size
        
        # Sobel kernels
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x, persistent=False)
        self.register_buffer('sobel_y', sobel_y, persistent=False)
    
    def _create_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create 2D Gaussian kernel."""
        x = torch.arange(size).float() - size // 2
        gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        gauss_2d = gauss_1d.outer(gauss_1d)
        gauss_2d = gauss_2d / gauss_2d.sum()
        return gauss_2d.view(1, 1, size, size)
    
    def _soft_nms(
        self, 
        magnitude: torch.Tensor, 
        gx: torch.Tensor, 
        gy: torch.Tensor
    ) -> torch.Tensor:
        """
        Soft non-maximum suppression (differentiable approximation).
        
        Instead of hard NMS, uses gradient-weighted attenuation.
        """
        b, c, h, w = magnitude.shape
        
        # Simplified soft NMS using max pooling comparison
        # This is more robust and efficient than grid sampling
        
        # Get local max in 3x3 neighborhood
        local_max = F.max_pool2d(magnitude, kernel_size=3, stride=1, padding=1)
        
        # Soft comparison: how close is this pixel to the local max?
        # Values close to local max get weight close to 1
        diff = magnitude - local_max
        soft_max = torch.sigmoid(20 * diff + 5)  # Shifted sigmoid for softer transition
        
        return magnitude * soft_max
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Canny-like edges.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Edge map [B, 1, H, W] in range [0, 1]
        """
        # Convert to grayscale if multi-channel
        if x.size(1) > 1:
            x = x.mean(dim=1, keepdim=True)
        
        # Gaussian smoothing
        padding = self.kernel_size // 2
        x_smooth = F.conv2d(x, self.gaussian, padding=padding)
        
        # Compute gradients
        gx = F.conv2d(x_smooth, self.sobel_x, padding=1)
        gy = F.conv2d(x_smooth, self.sobel_y, padding=1)
        
        # Gradient magnitude
        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        
        # Normalize to [0, 1]
        magnitude = magnitude / (magnitude.max() + 1e-8)
        
        # Soft non-maximum suppression
        nms_result = self._soft_nms(magnitude, gx, gy)
        
        # Soft hysteresis thresholding using sigmoid
        # High confidence edges
        high_edges = torch.sigmoid(20 * (nms_result - self.high_threshold))
        # Low confidence edges
        low_edges = torch.sigmoid(20 * (nms_result - self.low_threshold))
        
        # Combine with preference for high edges
        edges = 0.7 * high_edges + 0.3 * low_edges
        
        return edges


class HEDBlock(nn.Module):
    """
    Single HED-style side output block.
    """
    
    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.apply(_init_conv)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LightweightHED(nn.Module):
    """
    Lightweight Holistically-Nested Edge Detection.
    
    A simplified version of HED that uses a small encoder
    with multi-scale side outputs.
    """
    
    def __init__(self, in_channels: int = 1):
        super().__init__()
        
        # Encoder blocks (lightweight)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Side outputs
        self.side1 = HEDBlock(16, 1)
        self.side2 = HEDBlock(32, 1)
        self.side3 = HEDBlock(64, 1)
        self.side4 = HEDBlock(64, 1)
        
        # Fusion layer
        self.fuse = nn.Conv2d(4, 1, 1)
        
        self.apply(_init_conv)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute HED edges.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Edge map [B, 1, H, W] in range [0, 1]
        """
        h, w = x.shape[2:]
        
        # Convert to grayscale if multi-channel
        if x.size(1) > 1:
            x = x.mean(dim=1, keepdim=True)
        
        # Encoder with side outputs
        c1 = self.conv1(x)
        s1 = self.side1(c1)
        
        c2 = self.conv2(self.pool1(c1))
        s2 = self.side2(c2)
        s2 = F.interpolate(s2, size=(h, w), mode='bilinear', align_corners=False)
        
        c3 = self.conv3(self.pool2(c2))
        s3 = self.side3(c3)
        s3 = F.interpolate(s3, size=(h, w), mode='bilinear', align_corners=False)
        
        c4 = self.conv4(self.pool3(c3))
        s4 = self.side4(c4)
        s4 = F.interpolate(s4, size=(h, w), mode='bilinear', align_corners=False)
        
        # Fuse all side outputs
        fused = torch.cat([s1, s2, s3, s4], dim=1)
        edges = self.fuse(fused)
        
        # Apply sigmoid for [0, 1] output
        edges = torch.sigmoid(edges)
        
        return edges


class MultiScaleEdgeDetector(nn.Module):
    """
    Multi-scale edge detection combining multiple methods.
    
    Combines Canny and HED at multiple scales for robust edge detection.
    """
    
    def __init__(
        self,
        use_canny: bool = True,
        use_hed: bool = True,
        scales: Tuple[float, ...] = (1.0, 0.5, 0.25),
        canny_low: float = 0.1,
        canny_high: float = 0.3,
    ):
        super().__init__()
        self.use_canny = use_canny
        self.use_hed = use_hed
        self.scales = scales
        
        if use_canny:
            self.canny = CannyEdgeDetector(
                low_threshold=canny_low,
                high_threshold=canny_high,
            )
        
        if use_hed:
            self.hed = LightweightHED(in_channels=1)
        
        # Fusion for multi-scale results
        num_outputs = (len(scales) if use_canny else 0) + (len(scales) if use_hed else 0)
        if num_outputs > 1:
            self.fusion = nn.Conv2d(num_outputs, 1, 1)
            nn.init.constant_(self.fusion.weight, 1.0 / num_outputs)
            nn.init.zeros_(self.fusion.bias)
        else:
            self.fusion = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale edges.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Edge map [B, 1, H, W] in range [0, 1]
        """
        h, w = x.shape[2:]
        edge_maps = []
        
        for scale in self.scales:
            if scale != 1.0:
                x_scaled = F.interpolate(
                    x, scale_factor=scale, mode='bilinear', align_corners=False
                )
            else:
                x_scaled = x
            
            if self.use_canny:
                canny_edges = self.canny(x_scaled)
                if scale != 1.0:
                    canny_edges = F.interpolate(
                        canny_edges, size=(h, w), mode='bilinear', align_corners=False
                    )
                edge_maps.append(canny_edges)
            
            if self.use_hed:
                hed_edges = self.hed(x_scaled)
                if scale != 1.0:
                    hed_edges = F.interpolate(
                        hed_edges, size=(h, w), mode='bilinear', align_corners=False
                    )
                edge_maps.append(hed_edges)
        
        if len(edge_maps) == 1:
            return edge_maps[0]
        
        # Fuse all edge maps
        combined = torch.cat(edge_maps, dim=1)
        if self.fusion is not None:
            edges = self.fusion(combined)
            edges = torch.sigmoid(edges)
        else:
            edges = combined.mean(dim=1, keepdim=True)
        
        return edges


class EdgeEncoder(nn.Module):
    """
    Encoder to convert edge maps to feature representations.
    
    Used to encode edge maps for conditioning the U-Net.
    """
    
    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1),
        )
        self.apply(_init_conv)
    
    def forward(self, edge_map: torch.Tensor) -> torch.Tensor:
        """
        Encode edge map to features.
        
        Args:
            edge_map: Edge map [B, 1, H, W]
            
        Returns:
            Edge features [B, out_channels, H, W]
        """
        return self.encoder(edge_map)


__all__ = [
    "SobelEdgeDetector",
    "CannyEdgeDetector", 
    "LightweightHED",
    "MultiScaleEdgeDetector",
    "EdgeEncoder",
]

