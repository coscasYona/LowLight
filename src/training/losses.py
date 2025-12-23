"""
Loss functions for EMVA 1288 diffusion model training.

Implements hybrid loss combining MSE, L1, and gradient-based losses
for training diffusion models on image denoising tasks.

SOTA loss functions include:
- Charbonnier loss (smooth L1 alternative)
- Frequency-domain loss (FFT-based edge preservation)
- Multi-scale gradient loss (enhanced edge detection)
- Enhanced hybrid loss (combines all SOTA techniques)
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridDiffusionLoss(nn.Module):
    """
    Hybrid loss for diffusion model training.
    
    Combines:
    - MSE loss for overall reconstruction
    - L1 loss for sharper images (less blur)
    - Gradient loss for edge preservation (illuminance-invariant)
    
    Args:
        l1_weight: Weight for L1 loss component (0.0-1.0)
        gradient_weight: Weight for gradient loss (0.0 means disabled)
        loss_scale: Scaling factor for loss (increases gradient magnitude)
    """
    
    def __init__(
        self,
        l1_weight: float = 0.8,
        gradient_weight: float = 0.1,
        loss_scale: float = 1.0,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.gradient_weight = gradient_weight
        self.loss_scale = loss_scale
        
        # Pre-create Sobel kernels as buffers (more efficient than creating each forward)
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
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def gradient_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute illuminance-invariant gradient loss using Sobel filters.
        
        The gradients are normalized by local mean to make them
        invariant to illuminance variations.
        
        Args:
            pred: Predicted tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]
            
        Returns:
            Gradient loss value
        """
        n_channels = pred.size(1)
        
        # Expand Sobel kernels for all channels
        kernel_x = self.sobel_x.repeat(n_channels, 1, 1, 1)
        kernel_y = self.sobel_y.repeat(n_channels, 1, 1, 1)
        
        # Compute local mean for illuminance normalization
        pred_mean = F.avg_pool2d(torch.abs(pred), kernel_size=3, stride=1, padding=1)
        
        # Apply Sobel filters to prediction
        pred_gx = F.conv2d(pred, kernel_x, groups=n_channels, padding=1)
        pred_gy = F.conv2d(pred, kernel_y, groups=n_channels, padding=1)
        
        # Normalize by local illuminance
        epsilon = 1e-3
        pred_gx_norm = pred_gx / (pred_mean + epsilon)
        pred_gy_norm = pred_gy / (pred_mean + epsilon)
        
        # Compute target gradients without gradients (save memory)
        with torch.no_grad():
            target_mean = F.avg_pool2d(torch.abs(target), kernel_size=3, stride=1, padding=1)
            target_gx = F.conv2d(target, kernel_x, groups=n_channels, padding=1)
            target_gy = F.conv2d(target, kernel_y, groups=n_channels, padding=1)
            target_gx_norm = target_gx / (target_mean + epsilon)
            target_gy_norm = target_gy / (target_mean + epsilon)
        
        # L1 loss on normalized gradient components
        loss_x = F.l1_loss(pred_gx_norm, target_gx_norm)
        loss_y = F.l1_loss(pred_gy_norm, target_gy_norm)
        
        return (loss_x + loss_y) * 0.5
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Compute hybrid loss.
        
        Args:
            pred: Predicted noise/image [B, C, H, W]
            target: Target noise/image [B, C, H, W]
            return_components: If True, return dict with individual loss components
            
        Returns:
            Total loss (optionally with components dict)
        """
        # Ensure high precision for loss computation
        pred = pred.float()
        target = target.float()

        # Clean up NaN/inf values
        pred = torch.where(torch.isfinite(pred), pred, torch.zeros_like(pred))
        target = torch.where(torch.isfinite(target), target, torch.zeros_like(target))

        # Clamp to prevent extreme values
        pred = torch.clamp(pred, min=-10.0, max=10.0)
        target = torch.clamp(target, min=-10.0, max=10.0)
        
        # Compute individual losses
        loss_mse = self.mse_loss(pred, target)
        loss_l1 = self.l1_loss(pred, target)
        
        # Base loss: weighted combination of MSE and L1
        base_weight = 1.0 - self.gradient_weight
        loss = base_weight * ((1.0 - self.l1_weight) * loss_mse + self.l1_weight * loss_l1)
        
        # Add gradient loss if enabled
        if self.gradient_weight > 0:
            loss_grad = self.gradient_loss(pred, target)
            loss = loss + self.gradient_weight * loss_grad
        else:
            loss_grad = torch.tensor(0.0, device=pred.device)
        
        # Apply loss scaling
        loss = loss * self.loss_scale
        
        if return_components:
            return loss, {
                'mse': loss_mse.item(),
                'l1': loss_l1.item(),
                'gradient': loss_grad.item() if self.gradient_weight > 0 else 0.0,
                'total_unscaled': loss.item() / self.loss_scale,
            }
        
        return loss


class MinSNRWeightedLoss(nn.Module):
    """
    Loss with Min-SNR weighting for improved diffusion training.
    
    From "Efficient Diffusion Training via Min-SNR Weighting Strategy"
    
    Args:
        base_loss: Base loss function to weight
        gamma: SNR clipping value (default: 5.0)
    """
    
    def __init__(self, base_loss: nn.Module, gamma: float = 5.0):
        super().__init__()
        self.base_loss = base_loss
        self.gamma = gamma
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        snr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SNR-weighted loss.
        
        Args:
            pred: Predicted tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]
            snr: Signal-to-noise ratio for each sample [B]
            
        Returns:
            Weighted loss
        """
        # Compute base loss per sample
        base_loss = self.base_loss(pred, target)
        
        # Min-SNR weighting
        weight = torch.clamp(snr, max=self.gamma) / snr
        weight = weight.view(-1, 1, 1, 1)
        
        return (base_loss * weight).mean()


class VPredictionLoss(nn.Module):
    """
    Loss for v-prediction parameterization.
    
    v = sqrt(alpha_bar) * noise - sqrt(1 - alpha_bar) * x_start
    
    This can be more stable than epsilon-prediction in some cases.
    """
    
    def __init__(self, base_loss: nn.Module = None):
        super().__init__()
        self.base_loss = base_loss or nn.MSELoss()
    
    def forward(
        self,
        v_pred: torch.Tensor,
        noise: torch.Tensor,
        x_start: torch.Tensor,
        sqrt_alpha_bar: torch.Tensor,
        sqrt_one_minus_alpha_bar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute v-prediction loss.
        
        Args:
            v_pred: Predicted v
            noise: Actual noise
            x_start: Original clean data
            sqrt_alpha_bar: sqrt(alpha_bar) for timestep
            sqrt_one_minus_alpha_bar: sqrt(1 - alpha_bar) for timestep
            
        Returns:
            Loss value
        """
        v_target = sqrt_alpha_bar * noise - sqrt_one_minus_alpha_bar * x_start
        return self.base_loss(v_pred, v_target)


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (smooth L1 alternative).
    
    L = sqrt(x^2 + eps^2)
    
    Provides better gradient flow than L1 for small errors,
    and is more robust to outliers than MSE.
    
    Args:
        epsilon: Smoothing parameter (default: 1e-3)
    """
    
    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Charbonnier loss.
        
        Args:
            pred: Predicted tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]
            
        Returns:
            Charbonnier loss value
        """
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + self.epsilon ** 2)
        return loss.mean()


class FrequencyDomainLoss(nn.Module):
    """
    Frequency-domain loss for edge preservation.
    
    Uses FFT to compute loss in frequency domain, with separate
    weighting for low and high frequency components.
    
    High frequency components are particularly important for
    preserving sharp edges and fine details.
    
    Args:
        high_freq_weight: Weight for high frequency components (default: 2.0)
        magnitude_weight: Weight for magnitude loss (default: 1.0)
        phase_weight: Weight for phase loss (default: 0.1)
    """
    
    def __init__(
        self,
        high_freq_weight: float = 2.0,
        magnitude_weight: float = 1.0,
        phase_weight: float = 0.1,
    ):
        super().__init__()
        self.high_freq_weight = high_freq_weight
        self.magnitude_weight = magnitude_weight
        self.phase_weight = phase_weight
    
    def _create_frequency_mask(
        self, h: int, w: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create masks for low and high frequency components.
        
        Returns:
            low_freq_mask, high_freq_mask
        """
        # Create frequency coordinate grid
        fy = torch.fft.fftfreq(h, device=device).view(-1, 1)
        fx = torch.fft.fftfreq(w, device=device).view(1, -1)
        
        # Radial frequency
        freq_radius = torch.sqrt(fy ** 2 + fx ** 2)
        
        # Threshold at 0.25 of max frequency
        threshold = 0.25
        
        # Smooth transition masks
        low_freq_mask = torch.sigmoid(10 * (threshold - freq_radius))
        high_freq_mask = 1.0 - low_freq_mask
        
        return low_freq_mask, high_freq_mask
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency-domain loss.
        
        Args:
            pred: Predicted tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]
            
        Returns:
            Frequency-domain loss value
        """
        b, c, h, w = pred.shape
        
        # Compute 2D FFT
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        
        # Get magnitude and phase
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        
        # Create frequency masks
        low_mask, high_mask = self._create_frequency_mask(h, w, pred.device)
        low_mask = low_mask.view(1, 1, h, w)
        high_mask = high_mask.view(1, 1, h, w)
        
        # Magnitude loss with frequency weighting
        mag_diff = (pred_mag - target_mag) ** 2
        low_freq_loss = (mag_diff * low_mask).mean()
        high_freq_loss = (mag_diff * high_mask).mean()
        
        magnitude_loss = low_freq_loss + self.high_freq_weight * high_freq_loss
        
        # Phase loss (wrap-aware)
        phase_diff = pred_phase - target_phase
        # Wrap to [-pi, pi]
        phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
        phase_loss = (phase_diff ** 2 * high_mask).mean()  # Only high freq phase matters
        
        total_loss = (
            self.magnitude_weight * magnitude_loss + 
            self.phase_weight * phase_loss
        )
        
        return total_loss


class MultiScaleGradientLoss(nn.Module):
    """
    Multi-scale gradient loss for enhanced edge preservation.
    
    Combines gradient losses at multiple scales using different
    kernel sizes (3x3, 5x5, 7x7) and includes Laplacian of Gaussian.
    
    Args:
        scales: Tuple of kernel sizes to use (default: (3, 5, 7))
        use_log: Whether to include Laplacian of Gaussian (default: True)
        illuminance_invariant: Whether to normalize by local mean (default: True)
    """
    
    def __init__(
        self,
        scales: Tuple[int, ...] = (3, 5, 7),
        use_log: bool = True,
        illuminance_invariant: bool = True,
    ):
        super().__init__()
        self.scales = scales
        self.use_log = use_log
        self.illuminance_invariant = illuminance_invariant
        
        # Create Sobel kernels at different scales
        for size in scales:
            sobel_x, sobel_y = self._create_sobel_kernels(size)
            self.register_buffer(f'sobel_x_{size}', sobel_x, persistent=False)
            self.register_buffer(f'sobel_y_{size}', sobel_y, persistent=False)
        
        # Create Laplacian of Gaussian kernel
        if use_log:
            log_kernel = self._create_log_kernel(5, sigma=1.0)
            self.register_buffer('log_kernel', log_kernel, persistent=False)
    
    def _create_sobel_kernels(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create Sobel kernels of given size."""
        if size == 3:
            sobel_x = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                dtype=torch.float32
            )
            sobel_y = torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                dtype=torch.float32
            )
        elif size == 5:
            # Extended Sobel (Scharr-like)
            sobel_x = torch.tensor([
                [-1, -2, 0, 2, 1],
                [-4, -8, 0, 8, 4],
                [-6, -12, 0, 12, 6],
                [-4, -8, 0, 8, 4],
                [-1, -2, 0, 2, 1],
            ], dtype=torch.float32) / 12.0
            sobel_y = sobel_x.t()
        elif size == 7:
            # 7x7 Sobel approximation
            sobel_x = torch.tensor([
                [-1, -3, -5, 0, 5, 3, 1],
                [-3, -9, -15, 0, 15, 9, 3],
                [-5, -15, -25, 0, 25, 15, 5],
                [-6, -18, -30, 0, 30, 18, 6],
                [-5, -15, -25, 0, 25, 15, 5],
                [-3, -9, -15, 0, 15, 9, 3],
                [-1, -3, -5, 0, 5, 3, 1],
            ], dtype=torch.float32) / 60.0
            sobel_y = sobel_x.t()
        else:
            raise ValueError(f"Unsupported Sobel size: {size}")
        
        return sobel_x.view(1, 1, size, size), sobel_y.view(1, 1, size, size)
    
    def _create_log_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create Laplacian of Gaussian kernel."""
        x = torch.arange(size).float() - size // 2
        y = torch.arange(size).float() - size // 2
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # Gaussian
        gauss = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        
        # Laplacian of Gaussian
        log = ((xx ** 2 + yy ** 2 - 2 * sigma ** 2) / (sigma ** 4)) * gauss
        
        # Normalize
        log = log - log.mean()
        
        return log.view(1, 1, size, size)
    
    def _compute_gradient_loss(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        kernel_x: torch.Tensor,
        kernel_y: torch.Tensor,
        size: int,
    ) -> torch.Tensor:
        """Compute gradient loss for a single scale."""
        n_channels = pred.size(1)
        padding = size // 2
        
        # Expand kernels for all channels
        kx = kernel_x.repeat(n_channels, 1, 1, 1)
        ky = kernel_y.repeat(n_channels, 1, 1, 1)
        
        # Compute gradients
        pred_gx = F.conv2d(pred, kx, groups=n_channels, padding=padding)
        pred_gy = F.conv2d(pred, ky, groups=n_channels, padding=padding)
        
        with torch.no_grad():
            target_gx = F.conv2d(target, kx, groups=n_channels, padding=padding)
            target_gy = F.conv2d(target, ky, groups=n_channels, padding=padding)
        
        if self.illuminance_invariant:
            epsilon = 1e-3
            pred_mean = F.avg_pool2d(
                torch.abs(pred), kernel_size=size, stride=1, padding=padding
            )
            target_mean = F.avg_pool2d(
                torch.abs(target), kernel_size=size, stride=1, padding=padding
            )
            
            pred_gx = pred_gx / (pred_mean + epsilon)
            pred_gy = pred_gy / (pred_mean + epsilon)
            target_gx = target_gx / (target_mean + epsilon)
            target_gy = target_gy / (target_mean + epsilon)
        
        loss_x = F.l1_loss(pred_gx, target_gx)
        loss_y = F.l1_loss(pred_gy, target_gy)
        
        return (loss_x + loss_y) * 0.5
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale gradient loss.
        
        Args:
            pred: Predicted tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]
            
        Returns:
            Multi-scale gradient loss value
        """
        total_loss = 0.0
        
        # Multi-scale gradient losses
        for size in self.scales:
            kernel_x = getattr(self, f'sobel_x_{size}')
            kernel_y = getattr(self, f'sobel_y_{size}')
            
            # Weight smaller scales more (finer details)
            weight = 1.0 / (size / 3.0)
            loss = self._compute_gradient_loss(pred, target, kernel_x, kernel_y, size)
            total_loss = total_loss + weight * loss
        
        # Normalize by number of scales
        total_loss = total_loss / len(self.scales)
        
        # Add Laplacian of Gaussian loss if enabled
        if self.use_log:
            n_channels = pred.size(1)
            log_kernel = self.log_kernel.repeat(n_channels, 1, 1, 1)
            
            pred_log = F.conv2d(pred, log_kernel, groups=n_channels, padding=2)
            with torch.no_grad():
                target_log = F.conv2d(target, log_kernel, groups=n_channels, padding=2)
            
            log_loss = F.l1_loss(pred_log, target_log)
            total_loss = total_loss + 0.5 * log_loss
        
        return total_loss


class EnhancedHybridLoss(nn.Module):
    """
    Enhanced hybrid loss combining SOTA techniques for edge preservation.
    
    Combines:
    - MSE loss for overall reconstruction
    - L1 loss for reduced blur
    - Charbonnier loss for smooth gradients
    - Frequency-domain loss for edge preservation
    - Multi-scale gradient loss for edge sharpness
    
    Args:
        mse_weight: Weight for MSE loss (default: 0.25)
        l1_weight: Weight for L1 loss (default: 0.35)
        charbonnier_weight: Weight for Charbonnier loss (default: 0.25)
        frequency_weight: Weight for frequency-domain loss (default: 0.1)
        gradient_weight: Weight for multi-scale gradient loss (default: 0.05)
        loss_scale: Scaling factor for total loss (default: 1.0)
    """
    
    def __init__(
        self,
        mse_weight: float = 0.25,
        l1_weight: float = 0.35,
        charbonnier_weight: float = 0.25,
        frequency_weight: float = 0.1,
        gradient_weight: float = 0.05,
        loss_scale: float = 1.0,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.charbonnier_weight = charbonnier_weight
        self.frequency_weight = frequency_weight
        self.gradient_weight = gradient_weight
        self.loss_scale = loss_scale
        
        # Individual loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.charbonnier_loss = CharbonnierLoss()
        self.frequency_loss = FrequencyDomainLoss()
        self.gradient_loss = MultiScaleGradientLoss()
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Compute enhanced hybrid loss.
        
        Args:
            pred: Predicted tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]
            return_components: If True, return dict with individual loss components
            
        Returns:
            Total loss (optionally with components dict)
        """
        # Ensure high precision
        pred = pred.float()
        target = target.float()
        
        # Clean up NaN/inf values
        pred = torch.where(torch.isfinite(pred), pred, torch.zeros_like(pred))
        target = torch.where(torch.isfinite(target), target, torch.zeros_like(target))
        
        # Clamp to prevent extreme values
        pred = torch.clamp(pred, min=-10.0, max=10.0)
        target = torch.clamp(target, min=-10.0, max=10.0)
        
        # Compute individual losses
        loss_mse = self.mse_loss(pred, target)
        loss_l1 = self.l1_loss(pred, target)
        loss_charb = self.charbonnier_loss(pred, target)
        
        # Frequency and gradient losses (more expensive)
        loss_freq = torch.tensor(0.0, device=pred.device)
        loss_grad = torch.tensor(0.0, device=pred.device)
        
        if self.frequency_weight > 0:
            loss_freq = self.frequency_loss(pred, target)
        
        if self.gradient_weight > 0:
            loss_grad = self.gradient_loss(pred, target)
        
        # Weighted combination
        total_loss = (
            self.mse_weight * loss_mse +
            self.l1_weight * loss_l1 +
            self.charbonnier_weight * loss_charb +
            self.frequency_weight * loss_freq +
            self.gradient_weight * loss_grad
        )
        
        # Apply loss scaling
        total_loss = total_loss * self.loss_scale
        
        if return_components:
            return total_loss, {
                'mse': loss_mse.item(),
                'l1': loss_l1.item(),
                'charbonnier': loss_charb.item(),
                'frequency': loss_freq.item() if self.frequency_weight > 0 else 0.0,
                'gradient': loss_grad.item() if self.gradient_weight > 0 else 0.0,
                'total_unscaled': total_loss.item() / self.loss_scale,
            }
        
        return total_loss


__all__ = [
    "HybridDiffusionLoss", 
    "MinSNRWeightedLoss", 
    "VPredictionLoss",
    "CharbonnierLoss",
    "FrequencyDomainLoss",
    "MultiScaleGradientLoss",
    "EnhancedHybridLoss",
]

