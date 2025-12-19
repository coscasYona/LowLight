"""
Loss functions for EMVA 1288 diffusion model training.

Implements hybrid loss combining MSE, L1, and gradient-based losses
for training diffusion models on image denoising tasks.
"""

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


__all__ = ["HybridDiffusionLoss", "MinSNRWeightedLoss", "VPredictionLoss"]

