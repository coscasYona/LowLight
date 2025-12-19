"""
Visualization utilities for EMVA 1288 diffusion training.

Provides functions for image logging, RAW to RGB conversion,
and TensorBoard visualization.
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image


def raw_to_rgb(raw_4ch: torch.Tensor) -> torch.Tensor:
    """
    Convert 4-channel RAW (RGGB) to 3-channel RGB for visualization.
    
    RGGB format: [R, G1, B, G2] -> RGB: [R, (G1+G2)/2, B]
    
    Args:
        raw_4ch: Input tensor [B, 4, H, W] or [4, H, W]
        
    Returns:
        RGB tensor [B, 3, H, W] or [3, H, W]
    """
    squeeze = False
    if raw_4ch.dim() == 3:
        raw_4ch = raw_4ch.unsqueeze(0)
        squeeze = True
    
    B, C, H, W = raw_4ch.shape
    
    if C == 4:
        R = raw_4ch[:, 0:1, :, :]
        G1 = raw_4ch[:, 1:2, :, :]
        B_ch = raw_4ch[:, 2:3, :, :]
        G2 = raw_4ch[:, 3:4, :, :]
        G = (G1 + G2) / 2.0
        rgb = torch.cat([R, G, B_ch], dim=1)
    else:
        rgb = raw_4ch
    
    if squeeze:
        rgb = rgb.squeeze(0)
    
    return rgb


def apply_gamma(image: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    """
    Apply gamma correction for display.
    
    Args:
        image: Input image tensor
        gamma: Gamma value (default: 2.2)
        
    Returns:
        Gamma-corrected image
    """
    return torch.clamp(image, min=1e-8) ** (1 / gamma)


def log_images_to_tensorboard(
    writer,
    epoch: int,
    images: Dict[str, torch.Tensor],
    prefix: str = 'Train',
    apply_gamma_correction: bool = True,
):
    """
    Log images to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        epoch: Current epoch number
        images: Dict with 'clean', 'noisy', 'denoised' tensors
        prefix: Tag prefix ('Train' or 'Validation')
        apply_gamma_correction: Whether to apply gamma correction
    """
    for name, img in images.items():
        if img is None:
            continue
        
        # Ensure batch dimension
        if img.dim() == 3:
            img = img.unsqueeze(0)
        
        # Convert to RGB if needed
        if img.shape[1] == 4:
            img = raw_to_rgb(img)
        
        # Clamp to valid range
        img = torch.clamp(img, 0, 1)
        
        # Apply gamma correction for better visualization
        if apply_gamma_correction:
            img = apply_gamma(img)
        
        writer.add_images(f'{prefix}/{name}', img, epoch, dataformats='NCHW')


def create_comparison_grid(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    denoised: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Create a side-by-side comparison grid.
    
    Args:
        clean: Clean image [B, C, H, W]
        noisy: Noisy image [B, C, H, W]
        denoised: Denoised image [B, C, H, W]
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        Comparison grid [B, C, H, W*3]
    """
    # Convert to RGB if needed
    if clean.shape[1] == 4:
        clean = raw_to_rgb(clean)
        noisy = raw_to_rgb(noisy)
        denoised = raw_to_rgb(denoised)
    
    if normalize:
        clean = torch.clamp(clean, 0, 1)
        noisy = torch.clamp(noisy, 0, 1)
        denoised = torch.clamp(denoised, 0, 1)
    
    # Concatenate horizontally
    grid = torch.cat([clean, noisy, denoised], dim=3)
    
    return grid


def save_image(
    tensor: torch.Tensor,
    path: str,
    normalize: bool = True,
    apply_gamma_correction: bool = True,
):
    """
    Save tensor as image file.
    
    Args:
        tensor: Image tensor [C, H, W] or [B, C, H, W]
        path: Output file path
        normalize: Whether to normalize to [0, 1]
        apply_gamma_correction: Whether to apply gamma correction
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first in batch
    
    if tensor.shape[0] == 4:
        tensor = raw_to_rgb(tensor)
    
    if normalize:
        tensor = torch.clamp(tensor, 0, 1)
    
    if apply_gamma_correction:
        tensor = apply_gamma(tensor)
    
    # Convert to numpy
    img_np = tensor.cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    img_np = (img_np * 255).astype(np.uint8)
    
    # Save
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    Image.fromarray(img_np).save(path)


def save_comparison(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    denoised: torch.Tensor,
    path: str,
    apply_gamma_correction: bool = True,
):
    """
    Save side-by-side comparison image.
    
    Args:
        clean: Clean image
        noisy: Noisy image
        denoised: Denoised image
        path: Output file path
        apply_gamma_correction: Whether to apply gamma correction
    """
    grid = create_comparison_grid(clean, noisy, denoised)
    save_image(grid, path, apply_gamma_correction=apply_gamma_correction)


__all__ = [
    "raw_to_rgb",
    "apply_gamma",
    "log_images_to_tensorboard",
    "create_comparison_grid",
    "save_image",
    "save_comparison",
]

