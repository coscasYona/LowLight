#!/usr/bin/env python3
"""
Single image inference script for EMVA 1288 Physics-Guided Diffusion Model.

Denoise a single RAW image using a trained model.

Usage:
    python inference.py --checkpoint path/to/checkpoint.ckpt --input image.ARW --output denoised.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import rawpy
from PIL import Image

from training import EMVA1288LightningModule
from data.sid_dataset import pack_raw_bayer, metainfo
from utils.visualization import raw_to_rgb, apply_gamma


def load_raw_image(path: str, ratio: float = 1.0) -> tuple:
    """
    Load RAW image and metadata.
    
    Args:
        path: Path to RAW file
        ratio: Exposure ratio multiplier
        
    Returns:
        Tuple of (image tensor, iso, ratio)
    """
    # Get metadata
    try:
        iso, expo = metainfo(path)
    except Exception:
        iso = 6400
        expo = 0.1
    
    # Load and pack RAW
    with rawpy.imread(path) as raw:
        image = pack_raw_bayer(raw) * ratio
    
    # Clip and convert to tensor
    image = np.clip(image, 0, 1).astype(np.float32)
    image_tensor = torch.from_numpy(image).unsqueeze(0)
    
    return image_tensor, iso, ratio


def save_output(
    tensor: torch.Tensor,
    path: str,
    apply_gamma_correction: bool = True,
):
    """
    Save output tensor as image.
    
    Args:
        tensor: Image tensor [1, C, H, W]
        path: Output file path
        apply_gamma_correction: Whether to apply gamma
    """
    # Remove batch dimension
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Convert to RGB
    if tensor.shape[0] == 4:
        tensor = raw_to_rgb(tensor)
    
    # Clamp and apply gamma
    tensor = torch.clamp(tensor, 0, 1)
    if apply_gamma_correction:
        tensor = apply_gamma(tensor)
    
    # Convert to numpy and save
    img_np = tensor.cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    img_np = (img_np * 255).astype(np.uint8)
    
    Image.fromarray(img_np).save(path)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Single image inference for EMVA 1288 Diffusion Model'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input RAW image')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output image')
    parser.add_argument('--ratio', type=float, default=200.0,
                        help='Exposure ratio (default: 200)')
    parser.add_argument('--iso', type=int, default=None,
                        help='ISO value (auto-detect if not specified)')
    parser.add_argument('--num_steps', type=int, default=None,
                        help='Number of sampling steps (default: model default)')
    parser.add_argument('--no_gamma', action='store_true',
                        help='Disable gamma correction')
    parser.add_argument('--save_noisy', action='store_true',
                        help='Also save noisy input for comparison')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = EMVA1288LightningModule.load_from_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()
    
    # Load input image
    print(f"Loading input: {args.input}")
    image, detected_iso, ratio = load_raw_image(args.input, args.ratio)
    image = image.to(device)
    
    # Use provided or detected ISO
    iso_val = args.iso if args.iso is not None else detected_iso
    print(f"ISO: {iso_val}, Ratio: {ratio}")
    
    # Create tensors
    iso_tensor = torch.tensor([[iso_val]], device=device, dtype=torch.float32)
    ratio_tensor = torch.tensor([[ratio]], device=device, dtype=torch.float32)
    
    # Run inference
    print("Running denoising...")
    with torch.no_grad():
        num_steps = args.num_steps or model.num_steps
        # Pass noisy input as cond_image if model uses measurement conditioning
        cond_image = image if model.use_measurement_cond else None
        denoised = model.model.sample(
            image,
            iso=iso_tensor,
            ratio=ratio_tensor,
            num_steps=num_steps,
            cond_image=cond_image,
        )
    
    # Save output
    save_output(denoised, args.output, apply_gamma_correction=not args.no_gamma)
    
    # Optionally save noisy input
    if args.save_noisy:
        noisy_path = args.output.replace('.', '_noisy.')
        save_output(image, noisy_path, apply_gamma_correction=not args.no_gamma)
    
    print("Done!")


if __name__ == "__main__":
    main()

