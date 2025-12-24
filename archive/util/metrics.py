"""
Image quality metrics for denoising evaluation.

This module provides reusable metrics (PSNR, SNR) that can be used
across training, validation, and testing.
"""

import torch
from torchmetrics.image import PeakSignalNoiseRatio


class ImageQualityMetrics:
    """Compute image quality metrics for denoising tasks."""
    
    def __init__(self, device='cuda', data_range=1.0):
        """
        Initialize metrics.
        
        Args:
            device: Device to run metrics on ('cuda' or 'cpu')
            data_range: Maximum possible pixel value (1.0 for normalized images)
        """
        self.device = device
        self.psnr_metric = PeakSignalNoiseRatio(data_range=data_range).to(device)
    
    @staticmethod
    def compute_snr(image, reference, eps=1e-10):
        """
        Compute Signal-to-Noise Ratio (SNR) in dB.
        
        Args:
            image: Image to evaluate [B, C, H, W]
            reference: Clean reference image [B, C, H, W]
            eps: Small constant to avoid division by zero
        
        Returns:
            SNR in dB
        """
        signal_power = torch.mean(reference ** 2)
        noise = image - reference
        noise_power = torch.mean(noise ** 2)
        snr_linear = signal_power / (noise_power + eps)
        snr_db = 10 * torch.log10(snr_linear)
        return snr_db.item()
    
    def compute_all_metrics(self, clean, noisy, denoised):
        """
        Compute all metrics for clean, noisy, and denoised images.
        
        Args:
            clean: Clean/ground truth images [B, C, H, W], range [0, 1]
            noisy: Noisy input images [B, C, H, W], range [0, 1]
            denoised: Denoised output images [B, C, H, W], range [0, 1]
        
        Returns:
            dict: Dictionary containing all computed metrics
        """
        # Clamp all images to valid range
        clean = torch.clamp(clean, 0.0, 1.0)
        noisy = torch.clamp(noisy, 0.0, 1.0)
        denoised = torch.clamp(denoised, 0.0, 1.0)
        
        # Calculate PSNR
        psnr_noisy = self.psnr_metric(noisy, clean).item()
        psnr_denoised = self.psnr_metric(denoised, clean).item()
        psnr_improvement = psnr_denoised - psnr_noisy
        
        # Calculate SNR
        snr_noisy = self.compute_snr(noisy, clean)
        snr_denoised = self.compute_snr(denoised, clean)
        snr_enhancement_db = snr_denoised - snr_noisy
        
        # Calculate SNR enhancement ratio (linear scale)
        snr_enhancement_ratio = 10 ** (snr_enhancement_db / 10)
        
        return {
            'psnr_noisy': psnr_noisy,
            'psnr_denoised': psnr_denoised,
            'psnr_improvement': psnr_improvement,
            'snr_noisy': snr_noisy,
            'snr_denoised': snr_denoised,
            'snr_enhancement_db': snr_enhancement_db,
            'snr_enhancement_ratio': snr_enhancement_ratio,
        }


def log_metrics_to_tensorboard(writer, epoch, metrics, prefix='Validation'):
    """
    Log computed metrics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        epoch: Current epoch number
        metrics: Dictionary of metrics from ImageQualityMetrics.compute_all_metrics()
        prefix: Prefix for TensorBoard metric names (default: 'Validation')
    """
    writer.add_scalar(f'{prefix}/PSNR_Noisy_vs_Clean', metrics['psnr_noisy'], epoch)
    writer.add_scalar(f'{prefix}/PSNR_Denoised_vs_Clean', metrics['psnr_denoised'], epoch)
    writer.add_scalar(f'{prefix}/PSNR_Improvement', metrics['psnr_improvement'], epoch)
    writer.add_scalar(f'{prefix}/SNR_Noisy_vs_Clean', metrics['snr_noisy'], epoch)
    writer.add_scalar(f'{prefix}/SNR_Denoised_vs_Clean', metrics['snr_denoised'], epoch)
    writer.add_scalar(f'{prefix}/SNR_Enhancement_dB', metrics['snr_enhancement_db'], epoch)
    writer.add_scalar(f'{prefix}/SNR_Enhancement_Ratio', metrics['snr_enhancement_ratio'], epoch)


def print_metrics_summary(metrics, prefix='Validation'):
    """
    Print metrics summary to console.
    
    Args:
        metrics: Dictionary of metrics from ImageQualityMetrics.compute_all_metrics()
        prefix: Prefix for console output (default: 'Validation')
    """
    print(f"[{prefix}] PSNR - Noisy: {metrics['psnr_noisy']:.2f} dB | "
          f"Denoised: {metrics['psnr_denoised']:.2f} dB | "
          f"Improvement: {metrics['psnr_improvement']:.2f} dB")
    print(f"[{prefix}] SNR - Noisy: {metrics['snr_noisy']:.2f} dB | "
          f"Denoised: {metrics['snr_denoised']:.2f} dB | "
          f"Enhancement: {metrics['snr_enhancement_db']:.2f} dB "
          f"({metrics['snr_enhancement_ratio']:.2f}x)")
