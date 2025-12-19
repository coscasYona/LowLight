"""
Metrics for evaluating denoising quality.

Provides PSNR, SSIM, and SNR metrics with batch support
and optional TensorBoard logging.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torchmetrics import Metric
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class DenoisingMetrics(nn.Module):
    """
    Collection of metrics for evaluating denoising quality.
    
    Includes:
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index)
    - SNR improvement
    
    Args:
        data_range: Maximum value of the data (default: 1.0)
    """
    
    def __init__(self, data_range: float = 1.0):
        super().__init__()
        self.data_range = data_range
        
        self.psnr = PeakSignalNoiseRatio(data_range=data_range)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range)
    
    def compute_psnr(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute PSNR between prediction and target."""
        pred = torch.clamp(pred, 0, self.data_range)
        target = torch.clamp(target, 0, self.data_range)
        return self.psnr(pred, target)
    
    def compute_ssim(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute SSIM between prediction and target."""
        pred = torch.clamp(pred, 0, self.data_range)
        target = torch.clamp(target, 0, self.data_range)
        return self.ssim(pred, target)
    
    def compute_snr(
        self, 
        clean: torch.Tensor,
        noisy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SNR of noisy image relative to clean.
        
        SNR = 10 * log10(signal_power / noise_power)
        """
        signal_power = torch.mean(clean ** 2)
        noise_power = torch.mean((noisy - clean) ** 2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
        return snr
    
    def compute_snr_improvement(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        denoised: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SNR improvement from denoising.
        
        Returns the difference: SNR(denoised) - SNR(noisy)
        """
        snr_noisy = self.compute_snr(clean, noisy)
        snr_denoised = self.compute_snr(clean, denoised)
        return snr_denoised - snr_noisy
    
    def forward(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        denoised: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all metrics.
        
        Args:
            clean: Ground truth clean images [B, C, H, W]
            noisy: Noisy input images [B, C, H, W]
            denoised: Denoised output images [B, C, H, W]
            
        Returns:
            Dictionary of metric values
        """
        return {
            'psnr': self.compute_psnr(denoised, clean),
            'ssim': self.compute_ssim(denoised, clean),
            'snr_noisy': self.compute_snr(clean, noisy),
            'snr_denoised': self.compute_snr(clean, denoised),
            'snr_improvement': self.compute_snr_improvement(clean, noisy, denoised),
        }
    
    def reset(self):
        """Reset metric states."""
        self.psnr.reset()
        self.ssim.reset()


class BatchPSNR:
    """
    Batch-aware PSNR computation for validation.
    
    Computes PSNR for each sample in batch and returns average.
    """
    
    def __init__(self, data_range: float = 1.0):
        self.data_range = data_range
    
    def __call__(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> float:
        """
        Compute average PSNR across batch.
        
        Args:
            pred: Predicted images [B, C, H, W]
            target: Target images [B, C, H, W]
            
        Returns:
            Average PSNR value
        """
        pred = pred.detach().cpu().float().numpy()
        target = target.detach().cpu().float().numpy()
        
        batch_size = pred.shape[0]
        psnr_sum = 0.0
        
        for i in range(batch_size):
            mse = ((pred[i] - target[i]) ** 2).mean()
            if mse == 0:
                psnr_sum += 100.0
            else:
                psnr_sum += 10 * (
                    (self.data_range ** 2) / mse
                ).__log10__() if hasattr(mse, '__log10__') else 10 * float(
                    __import__('math').log10((self.data_range ** 2) / mse)
                )
        
        return psnr_sum / batch_size


class IlluminanceCorrect(nn.Module):
    """
    Illuminance correction for fair metric comparison.
    
    Adjusts prediction brightness to match target before computing metrics.
    This accounts for global brightness differences that don't affect
    perceptual quality.
    """
    
    def forward(
        self, predict: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply illuminance correction.
        
        Args:
            predict: Predicted image [B, C, H, W]
            source: Reference image [B, C, H, W]
            
        Returns:
            Brightness-corrected prediction
        """
        if predict.shape[0] != 1:
            output = torch.zeros_like(predict)
            if source.shape[0] != 1:
                for i in range(predict.shape[0]):
                    output[i:i+1] = self._correct_single(
                        predict[i:i+1], source[i:i+1]
                    )
            else:
                for i in range(predict.shape[0]):
                    output[i:i+1] = self._correct_single(predict[i:i+1], source)
        else:
            output = self._correct_single(predict, source)
        return output
    
    def _correct_single(
        self, predict: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        """Correct single sample."""
        predict = torch.clamp(predict, 0, 1)
        
        pred_flat = predict[source != 1]
        source_flat = source[source != 1]
        
        if pred_flat.numel() == 0:
            return predict
            
        num = torch.dot(pred_flat, source_flat)
        den = torch.dot(pred_flat, pred_flat)
        
        if den > 1e-10:
            return num / den * predict
        return predict


__all__ = [
    "DenoisingMetrics", 
    "BatchPSNR", 
    "IlluminanceCorrect",
]

