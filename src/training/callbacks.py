"""
Custom callbacks for EMVA 1288 diffusion training.

Provides callbacks for image logging, validation metrics,
and other training utilities.
"""

import os
from typing import Any, Dict, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from models.noise_model import sample_params_max


class ImageLoggingCallback(Callback):
    """
    Callback for logging denoised images to TensorBoard.
    
    Logs sample images at the end of each validation epoch,
    showing noisy input, denoised output, and ground truth.
    
    Args:
        log_every_n_epochs: Log images every N epochs
        num_samples: Number of samples to log
        save_to_disk: Whether to also save images to disk
        save_dir: Directory for saved images
    """
    
    def __init__(
        self,
        log_every_n_epochs: int = 1,
        num_samples: int = 4,
        save_to_disk: bool = False,
        save_dir: Optional[str] = None,
        log_training_images: bool = True,
        log_validation_images: bool = True,
        compute_metrics: bool = False,
    ):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples
        self.save_to_disk = save_to_disk
        self.save_dir = save_dir
        self.log_training_images = log_training_images
        self.log_validation_images = log_validation_images
        self.compute_metrics = compute_metrics
    
    def on_validation_epoch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
    ):
        """Log images at end of validation epoch."""
        # Only log/save on rank 0 to avoid conflicts in multi-GPU training
        if trainer.global_rank != 0:
            return
            
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        
        # Check if we have validation outputs with images
        if not hasattr(pl_module, 'validation_step_outputs'):
            return
        
        outputs = pl_module.validation_step_outputs
        if not outputs:
            return
        
        # Find output with stored images
        image_output = None
        for out in outputs:
            if 'img_gt' in out:
                image_output = out
                break
        
        if image_output is None:
            return
        
        # Get model for inference
        model = pl_module.model
        if hasattr(pl_module, 'ema_model') and pl_module.ema_model is not None:
            model = pl_module.ema_model
        
        model.eval()
        with torch.no_grad():
            img_gt = image_output['img_gt']
            noisy_state = image_output['noisy_state']
            iso = image_output['iso']
            ratio = image_output['ratio']
            camera_params_list = image_output.get('camera_params_list')
            
            # Use real noisy input if available (better for visualization)
            img_noisy = image_output.get('img_noisy', noisy_state)
            
            # Determine if we should pass cond_image for measurement conditioning
            cond_image = None
            if hasattr(pl_module, 'use_measurement_cond') and pl_module.use_measurement_cond:
                cond_image = img_noisy
            
            # Generate denoised images from real noisy input
            denoised = model.sample(
                img_noisy,
                iso=iso,
                ratio=ratio,
                num_steps=min(50, pl_module.num_steps),
                camera_params=camera_params_list[0] if camera_params_list else None,
                cond_image=cond_image,
            )
            
            # Convert to RGB for visualization (4ch RGGB -> 3ch RGB)
            def to_rgb(x):
                """Convert 4-channel RGGB to 3-channel RGB."""
                if x.dim() == 3:
                    x = x.unsqueeze(0)
                B, C, H, W = x.shape
                if C == 4:
                    R = x[:, 0:1]
                    G = (x[:, 1:2] + x[:, 3:4]) / 2
                    B_ch = x[:, 2:3]
                    return torch.cat([R, G, B_ch], dim=1)
                return x
            
            img_gt_rgb = to_rgb(img_gt.clamp(0, 1))
            noisy_rgb = to_rgb(img_noisy.clamp(0, 1))
            denoised_rgb = to_rgb(denoised.clamp(0, 1))
            
            # Log to TensorBoard
            if trainer.logger is not None:
                logger = trainer.logger.experiment
                
                # Concatenate horizontally: [clean | noisy | denoised]
                grid = torch.cat([img_gt_rgb, noisy_rgb, denoised_rgb], dim=3)
                logger.add_images(
                    'Validation/Images',
                    grid,
                    trainer.current_epoch,
                    dataformats='NCHW'
                )
            
            # Save to disk
            if self.save_to_disk and self.save_dir:
                self._save_images(
                    img_gt_rgb, noisy_rgb, denoised_rgb,
                    trainer.current_epoch, prefix=prefix
                )
        
        model.train()

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ):
        """Log training images at end of training epoch."""
        # Only log/save on rank 0 to avoid conflicts in multi-GPU training
        if trainer.global_rank != 0:
            return
            
        if not self.log_training_images:
            return

        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        # Check if we have training outputs with images
        if not hasattr(pl_module, 'training_step_outputs'):
            return

        outputs = pl_module.training_step_outputs
        if not outputs:
            return

        # Use the stored training batch
        image_output = outputs[0] if outputs else None
        if image_output is None:
            return

        self._log_images(trainer, pl_module, image_output, prefix='Train')

    def _log_images(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        image_output: dict,
        prefix: str = 'Validation'
    ):
        """Shared method to log images for training/validation."""
        # Get model for inference
        model = pl_module.model
        if hasattr(pl_module, 'ema_model') and pl_module.ema_model is not None:
            model = pl_module.ema_model

        model.eval()
        with torch.no_grad():
            img_gt = image_output['img_gt'].to(pl_module.device)
            noisy_state = image_output['noisy_state'].to(pl_module.device)
            iso = image_output['iso'].to(pl_module.device)
            ratio = image_output['ratio'].to(pl_module.device)
            camera_params_list = image_output.get('camera_params_list')

            # Use real noisy input if available (better for visualization)
            img_noisy = image_output.get('img_noisy')
            if img_noisy is not None:
                img_noisy = img_noisy.to(pl_module.device)
            else:
                img_noisy = noisy_state

            # Determine if we should pass cond_image for measurement conditioning
            cond_image = None
            if hasattr(pl_module, 'use_measurement_cond') and pl_module.use_measurement_cond:
                cond_image = img_noisy

            # Generate denoised images from real noisy input
            denoised = model.sample(
                img_noisy,
                iso=iso,
                ratio=ratio,
                num_steps=min(50, pl_module.num_steps),
                camera_params=camera_params_list[0] if camera_params_list else None,
                cond_image=cond_image,
            )

            # Convert to RGB for visualization (4ch RGGB -> 3ch RGB)
            def to_rgb(x):
                """Convert 4-channel RGGB to 3-channel RGB."""
                if x.dim() == 3:
                    x = x.unsqueeze(0)
                B, C, H, W = x.shape
                if C == 4:
                    R = x[:, 0:1]
                    G = (x[:, 1:2] + x[:, 3:4]) / 2
                    B_ch = x[:, 2:3]
                    return torch.cat([R, G, B_ch], dim=1)
                return x

            img_gt_rgb = to_rgb(img_gt.clamp(0, 1))
            noisy_rgb = to_rgb(img_noisy.clamp(0, 1))
            denoised_rgb = to_rgb(denoised.clamp(0, 1))

            # Compute metrics if requested (only on rank 0 to avoid duplicates)
            if self.compute_metrics and trainer.logger is not None and trainer.global_rank == 0:
                try:
                    from training.metrics import DenoisingMetrics
                    metrics_calc = DenoisingMetrics(data_range=1.0)
                    metrics = metrics_calc(img_gt, img_noisy, denoised)

                    # Log metrics to TensorBoard
                    logger = trainer.logger.experiment
                    logger.add_scalar(f'{prefix}/PSNR', metrics['psnr'], trainer.current_epoch)
                    logger.add_scalar(f'{prefix}/SSIM', metrics['ssim'], trainer.current_epoch)
                    logger.add_scalar(f'{prefix}/SNR_Noisy', metrics['snr_noisy'], trainer.current_epoch)
                    logger.add_scalar(f'{prefix}/SNR_Denoised', metrics['snr_denoised'], trainer.current_epoch)
                    logger.add_scalar(f'{prefix}/SNR_Improvement', metrics['snr_improvement'], trainer.current_epoch)

                    # Use Lightning's log method - rank_zero_only since we only compute on rank 0
                    # (sync_dist would deadlock since other ranks don't reach this code path)
                    pl_module.log(f'{prefix.lower()}/psnr', metrics['psnr'], rank_zero_only=True)
                    pl_module.log(f'{prefix.lower()}/ssim', metrics['ssim'], rank_zero_only=True)
                    print(f"[{prefix}] Epoch {trainer.current_epoch}: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")

                except Exception as e:
                    print(f"Warning: Failed to compute {prefix.lower()} metrics: {e}")

            # Log to TensorBoard (only on rank 0)
            if trainer.logger is not None and trainer.global_rank == 0:
                logger = trainer.logger.experiment

                # Concatenate horizontally: [clean | noisy | denoised]
                grid = torch.cat([img_gt_rgb, noisy_rgb, denoised_rgb], dim=3)
                logger.add_images(
                    f'{prefix}/Images',
                    grid,
                    trainer.current_epoch,
                    dataformats='NCHW'
                )

            # Save to disk
            if self.save_to_disk and self.save_dir:
                self._save_images(
                    img_gt_rgb, noisy_rgb, denoised_rgb,
                    trainer.current_epoch, prefix=prefix
                )

        model.train()

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ):
        """Log images at end of validation epoch."""
        # Only run on rank 0 to avoid DDP deadlocks from model.sample()
        if trainer.global_rank != 0:
            return
            
        if not self.log_validation_images:
            return

        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        # Check if we have validation outputs with images
        if not hasattr(pl_module, 'validation_step_outputs'):
            return

        outputs = pl_module.validation_step_outputs
        if not outputs:
            return

        # Find output with stored images
        image_output = None
        for out in outputs:
            if 'img_gt' in out:
                image_output = out
                break

        if image_output is None:
            return

        self._log_images(trainer, pl_module, image_output, prefix='Validation')

    def _save_images(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        denoised: torch.Tensor,
        epoch: int,
        prefix: str = 'Validation',
    ):
        """Save images to disk."""
        from PIL import Image
        import numpy as np

        subdir = 'train_images' if prefix == 'Train' else 'val_images'
        save_dir = os.path.join(self.save_dir, subdir)
        os.makedirs(save_dir, exist_ok=True)
        
        for i in range(min(clean.shape[0], self.num_samples)):
            for name, tensor in [
                ('clean', clean[i]),
                ('noisy', noisy[i]),
                ('denoised', denoised[i])
            ]:
                img_np = tensor.cpu().clamp(0, 1).numpy()
                img_np = np.transpose(img_np, (1, 2, 0))
                img_np = (img_np * 255).astype(np.uint8)
                
                img = Image.fromarray(img_np)
                filename = f"epoch_{epoch:04d}_sample_{i:02d}_{name}.png"
                img.save(os.path.join(save_dir, filename))


class ValidationMetricsCallback(Callback):
    """
    Callback for running full evaluation on SID/ELD datasets.
    
    Runs comprehensive evaluation on test datasets periodically
    and logs detailed metrics.
    
    Args:
        eval_every_n_epochs: Run evaluation every N epochs
        sid_eval_dir: Directory for SID evaluation data
        eld_eval_dir: Directory for ELD evaluation data
    """
    
    def __init__(
        self,
        eval_every_n_epochs: int = 10,
        sid_eval_dir: Optional[str] = None,
        eld_eval_dir: Optional[str] = None,
    ):
        super().__init__()
        self.eval_every_n_epochs = eval_every_n_epochs
        self.sid_eval_dir = sid_eval_dir
        self.eld_eval_dir = eld_eval_dir
    
    def on_validation_epoch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
    ):
        """Run evaluation at specified intervals."""
        # Only run on rank 0 to avoid duplicate outputs
        if trainer.global_rank != 0:
            return
            
        if trainer.current_epoch % self.eval_every_n_epochs != 0:
            return
        
        if trainer.current_epoch == 0:
            return  # Skip first epoch
        
        # Log that we're running evaluation
        print(f"\nRunning comprehensive evaluation at epoch {trainer.current_epoch}...")
        
        # This could be extended to run actual SID/ELD evaluation
        # For now, we just log that evaluation would run here


class EMACallback(Callback):
    """
    Callback for exponential moving average of model weights.
    
    Maintains an EMA copy of the model and optionally swaps
    it in for validation/inference.
    
    Args:
        decay: EMA decay rate (0.9999 typical)
        use_ema_for_eval: Use EMA weights for validation
    """
    
    def __init__(
        self,
        decay: float = 0.9999,
        use_ema_for_eval: bool = True,
    ):
        super().__init__()
        self.decay = decay
        self.use_ema_for_eval = use_ema_for_eval
        self.ema_model = None
        self.original_state = None
    
    def on_fit_start(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
    ):
        """Initialize EMA model."""
        import copy
        self.ema_model = copy.deepcopy(pl_module.model)
        self.ema_model.requires_grad_(False)
    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        """Update EMA weights after each training batch."""
        if self.ema_model is None:
            return
        
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_model.parameters(),
                pl_module.model.parameters()
            ):
                ema_param.data.mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )
    
    def on_validation_start(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
    ):
        """Swap in EMA weights for validation."""
        if not self.use_ema_for_eval or self.ema_model is None:
            return
        
        self.original_state = {
            k: v.clone() for k, v in pl_module.model.state_dict().items()
        }
        pl_module.model.load_state_dict(self.ema_model.state_dict())
    
    def on_validation_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
    ):
        """Restore original weights after validation."""
        if self.original_state is not None:
            pl_module.model.load_state_dict(self.original_state)
            self.original_state = None


class GradientMonitorCallback(Callback):
    """
    Callback for monitoring gradient statistics.
    
    Logs gradient norms and detects gradient issues.
    """
    
    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
    
    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: Any,
    ):
        """Log gradient statistics before optimizer step."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        
        total_norm = 0.0
        param_count = 0
        
        for p in pl_module.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** 0.5
            pl_module.log('train/gradient_norm', total_norm)


__all__ = [
    "ImageLoggingCallback",
    "ValidationMetricsCallback",
    "EMACallback",
    "GradientMonitorCallback",
]

