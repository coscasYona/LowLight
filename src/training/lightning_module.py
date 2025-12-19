"""
PyTorch Lightning module for EMVA 1288 diffusion model training.

Encapsulates training, validation, and inference logic with proper
logging, checkpointing, and metric tracking.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig

from models import EMVA1288Diffusion
from models.noise_model import sample_params_max
from training.losses import HybridDiffusionLoss
from training.metrics import DenoisingMetrics


class EMVA1288LightningModule(pl.LightningModule):
    """
    Lightning module for EMVA 1288 physics-guided diffusion training.
    
    Handles:
    - Forward/backward passes with physics-based noise
    - Validation with image logging
    - Learning rate scheduling
    - Metric tracking and logging
    
    Args:
        model_config: Model configuration
        training_config: Training hyperparameters
    """
    
    def __init__(
        self,
        model_config: Optional[DictConfig] = None,
        training_config: Optional[DictConfig] = None,
        # Direct arguments for non-Hydra usage
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 32,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        num_steps: int = 4,
        time_embed_dim: int = 64,
        cond_embed_dim: int = 64,
        attn_type: str = "linear",
        scheduler: str = "ddpm",
        camera_type: str = "SonyA7S2",
        noise_code: str = "prq",
        learning_rate: float = 1e-4,
        l1_weight: float = 0.8,
        gradient_weight: float = 0.1,
        loss_scale: float = 10.0,
        use_ema: bool = False,
        ema_decay: float = 0.9999,
    ):
        super().__init__()
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters()
        
        # Extract config values if provided, else use direct args
        if model_config is not None:
            in_channels = getattr(model_config, 'in_channels', in_channels)
            out_channels = getattr(model_config, 'out_channels', out_channels)
            base_channels = getattr(model_config, 'base_channels', base_channels)
            channel_mults = getattr(model_config, 'channel_mults', channel_mults)
            num_steps = getattr(model_config, 'num_steps', num_steps)
            time_embed_dim = getattr(model_config, 'time_embed_dim', time_embed_dim)
            cond_embed_dim = getattr(model_config, 'cond_embed_dim', cond_embed_dim)
            attn_type = getattr(model_config, 'attn_type', attn_type)
            scheduler = getattr(model_config, 'scheduler', scheduler)
            camera_type = getattr(model_config, 'camera_type', camera_type)
            noise_code = getattr(model_config, 'noise_code', noise_code)
        
        if training_config is not None:
            learning_rate = getattr(training_config, 'learning_rate', learning_rate)
            l1_weight = getattr(training_config, 'l1_weight', l1_weight)
            gradient_weight = getattr(training_config, 'gradient_weight', gradient_weight)
            loss_scale = getattr(training_config, 'loss_scale', loss_scale)
            use_ema = getattr(training_config, 'use_ema', use_ema)
            ema_decay = getattr(training_config, 'ema_decay', ema_decay)
        
        # Store config
        self.learning_rate = learning_rate
        self.camera_type = camera_type
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.num_steps = num_steps
        
        # Ensure channel_mults is a tuple
        if isinstance(channel_mults, str):
            channel_mults = tuple(int(x) for x in channel_mults.split(','))
        elif not isinstance(channel_mults, tuple):
            channel_mults = tuple(channel_mults)
        
        # Create model
        self.model = EMVA1288Diffusion(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_steps=num_steps,
            time_embed_dim=time_embed_dim,
            cond_embed_dim=cond_embed_dim,
            attn_type=attn_type,
            scheduler=scheduler,
            camera_type=camera_type,
            noise_code=noise_code,
        )
        
        # Loss function
        self.loss_fn = HybridDiffusionLoss(
            l1_weight=l1_weight,
            gradient_weight=gradient_weight,
            loss_scale=loss_scale,
        )
        
        # Metrics
        self.metrics = DenoisingMetrics(data_range=1.0)
        
        # EMA model (optional)
        if use_ema:
            self.ema_model = self._create_ema_model()
        else:
            self.ema_model = None
        
        # Store validation outputs for epoch-end processing
        self.validation_step_outputs = []
    
    def _create_ema_model(self):
        """Create EMA copy of model."""
        import copy
        ema = copy.deepcopy(self.model)
        ema.requires_grad_(False)
        return ema
    
    def _update_ema(self):
        """Update EMA model weights."""
        if self.ema_model is None:
            return
        
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_model.parameters(), 
                self.model.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    param.data, alpha=1 - self.ema_decay
                )
    
    def forward(
        self, 
        x: torch.Tensor, 
        iso: torch.Tensor, 
        ratio: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass for inference."""
        return self.model(x, iso=iso, ratio=ratio, **kwargs)
    
    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Dict with 'clean', 'noisy', 'ratio', 'ISO'
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        img_gt = batch['clean']
        ratio = batch['ratio']
        iso = batch['ISO']
        
        batch_size = img_gt.size(0)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.num_steps, (batch_size,), 
            device=self.device, dtype=torch.long
        )
        
        # Generate base noise
        base_noise = torch.randn_like(img_gt)
        
        # Get camera parameters
        iso_np = iso.cpu().numpy().flatten()
        ratio_np = ratio.cpu().numpy().flatten()
        iso_val = int(iso_np[0]) if len(iso_np) > 0 else 6400
        ratio_val = float(ratio_np[0]) if len(ratio_np) > 0 else 200.0
        
        camera_params = sample_params_max(
            camera_type=self.camera_type,
            iso=iso_val,
            ratio=ratio_val
        )
        
        # Forward diffusion (simple Gaussian for training efficiency)
        sqrt_alpha = self.model._extract(
            self.model.sqrt_alphas_cumprod, timesteps, img_gt.shape
        )
        sqrt_one_minus_alpha = self.model._extract(
            self.model.sqrt_one_minus_alphas_cumprod, timesteps, img_gt.shape
        )
        noisy_state = sqrt_alpha * img_gt + sqrt_one_minus_alpha * base_noise
        
        # Predict noise
        pred_noise = self.model(
            noisy_state,
            iso=iso,
            ratio=ratio,
            timesteps=timesteps,
            predict_noise=True,
            camera_params=camera_params,
        )
        
        # Compute actual noise
        sqrt_one_minus_alpha_safe = torch.clamp(sqrt_one_minus_alpha, min=1e-6)
        actual_noise = (noisy_state - sqrt_alpha * img_gt) / sqrt_one_minus_alpha_safe
        actual_noise = torch.where(
            torch.isfinite(actual_noise), actual_noise, torch.zeros_like(actual_noise)
        )
        actual_noise = torch.clamp(actual_noise, min=-10.0, max=10.0)
        
        pred_noise = torch.where(
            torch.isfinite(pred_noise), pred_noise, torch.zeros_like(pred_noise)
        )
        
        # Compute loss
        loss = self.loss_fn(pred_noise, actual_noise)
        
        # Check for NaN loss
        if not torch.isfinite(loss):
            self.log('train/nan_loss', 1.0)
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Update EMA
        if self.use_ema and self.training:
            self._update_ema()
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_unscaled', loss / self.loss_fn.loss_scale)
        
        return loss
    
    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Validation step with physics-based noise.
        
        Args:
            batch: Dict with 'clean', 'noisy', 'ratio', 'ISO'
            batch_idx: Batch index
            
        Returns:
            Dict with validation metrics
        """
        img_gt = batch['clean']
        ratio = batch['ratio']
        iso = batch['ISO']
        
        batch_size = img_gt.size(0)
        
        # Sample timesteps
        timesteps = torch.randint(
            0, self.num_steps, (batch_size,), 
            device=self.device, dtype=torch.long
        )
        
        base_noise = torch.randn_like(img_gt)
        
        # Get camera params
        iso_np = iso.cpu().numpy().flatten()
        ratio_np = ratio.cpu().numpy().flatten()
        iso_val = int(iso_np[0]) if len(iso_np) > 0 else 6400
        ratio_val = float(ratio_np[0]) if len(ratio_np) > 0 else 200.0
        
        camera_params = sample_params_max(
            camera_type=self.camera_type,
            iso=iso_val,
            ratio=ratio_val
        )
        
        # Forward with physics noise for validation
        noisy_state = self.model.q_sample(
            img_gt, base_noise, timesteps,
            iso=iso, ratio=ratio,
            camera_params=camera_params,
            use_physics_noise=True,
        )
        
        # Predict noise
        pred_noise = self.model(
            noisy_state,
            iso=iso,
            ratio=ratio,
            timesteps=timesteps,
            predict_noise=True,
            camera_params=camera_params,
        )
        
        # Compute actual noise
        sqrt_alpha = self.model._extract(
            self.model.sqrt_alphas_cumprod, timesteps, img_gt.shape
        )
        sqrt_one_minus_alpha = self.model._extract(
            self.model.sqrt_one_minus_alphas_cumprod, timesteps, img_gt.shape
        )
        sqrt_one_minus_alpha = torch.clamp(sqrt_one_minus_alpha, min=1e-6)
        
        actual_noise = (noisy_state - sqrt_alpha * img_gt) / sqrt_one_minus_alpha
        actual_noise = torch.where(
            torch.isfinite(actual_noise), actual_noise, torch.zeros_like(actual_noise)
        )
        actual_noise = torch.clamp(actual_noise, min=-10.0, max=10.0)
        pred_noise = torch.where(
            torch.isfinite(pred_noise), pred_noise, torch.zeros_like(pred_noise)
        )
        
        # Compute loss
        loss = self.loss_fn(pred_noise, actual_noise)
        
        if torch.isfinite(loss):
            output = {
                'val_loss': loss,
                'batch_idx': batch_idx,
            }
            
            # Store first batch for image logging
            if batch_idx == 0:
                output['img_gt'] = img_gt[:min(4, batch_size)].detach()
                output['noisy_state'] = noisy_state[:min(4, batch_size)].detach()
                output['iso'] = iso[:min(4, batch_size)]
                output['ratio'] = ratio[:min(4, batch_size)]
            
            self.validation_step_outputs.append(output)
            return output
        
        return {'val_loss': torch.tensor(0.0, device=self.device)}
    
    def on_validation_epoch_end(self):
        """Aggregate validation metrics at epoch end."""
        if not self.validation_step_outputs:
            return
        
        # Average validation loss
        val_losses = [
            x['val_loss'] for x in self.validation_step_outputs 
            if torch.isfinite(x['val_loss'])
        ]
        
        if val_losses:
            avg_val_loss = torch.stack(val_losses).mean()
            self.log('val/loss', avg_val_loss, prog_bar=True, sync_dist=True)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate
        )
        
        # Milestone-based LR scheduler (matching original implementation)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[100, 180],
            gamma=0.5,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            },
        }
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        """Add EMA state to checkpoint."""
        if self.ema_model is not None:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        """Load EMA state from checkpoint."""
        if self.ema_model is not None and 'ema_state_dict' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])


__all__ = ["EMVA1288LightningModule"]

