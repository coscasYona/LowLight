"""
PyTorch Lightning module for EMVA 1288 diffusion model training.

Encapsulates training, validation, and inference logic with proper
logging, checkpointing, and metric tracking.

Supports edge conditioning and enhanced SOTA loss functions.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig

from models import EMVA1288Diffusion
from models.noise_model import sample_params_max
from training.losses import HybridDiffusionLoss, EnhancedHybridLoss
from training.metrics import DenoisingMetrics


class EMVA1288LightningModule(pl.LightningModule):
    """
    Lightning module for EMVA 1288 physics-guided diffusion training.
    
    Handles:
    - Forward/backward passes with physics-based noise
    - Validation with image logging
    - Learning rate scheduling
    - Metric tracking and logging
    - Optional edge conditioning for improved edge preservation
    
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
        use_measurement_cond: bool = False,
        use_edge_cond: bool = False,
        edge_detector: str = "canny",
        learning_rate: float = 1e-4,
        l1_weight: float = 0.8,
        gradient_weight: float = 0.1,
        loss_scale: float = 10.0,
        use_ema: bool = False,
        ema_decay: float = 0.9999,
        # Enhanced loss options
        use_enhanced_loss: bool = False,
        mse_weight: float = 0.25,
        charbonnier_weight: float = 0.25,
        frequency_weight: float = 0.1,
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
            use_measurement_cond = getattr(model_config, 'use_measurement_cond', use_measurement_cond)
            use_edge_cond = getattr(model_config, 'use_edge_cond', use_edge_cond)
            edge_detector = getattr(model_config, 'edge_detector', edge_detector)
        
        if training_config is not None:
            learning_rate = getattr(training_config, 'learning_rate', learning_rate)
            l1_weight = getattr(training_config, 'l1_weight', l1_weight)
            gradient_weight = getattr(training_config, 'gradient_weight', gradient_weight)
            loss_scale = getattr(training_config, 'loss_scale', loss_scale)
            use_ema = getattr(training_config, 'use_ema', use_ema)
            ema_decay = getattr(training_config, 'ema_decay', ema_decay)
            use_enhanced_loss = getattr(training_config, 'use_enhanced_loss', use_enhanced_loss)
            mse_weight = getattr(training_config, 'mse_weight', mse_weight)
            charbonnier_weight = getattr(training_config, 'charbonnier_weight', charbonnier_weight)
            frequency_weight = getattr(training_config, 'frequency_weight', frequency_weight)
        
        # Store config
        self.learning_rate = learning_rate
        self.camera_type = camera_type
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.num_steps = num_steps
        self.use_measurement_cond = use_measurement_cond
        self.use_edge_cond = use_edge_cond
        
        # Ensure channel_mults is a tuple
        if isinstance(channel_mults, str):
            channel_mults = tuple(int(x) for x in channel_mults.split(','))
        elif not isinstance(channel_mults, tuple):
            channel_mults = tuple(channel_mults)
        
        # Double input channels when using measurement conditioning (x_t concat with noisy)
        model_in_channels = in_channels * 2 if use_measurement_cond else in_channels
        
        # Create model with edge conditioning support
        self.model = EMVA1288Diffusion(
            in_channels=model_in_channels,
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
            use_measurement_cond=use_measurement_cond,
            use_edge_cond=use_edge_cond,
            edge_detector=edge_detector,
        )
        
        # Loss function (enhanced or standard)
        if use_enhanced_loss:
            self.loss_fn = EnhancedHybridLoss(
                mse_weight=mse_weight,
                l1_weight=l1_weight,
                charbonnier_weight=charbonnier_weight,
                frequency_weight=frequency_weight,
                gradient_weight=gradient_weight,
                loss_scale=loss_scale,
            )
        else:
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
        self.training_step_outputs = []
    
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
        Training step with simple Gaussian diffusion (matching legacy implementation).
        
        Note: Legacy code used simple Gaussian for training (more stable convergence)
        while physics noise is used for validation. This asymmetry worked well
        because the model learns the noise structure from the conditioning (ISO/ratio).
        
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
        
        # Get camera parameters for conditioning (uses first sample's ISO/ratio like legacy)
        iso_np = iso.cpu().numpy().flatten()
        ratio_np = ratio.cpu().numpy().flatten()
        iso_val = int(iso_np[0]) if len(iso_np) > 0 else 6400
        ratio_val = float(ratio_np[0]) if len(ratio_np) > 0 else 200.0
        camera_params = sample_params_max(
            camera_type=self.camera_type,
            iso=iso_val,
            ratio=ratio_val
        )
        
        # Get real noisy measurement for conditioning (if available and enabled)
        img_noisy = batch.get('noisy', None)
        cond_image = img_noisy if self.use_measurement_cond and img_noisy is not None else None
        
        # Compute edge features from clean image for training (if enabled)
        edge_feat = None
        if self.use_edge_cond:
            edge_feat = self.model.compute_edge_features(img_gt)
        
        # Use physics-based noise for training to match validation
        # This ensures the model learns the correct noise distribution
        noisy_state = self.model.q_sample(
            img_gt, base_noise, timesteps,
            iso=iso, ratio=ratio,
            camera_params=camera_params,
            use_physics_noise=True,
        )

        # Predict noise (with measurement and edge conditioning if enabled)
        pred_noise = self.model(
            noisy_state,
            iso=iso,
            ratio=ratio,
            timesteps=timesteps,
            predict_noise=True,
            camera_params=camera_params,
            cond_image=cond_image,
            edge_feat=edge_feat,
        )

        # Compute actual noise using the same formula as validation
        # Even with physics noise, we compute the theoretical noise for loss computation
        # Ensure high precision for noise computation to avoid gradient issues
        sqrt_alpha = self.model._extract(
            self.model.sqrt_alphas_cumprod, timesteps, img_gt.shape
        ).float()  # Ensure float32
        sqrt_one_minus_alpha = self.model._extract(
            self.model.sqrt_one_minus_alphas_cumprod, timesteps, img_gt.shape
        ).float()  # Ensure float32
        sqrt_one_minus_alpha_safe = torch.clamp(sqrt_one_minus_alpha, min=1e-6)

        # Compute noise in high precision
        actual_noise = (noisy_state.float() - sqrt_alpha * img_gt.float()) / sqrt_one_minus_alpha_safe
        actual_noise = torch.where(
            torch.isfinite(actual_noise), actual_noise, torch.zeros_like(actual_noise)
        )
        actual_noise = torch.clamp(actual_noise, min=-10.0, max=10.0)

        # Ensure predicted noise is also in high precision
        pred_noise = pred_noise.float()
        pred_noise = torch.where(
            torch.isfinite(pred_noise), pred_noise, torch.zeros_like(pred_noise)
        )

        # Compute loss with high precision tensors
        loss = self.loss_fn(pred_noise, actual_noise)
        
        # Check for NaN loss
        if not torch.isfinite(loss):
            self.log('train/nan_loss', 1.0)
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Update EMA
        if self.use_ema and self.training:
            self._update_ema()
        
        # Log metrics (step-level only to reduce clutter)
        self.log('train/loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train/loss_unscaled', loss / self.loss_fn.loss_scale)

        # Store last batch for training image logging (only in training mode)
        if self.training and batch_idx == 0:  # Store first batch of each epoch
            train_output = {
                'batch_idx': batch_idx,
                'img_gt': img_gt[:min(4, batch_size)].detach().cpu(),
                'noisy_state': noisy_state[:min(4, batch_size)].detach().cpu(),
                'iso': iso[:min(4, batch_size)].cpu(),
                'ratio': ratio[:min(4, batch_size)].cpu(),
                'camera_params_list': [camera_params] * min(4, batch_size),
            }
            if img_noisy is not None:
                train_output['img_noisy'] = img_noisy[:min(4, batch_size)].detach().cpu()
            self.training_step_outputs = [train_output]  # Replace with latest batch

        return loss
    
    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Validation step with physics-based noise and real denoising metrics.
        
        Args:
            batch: Dict with 'clean', 'noisy', 'ratio', 'ISO'
            batch_idx: Batch index
            
        Returns:
            Dict with validation metrics
        """
        img_gt = batch['clean']
        img_noisy = batch.get('noisy', None)  # Real noisy measurement from dataset
        ratio = batch['ratio']
        iso = batch['ISO']
        
        batch_size = img_gt.size(0)
        
        # Sample timesteps
        timesteps = torch.randint(
            0, self.num_steps, (batch_size,), 
            device=self.device, dtype=torch.long
        )
        
        base_noise = torch.randn_like(img_gt)
        
        # Get per-sample camera params
        iso_np = iso.cpu().numpy().flatten()
        ratio_np = ratio.cpu().numpy().flatten()
        camera_params_list = []
        for i in range(batch_size):
            iso_val = int(iso_np[i]) if i < len(iso_np) else 6400
            ratio_val = float(ratio_np[i]) if i < len(ratio_np) else 200.0
            params = sample_params_max(
                camera_type=self.camera_type,
                iso=iso_val,
                ratio=ratio_val
            )
            camera_params_list.append(params)
        
        # Get real noisy measurement for conditioning (if available and enabled)
        cond_image = img_noisy if self.use_measurement_cond and img_noisy is not None else None
        
        # Compute edge features from clean image for validation (if enabled)
        # During validation, we use clean image edges as the ground truth target
        edge_feat = None
        if self.use_edge_cond:
            edge_feat = self.model.compute_edge_features(img_gt)
        
        # Forward with physics noise for validation
        noisy_state = self.model.q_sample(
            img_gt, base_noise, timesteps,
            iso=iso, ratio=ratio,
            camera_params=camera_params_list,
            use_physics_noise=True,
        )
        
        # Predict noise (with measurement and edge conditioning if enabled)
        pred_noise = self.model(
            noisy_state,
            iso=iso,
            ratio=ratio,
            timesteps=timesteps,
            predict_noise=True,
            camera_params=camera_params_list,
            cond_image=cond_image,
            edge_feat=edge_feat,
        )
        
        # Compute actual noise with high precision
        sqrt_alpha = self.model._extract(
            self.model.sqrt_alphas_cumprod, timesteps, img_gt.shape
        ).float()  # Ensure float32
        sqrt_one_minus_alpha = self.model._extract(
            self.model.sqrt_one_minus_alphas_cumprod, timesteps, img_gt.shape
        ).float()  # Ensure float32
        sqrt_one_minus_alpha = torch.clamp(sqrt_one_minus_alpha, min=1e-6)

        actual_noise = (noisy_state.float() - sqrt_alpha * img_gt.float()) / sqrt_one_minus_alpha
        actual_noise = torch.where(
            torch.isfinite(actual_noise), actual_noise, torch.zeros_like(actual_noise)
        )
        actual_noise = torch.clamp(actual_noise, min=-10.0, max=10.0)
        pred_noise = pred_noise.float()  # Ensure float32
        pred_noise = torch.where(
            torch.isfinite(pred_noise), pred_noise, torch.zeros_like(pred_noise)
        )
        
        # Compute noise prediction loss
        loss = self.loss_fn(pred_noise, actual_noise)
        
        if torch.isfinite(loss):
            output = {
                'val_loss': loss,
                'batch_idx': batch_idx,
            }
            
            # Store first batch for image logging and real denoising metrics
            if batch_idx == 0:
                output['img_gt'] = img_gt[:min(4, batch_size)].detach()
                output['noisy_state'] = noisy_state[:min(4, batch_size)].detach()
                output['iso'] = iso[:min(4, batch_size)]
                output['ratio'] = ratio[:min(4, batch_size)]
                output['camera_params_list'] = camera_params_list[:min(4, batch_size)]
                
                # Store real noisy input for denoising evaluation
                if img_noisy is not None:
                    output['img_noisy'] = img_noisy[:min(4, batch_size)].detach()
            
            self.validation_step_outputs.append(output)
            return output
        
        return {'val_loss': torch.tensor(0.0, device=self.device)}
    
    def on_validation_epoch_end(self):
        """Aggregate validation metrics at epoch end, including real denoising metrics."""
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
        
        # Compute denoising metrics on first batch (if real noisy input available)
        first_batch = self.validation_step_outputs[0] if self.validation_step_outputs else None
        if first_batch and 'img_noisy' in first_batch and 'img_gt' in first_batch:
            img_noisy = first_batch['img_noisy']
            img_gt = first_batch['img_gt']
            iso = first_batch.get('iso')
            ratio = first_batch.get('ratio')
            camera_params_list = first_batch.get('camera_params_list')

            # Sample denoised output from real noisy measurement
            with torch.no_grad():
                # Compute edge features from noisy image for inference
                # (In real inference, we only have noisy input)
                edge_feat = None
                if self.use_edge_cond:
                    edge_feat = self.model.compute_edge_features(img_noisy)
                
                denoised = self.model.sample(
                    img_noisy,
                    iso=iso,
                    ratio=ratio,
                    num_steps=self.num_steps,
                    camera_params=camera_params_list[0] if camera_params_list else None,
                    cond_image=img_noisy if self.model.use_measurement_cond else None,
                    edge_feat=edge_feat,
                )

                # Compute PSNR and SSIM on denoised output
                metrics = self.metrics(img_gt, img_noisy, denoised)

                self.log('val/psnr', metrics['psnr'], prog_bar=True, sync_dist=True)
                self.log('val/ssim', metrics['ssim'], sync_dist=True)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate
        )
        
        # LR schedule with warmup for physics noise training:
        # - Epoch 1-9: warmup from 2e-5 to 2e-4 (×0.1 to ×1.0)
        # - Epoch 10-99: 2e-4
        # - Epoch 100-179: 1e-4 (×0.5)
        # - Epoch 180+: 2e-5 (×0.1)
        def lr_lambda(epoch):
            if epoch < 10:
                return 0.1 + 0.9 * (epoch / 9)  # Warmup from 0.1x to 1.0x
            elif epoch < 100:
                return 1.0
            elif epoch < 180:
                return 0.5
            else:
                return 0.1
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
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

