#!/usr/bin/env python3
"""
Main training entry point for EMVA 1288 Physics-Guided Diffusion Model.

Uses Hydra for configuration management and PyTorch Lightning for training.

Usage:
    python train.py                           # Use default config
    python train.py training.epochs=100       # Override epochs
    python train.py model=unet                # Use UNet model config
    python train.py data=eld                  # Use ELD dataset config
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from training import EMVA1288LightningModule
from training.callbacks import ImageLoggingCallback, GradientMonitorCallback
from data import EMVA1288DataModule


def create_callbacks(cfg: DictConfig) -> list:
    """Create training callbacks from config."""
    callbacks = []
    
    # Model checkpoint - saves best (by PSNR) and optionally last
    # every_n_epochs reduces I/O overhead for large checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.paths.save_dir, 'checkpoints'),
        filename=cfg.checkpoint.filename,
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=cfg.checkpoint.save_last,
        every_n_epochs=cfg.checkpoint.get('every_n_epochs', 1),
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if cfg.early_stopping.enabled:
        early_stop_callback = EarlyStopping(
            monitor=cfg.early_stopping.monitor,
            patience=cfg.early_stopping.patience,
            min_delta=cfg.early_stopping.min_delta,
            mode=cfg.early_stopping.mode,
            verbose=True,
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Image logging (training and validation)
    image_logger = ImageLoggingCallback(
        log_every_n_epochs=cfg.logging.image_log_every_n_epochs,
        num_samples=cfg.logging.num_sample_images,
        save_to_disk=True,
        save_dir=cfg.paths.save_dir,
        log_training_images=True,
        log_validation_images=True,
        compute_metrics=True,
    )
    callbacks.append(image_logger)
    
    # Gradient monitoring
    gradient_monitor = GradientMonitorCallback(
        log_every_n_steps=cfg.logging.log_every_n_steps
    )
    callbacks.append(gradient_monitor)
    
    # Progress bar
    callbacks.append(RichProgressBar())
    
    return callbacks


def train_with_config(cfg: DictConfig) -> float:
    """
    Core training function that can be called with a config object.
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Best validation loss achieved
    """
    # Print configuration
    print("=" * 60)
    print("EMVA 1288 Physics-Guided Diffusion Training")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    
    # Set seed for reproducibility
    seed = int(cfg.get('seed', 42))
    pl.seed_everything(seed, workers=True)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Set deterministic flags before model creation
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create directories
    os.makedirs(cfg.paths.save_dir, exist_ok=True)
    os.makedirs(cfg.paths.log_dir, exist_ok=True)
    
    # Create data module
    data_config = cfg.data if hasattr(cfg, 'data') else cfg

    # Resolve paths relative to workspace root (Hydra changes cwd)
    # Use script location to find workspace root (script is at src/scripts/train.py, workspace is 2 levels up)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Fallback to get_original_cwd if available, but prefer workspace_root
    try:
        original_cwd = get_original_cwd()
        # Use workspace_root if it's more specific (contains 'lowlight')
        if 'lowlight' in workspace_root:
            base_dir = workspace_root
        else:
            base_dir = original_cwd
    except:
        base_dir = workspace_root
    
    def resolve_path(path: str) -> str:
        """Resolve path relative to workspace root if relative."""
        if os.path.isabs(path):
            return os.path.normpath(path)
        # Handle paths starting with ../
        if path.startswith('../'):
            # Remove ../ and resolve relative to workspace root
            rel_path = path[3:]  # Remove '../'
            return os.path.normpath(os.path.join(base_dir, rel_path))
        # Regular relative path
        return os.path.normpath(os.path.join(base_dir, path))
    
    train_dir = resolve_path(data_config.train_dir)
    train_list = resolve_path(data_config.train_list)
    val_dir = resolve_path(data_config.get('val_dir', data_config.train_dir)) if data_config.get('val_dir') else None
    val_list = resolve_path(data_config.val_list) if data_config.get('val_list') else None

    datamodule = EMVA1288DataModule(
        train_dir=train_dir,
        val_dir=val_dir,
        train_list=train_list,
        val_list=val_list,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.hardware.num_workers,
        patch_size=data_config.patch_size,
        pin_memory=data_config.pin_memory,
        use_sid_raw=data_config.get('use_sid_raw', True),
        use_fuji_raw=data_config.get('use_fuji_raw', False),
        val_split=data_config.get('val_split', 0.1),
    )
    
    # Create model
    model_config = cfg.model if hasattr(cfg, 'model') else cfg
    
    # Convert channel_mults to tuple if needed - ensure consistent type
    channel_mults = model_config.channel_mults
    if isinstance(channel_mults, str):
        channel_mults = tuple(int(x) for x in channel_mults.split(','))
    elif not isinstance(channel_mults, tuple):
        channel_mults = tuple(channel_mults)
    
    # Ensure all config values are explicit types to avoid Hydra/OmegaConf issues
    use_ema = bool(cfg.training.get('use_ema', False))
    ema_decay = float(cfg.training.get('ema_decay', 0.9999))
    
    model = EMVA1288LightningModule(
        in_channels=int(model_config.in_channels),
        out_channels=int(model_config.out_channels),
        base_channels=int(model_config.base_channels),
        channel_mults=channel_mults,
        num_steps=int(model_config.num_steps),
        time_embed_dim=int(model_config.time_embed_dim),
        cond_embed_dim=int(model_config.cond_embed_dim),
        attn_type=str(model_config.attn_type),
        scheduler=str(model_config.scheduler),
        camera_type=str(model_config.get('camera_type', 'SonyA7S2')),
        noise_code=str(model_config.get('noise_code', 'prq')),
        learning_rate=float(cfg.training.learning_rate),
        l1_weight=float(cfg.training.l1_weight),
        gradient_weight=float(cfg.training.gradient_weight),
        loss_scale=float(cfg.training.loss_scale),
        use_ema=use_ema,  # Explicitly convert to bool
        ema_decay=ema_decay,
    )
    
    # Log model parameter count
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameter count: {param_count} (trainable: {trainable_count})")
    
    # Create callbacks
    callbacks = create_callbacks(cfg)
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=cfg.paths.log_dir,
        name='emva1288_diffusion',
        version=None,
    )
    
    # Create trainer
    # Trainer configuration
    trainer_kwargs = dict(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        precision=cfg.hardware.precision,
        strategy='auto',  # Let Lightning choose strategy based on device count
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        gradient_clip_val=cfg.training.grad_clip_max if cfg.training.use_grad_clip else None,
        deterministic=False,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Optional debug knobs (can be passed via CLI as e.g. trainer.limit_train_batches=1)
    if hasattr(cfg, "trainer"):
        for key in (
            "fast_dev_run",
            "limit_train_batches",
            "limit_val_batches",
            "num_sanity_val_steps",
            "max_steps",
        ):
            if key in cfg.trainer and cfg.trainer.get(key) is not None:
                trainer_kwargs[key] = cfg.trainer.get(key)

    trainer = pl.Trainer(**trainer_kwargs)
    
    # Train
    print("\nStarting training...")
    try:
        trainer.fit(model, datamodule)
    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Get best checkpoint
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_val_loss = trainer.checkpoint_callback.best_model_score
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best model: {best_model_path}")
    if best_val_loss is not None:
        print(f"Best validation loss: {best_val_loss:.4f}")
    else:
        print("Best validation loss: N/A")
    print("=" * 60)
    
    return float(best_val_loss) if best_val_loss is not None else float('inf')


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> float:
    """
    Main training function (Hydra entry point).
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Best validation loss achieved
    """
    return train_with_config(cfg)


if __name__ == "__main__":
    main()

