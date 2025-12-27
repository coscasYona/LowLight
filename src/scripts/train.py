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
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.paths.save_dir, 'checkpoints'),
        filename=cfg.checkpoint.filename,
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=cfg.checkpoint.save_last,
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
    pl.seed_everything(cfg.seed, workers=True)
    
    # Create directories
    os.makedirs(cfg.paths.save_dir, exist_ok=True)
    os.makedirs(cfg.paths.log_dir, exist_ok=True)
    
    # Create data module
    data_config = cfg.data if hasattr(cfg, 'data') else cfg


    datamodule = EMVA1288DataModule(
        train_dir=data_config.train_dir,
        val_dir=data_config.get('val_dir', data_config.train_dir),
        train_list=data_config.train_list,
        val_list=data_config.get('val_list', None),
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
    
    # Convert channel_mults to tuple if needed
    channel_mults = model_config.channel_mults
    if isinstance(channel_mults, str):
        channel_mults = tuple(int(x) for x in channel_mults.split(','))
    elif not isinstance(channel_mults, tuple):
        channel_mults = tuple(channel_mults)
    
    model = EMVA1288LightningModule(
        in_channels=model_config.in_channels,
        out_channels=model_config.out_channels,
        base_channels=model_config.base_channels,
        channel_mults=channel_mults,
        num_steps=model_config.num_steps,
        time_embed_dim=model_config.time_embed_dim,
        cond_embed_dim=model_config.cond_embed_dim,
        attn_type=model_config.attn_type,
        scheduler=model_config.scheduler,
        camera_type=model_config.get('camera_type', 'SonyA7S2'),
        noise_code=model_config.get('noise_code', 'prq'),
        learning_rate=cfg.training.learning_rate,
        l1_weight=cfg.training.l1_weight,
        gradient_weight=cfg.training.gradient_weight,
        loss_scale=cfg.training.loss_scale,
        use_ema=cfg.training.use_ema,
        ema_decay=cfg.training.ema_decay,
    )
    
    # Create callbacks
    callbacks = create_callbacks(cfg)
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=cfg.paths.log_dir,
        name='emva1288_diffusion',
        version=None,
    )
    
    # Create trainer
    trainer_kwargs = dict(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        precision=cfg.hardware.precision,
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

