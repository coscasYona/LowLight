"""
Tests for PyTorch Lightning module.
"""

import pytest
import torch
import pytorch_lightning as pl

from src.training import EMVA1288LightningModule
from src.training.losses import HybridDiffusionLoss


class TestHybridDiffusionLoss:
    """Tests for hybrid loss function."""
    
    def test_mse_only(self):
        """Test with L1 weight = 0 (MSE only)."""
        loss_fn = HybridDiffusionLoss(l1_weight=0.0, gradient_weight=0.0)
        
        pred = torch.randn(2, 4, 32, 32)
        target = torch.randn(2, 4, 32, 32)
        
        loss = loss_fn(pred, target)
        
        assert loss.ndim == 0  # Scalar
        assert loss >= 0
    
    def test_l1_only(self):
        """Test with L1 weight = 1."""
        loss_fn = HybridDiffusionLoss(l1_weight=1.0, gradient_weight=0.0)
        
        pred = torch.randn(2, 4, 32, 32)
        target = torch.randn(2, 4, 32, 32)
        
        loss = loss_fn(pred, target)
        
        assert loss >= 0
    
    def test_with_gradient(self):
        """Test with gradient loss enabled."""
        loss_fn = HybridDiffusionLoss(l1_weight=0.5, gradient_weight=0.2)
        
        pred = torch.randn(2, 4, 32, 32)
        target = torch.randn(2, 4, 32, 32)
        
        loss, components = loss_fn(pred, target, return_components=True)
        
        assert loss >= 0
        assert 'mse' in components
        assert 'l1' in components
        assert 'gradient' in components
    
    def test_loss_scaling(self):
        """Test loss scaling."""
        loss_fn_scaled = HybridDiffusionLoss(loss_scale=10.0)
        loss_fn_unscaled = HybridDiffusionLoss(loss_scale=1.0)
        
        pred = torch.randn(2, 4, 32, 32)
        target = torch.randn(2, 4, 32, 32)
        
        loss_scaled = loss_fn_scaled(pred, target)
        loss_unscaled = loss_fn_unscaled(pred, target)
        
        # Scaled loss should be approximately 10x larger
        assert abs(loss_scaled / loss_unscaled - 10.0) < 0.1


class TestEMVA1288LightningModule:
    """Tests for Lightning module."""
    
    @pytest.fixture
    def model(self):
        """Create test model."""
        return EMVA1288LightningModule(
            in_channels=4,
            out_channels=4,
            base_channels=16,
            channel_mults=(1, 2),
            num_steps=5,
            time_embed_dim=32,
            cond_embed_dim=32,
            learning_rate=1e-4,
        )
    
    def test_forward(self, model):
        """Test forward pass."""
        x = torch.randn(2, 4, 32, 32)
        iso = torch.tensor([[6400.0], [3200.0]])
        ratio = torch.tensor([[200.0], [100.0]])
        
        output = model(x, iso, ratio)
        
        assert output.shape == x.shape
    
    def test_training_step(self, model):
        """Test training step."""
        batch = {
            'clean': torch.randn(2, 4, 32, 32).clamp(0, 1),
            'noisy': torch.randn(2, 4, 32, 32).clamp(0, 1),
            'ratio': torch.tensor([[200.0], [100.0]]),
            'ISO': torch.tensor([[6400.0], [3200.0]]),
        }
        
        loss = model.training_step(batch, 0)
        
        assert loss.ndim == 0
        assert torch.isfinite(loss)
    
    def test_validation_step(self, model):
        """Test validation step."""
        batch = {
            'clean': torch.randn(2, 4, 32, 32).clamp(0, 1),
            'noisy': torch.randn(2, 4, 32, 32).clamp(0, 1),
            'ratio': torch.tensor([[200.0], [100.0]]),
            'ISO': torch.tensor([[6400.0], [3200.0]]),
        }
        
        output = model.validation_step(batch, 0)
        
        assert 'val_loss' in output
    
    def test_configure_optimizers(self, model):
        """Test optimizer configuration."""
        config = model.configure_optimizers()
        
        assert 'optimizer' in config
        assert 'lr_scheduler' in config
        
        optimizer = config['optimizer']
        assert isinstance(optimizer, torch.optim.Adam)
    
    def test_hyperparameters_saved(self, model):
        """Test that hyperparameters are saved."""
        hparams = model.hparams
        
        assert 'in_channels' in hparams
        assert 'learning_rate' in hparams
        assert hparams['in_channels'] == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

