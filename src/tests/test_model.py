"""
Tests for EMVA 1288 diffusion model components.
"""

import pytest
import torch

from src.models import (
    EMVA1288Diffusion,
    SlimUNet,
    EMVA1288PhysicsEncoder,
    EMVA1288NoiseModel,
    DiffusionScheduler,
)


class TestSlimUNet:
    """Tests for SlimUNet architecture."""
    
    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        unet = SlimUNet(
            in_ch=4,
            out_ch=4,
            base_ch=32,
            channel_mults=(1, 2, 4),
            emb_dim=128,
        )
        
        x = torch.randn(2, 4, 64, 64)
        emb = torch.randn(2, 128)
        
        output = unet(x, emb)
        
        assert output.shape == x.shape
    
    def test_different_sizes(self):
        """Test with different input sizes."""
        unet = SlimUNet(
            in_ch=4,
            out_ch=4,
            base_ch=16,
            channel_mults=(1, 2),
            emb_dim=64,
        )
        
        for size in [32, 64, 128]:
            x = torch.randn(1, 4, size, size)
            emb = torch.randn(1, 64)
            output = unet(x, emb)
            assert output.shape == x.shape


class TestPhysicsEncoder:
    """Tests for EMVA 1288 physics encoder."""
    
    def test_forward_shape(self):
        """Test output shape."""
        encoder = EMVA1288PhysicsEncoder(cond_dim=64)
        
        iso = torch.tensor([[6400.0]])
        ratio = torch.tensor([[200.0]])
        
        output = encoder(iso, ratio)
        
        assert output.shape == (1, 64)
    
    def test_batch_processing(self):
        """Test with batch of inputs."""
        encoder = EMVA1288PhysicsEncoder(cond_dim=32)
        
        batch_size = 4
        iso = torch.rand(batch_size, 1) * 10000 + 100
        ratio = torch.rand(batch_size, 1) * 300 + 1
        
        output = encoder(iso, ratio)
        
        assert output.shape == (batch_size, 32)
    
    def test_with_camera_params(self):
        """Test with explicit camera parameters."""
        encoder = EMVA1288PhysicsEncoder(cond_dim=64)
        
        iso = torch.tensor([[6400.0]])
        ratio = torch.tensor([[200.0]])
        camera_params = {
            'K': 6.12032,
            'sigGs': 7.1163535,
            'sigR': 0.7218788,
        }
        
        output = encoder(iso, ratio, camera_params)
        
        assert output.shape == (1, 64)
        assert torch.isfinite(output).all()


class TestDiffusionScheduler:
    """Tests for diffusion scheduler."""
    
    def test_linear_schedule(self):
        """Test linear noise schedule."""
        scheduler = DiffusionScheduler(
            num_steps=100,
            schedule_type='linear',
        )
        
        # Alphas should decrease
        assert scheduler.alphas_cumprod[0] > scheduler.alphas_cumprod[-1]
        
        # All values should be in (0, 1)
        assert (scheduler.alphas_cumprod > 0).all()
        assert (scheduler.alphas_cumprod < 1).all()
    
    def test_q_sample(self):
        """Test forward diffusion."""
        scheduler = DiffusionScheduler(num_steps=100)
        
        x_start = torch.randn(2, 4, 32, 32)
        noise = torch.randn_like(x_start)
        timesteps = torch.tensor([50, 75])
        
        noisy = scheduler.q_sample(x_start, noise, timesteps)
        
        assert noisy.shape == x_start.shape
        assert torch.isfinite(noisy).all()
    
    def test_ddpm_step(self):
        """Test DDPM reverse step."""
        scheduler = DiffusionScheduler(num_steps=100)
        
        x_t = torch.randn(2, 4, 32, 32)
        predicted_noise = torch.randn_like(x_t)
        timestep = torch.tensor([50, 50])
        
        x_prev = scheduler.ddpm_step(x_t, predicted_noise, timestep)
        
        assert x_prev.shape == x_t.shape
        assert torch.isfinite(x_prev).all()


class TestEMVA1288Diffusion:
    """Tests for main diffusion model."""
    
    def test_forward_noise_prediction(self):
        """Test noise prediction mode."""
        model = EMVA1288Diffusion(
            in_channels=4,
            out_channels=4,
            base_channels=16,
            channel_mults=(1, 2),
            num_steps=10,
        )
        
        x = torch.randn(2, 4, 32, 32)
        iso = torch.tensor([[6400.0], [3200.0]])
        ratio = torch.tensor([[200.0], [100.0]])
        timesteps = torch.tensor([5, 3])
        
        noise_pred = model(
            x, iso=iso, ratio=ratio, 
            timesteps=timesteps, predict_noise=True
        )
        
        assert noise_pred.shape == x.shape
        assert torch.isfinite(noise_pred).all()
    
    def test_sampling(self):
        """Test denoising sampling."""
        model = EMVA1288Diffusion(
            in_channels=4,
            out_channels=4,
            base_channels=16,
            channel_mults=(1, 2),
            num_steps=5,
        )
        
        x = torch.randn(1, 4, 32, 32).clamp(0, 1)
        iso = torch.tensor([[6400.0]])
        ratio = torch.tensor([[200.0]])
        
        denoised = model.sample(x, iso=iso, ratio=ratio, num_steps=5)
        
        assert denoised.shape == x.shape
        assert (denoised >= 0).all() and (denoised <= 1).all()


class TestNoiseModel:
    """Tests for EMVA 1288 noise model."""
    
    def test_get_params(self):
        """Test parameter sampling."""
        noise_model = EMVA1288NoiseModel(
            camera_type='SonyA7S2',
            noise_code='prq',
        )
        
        params = noise_model.get_params(iso=6400, ratio=200)
        
        assert 'K' in params
        assert 'sigGs' in params
        assert 'ratio' in params
        assert params['ratio'] == 200
    
    def test_generate_noise_torch(self):
        """Test PyTorch noise generation."""
        noise_model = EMVA1288NoiseModel(
            camera_type='SonyA7S2',
            noise_code='prq',
        )
        
        clean = torch.rand(1, 4, 64, 64)
        params = noise_model.get_params(iso=6400, ratio=200)
        
        noisy = noise_model.generate_noise_torch(clean, params)
        
        assert noisy.shape == clean.shape
        assert torch.isfinite(noisy).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

