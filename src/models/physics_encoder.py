"""
EMVA 1288 Physics Encoder for diffusion conditioning.

Encodes camera parameters (ISO, ratio, exposure) into features
that represent the noise characteristics based on the EMVA 1288 standard.
"""

from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn


def _init_linear(layer: nn.Module) -> None:
    """Initialize linear layers."""
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class EMVA1288PhysicsEncoder(nn.Module):
    """
    Physics encoder based on EMVA 1288 standard.
    
    Encodes camera parameters (ISO, ratio, exposure) into features
    that represent the noise characteristics for conditioning the
    diffusion model.
    
    Args:
        cond_dim: Output conditioning dimension
    """
    
    def __init__(self, cond_dim: int):
        super().__init__()
        # Input features: [iso_norm, ratio_norm, log_K, log_sigGs, log_sigR, 
        #                  shot_variance, read_variance, snr]
        self.embed = nn.Sequential(
            nn.Linear(8, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.apply(_init_linear)

    def forward(
        self, 
        iso: torch.Tensor, 
        ratio: torch.Tensor, 
        camera_params: Optional[Union[Dict, Sequence[Dict]]] = None
    ) -> torch.Tensor:
        """
        Encode ISO and ratio into physics-based features.
        
        Args:
            iso: ISO sensitivity [B] or [B, 1]
            ratio: Exposure ratio [B] or [B, 1]
            camera_params: Optional dict or list of dicts (per-sample) with K, sigGs, sigR, etc.
            
        Returns:
            Conditioning embedding [B, cond_dim]
        """
        batch_size = iso.size(0)
        device = iso.device
        dtype = iso.dtype
        
        iso = torch.clamp(iso.view(batch_size, -1), min=1.0)
        ratio = torch.clamp(ratio.view(batch_size, -1), min=1.0)
        
        # Normalize
        iso_norm = iso / 6400.0
        ratio_norm = ratio / 300.0
        
        if camera_params is not None:
            # Handle per-sample (list) or single (dict) camera params
            if isinstance(camera_params, (list, tuple)):
                # Per-sample: extract values for each sample
                K_vals = [p['K'] for p in camera_params]
                sigGs_vals = [p['sigGs'] for p in camera_params]
                sigR_vals = [p.get('sigR', 0.0) for p in camera_params]
                K = torch.tensor(K_vals, device=device, dtype=dtype).view(batch_size, 1)
                sigGs = torch.tensor(sigGs_vals, device=device, dtype=dtype).view(batch_size, 1)
                sigR = torch.tensor(sigR_vals, device=device, dtype=dtype).view(batch_size, 1)
            else:
                # Single dict: broadcast to all samples
                K = torch.tensor(
                    camera_params['K'], device=device, dtype=dtype
                ).expand(batch_size, 1)
                sigGs = torch.tensor(
                    camera_params['sigGs'], device=device, dtype=dtype
                ).expand(batch_size, 1)
                sigR = torch.tensor(
                    camera_params.get('sigR', 0.0), device=device, dtype=dtype
                ).expand(batch_size, 1)
        else:
            # Estimate from ISO (fallback - should use actual calibration)
            log_K = torch.log(iso_norm * 8.0 + 0.1)
            K = torch.exp(log_K)
            sigGs = torch.exp(log_K * 0.85 - 0.18)
            sigR = torch.exp(log_K * 0.88 - 2.11)
        
        # EMVA 1288 calculations
        # Shot noise variance is proportional to signal: var_shot = signal / K
        shot_variance = torch.clamp(iso_norm * ratio_norm / (K + 1e-8), min=1e-6)
        read_variance = sigGs ** 2
        
        # SNR approximation: SNR ≈ signal / sqrt(shot_var + read_var)
        snr = torch.clamp(
            (iso_norm * ratio_norm) / torch.sqrt(shot_variance + read_variance + 1e-8),
            min=1e-6
        )
        
        # Feature vector
        features = torch.cat([
            iso_norm,
            ratio_norm,
            torch.log1p(K),
            torch.log1p(sigGs),
            torch.log1p(sigR + 1e-6),
            torch.log1p(shot_variance),
            torch.log1p(read_variance),
            torch.log1p(snr),
        ], dim=-1)
        
        return self.embed(features)


class SimplePhysicsEncoder(nn.Module):
    """
    Simpler physics encoder for backward compatibility.
    
    Uses basic ISO/ratio features without full EMVA 1288 modeling.
    """
    
    def __init__(self, cond_dim: int):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(5, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )
        self.apply(_init_linear)

    def forward(self, iso: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
        iso = torch.log1p(torch.clamp(iso, min=0.0))
        ratio = torch.log1p(torch.clamp(ratio, min=0.0))
        iso_norm = iso / 10.0
        ratio_norm = ratio / 10.0
        shot = torch.sqrt(torch.clamp(iso_norm * ratio_norm, min=1e-4))
        read = torch.log1p(iso_norm + 1e-4)
        inverse_ratio = torch.reciprocal(torch.clamp(ratio + 1e-4, min=1e-4))
        feats = torch.cat([iso_norm, ratio_norm, shot, read, inverse_ratio], dim=-1)
        return self.embed(feats)


__all__ = ["EMVA1288PhysicsEncoder", "SimplePhysicsEncoder"]

