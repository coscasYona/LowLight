"""
EMVA 1288 Physics-Based Noise Model.

Implements CMOS camera noise generation following the EMVA 1288 standard,
including shot noise (Poisson), read noise, row noise, and quantization noise.

Reference: https://kmdouglass.github.io/posts/modeling-noise-for-image-simulations/
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributions as tdist


# Camera parameters for different sensors
CAMERA_PARAMS = {
    'SonyA7S2_50': {'Kmax': 0.047815, 'lam': 0.1474653, 'sigGs': 1.0164667, 'sigGssig': 0.005272454, 'sigTL': 0.70727646, 'sigTLsig': 0.004360543, 'sigR': 0.13997398, 'sigRsig': 0.0064381803, 'bias': 0, 'biassig': 0.010093017, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
    'SonyA7S2_100': {'Kmax': 0.09563, 'lam': 0.14875287, 'sigGs': 1.0067395, 'sigGssig': 0.0033682834, 'sigTL': 0.70181876, 'sigTLsig': 0.0037532174, 'sigR': 0.1391465, 'sigRsig': 0.006530218, 'bias': 0, 'biassig': 0.007235429, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
    'SonyA7S2_200': {'Kmax': 0.19126, 'lam': 0.07902429, 'sigGs': 1.2926387, 'sigGssig': 0.012171176, 'sigTL': 0.8117464, 'sigTLsig': 0.010250768, 'sigR': 0.22815849, 'sigRsig': 0.010726711, 'bias': 0, 'biassig': 0.011413908, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
    'SonyA7S2_400': {'Kmax': 0.38252, 'lam': 0.0222538, 'sigGs': 2.0595572, 'sigGssig': 0.024872316, 'sigTL': 1.1816813, 'sigTLsig': 0.02505812, 'sigR': 0.36209714, 'sigRsig': 0.01994737, 'bias': 0, 'biassig': 0.021005306, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
    'SonyA7S2_800': {'Kmax': 0.76504, 'lam': -0.008199721, 'sigGs': 3.5475867, 'sigGssig': 0.052318197, 'sigTL': 1.9346539, 'sigTLsig': 0.046128694, 'sigR': 0.5723769, 'sigRsig': 0.037824076, 'bias': 0, 'biassig': 0.025339302, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
    'SonyA7S2_1600': {'Kmax': 1.53008, 'lam': -0.0441045, 'sigGs': 6.29925, 'sigGssig': 0.1153261, 'sigTL': 3.2283993, 'sigTLsig': 0.09118158, 'sigR': 0.988786, 'sigRsig': 0.078567736, 'bias': 0, 'biassig': 0.03877672, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
    'SonyA7S2_3200': {'Kmax': 3.06016, 'lam': -0.034863412, 'sigGs': 3.9193838, 'sigGssig': 0.02649232, 'sigTL': 2.0417721, 'sigTLsig': 0.032873377, 'sigR': 0.44543457, 'sigRsig': 0.030114045, 'bias': 0, 'biassig': 0.021355819, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
    'SonyA7S2_6400': {'Kmax': 6.12032, 'lam': -0.07517104, 'sigGs': 7.1163535, 'sigGssig': 0.08435366, 'sigTL': 3.4502964, 'sigTLsig': 0.08226275, 'sigR': 0.7218788, 'sigRsig': 0.0642334, 'bias': 0, 'biassig': 0.059074216, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
    'SonyA7S2_12800': {'Kmax': 12.24064, 'lam': -0.06495205, 'sigGs': 14.245901, 'sigGssig': 0.17283991, 'sigTL': 7.038261, 'sigTLsig': 0.18822834, 'sigR': 1.2749791, 'sigRsig': 0.120479785, 'bias': 0, 'biassig': 0.0944684, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
    'SonyA7S2_25600': {'Kmax': 24.48128, 'lam': -0.09089118, 'sigGs': 25.853043, 'sigGssig': 0.35371417, 'sigTL': 12.175712, 'sigTLsig': 0.4215717, 'sigR': 2.2760193, 'sigRsig': 0.2609267, 'bias': 0, 'biassig': 0.37568903, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
}

# Camera type to dual-ISO mapping
DUAL_ISO_CAMERAS = ['SonyA7S2']

# Fallback camera parameters for regression-based estimation
CAMERA_PARAMS_REGRESSION = {
    'NikonD850': {
        'Kmin': 1.2, 'Kmax': 2.4828, 'lam': -0.26, 'q': 1/(2**14), 'wp': 16383, 'bl': 512,
        'sigTLk': 0.906, 'sigTLb': -0.6754, 'sigTLsig': 0.035165,
        'sigRk': 0.8322, 'sigRb': -2.3326, 'sigRsig': 0.301333,
        'sigGsk': 0.8322, 'sigGsb': -0.1754, 'sigGssig': 0.035165,
    },
    'SonyA7S2_lowISO': {
        'Kmin': -1.67214, 'Kmax': 0.42228, 'lam': -0.026, 'q': 1/(2**14), 'wp': 16383, 'bl': 512,
        'sigRk': 0.78782, 'sigRb': -0.34227, 'sigRsig': 0.02832,
        'sigTLk': 0.74043, 'sigTLb': 0.86182, 'sigTLsig': 0.00712,
        'sigGsk': 0.82966, 'sigGsb': 1.49343, 'sigGssig': 0.00359,
    },
    'SonyA7S2_highISO': {
        'Kmin': 0.64567, 'Kmax': 2.51606, 'lam': -0.025, 'q': 1/(2**14), 'wp': 16383, 'bl': 512,
        'sigRk': 0.62945, 'sigRb': -1.51040, 'sigRsig': 0.02609,
        'sigTLk': 0.74901, 'sigTLb': -0.12348, 'sigTLsig': 0.00638,
        'sigGsk': 0.82878, 'sigGsb': 0.44162, 'sigGssig': 0.00153,
    },
}


class EMVA1288NoiseModel:
    """
    Physics-based CMOS noise generation per EMVA 1288 standard.
    
    Generates realistic camera noise including:
    - Shot noise (Poisson distributed, signal-dependent)
    - Read noise (Gaussian, signal-independent)
    - Row noise (correlated along rows)
    - Quantization noise (uniform)
    - Bias/offset
    
    Args:
        camera_type: Camera model identifier
        noise_code: Noise components to include:
            - 'p': Poisson shot noise
            - 'r': Row noise
            - 'q': Quantization noise
            - 'g': Tukey-Lambda (generalized) read noise
            - 'd': Dark current/bias
        device: Torch device for tensor operations
    """
    
    def __init__(
        self, 
        camera_type: str = 'SonyA7S2',
        noise_code: str = 'prq',
        device: Optional[torch.device] = None,
    ):
        self.camera_type = camera_type
        self.noise_code = noise_code.lower()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def get_params(self, iso: int, ratio: Optional[float] = None) -> Dict:
        """
        Get noise parameters for given ISO and ratio.
        
        Args:
            iso: ISO sensitivity value
            ratio: Exposure ratio (optional)
            
        Returns:
            Dictionary of noise parameters
        """
        camera_iso_key = f"{self.camera_type}_{iso}"
        
        # Try exact ISO match first
        if camera_iso_key in CAMERA_PARAMS:
            params = CAMERA_PARAMS[camera_iso_key].copy()
            params['K'] = params['Kmax'] * (1 + np.random.uniform(-0.01, 0.01))
            params['sigGs'] = np.random.normal(
                params['sigGs'], 
                params.get('sigGssig', 0)
            ) if 'sigGssig' in params else params['sigGs']
            params['sigTL'] = np.random.normal(
                params.get('sigTL', params['sigGs']), 
                params.get('sigTLsig', 0)
            ) if 'sigTLsig' in params else params.get('sigTL', params['sigGs'])
            params['sigR'] = np.random.normal(
                params.get('sigR', 0), 
                params.get('sigRsig', 0)
            ) if 'sigRsig' in params else params.get('sigR', 0)
        else:
            # Use regression-based estimation
            params = self._estimate_params_from_regression(iso)
            
        # Set ratio
        if ratio is None:
            if 'SonyA7S2' in self.camera_type:
                ratio = np.random.uniform(100, 300)
            else:
                ratio = np.exp(np.random.uniform(0, 2.08))
        params['ratio'] = ratio
        
        return params
    
    def _estimate_params_from_regression(self, iso: int) -> Dict:
        """Estimate parameters using regression model."""
        camera_type = self.camera_type
        
        if camera_type in DUAL_ISO_CAMERAS:
            if iso <= 1600:
                camera_type = f"{camera_type}_lowISO"
            else:
                camera_type = f"{camera_type}_highISO"
        
        if camera_type not in CAMERA_PARAMS_REGRESSION:
            camera_type = 'NikonD850'  # fallback
            
        base_params = CAMERA_PARAMS_REGRESSION[camera_type]
        
        log_K = base_params['Kmax'] + np.random.uniform(-0.01, 0.01)
        K = np.exp(log_K)
        
        mu_TL = base_params.get('sigTLk', 0) * log_K + base_params.get('sigTLb', 0)
        mu_R = base_params.get('sigRk', 0) * log_K + base_params.get('sigRb', 0)
        mu_Gs = base_params.get('sigGsk', 0) * log_K + base_params.get('sigGsb', 0)
        
        return {
            'K': K,
            'sigTL': np.exp(mu_TL),
            'sigR': np.exp(mu_R),
            'sigGs': np.exp(np.random.normal(mu_Gs, base_params.get('sigGssig', 0))),
            'bias': 0,
            'lam': base_params['lam'],
            'q': base_params['q'],
            'wp': base_params['wp'],
            'bl': base_params['bl'],
        }
    
    def generate_noise_numpy(
        self,
        clean: np.ndarray,
        params: Dict,
        multi_frame_mean: int = 1,
    ) -> np.ndarray:
        """
        Generate noisy observation (NumPy version).
        
        Args:
            clean: Clean image in [0, 1] range
            params: Noise parameters dictionary
            multi_frame_mean: Number of frames to average
            
        Returns:
            Noisy image
        """
        p = params
        y = clean * (p['wp'] - p['bl']) / p['ratio']
        mfm = multi_frame_mean ** 0.5
        
        use_P = 'p' in self.noise_code
        use_R = 'r' in self.noise_code
        use_Q = 'q' in self.noise_code
        use_TL = 'g' in self.noise_code
        use_D = 'd' in self.noise_code
        
        # Shot noise (Poisson)
        if use_P:
            noisy_shot = np.random.poisson(mfm * y / p['K']).astype(np.float32) * p['K'] / mfm
        else:
            noisy_shot = y + np.random.randn(*y.shape).astype(np.float32) * \
                np.sqrt(np.maximum(y / p['K'], 1e-10)) * p['K'] / mfm
        
        # Read noise (Gaussian or Tukey-Lambda)
        if use_TL:
            from scipy import stats
            noisy_read = stats.tukeylambda.rvs(
                p['lam'], scale=p.get('sigTL', p['sigGs']) / mfm, size=y.shape
            ).astype(np.float32)
        else:
            noisy_read = np.random.randn(*y.shape).astype(np.float32) * p['sigGs'] / mfm
        
        # Row noise
        noisy_row = np.random.randn(y.shape[-3], y.shape[-2], 1).astype(np.float32) * \
            p.get('sigR', 0) / mfm if use_R else 0
        
        # Quantization noise
        noisy_q = np.random.uniform(-0.5, 0.5, size=y.shape) if use_Q else 0
        
        # Bias
        noisy_bias = p.get('bias', 0) if use_D else 0
        
        # Combine and normalize
        z = (noisy_shot + noisy_read + noisy_row + noisy_q + noisy_bias) / (p['wp'] - p['bl'])
        z = np.clip(z, -p['bl'] / p['wp'], 1)
        z = z * p['ratio']
        
        return z.astype(np.float32)
    
    def generate_noise_torch(
        self,
        clean: torch.Tensor,
        params: Dict,
        multi_frame_mean: int = 1,
    ) -> torch.Tensor:
        """
        Generate noisy observation (PyTorch version).
        
        Args:
            clean: Clean image tensor in [0, 1] range [B, C, H, W]
            params: Noise parameters dictionary
            multi_frame_mean: Number of frames to average
            
        Returns:
            Noisy image tensor
        """
        p = params
        device = clean.device
        
        y = clean * (p['wp'] - p['bl']) / p['ratio']
        mfm = multi_frame_mean ** 0.5
        
        use_P = 'p' in self.noise_code
        use_R = 'r' in self.noise_code
        use_Q = 'q' in self.noise_code
        use_D = 'd' in self.noise_code
        
        # Shot noise (Poisson)
        if use_P:
            noisy_shot = tdist.Poisson(mfm * y / p['K']).sample() * p['K'] / mfm
        else:
            noisy_shot = tdist.Normal(y, torch.sqrt(torch.clamp(y / p['K'], min=1e-10)) * p['K'] / mfm).sample()
        
        # Read noise (Gaussian)
        noisy_read = tdist.Normal(
            torch.zeros_like(y), 
            p['sigGs'] / mfm
        ).sample()
        
        # Row noise
        if use_R and p.get('sigR', 0) > 0:
            noisy_row = torch.randn(
                y.shape[-3], y.shape[-2], 1, device=device
            ) * p['sigR'] / mfm
        else:
            noisy_row = 0
        
        # Quantization noise
        if use_Q:
            noisy_q = (torch.rand(y.shape, device=device) - 0.5) * \
                p['q'] * (p['wp'] - p['bl'])
        else:
            noisy_q = 0
        
        # Bias
        noisy_bias = p.get('bias', 0) if use_D else 0
        
        # Combine and normalize
        z = (noisy_shot + noisy_read + noisy_row + noisy_q + noisy_bias) / (p['wp'] - p['bl'])
        z = torch.clamp(z, -p['bl'] / p['wp'], 1)
        z = z * p['ratio']
        
        return z
    
    def __call__(
        self,
        clean: torch.Tensor,
        iso: int = 6400,
        ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Generate noisy image from clean input.
        
        Args:
            clean: Clean image tensor [B, C, H, W] in [0, 1]
            iso: ISO sensitivity
            ratio: Exposure ratio
            
        Returns:
            Tuple of (noisy image, noise parameters)
        """
        params = self.get_params(iso, ratio)
        noisy = self.generate_noise_torch(clean, params)
        return noisy, params


def sample_params_max(
    camera_type: str = 'SonyA7S2',
    iso: Optional[int] = None,
    ratio: Optional[float] = None,
) -> Dict:
    """
    Sample noise parameters for given camera and ISO.
    
    Convenience function for backward compatibility.
    """
    noise_model = EMVA1288NoiseModel(camera_type=camera_type)
    return noise_model.get_params(iso or 6400, ratio)


__all__ = ["EMVA1288NoiseModel", "sample_params_max", "CAMERA_PARAMS"]

