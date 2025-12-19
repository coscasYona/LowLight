"""
Camera noise parameters for EMVA 1288 noise model.

Contains calibrated noise parameters for various camera models.
"""

from typing import Dict, Optional
import numpy as np


# Camera parameters indexed by camera_type_ISO
CAMERA_NOISE_PARAMS = {
    # Sony A7S2 at various ISO values
    'SonyA7S2_100': {
        'Kmax': 0.09563, 'lam': 0.14875287, 
        'sigGs': 1.0067395, 'sigGssig': 0.0033682834,
        'sigTL': 0.70181876, 'sigTLsig': 0.0037532174,
        'sigR': 0.1391465, 'sigRsig': 0.006530218,
        'bias': 0, 'biassig': 0.007235429,
        'q': 6.103515625e-05, 'wp': 16383, 'bl': 512
    },
    'SonyA7S2_800': {
        'Kmax': 0.76504, 'lam': -0.008199721,
        'sigGs': 3.5475867, 'sigGssig': 0.052318197,
        'sigTL': 1.9346539, 'sigTLsig': 0.046128694,
        'sigR': 0.5723769, 'sigRsig': 0.037824076,
        'bias': 0, 'biassig': 0.025339302,
        'q': 6.103515625e-05, 'wp': 16383, 'bl': 512
    },
    'SonyA7S2_1600': {
        'Kmax': 1.53008, 'lam': -0.0441045,
        'sigGs': 6.29925, 'sigGssig': 0.1153261,
        'sigTL': 3.2283993, 'sigTLsig': 0.09118158,
        'sigR': 0.988786, 'sigRsig': 0.078567736,
        'bias': 0, 'biassig': 0.03877672,
        'q': 6.103515625e-05, 'wp': 16383, 'bl': 512
    },
    'SonyA7S2_3200': {
        'Kmax': 3.06016, 'lam': -0.034863412,
        'sigGs': 3.9193838, 'sigGssig': 0.02649232,
        'sigTL': 2.0417721, 'sigTLsig': 0.032873377,
        'sigR': 0.44543457, 'sigRsig': 0.030114045,
        'bias': 0, 'biassig': 0.021355819,
        'q': 6.103515625e-05, 'wp': 16383, 'bl': 512
    },
    'SonyA7S2_6400': {
        'Kmax': 6.12032, 'lam': -0.07517104,
        'sigGs': 7.1163535, 'sigGssig': 0.08435366,
        'sigTL': 3.4502964, 'sigTLsig': 0.08226275,
        'sigR': 0.7218788, 'sigRsig': 0.0642334,
        'bias': 0, 'biassig': 0.059074216,
        'q': 6.103515625e-05, 'wp': 16383, 'bl': 512
    },
    'SonyA7S2_12800': {
        'Kmax': 12.24064, 'lam': -0.06495205,
        'sigGs': 14.245901, 'sigGssig': 0.17283991,
        'sigTL': 7.038261, 'sigTLsig': 0.18822834,
        'sigR': 1.2749791, 'sigRsig': 0.120479785,
        'bias': 0, 'biassig': 0.0944684,
        'q': 6.103515625e-05, 'wp': 16383, 'bl': 512
    },
}

# Regression parameters for cameras without exact ISO calibration
CAMERA_REGRESSION_PARAMS = {
    'NikonD850': {
        'Kmin': 1.2, 'Kmax': 2.4828, 'lam': -0.26,
        'q': 1/(2**14), 'wp': 16383, 'bl': 512,
        'sigTLk': 0.906, 'sigTLb': -0.6754, 'sigTLsig': 0.035165,
        'sigRk': 0.8322, 'sigRb': -2.3326, 'sigRsig': 0.301333,
        'sigGsk': 0.8322, 'sigGsb': -0.1754, 'sigGssig': 0.035165,
    },
    'SonyA7S2_lowISO': {
        'Kmin': -1.67214, 'Kmax': 0.42228, 'lam': -0.026,
        'q': 1/(2**14), 'wp': 16383, 'bl': 512,
        'sigRk': 0.78782, 'sigRb': -0.34227, 'sigRsig': 0.02832,
        'sigTLk': 0.74043, 'sigTLb': 0.86182, 'sigTLsig': 0.00712,
        'sigGsk': 0.82966, 'sigGsb': 1.49343, 'sigGssig': 0.00359,
    },
    'SonyA7S2_highISO': {
        'Kmin': 0.64567, 'Kmax': 2.51606, 'lam': -0.025,
        'q': 1/(2**14), 'wp': 16383, 'bl': 512,
        'sigRk': 0.62945, 'sigRb': -1.51040, 'sigRsig': 0.02609,
        'sigTLk': 0.74901, 'sigTLb': -0.12348, 'sigTLsig': 0.00638,
        'sigGsk': 0.82878, 'sigGsb': 0.44162, 'sigGssig': 0.00153,
    },
}


def get_camera_params(
    camera_type: str, 
    iso: Optional[int] = None
) -> Optional[Dict]:
    """
    Get calibrated noise parameters for camera at given ISO.
    
    Args:
        camera_type: Camera model identifier
        iso: ISO sensitivity value
        
    Returns:
        Dict of noise parameters or None if not found
    """
    if iso is not None:
        key = f"{camera_type}_{iso}"
        if key in CAMERA_NOISE_PARAMS:
            return CAMERA_NOISE_PARAMS[key].copy()
    
    return None


def sample_camera_params(
    camera_type: str = 'SonyA7S2',
    iso: Optional[int] = None,
    ratio: Optional[float] = None,
) -> Dict:
    """
    Sample noise parameters for given camera configuration.
    
    If exact ISO calibration exists, uses it with small perturbations.
    Otherwise, uses regression model to estimate parameters.
    
    Args:
        camera_type: Camera model identifier
        iso: ISO sensitivity value
        ratio: Exposure ratio
        
    Returns:
        Dict of sampled noise parameters
    """
    # Try exact calibration first
    params = get_camera_params(camera_type, iso)
    
    if params is not None:
        # Add small perturbations for diversity
        params['K'] = params['Kmax'] * (1 + np.random.uniform(-0.01, 0.01))
        
        if 'sigGssig' in params:
            params['sigGs'] = np.random.normal(
                params['sigGs'], params['sigGssig']
            )
        if 'sigTLsig' in params:
            params['sigTL'] = np.random.normal(
                params.get('sigTL', params['sigGs']),
                params['sigTLsig']
            )
        if 'sigRsig' in params:
            params['sigR'] = np.random.normal(
                params.get('sigR', 0), params['sigRsig']
            )
    else:
        # Use regression model
        params = _sample_from_regression(camera_type, iso)
    
    # Set ratio
    if ratio is None:
        if 'SonyA7S2' in camera_type:
            ratio = np.random.uniform(100, 300)
        else:
            ratio = np.exp(np.random.uniform(0, 2.08))
    
    params['ratio'] = ratio
    
    return params


def _sample_from_regression(
    camera_type: str, 
    iso: Optional[int]
) -> Dict:
    """Sample parameters using regression model."""
    # Select appropriate regression model
    if camera_type.startswith('SonyA7S2'):
        if iso is not None and iso <= 1600:
            reg_type = 'SonyA7S2_lowISO'
        else:
            reg_type = 'SonyA7S2_highISO'
    elif camera_type in CAMERA_REGRESSION_PARAMS:
        reg_type = camera_type
    else:
        reg_type = 'NikonD850'  # fallback
    
    base = CAMERA_REGRESSION_PARAMS[reg_type]
    
    # Sample log_K from range
    log_K = base['Kmax'] + np.random.uniform(-0.01, 0.01)
    K = np.exp(log_K)
    
    # Compute other parameters from regression
    mu_TL = base.get('sigTLk', 0) * log_K + base.get('sigTLb', 0)
    mu_R = base.get('sigRk', 0) * log_K + base.get('sigRb', 0)
    mu_Gs = base.get('sigGsk', 0) * log_K + base.get('sigGsb', 0)
    
    return {
        'K': K,
        'sigTL': np.exp(mu_TL),
        'sigR': np.exp(mu_R),
        'sigGs': np.exp(np.random.normal(
            mu_Gs, base.get('sigGssig', 0)
        )),
        'bias': 0,
        'lam': base['lam'],
        'q': base['q'],
        'wp': base['wp'],
        'bl': base['bl'],
    }


__all__ = [
    "get_camera_params",
    "sample_camera_params",
    "CAMERA_NOISE_PARAMS",
    "CAMERA_REGRESSION_PARAMS",
]

