# EMVA 1288 Physics-Guided Diffusion Training

This document describes the new training approach that integrates the EMVA 1288 standard for accurate CMOS noise modeling with stable diffusion.

## Overview

The `EMVA1288Diffusion` network and `stg2_emva1288_train.py` training script implement a diffusion model that uses the actual CMOS noise distribution (shot + read + row + quantization) instead of simple Gaussian scaling. This improves SNR by learning the true noise characteristics.

## Key Features

1. **Physics-Based Noise Generation**: Uses `generate_noisy_torch` from `data_process/process.py` to generate accurate CMOS noise
2. **EMVA 1288 Standard**: Follows the mathematical model from https://kmdouglass.github.io/posts/modeling-noise-for-image-simulations/
3. **Diffusion Forward Process**: The `q_sample` method uses actual CMOS noise distribution in the forward diffusion process
4. **Camera-Specific Parameters**: Supports different camera types (SonyA7S2, IMX686, NikonD850, etc.)

## Files

- `net/EMVA1288Diffusion.py`: Network architecture with EMVA 1288 physics
- `stg2_emva1288_train.py`: Training script that uses physics-based noise
- `stg2_denoise_options.py`: Updated with `--emva_camera_type` and `--emva_noise_code` arguments

## Usage

### Basic Training

```bash
python stg2_emva1288_train.py \
    --trainset_path /path/to/train/data \
    --train_list ./dataset/Sony_train.txt \
    --use_sid_raw \
    --emva_camera_type SonyA7S2 \
    --emva_noise_code prq \
    --batch_size 4 \
    --epoch 200 \
    --save_path ./checkpoints/emva1288/
```

### Arguments

- `--emva_camera_type`: Camera type for noise parameter lookup (default: `SonyA7S2`)
  - Options: `SonyA7S2`, `IMX686`, `NikonD850`, etc.
  - Must match camera types defined in `data_process/process.py`

- `--emva_noise_code`: Noise components to include (default: `prq`)
  - `p`: Poisson shot noise
  - `r`: Row noise (correlated across rows)
  - `q`: Quantization noise
  - `g`: Tukey-Lambda read noise (alternative to Gaussian)
  - `d`: Bias/dark current

### How It Works

1. **Noise Generation**: During training, for each batch:
   - Camera parameters are sampled based on ISO using `sample_params_max()`
   - CMOS noise is generated using `generate_noisy_torch()` with physics parameters
   - This includes shot noise (Poisson), read noise (Gaussian), row noise, and quantization

2. **Forward Diffusion**: The `q_sample` method:
   - Generates CMOS noise matching the physics model
   - Blends it with standard Gaussian noise based on timestep
   - Early timesteps use more physics noise (realistic)
   - Later timesteps use more standard noise (for diffusion schedule)

3. **Noise Prediction**: The network learns to predict the actual noise (CMOS + Gaussian blend) rather than just Gaussian noise

## Differences from Standard Diffusion

| Aspect | Standard Diffusion | EMVA 1288 Diffusion |
|--------|-------------------|---------------------|
| Noise Type | Gaussian only | CMOS (shot + read + row + quant) |
| Noise Scale | Simple heuristic | Physics-based from camera params |
| Forward Process | `x_t = sqrt(α) * x_0 + sqrt(1-α) * ε` | Uses actual CMOS noise distribution |
| Training | Predicts Gaussian noise | Predicts blended CMOS+Gaussian noise |

## Benefits

1. **Accurate Noise Modeling**: Uses real CMOS noise characteristics
2. **Better SNR**: Learns to denoise actual camera noise patterns
3. **Camera-Specific**: Adapts to different camera noise profiles
4. **Physics-Informed**: Grounded in EMVA 1288 standard

## Integration with Existing Code

The new network:
- Uses the same UNet architecture as `PhysicsGuidedStableDiffusion`
- Integrates with existing `data_process/process.py` functions
- Compatible with existing dataset loaders
- Uses same training infrastructure (TensorBoard, checkpoints, etc.)

## Example Training Output

```
Using attention type: linear (memory-efficient)
Camera type: SonyA7S2, Noise code: prq
Model architecture: attention_type=linear, scheduler=ddpm
EMVA 1288 physics: camera_type=SonyA7S2, noise_code=prq
Epoch:[1/200] Batch: [1/1000] loss = 0.0234
...
```

## Notes

- The noise generation uses `generate_noisy_torch` which requires proper camera parameters
- Make sure your camera type is defined in `get_camera_noisy_params_max()` or `get_camera_noisy_params()`
- The physics noise is blended with Gaussian noise to maintain diffusion schedule properties
- For best results, use camera parameters calibrated from integration sphere measurements

