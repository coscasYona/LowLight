# EMVA 1288 Physics-Guided Diffusion Model

A PyTorch Lightning implementation of a diffusion model for low-light image denoising using physics-based noise modeling from the EMVA 1288 standard.

## Project Structure

```
src/
├── configs/                    # Hydra YAML configurations
│   ├── train.yaml             # Main training config
│   ├── model/
│   │   ├── emva1288.yaml      # EMVA 1288 diffusion model
│   │   └── unet.yaml          # Basic UNet config
│   └── data/
│       ├── sid.yaml           # SID dataset config
│       └── eld.yaml           # ELD dataset config
├── models/                     # Model components
│   ├── diffusion.py           # EMVA1288Diffusion model
│   ├── unet.py                # SlimUNet backbone
│   ├── physics_encoder.py     # Physics-based conditioning
│   ├── noise_model.py         # EMVA 1288 noise generation
│   └── scheduler.py           # DDPM/DDIM scheduler
├── data/                       # Data loading
│   ├── datamodule.py          # Lightning DataModule
│   ├── sid_dataset.py         # SID/ELD dataset classes
│   └── transforms.py          # Augmentations
├── training/                   # Training components
│   ├── lightning_module.py    # Main LightningModule
│   ├── losses.py              # Hybrid loss functions
│   ├── callbacks.py           # Custom callbacks
│   └── metrics.py             # PSNR/SSIM metrics
├── utils/                      # Utilities
│   ├── camera_params.py       # Camera noise parameters
│   └── visualization.py       # Image logging/display
├── scripts/                    # Entry points
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   └── inference.py           # Single-image inference
└── tests/                      # Unit tests
    ├── test_model.py
    └── test_lightning_module.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

### Basic Training

```bash
# From project root
python src/scripts/train.py

# With custom config overrides
python src/scripts/train.py training.epochs=100 training.batch_size=2

# With different model config
python src/scripts/train.py model=unet

# With different dataset
python src/scripts/train.py data=eld
```

### Environment Variables

```bash
export SID_DATASET_PATH=/path/to/SID/Sony
export ELD_DATASET_PATH=/path/to/ELD_new
```

## Evaluation

```bash
# Evaluate on all datasets
python src/scripts/evaluate.py --checkpoint path/to/checkpoint.ckpt

# Evaluate on specific dataset
python src/scripts/evaluate.py --checkpoint path/to/checkpoint.ckpt --dataset sid
python src/scripts/evaluate.py --checkpoint path/to/checkpoint.ckpt --dataset eld
```

## Single Image Inference

```bash
python src/scripts/inference.py \
    --checkpoint path/to/checkpoint.ckpt \
    --input image.ARW \
    --output denoised.png \
    --ratio 200
```

## Key Features

### EMVA 1288 Physics-Based Noise

The model uses the EMVA 1288 standard for CMOS camera noise modeling:
- **Shot noise**: Poisson-distributed, signal-dependent
- **Read noise**: Gaussian, sensor-dependent
- **Row noise**: Correlated along image rows
- **Quantization noise**: Due to ADC

### Hybrid Loss Function

Combines multiple loss components:
- **MSE loss**: Overall reconstruction
- **L1 loss**: Sharper images (less blur)
- **Gradient loss**: Edge preservation (illuminance-invariant)

### Diffusion Schedulers

Supports both:
- **DDPM**: Stochastic sampling (original)
- **DDIM**: Deterministic/accelerated sampling

## Configuration

Key configuration options in `configs/train.yaml`:

```yaml
training:
  epochs: 200
  batch_size: 1
  learning_rate: 1e-4
  l1_weight: 0.8
  gradient_weight: 0.1
  loss_scale: 10.0

model:
  base_channels: 32
  channel_mults: [1, 2, 4]
  num_steps: 4
  attn_type: linear
  scheduler: ddpm
  camera_type: SonyA7S2
  noise_code: prq
```

## Running Tests

```bash
# From project root
pytest src/tests/ -v
```

## Migration from Legacy Code

The original `stg2_emva1288_train.py` has been archived as `stg2_emva1288_train.py.legacy`. 
Key improvements in the refactored version:

1. **PyTorch Lightning**: Automatic training loop, checkpointing, logging
2. **Hydra Configuration**: Flexible config management via YAML
3. **Modular Design**: Separate modules for models, data, training
4. **Better Structure**: Clear separation of concerns
5. **Testability**: Unit tests for components

