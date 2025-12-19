# EMVA 1288 Physics-Guided Diffusion - Training Commands

This document contains CLI commands for training the EMVA 1288 diffusion model. All commands should be run from the `src/` directory.

## Setup

```bash
cd /workspace/src
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

---

## Training Configurations

### 1. Default Training (SID Dataset)

Basic training with default settings on the SID (See in the Dark) dataset.

```bash
python scripts/train.py
```

---

### 2. Training with Measurement Conditioning

Enable measurement conditioning for better image quality. The model learns to leverage the actual sensor noise structure by concatenating the real noisy input with the diffusion state.

```bash
python scripts/train.py model.use_measurement_cond=true
```

---

### 3. Training on ELD Dataset

Train on the Extreme Low-light Denoising (ELD) dataset instead of SID.

```bash
python scripts/train.py data=eld
```

---

### 4. Fast/Debug Training

Quick training run for debugging or testing changes (fewer epochs, smaller model).

```bash
python scripts/train.py \
    training.epochs=5 \
    model.base_channels=16 \
    model.num_steps=2 \
    logging.log_every_n_steps=10
```

---

### 5. Monitor PSNR for Checkpoints

Use PSNR (Peak Signal-to-Noise Ratio) instead of loss for checkpoint selection and early stopping. Better for image quality optimization.

```bash
python scripts/train.py \
    checkpoint.monitor=val/psnr \
    checkpoint.mode=max \
    early_stopping.monitor=val/psnr \
    early_stopping.mode=max
```

---

### 6. Mixed Precision Training (FP16)

Enable mixed precision training for faster training and lower memory usage on modern GPUs.

```bash
python scripts/train.py hardware.precision=16-mixed
```

---

### 7. Training with EMA (Exponential Moving Average)

Enable EMA for smoother model weights and potentially better generalization.

```bash
python scripts/train.py \
    training.use_ema=true \
    training.ema_decay=0.9999
```

---

### 8. Full Quality Training (Recommended for Best Results)

Combines measurement conditioning, PSNR monitoring, and EMA for optimal image enhancement quality.

```bash
python scripts/train.py \
    model.use_measurement_cond=true \
    checkpoint.monitor=val/psnr \
    checkpoint.mode=max \
    early_stopping.monitor=val/psnr \
    early_stopping.mode=max \
    training.use_ema=true
```

---

### 9. Higher Learning Rate with Gradient Clipping

Use a higher learning rate with gradient clipping for potentially faster convergence.

```bash
python scripts/train.py \
    training.learning_rate=5e-4 \
    training.use_grad_clip=true \
    training.grad_clip_max=1.0
```

---

### 10. Custom Paths

Specify custom dataset paths and output directories.

```bash
# Set dataset path via environment variable
export SID_DATASET_PATH=/path/to/SID/Sony

python scripts/train.py \
    paths.save_dir=./output/experiment_1 \
    paths.log_dir=./output/logs
```

---

### 11. Long Training with More Epochs

Extended training for better convergence.

```bash
python scripts/train.py \
    training.epochs=500 \
    early_stopping.patience=50
```

---

### 12. Larger Model (More Channels)

Train a larger model with more capacity.

```bash
python scripts/train.py \
    model.base_channels=64 \
    model.channel_mults=[1,2,4,8]
```

---

### 13. More Diffusion Steps

Use more diffusion steps for potentially better quality (slower training).

```bash
python scripts/train.py model.num_steps=10
```

---

### 14. DDIM Scheduler (Faster Sampling)

Use DDIM scheduler for faster sampling during inference.

```bash
python scripts/train.py model.scheduler=ddim
```

---

### 15. Different Camera Type

Train for a specific camera type (affects noise model parameters).

```bash
# For Sony A7S2 (default)
python scripts/train.py model.camera_type=SonyA7S2

# For Nikon D850
python scripts/train.py model.camera_type=NikonD850
```

---

## Inference Commands

### Single Image Denoising

```bash
python scripts/inference.py \
    --checkpoint ./checkpoints/best.ckpt \
    --input /path/to/noisy_image.ARW \
    --output /path/to/denoised.png \
    --ratio 200
```

### With Custom ISO

```bash
python scripts/inference.py \
    --checkpoint ./checkpoints/best.ckpt \
    --input /path/to/noisy_image.ARW \
    --output /path/to/denoised.png \
    --ratio 200 \
    --iso 6400
```

### Save Noisy Input for Comparison

```bash
python scripts/inference.py \
    --checkpoint ./checkpoints/best.ckpt \
    --input /path/to/noisy_image.ARW \
    --output /path/to/denoised.png \
    --ratio 200 \
    --save_noisy
```

---

## Evaluation Commands

### Evaluate on SID Dataset

```bash
python scripts/evaluate.py \
    --checkpoint ./checkpoints/best.ckpt \
    --dataset sid
```

### Evaluate on ELD Dataset

```bash
python scripts/evaluate.py \
    --checkpoint ./checkpoints/best.ckpt \
    --dataset eld
```

### Evaluate on Both Datasets

```bash
python scripts/evaluate.py \
    --checkpoint ./checkpoints/best.ckpt \
    --dataset all
```

### Custom Dataset Paths

```bash
python scripts/evaluate.py \
    --checkpoint ./checkpoints/best.ckpt \
    --dataset sid \
    --sid_dir /custom/path/to/SID/Sony \
    --val_list ./dataset/Sony_val.txt \
    --test_list ./dataset/Sony_test.txt
```

---

## Testing Commands

### Run All Tests

```bash
cd /workspace
PYTHONPATH=$PYTHONPATH:src python -m pytest src/tests/ -v
```

### Run Specific Test File

```bash
PYTHONPATH=$PYTHONPATH:src python -m pytest src/tests/test_lightning_module.py -v
```

### Run Tests with Coverage

```bash
PYTHONPATH=$PYTHONPATH:src python -m pytest src/tests/ -v --cov=src --cov-report=html
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SID_DATASET_PATH` | Path to SID dataset | `../dataset/SID/Sony` |
| `ELD_DATASET_PATH` | Path to ELD dataset | `../dataset/ELD_new` |
| `CUDA_VISIBLE_DEVICES` | GPU device(s) to use | All available |

Example:
```bash
export SID_DATASET_PATH=/data/SID/Sony
export CUDA_VISIBLE_DEVICES=0,1
python scripts/train.py
```

---

## Hydra Configuration Tips

Hydra allows flexible configuration overrides from the command line:

```bash
# Override nested values with dot notation
python scripts/train.py training.learning_rate=1e-3

# Override multiple values
python scripts/train.py training.epochs=100 model.num_steps=8

# Use different config files
python scripts/train.py model=unet data=eld

# Show effective config without running
python scripts/train.py --cfg job

# Print help
python scripts/train.py --help
```

---

## TensorBoard Monitoring

Start TensorBoard to monitor training progress:

```bash
tensorboard --logdir ./logs
```

Then open http://localhost:6006 in your browser.
