# Docker Setup Guide

This project includes Docker configuration for easy deployment and training.

## Prerequisites

- Docker (version 20.10+)
- Docker Compose (optional, but recommended)
- NVIDIA Docker runtime (nvidia-docker2) for GPU support
- NVIDIA GPU with CUDA support

## Quick Start

### Using Docker Compose (Recommended)

1. **Build the image:**
   ```bash
   docker-compose build
   ```

2. **Run training:**
   ```bash
   docker-compose run --rm stg2-train python stg2_denoise_train.py \
     --trainset_path /workspace/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony \
     --train_list /workspace/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony_train_list.txt \
     --use_sid_raw \
     --skip_eval \
     --epoch 500 \
     --batch_size 8 \
     --load_thread 8 \
     --patch_size 128 \
     --sd_base_channels 16 \
     --sd_channel_mults 1,2 \
     --save_path /workspace/Stg2_LLD_Noise_Model/runs/sid_try/ \
     --save_prefix sid_try_epoch_ \
     --resume new
   ```

3. **Interactive shell:**
   ```bash
   docker-compose run --rm stg2-train bash
   ```

### Using Docker Directly

1. **Build the image:**
   ```bash
   docker build -t stg2-lld-noise-model:latest .
   ```

2. **Run training:**
   ```bash
   docker run --rm -it \
     --gpus all \
     --shm-size=8g \
     -v $(pwd)/data:/workspace/Stg2_LLD_Noise_Model/data:ro \
     -v $(pwd)/dataset:/workspace/Stg2_LLD_Noise_Model/dataset:ro \
     -v $(pwd)/runs:/workspace/Stg2_LLD_Noise_Model/runs \
     -v $(pwd)/denoise_last_ckpt:/workspace/Stg2_LLD_Noise_Model/denoise_last_ckpt \
     stg2-lld-noise-model:latest \
     python stg2_denoise_train.py [your arguments]
   ```

### Using the Helper Script

```bash
./docker-run.sh
```

## Configuration

### Memory Settings

The `docker-compose.yml` includes:
- `shm_size: 8gb` - Shared memory for PyTorch DataLoader
- `mem_limit: 32g` - Container memory limit

Adjust these in `docker-compose.yml` based on your system.

### GPU Configuration

The container is configured to use:
- All available GPUs (can be limited via `CUDA_VISIBLE_DEVICES`)
- NVIDIA runtime for GPU access

### Volume Mounts

- `./data` - Read-only mount for training data
- `./dataset` - Read-only mount for dataset files
- `./runs` - Writable mount for training outputs
- `./denoise_last_ckpt` - Writable mount for checkpoints

## Base Image

The Dockerfile uses:
- **Base:** `nvcr.io/nvidia/pytorch:24.01-py3`
- **CUDA:** Included in base image
- **Python:** 3.10

## Troubleshooting

### Out of Memory Errors

1. Reduce `batch_size` in training arguments
2. Reduce `patch_size` (e.g., 64 instead of 128)
3. Increase `shm_size` in docker-compose.yml
4. Reduce `load_thread` to 0 or 1

### GPU Not Detected

1. Verify NVIDIA Docker runtime:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
   ```

2. Check docker-compose GPU configuration

### Shared Memory Errors

Increase `shm_size` in docker-compose.yml:
```yaml
shm_size: 16gb  # Increase from 8gb
```

## Notes

- The container uses NumPy 1.26.4 for compatibility with rawpy/OpenCV
- All checkpoints and outputs are saved to mounted volumes
- Training data should be placed in `./data` directory before running

