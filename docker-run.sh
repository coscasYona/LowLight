#!/bin/bash
# Helper script to run training in Docker container

# Default training command
TRAIN_CMD="python stg2_denoise_train.py \
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
  --resume new"

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    echo "Using docker-compose..."
    docker-compose run --rm stg2-train bash -c "$TRAIN_CMD"
elif command -v docker &> /dev/null; then
    echo "Using docker directly..."
    docker run --rm -it \
        --gpus all \
        --shm-size=8g \
        -v $(pwd)/data:/workspace/Stg2_LLD_Noise_Model/data:ro \
        -v $(pwd)/dataset:/workspace/Stg2_LLD_Noise_Model/dataset:ro \
        -v $(pwd)/runs:/workspace/Stg2_LLD_Noise_Model/runs \
        -v $(pwd)/denoise_last_ckpt:/workspace/Stg2_LLD_Noise_Model/denoise_last_ckpt \
        stg2-lld-noise-model:latest \
        bash -c "$TRAIN_CMD"
else
    echo "Error: Docker not found. Please install Docker first."
    exit 1
fi

