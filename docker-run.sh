#!/bin/bash
# Helper script to run training in Docker container
# Supports both interactive container mode and training execution

# Default training command
TRAIN_CMD="python stg2_denoise_train.py \
  --trainset_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony \
  --train_list /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/data/SID/Sony2025/Sony_train_list.txt \
  --use_sid_raw \
  --skip_eval \
  --epoch 500 \
  --batch_size 8 \
  --load_thread 8 \
  --patch_size 128 \
  --sd_base_channels 16 \
  --sd_channel_mults 1,2 \
  --save_path /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/runs/sid_try/ \
  --save_prefix sid_try_epoch_ \
  --resume new"

# Check if container already exists
CONTAINER_NAME="lowlight-train"
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container '${CONTAINER_NAME}' is already running."
        echo "To attach to it, run: docker exec -it ${CONTAINER_NAME} bash"
        exit 0
    else
        echo "Container '${CONTAINER_NAME}' exists but is stopped. Starting it..."
        docker start ${CONTAINER_NAME}
        echo "Container started. To attach, run: docker exec -it ${CONTAINER_NAME} bash"
        exit 0
    fi
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    echo "Using docker-compose..."
    docker-compose run --rm -p 6006:6006 stg2-train bash -c "$TRAIN_CMD"
elif command -v docker &> /dev/null; then
    echo "Creating and starting container '${CONTAINER_NAME}'..."
    echo "TensorBoard will be accessible on port 6006"
    echo ""
    
    docker run -d --gpus all \
        --shm-size=16g \
        -p 6006:6006 \
        -v "${PROJECT_ROOT}:/workspace/LowLight" \
        -w /workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model \
        --name ${CONTAINER_NAME} \
        nvcr.io/nvidia/pytorch:24.05-py3 \
        sleep infinity
    
    echo "Container '${CONTAINER_NAME}' created and started."
    echo ""
    echo "To attach to the container, run:"
    echo "  docker exec -it ${CONTAINER_NAME} bash"
    echo ""
    echo "To run training inside the container:"
    echo "  docker exec -it ${CONTAINER_NAME} bash -c \"${TRAIN_CMD}\""
    echo ""
    echo "To start TensorBoard inside the container:"
    echo "  docker exec -it ${CONTAINER_NAME} bash -c \"tensorboard --logdir=/workspace/LowLight/LLD/Codes/Stg2_LLD_Noise_Model/runs --port=6006 --host=0.0.0.0\""
    echo ""
    echo "Then access TensorBoard at: http://localhost:6006"
else
    echo "Error: Docker not found. Please install Docker first."
    exit 1
fi

