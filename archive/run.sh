#!/bin/bash
# Script to run a detached Docker container with the codebase mounted as a volume
# This allows code changes to be reflected immediately without rebuilding the image

set -e

# Configuration
IMAGE_NAME="stg2-lld-noise-model"
CONTAINER_NAME="stg2_denoise_train"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Stg2 LLD Noise Model - Docker Container Runner${NC}"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker not found. Please install Docker first."
    exit 1
fi

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${YELLOW}Container '${CONTAINER_NAME}' is already running.${NC}"
        echo "To attach to it, run: docker exec -it ${CONTAINER_NAME} bash"
        exit 0
    else
        echo "Container '${CONTAINER_NAME}' exists but is stopped. Starting it..."
        docker start ${CONTAINER_NAME}
        echo -e "${GREEN}Container started.${NC}"
        echo "To attach, run: docker exec -it ${CONTAINER_NAME} bash"
        exit 0
    fi
fi

# Check if image exists, if not build it
if ! docker images --format '{{.Repository}}' | grep -q "^${IMAGE_NAME}$"; then
    echo "Building Docker image '${IMAGE_NAME}'..."
    docker build -t ${IMAGE_NAME} -f "${PROJECT_ROOT}/Dockerfile" "${PROJECT_ROOT}"
    echo -e "${GREEN}Image built successfully.${NC}"
    echo ""
fi

# Create and start detached container
echo "Creating and starting detached container '${CONTAINER_NAME}'..."
echo ""

docker run -d \
    --name ${CONTAINER_NAME} \
    --gpus all \
    --shm-size=16g \
    -p 6006:6006 \
    -v "${PROJECT_ROOT}:/workspace/Stg2_LLD_Noise_Model" \
    -v "${PROJECT_ROOT}/data:/workspace/Stg2_LLD_Noise_Model/data:ro" \
    -v "${PROJECT_ROOT}/dataset:/workspace/Stg2_LLD_Noise_Model/dataset:ro" \
    -v "${PROJECT_ROOT}/runs:/workspace/Stg2_LLD_Noise_Model/runs" \
    -v "${PROJECT_ROOT}/denoise_last_ckpt:/workspace/Stg2_LLD_Noise_Model/denoise_last_ckpt" \
    -w /workspace/Stg2_LLD_Noise_Model \
    -e CUDA_VISIBLE_DEVICES=0 \
    ${IMAGE_NAME} \
    sleep infinity

echo -e "${GREEN}Container '${CONTAINER_NAME}' created and started in detached mode.${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Useful commands:"
echo ""
echo "  Attach to container:"
echo "    docker exec -it ${CONTAINER_NAME} bash"
echo ""
echo "  Run training:"
echo "    docker exec -it ${CONTAINER_NAME} bash -c 'python stg2_denoise_train.py [args]'"
echo ""
echo "  Start TensorBoard:"
echo "    docker exec -d ${CONTAINER_NAME} bash -c 'tensorboard --logdir=/workspace/Stg2_LLD_Noise_Model/runs --port=6006 --host=0.0.0.0'"
echo "    Then access at: http://localhost:6006"
echo ""
echo "  Stop container:"
echo "    docker stop ${CONTAINER_NAME}"
echo ""
echo "  Remove container:"
echo "    docker rm ${CONTAINER_NAME}"
echo ""
echo "  View logs:"
echo "    docker logs ${CONTAINER_NAME}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

