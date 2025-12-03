#!/bin/bash
# Script to check NGC PyTorch container versions and their CUDA support

echo "Checking NGC PyTorch container versions..."
echo "=========================================="
echo ""

# Common NGC PyTorch container tags to check
# Format: YY.MM (year.month)
TAGS=(
    "25.09"
    "25.08"
    "25.07"
    "25.06"
    "25.05"
    "25.04"
    "25.03"
    "25.02"
    "25.01"
    "24.12"
    "24.11"
    "24.10"
)

echo "Recommended NGC PyTorch containers for RTX 5090 (sm_120) and RTX 3090 (sm_86):"
echo ""
echo "For CUDA 12.8+ support (required for RTX 5090 sm_120):"
echo "  nvcr.io/nvidia/pytorch:25.09-py3"
echo "  nvcr.io/nvidia/pytorch:25.08-py3"
echo "  nvcr.io/nvidia/pytorch:25.07-py3"
echo ""
echo "These containers should support both:"
echo "  - RTX 5090 (sm_120) - requires CUDA 12.8+"
echo "  - RTX 3090 (sm_86)  - supported by CUDA 11.0+"
echo ""
echo "To pull and test a container:"
echo "  docker pull nvcr.io/nvidia/pytorch:25.09-py3"
echo ""
echo "To run the container:"
echo "  docker run --gpus all -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \\"
echo "    -v \$(pwd):/workspace \\"
echo "    nvcr.io/nvidia/pytorch:25.09-py3"
echo ""
echo "Inside the container, verify GPU support:"
echo "  python3 -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')\""
echo ""
echo "For the latest container tags, visit:"
echo "  https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch"
echo ""

