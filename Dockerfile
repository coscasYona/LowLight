# Use NVIDIA PyTorch container as base
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set working directory
WORKDIR /workspace

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies for RAW image processing
RUN apt-get update && apt-get install -y \
    libraw-dev \
    libraw-bin \
    libjpeg-dev \
    libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Note: numpy 1.26.4 is pinned for compatibility with rawpy/opencv
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    rawpy \
    exifread \
    natsort \
    scikit-image \
    pillow \
    scipy \
    matplotlib \
    opencv-python==4.8.1.78

# Copy project files
COPY . /workspace/Stg2_LLD_Noise_Model/

# Set working directory to project root
WORKDIR /workspace/Stg2_LLD_Noise_Model

# Create directories for outputs
RUN mkdir -p runs denoise_last_ckpt

# Default command (can be overridden)
CMD ["/bin/bash"]

