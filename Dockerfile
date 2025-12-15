# Use NVIDIA PyTorch container as base
# Build argument to select appropriate image based on GPU
# RTX 5090: Use pytorch:25.09-py3 (CUDA 13.0)
# RTX 3090: Use pytorch:24.07-py3 (CUDA 12.4)
# Usage: docker build --build-arg PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:24.07-py3 .
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:24.07-py3
FROM ${PYTORCH_IMAGE}

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV DONT_PROMPT_WSL_INSTALL=1

# Install system dependencies for RAW image processing and VS Code/Cursor Server
RUN apt-get update && apt-get install -y \
    libraw-dev \
    libraw-bin \
    libjpeg-dev \
    libtiff-dev \
    libgl1 \
    libglib2.0-0 \
    wget \
    curl \
    ca-certificates \
    git \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (required for VS Code/Cursor Server)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
# Note: numpy 1.26.4 is pinned for compatibility with rawpy/opencv
COPY requirements.txt /tmp/requirements.txt
# Remove any pre-installed OpenCV packages from base image to avoid conflicts
RUN pip uninstall -y opencv opencv-python opencv-python-headless opencv-contrib-python || true \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Install debugpy for remote debugging support
RUN pip install --no-cache-dir debugpy

# Expose ports for TensorBoard and debugger
EXPOSE 6006 5678

# Set working directory
WORKDIR /workspace

# Default command (can be overridden)
CMD ["/bin/bash"]

