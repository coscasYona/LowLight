# Use NVIDIA PyTorch container as base
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies for RAW image processing
RUN apt-get update && apt-get install -y \
    libraw-dev \
    libraw-bin \
    libjpeg-dev \
    libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
# Note: numpy 1.26.4 is pinned for compatibility with rawpy/opencv
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Default command (can be overridden)
CMD ["/bin/bash"]

