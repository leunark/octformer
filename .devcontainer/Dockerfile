# Use the NVIDIA CUDA base image with a specific version
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Build arguments for configurable versions
ARG PYTHON_VERSION=3.8
ARG PYTORCH_VERSION=1.12.1
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"

# Environment variables
ENV PYTHON_VERSION=${PYTHON_VERSION} \
    PYTORCH_VERSION=${PYTORCH_VERSION} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    DEBIAN_FRONTEND=noninteractive

# Labels for metadata
LABEL maintainer="Leunar Kalludra <leunarkalludra@gmail.com>" \
      description="Docker image for OctFormer with PyTorch and CUDA (using system Python ${PYTHON_VERSION})"

# Set the working directory
WORKDIR /workspace

# Install system dependencies and Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        git \
        build-essential \
        ca-certificates \
        zstd \ 
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-distutils && \
    rm -rf /var/lib/apt/lists/*

# Ensure that the "python" command uses the configured version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Install pip for the configured Python version
RUN wget -qO get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py && \
    python -m pip install --upgrade pip

# Install PyTorch (with CUDA 11.3 support) and related packages via pip.
# Note: The official PyTorch pip wheels use a '+' in the version tag.
RUN python -m pip install --no-cache-dir \
    torch==${PYTORCH_VERSION}+cu113 \
    torchvision \
    torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Install OpenMIM and use it to install MMCV, MMDetection, and MMSegmentation
RUN python -m pip install --no-cache-dir openmim==0.3.3 && \
    mim install mmcv-full==1.6.2 && \
    mim install mmdet==2.26.0 && \
    mim install mmsegmentation==0.29.1

# Clone and install MMDetection3D
RUN git clone --branch v1.0.0rc5 --depth 1 https://github.com/open-mmlab/mmdetection3d.git && \
    cd mmdetection3d && \
    python -m pip install --no-cache-dir -e .

# Clone and install Octree Depthwise Convolution library
RUN git clone https://github.com/octree-nn/dwconv.git && \
    python -m pip install --no-cache-dir ./dwconv

# Set the final working directory to your application directory.
WORKDIR /workspace/octformer

# Copy your application's requirements.txt into the image
# (Make sure that requirements.txt is in the same folder as this Dockerfile.)
COPY requirements.txt .

# Install pip packages from your requirements.txt file
RUN python -m pip install --no-cache-dir -r requirements.txt

# Set the default command to bash
CMD ["/bin/bash"]