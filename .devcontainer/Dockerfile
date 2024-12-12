# Use the NVIDIA CUDA base image with a specific version
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Build arguments for configurable versions
ARG PYTHON_VERSION=3.8
ARG PYTORCH_VERSION=1.12.1
ARG CONDA_ENV_NAME=octformer
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"

# Set environment variables
ENV PYTHON_VERSION=${PYTHON_VERSION} \
    PYTORCH_VERSION=${PYTORCH_VERSION} \
    CONDA_ENV_NAME=${CONDA_ENV_NAME} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    DEBIAN_FRONTEND=noninteractive

# Set labels for metadata
LABEL maintainer="Leunar Kalludra <leunarkalludra@gmail.com>" \
      description="Docker image for OctFormer with PyTorch and CUDA"

# Set the working directory
WORKDIR /workspace

# Install system dependencies and Miniconda in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        git \
        build-essential \
        ca-certificates && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/conda/bin:${PATH}"

# Create the conda environment
RUN conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y && \
    conda clean -ya

# Use bash as the default shell
SHELL ["/bin/bash", "-c"]

# Install PyTorch and related packages
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate ${CONDA_ENV_NAME} && \
    conda install -y \
        pytorch=${PYTORCH_VERSION} \
        torchvision \
        torchaudio \
        cudatoolkit=11.3 \
        -c pytorch && \
    conda clean -ya

# Install pip packages and OpenMIM dependencies
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate ${CONDA_ENV_NAME} && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir openmim==0.3.3 && \
    mim install mmcv-full==1.6.2 && \
    mim install mmdet==2.26.0 && \
    mim install mmsegmentation==0.29.1

# Clone and install MMDetection3D
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate ${CONDA_ENV_NAME} && \
    git clone --branch v1.0.0rc5 --depth 1 https://github.com/open-mmlab/mmdetection3d.git && \
    cd mmdetection3d && \
    pip install --no-cache-dir -e .

# Clone and install Octree Depthwise Convolution library
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate ${CONDA_ENV_NAME} && \
    git clone https://github.com/octree-nn/dwconv.git && \
    pip install --no-cache-dir ./dwconv

# Set the final working directory
WORKDIR /workspace/octformer

# Final modifications at the end of the Dockerfile
SHELL ["/bin/bash", "-c"]

# Activate conda environment by default
ENTRYPOINT ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate octformer && exec \"$@\"", "bash"]

CMD ["/bin/bash"]
