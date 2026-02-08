# ---------------------------------------------------------------------------
# Dockerfile – reproduces the "stem" environment
# Self-contained: clones the repo from GitHub, no local files needed.
# Base: NVIDIA CUDA 12.8 + cuDNN on Ubuntu 22.04
#
# Build:
#   docker build -t stem:latest -f Dockerfile .
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# 1. System dependencies + Python
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        wget git build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. Install PyTorch (CUDA 12.8), xformers, and ninja
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
        torch==2.8.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir xformers==0.0.32.post1 ninja

# ---------------------------------------------------------------------------
# 3. Install project dependencies
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
        numpy omegaconf msgspec rouge-score sacrebleu \
        sentencepiece tiktoken blobfile wandb viztracer \
        lm-eval scipy pynvml datatrove orjson

CMD ["/bin/bash"]

ARG USERNAME="agi-user"
ARG USER_UID=1001
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

RUN pip install --no-cache-dir \
    https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp312-cp312-manylinux2014_x86_64.whl

# Install EFA installer deps (keep apt lists for the installer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates tar \
    pciutils environment-modules tcl \
    && apt-get clean

RUN curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz && \
    tar -xf aws-efa-installer-latest.tar.gz && \
    cd aws-efa-installer && \
    ./efa_installer.sh -y --skip-kmod --no-verify

# Now it's safe to delete apt lists
RUN rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/amazon/efa/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH