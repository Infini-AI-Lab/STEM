# ---------------------------------------------------------------------------
# Dockerfile – reproduces the "stem" conda environment
# Self-contained: clones the repo from GitHub, no local files needed.
# Base: NVIDIA CUDA 12.8 + cuDNN on Ubuntu 22.04
#
# Build from anywhere:
#   docker build -t stem:latest -f Dockerfile .
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# 1. System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        bzip2 \
        ca-certificates \
        curl \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. Install Miniconda
# ---------------------------------------------------------------------------
ENV CONDA_DIR=/opt/conda
RUN wget -qO /tmp/miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh
ENV PATH=${CONDA_DIR}/bin:${PATH}

# ---------------------------------------------------------------------------
# 3. Create the "stem" conda environment with Python 3.11
# ---------------------------------------------------------------------------
RUN conda create -n stem python=3.11 -y && \
    conda clean -afy

# Make every subsequent RUN use the stem environment
SHELL ["conda", "run", "-n", "stem", "/bin/bash", "-c"]

# ---------------------------------------------------------------------------
# 4. Install PyTorch (CUDA 12.8), xformers, and ninja
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
        torch==2.8.0 \
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir xformers==0.0.32.post1 ninja

# ---------------------------------------------------------------------------
# 5. Clone the STEM repo and install requirements
# ---------------------------------------------------------------------------
WORKDIR /workspace
RUN git clone https://github.com/facebookresearch/STEM.git
WORKDIR /workspace/STEM
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# 6. Default entrypoint activates the conda env
# ---------------------------------------------------------------------------
# Reset shell so ENTRYPOINT/CMD don't require conda run
SHELL ["/bin/bash", "-c"]

# Activate stem env by default for interactive and non-interactive shells
RUN echo "source activate stem" >> ~/.bashrc
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "stem"]
CMD ["/bin/bash"]
