# -----------------------------------------------------------------------------
# Base Image: NVIDIA PyTorch Container
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
# -----------------------------------------------------------------------------
FROM nvcr.io/nvidia/pytorch:25.08-py3

# -----------------------------------------------------------------------------
# Set basic environment variables
# -----------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Seoul

WORKDIR /workspace

# -----------------------------------------------------------------------------
# Install system dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ffmpeg \
    libsndfile1 \
    vim \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Install Python dependencies
# -----------------------------------------------------------------------------
RUN pip install --upgrade pip && \
    pip install \
        numpy \
        scipy \
        tqdm \
        datasets \
        jiwer \
        hydra-core \
        omegaconf \
        wandb \
        soundfile \
        librosa

# (Optional) Install RNNT loss implementations
# torchaudio has rnnt_loss, but warp-rnnt is an alternative
# RUN pip install warp-rnnt --no-build-isolation || true

# -----------------------------------------------------------------------------
# Copy project files (optional — adjust path based on your folder structure)
# -----------------------------------------------------------------------------
# COPY . /workspace

# -----------------------------------------------------------------------------
# Default command
# -----------------------------------------------------------------------------
CMD ["/bin/bash"]
