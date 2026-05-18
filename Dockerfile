# Base image: CUDA 12.1 is compatible with host driver >= 525 via minor version compatibility.
# If server driver >= 550 (check with nvidia-smi), upgrade to:
#   nvidia/cuda:12.4.1-cudnn9-devel-ubuntu22.04
# and change the torch install index to --index-url https://download.pytorch.org/whl/cu124
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# System deps: Python 3.10, MuJoCo EGL rendering, C extension compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3-setuptools \
    git build-essential patchelf curl \
    libegl1-mesa-dev libgl1-mesa-glx libosmesa6-dev libglew-dev \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# Layer 1: PyTorch with CUDA wheels — heavy layer, cached separately from app deps
RUN python -m pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# Layer 2: All other pip deps (numpy pinned before loris build in layer 3)
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Layer 3: loris patched install — must run after numpy==1.26.4 is present
COPY install_loris.sh .
RUN bash install_loris.sh

# Source code
COPY . .

CMD ["python", "dreamer.py", "--configs", "dmc_vision", "--task", "dmc_walker_walk"]
