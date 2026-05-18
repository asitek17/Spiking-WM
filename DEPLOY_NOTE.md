# Spiking World Model — Docker Deployment Guide (Ubuntu 24.04 LTS)

Step-by-step guide for building and running the project on a server with an NVIDIA GPU.
The container uses pure pip (no Conda) based on the official CUDA runtime image.

## System specification (reference)
* **OS:** Ubuntu 24.04 LTS
* **GPU:** NVIDIA RTX 4090 (24 GB VRAM) or similar
* **Host driver:** check with `nvidia-smi` — see "CUDA Version" in top-right corner
  * Driver >= 550 → CUDA 12.4 in container (matches `environment.yml`)
  * Driver 525–549 → use CUDA 12.1 (minor version compatibility)

---

## Step 1. Host setup

### Install Docker
```bash
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### Install NVIDIA Container Toolkit
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime and restart daemon
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU passthrough: `docker run --gpus all --rm nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`

---

## Step 2. Project files

The following files are already present in `Spiking-WM-cluster/`:

| File | Purpose |
|------|---------|
| `Dockerfile` | Container definition — CUDA 12.1 + Python 3.10 + EGL rendering |
| `requirements.txt` | All pip deps with `numpy==1.26.4` pinned |
| `install_loris.sh` | Patched loris install (incompatible with modern setuptools) |
| `.dockerignore` | Excludes logs, data, checkpoints from build context |
| `docker/build.sh` | Build helper |
| `docker/run.sh` | Run helper with correct GPU and IPC flags |

Key Dockerfile design decisions:
- `MUJOCO_GL=egl` — EGL headless rendering (faster than OSMesa on NVIDIA GPUs, matches `dreamer.py`)
- `libegl1-mesa-dev` in system deps — required for EGL support
- `loris` installed via `install_loris.sh`, not from requirements.txt (needs source patch)
- `--ipc=host --shm-size=16g` in run script — required for PyTorch multiprocessing DataLoader

---

## Step 3. Build and run

```bash
cd Spiking-WM-cluster

# Build image
bash docker/build.sh

# Interactive shell with GPU and mounted volumes
bash docker/run.sh

# Inside the container — smoke test
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# Inside the container — training
python dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir /workspace/logs --seed 0
```

### Background training (survives SSH disconnect)
```bash
docker run --gpus all -d \
  --name swm_training \
  --ipc=host \
  --shm-size=16g \
  -v $(pwd):/workspace \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e WANDB_API_KEY=<your_key> \
  -w /workspace \
  spiking-wm \
  python dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir /workspace/logs

# Monitor
docker logs -f swm_training

# Stop
docker stop swm_training
```

---

## Troubleshooting

1. **OOM on CPU RAM**: reduce `num_workers` in DataLoader if the server has many CPU cores.
2. **WandB**: pass API key via `-e WANDB_API_KEY=<key>` on `docker run`.
3. **CUDA version mismatch**: run `nvidia-smi` on the host and compare "CUDA Version" against the
   version in the Dockerfile base image. If driver >= 550, switch to the `cu124` base image and
   wheel index (see comment at top of `Dockerfile`).
