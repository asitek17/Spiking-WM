# Spiking World Model — Deployment Guide (uv, no sudo)

Step-by-step guide for deploying on a server with an NVIDIA GPU using
[uv](https://docs.astral.sh/uv/) — no Docker, no Conda, no sudo required.

## System specification (reference)

| Item | Value |
|------|-------|
| OS | Ubuntu 24.04 LTS |
| GPU | NVIDIA RTX 4090 (24 GB VRAM) |
| Driver / CUDA | ≥ 525 (CUDA 12.1 minor-version compatible) |
| Python | 3.10 (installed by uv, no system Python needed) |
| EGL | `libEGL.so.1` + `libEGL_nvidia.so.0` present → MuJoCo EGL headless works |
| build-essential | installed (required for loris C++ extension) |

---

## Step 1. Get uv (no sudo)

If `uv` is not on the server, copy the binary from a machine that has it:

```bash
# On local machine — copy to server
scp ~/.local/bin/uv user@server:~/.local/bin/uv

# On the server — add to PATH (add to ~/.bashrc for persistence)
export PATH="$HOME/.local/bin:$PATH"
uv --version   # should print uv 0.x.y
```

Or install directly on the server if internet is available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Step 2. Clone and set up

```bash
git clone <repo-url> ~/projects/Spiking-WM
cd ~/projects/Spiking-WM
bash setup.sh
```

`setup.sh` does three things in order:
1. `uv python install 3.10` — downloads a standalone CPython 3.10 (no sudo)
2. `uv sync` — creates `.venv/`, installs torch cu121 + all other deps from `pyproject.toml`
3. `bash install_loris.sh` — patches and builds loris 0.5.3 from source

Expected time: ~5–10 min (dominated by torch download ~2 GB).

---

## Step 3. Verify

```bash
source .venv/bin/activate

# CUDA check
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# Expected: True 12.1

# MuJoCo EGL check
MUJOCO_GL=egl python -c "import mujoco; print('mujoco ok')"
```

---

## Step 4. Training

```bash
source .venv/bin/activate

# Smoke test (~15-30 min)
bash scripts/train.sh

# Background run (survives SSH disconnect)
nohup bash scripts/train.sh > logs/train.log 2>&1 &
tail -f logs/train.log
```

Or run directly:

```bash
MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python dreamer.py --configs dmc_vision walker_bs32_500k \
  --task dmc_walker_walk --seed 0 --logdir ./logs/walker_bs32_500k
```

---

## Key files

| File | Purpose |
|------|---------|
| `pyproject.toml` | All Python deps; torch pulled from PyTorch cu121 index |
| `.python-version` | Pins Python 3.10 for uv |
| `setup.sh` | One-command setup: `uv python install` + `uv sync` + loris |
| `install_loris.sh` | Patched loris 0.5.3 source build (modern setuptools workaround) |
| `scripts/train.sh` | Training launcher — activates venv, sets `MUJOCO_GL=egl` |
| `uv.lock` | Locked dependency graph (committed for reproducibility) |

Docker files (`Dockerfile`, `docker/`) remain as reference but are not used.

---

## Dependency notes

| Problem | Root cause | Fix |
|---------|-----------|-----|
| `loris==0.5.3` install fails | `setup.py` uses `__builtins__.__NUMPY_SETUP__` — a dict under modern setuptools | `install_loris.sh` patches the line before building |
| `loris` C++ build error | Uses `PyArray_DESCR->fields`, removed in numpy 2.0 | `numpy==1.26.4` pinned first in `pyproject.toml` |
| `undefined symbol: iJIT_NotifyEvent` | pip `mkl==2026.x` missing a VTune symbol that torch 2.4.1 expects | `mkl-service` excluded from deps |

## Troubleshooting

**uv not found after setup:**
```bash
export PATH="$HOME/.local/bin:$PATH"
```

**EGL error on `import mujoco`:**
```bash
ls /usr/lib/x86_64-linux-gnu/libEGL*   # should show libEGL.so.1 and libEGL_nvidia.so.0
```
If missing, contact the server admin — NVIDIA driver must be installed with EGL support.

**WandB:**
```bash
export WANDB_API_KEY=<your_key>
```
