#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-spiking-wm}"
PROJECT_ROOT="$(realpath "$(dirname "$0")/..")"

# Create output dirs on the host so volumes mount cleanly
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/data"

docker run --gpus all --rm -it \
  --ipc=host \
  --shm-size=16g \
  -v "${PROJECT_ROOT}":/workspace \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -w /workspace \
  "${IMAGE_NAME}" \
  bash
