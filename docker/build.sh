#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-spiking-wm}"

# Run from the project root, not from docker/
cd "$(dirname "$0")/.."

docker build -t "${IMAGE_NAME}" .
echo "Built image: ${IMAGE_NAME}"
