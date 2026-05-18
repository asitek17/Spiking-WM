#!/usr/bin/env bash
# One-shot environment setup: Python 3.10 + all pip deps + loris (patched).
# Run once after cloning, then: source .venv/bin/activate
set -euo pipefail

if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found. Copy the binary to ~/.local/bin/ and add it to PATH."
    echo "       curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "==> Installing Python 3.10"
uv python install 3.10

echo "==> Syncing dependencies (torch cu121 + all pip deps)"
uv sync

echo "==> Installing loris 0.5.3 (patched source build)"
source .venv/bin/activate
bash install_loris.sh

echo ""
echo "Setup complete."
echo "  source .venv/bin/activate"
echo "  python -c \"import torch; print(torch.cuda.is_available(), torch.version.cuda)\""
