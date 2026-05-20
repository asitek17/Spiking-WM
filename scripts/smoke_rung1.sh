#!/usr/bin/env bash
# Rung 1 smoke test (~2 min): verify the reinforce + backward-ET path runs
# end-to-end and the eligibility-trace shapes line up before the overnight run.
# Uses the reinforce_back_et config (the most complex path) for a few hundred
# steps with --compile False for fast startup.
#
# Usage:
#   bash scripts/smoke_rung1.sh
# Expect: no shape/NaN errors, reaches "Start training." and logs a few steps.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

echo "=== [$(date '+%F %T')] Rung1 smoke: reinforce + backward-ET, 400 steps ==="
if python dreamer.py \
        --configs dmc_vision r1_reinforce_back_et \
        --task dmc_walker_walk --seed 0 \
        --steps 400 --compile False \
        --logdir logs/smoke_r1; then
    echo "=== SMOKE OK — reinforce+ET runs, shapes fine. Safe to run scripts/run_rung1.sh ==="
else
    echo "=== SMOKE FAILED (exit $?) — fix before the overnight run ===" >&2
    exit 1
fi
