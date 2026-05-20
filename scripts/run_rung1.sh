#!/usr/bin/env bash
# Rung 1: actor without backprop through the world model (imag_gradient=reinforce).
#
# Run 1: reinforce, no ET            (cost of removing backprop through the WM)
# Run 2: reinforce + backward ET     (does local credit assignment recover the gap)
#
# Compare both vs the existing dynamics baseline (seed 0, ~363 return).
#
# Run scripts/smoke_rung1.sh first (~2 min) to verify the reinforce+ET path.
#
# Usage:
#   nohup bash scripts/run_rung1.sh > logs/rung1_overnight.log 2>&1 &
#   tail -f logs/rung1_overnight.log           # overall progress
#   tail -f logs/r1_reinforce_s0.log           # run 1 details
#   tail -f logs/r1_reinforce_back_et_s0.log   # run 2 details

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

ts() { date '+%F %T'; }

run_exp() {
    local tag="$1"; shift
    local start=$SECONDS
    echo "=== [$(ts)] START  $tag ==="
    if python dreamer.py "$@" > "logs/${tag}.log" 2>&1; then
        echo "=== [$(ts)] OK     $tag (wall time: $((SECONDS - start))s) ==="
    else
        echo "=== [$(ts)] FAILED $tag (exit $?, wall time: $((SECONDS - start))s) — see logs/${tag}.log ==="
    fi
}

run_exp r1_reinforce_s0 \
    --configs dmc_vision r1_reinforce \
    --task dmc_walker_walk --seed 0 \
    --logdir logs/r1_reinforce_s0

run_exp r1_reinforce_back_et_s0 \
    --configs dmc_vision r1_reinforce_back_et \
    --task dmc_walker_walk --seed 0 \
    --logdir logs/r1_reinforce_back_et_s0

echo "=== [$(ts)] ALL DONE ==="
