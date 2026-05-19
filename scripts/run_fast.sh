#!/usr/bin/env bash
# Run fast baseline + ET experiments sequentially.
#
# Usage:
#   nohup bash scripts/run_fast.sh > logs/fast_overnight.log 2>&1 &
#   tail -f logs/fast_overnight.log   # overall progress
#   tail -f logs/fast_baseline.log    # baseline details
#   tail -f logs/fast_et.log          # ET details

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

run_exp fast_baseline \
    --configs dmc_vision fast_exp_baseline \
    --task dmc_walker_walk --seed 0 \
    --logdir logs/fast_baseline

run_exp fast_et \
    --configs dmc_vision fast_exp_et \
    --task dmc_walker_walk --seed 0 \
    --logdir logs/fast_et

echo "=== [$(ts)] ALL DONE ==="
