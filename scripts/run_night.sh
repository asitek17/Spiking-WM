#!/usr/bin/env bash
# Run two experiments sequentially; each gets its own log file.
#
# Usage:
#   nohup bash scripts/run_night.sh > logs/overnight.log 2>&1 &
#   tail -f logs/overnight.log          # overall progress
#   tail -f logs/baseline_500k.log      # baseline details
#   tail -f logs/et_500k.log            # ET details

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
    echo "=== [$(ts)] START  $tag ==="
    if python dreamer.py "$@" > "logs/${tag}.log" 2>&1; then
        echo "=== [$(ts)] OK     $tag ==="
    else
        echo "=== [$(ts)] FAILED $tag (exit $?) — see logs/${tag}.log ==="
    fi
}

run_exp baseline_500k \
    --configs dmc_vision walker_bs32_500k \
    --task dmc_walker_walk --seed 0 \
    --logdir logs/baseline_500k

run_exp et_500k \
    --configs dmc_vision walker_et_500k \
    --task dmc_walker_walk --seed 0 \
    --logdir logs/et_500k

echo "=== [$(ts)] ALL DONE ==="
