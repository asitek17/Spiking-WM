#!/usr/bin/env bash
# Night 2: lambda sweep + seed variance for ET.
#
# Run 1: ET with lambda=0.7, seed=0   (tests milder re-weighting)
# Run 2: ET with lambda=0.95, seed=1  (tests seed variance vs previous failing run)
#
# Compare both vs existing baseline (seed=0, ~363 return).
#
# Usage:
#   nohup bash scripts/run_night2.sh > logs/night2_overnight.log 2>&1 &
#   tail -f logs/night2_overnight.log         # overall progress
#   tail -f logs/fast_et_l07_s0.log           # run 1 details
#   tail -f logs/fast_et_l095_s1.log          # run 2 details

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

run_exp fast_et_l07_s0 \
    --configs dmc_vision fast_exp_et_lambda07 \
    --task dmc_walker_walk --seed 0 \
    --logdir logs/fast_et_l07_s0

run_exp fast_et_l095_s1 \
    --configs dmc_vision fast_exp_et \
    --task dmc_walker_walk --seed 1 \
    --logdir logs/fast_et_l095_s1

echo "=== [$(ts)] ALL DONE ==="
