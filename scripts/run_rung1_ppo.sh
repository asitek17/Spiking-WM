#!/usr/bin/env bash
# Rung 1.5: PPO-style actor without backprop through the world model.
#
# Run 1: r1_ppo (seed 0)          — PPO clip (epsilon=0.2, K=4 epochs), no ET.
# Run 2: r1_ppo_back_et (seed 0)  — PPO clip + backward ET (lambda=0.95).
#
# Compare against existing baselines: logs/r1_reinforce_s0, logs/r1_reinforce_back_et_s0.
#
# Usage:
#   nohup bash scripts/run_rung1_ppo.sh > logs/rung1_ppo.log 2>&1 &
#   tail -f logs/rung1_ppo.log
#   tail -f logs/r1_ppo_s0.log
#   tail -f logs/r1_ppo_back_et_s0.log

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

run_exp r1_ppo_s0 \
    --configs dmc_vision r1_ppo \
    --task dmc_walker_walk --seed 0 \
    --logdir logs/r1_ppo_s0

run_exp r1_ppo_back_et_s0 \
    --configs dmc_vision r1_ppo_back_et \
    --task dmc_walker_walk --seed 0 \
    --logdir logs/r1_ppo_back_et_s0

echo "=== [$(ts)] ALL DONE ==="
