#!/usr/bin/env bash
# Rung 1.5 stabilized: PPO with entropy floor fix + conservative clip/epochs.
#
# Addresses instability from r1_ppo_s0 (train_return 40↔220 oscillation):
#   - actor_min_std=0.3  : keeps score-function gradient alive (was at σ=0.1 floor)
#   - ppo_epsilon=0.1    : tighter clip, fewer wild r_t swings
#   - ppo_epochs=2       : less reuse on stale importance ratios
#   - actor_reinforce_norm_adv=True : scale normalization via RewardEMA
#   - steps=80000        : 2× longer for convergence plateau
#
# Expected: ~250–320 train_return, much smoother curves than r1_ppo_s0.
# Baseline refs: dynamics ~369, ppo ~175 (noisy), reinforce ~28.
#
# Usage:
#   nohup bash scripts/run_rung1_ppo_stab.sh > logs/rung1_ppo_stab.log 2>&1 &
#   tail -f logs/rung1_ppo_stab.log
#   tail -f logs/r1_ppo_stab_s0.log

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

run_exp r1_ppo_stab_s0 \
    --configs dmc_vision r1_ppo_stab \
    --task dmc_walker_walk --seed 0 \
    --logdir logs/r1_ppo_stab_s0

echo "=== [$(ts)] ALL DONE ==="
