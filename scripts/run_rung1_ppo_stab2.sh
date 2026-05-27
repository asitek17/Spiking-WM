#!/usr/bin/env bash
# Rung 1.5 stabilized v2: только actor_min_std=0.3, остальное как в r1_ppo.
#
# Гипотеза: нестабильность ppo_s0 шла от энтропийного коллапса (σ=0.1),
# а не от K=4/ε=0.2. С min_std=0.3 r_t остаётся у 1 → K=4 безопасны.
# Ожидание: стабильность ppo_stab + скорость ppo → ~200+ за 40k шагов.
#
# Usage:
#   nohup bash scripts/run_rung1_ppo_stab2.sh > logs/rung1_ppo_stab2.log 2>&1 &
#   tail -f logs/rung1_ppo_stab2.log
#   tail -f logs/r1_ppo_stab2_s0.log

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

run_exp r1_ppo_stab2_s0 \
    --configs dmc_vision r1_ppo_stab2 \
    --task dmc_walker_walk --seed 0 \
    --logdir logs/r1_ppo_stab2_s0

echo "=== [$(ts)] ALL DONE ==="
