#!/usr/bin/env bash
# LI readout experiment: compare dynamics + li_readout vs ppo_stab2 + li_readout.
#
# Гипотеза: замена линейного readout на leaky integrator (без спайков) делает
# пайплайн нейроморфным и может улучшить градиентный поток через readout-слой.
#
# Условия:
#   dynamics_li  — world-model gradient + LI readout   (baseline + li)
#   ppo_stab2_li — ppo_stab2 + LI readout              (best PPO + li)
#
# Usage:
#   nohup bash scripts/run_li_readout.sh > logs/li_readout.log 2>&1 &
#   tail -f logs/li_readout.log

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

run_exp dynamics_li_s0 \
    --configs dmc_vision fast_exp_baseline li_readout \
    --task dmc_walker_walk --seed 0 \
    --logdir logs/dynamics_li_s0

run_exp ppo_stab2_li_s0 \
    --configs dmc_vision r1_ppo_stab2 li_readout \
    --task dmc_walker_walk --seed 0 \
    --logdir logs/ppo_stab2_li_s0

echo "=== [$(ts)] ALL DONE ==="
