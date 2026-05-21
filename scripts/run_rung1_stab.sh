#!/usr/bin/env bash
# Rung 1 stabilized REINFORCE: norm_adv + entropy x10 (3e-3), no ET.
#
# Goal: does REINFORCE learn at all once we fight the entropy collapse seen in
# the first pilot (return stuck ~28 vs dynamics ~369)? If yes, re-run the ET
# comparison under these settings.
#
# Optional smoke first (~2 min):
#   python dreamer.py --configs dmc_vision r1_reinforce_stab \
#     --task dmc_walker_walk --seed 0 --steps 400 --compile False --logdir logs/smoke_r1_stab
#
# Usage:
#   nohup bash scripts/run_rung1_stab.sh > logs/rung1_stab_overnight.log 2>&1 &
#   tail -f logs/r1_reinforce_stab_s0.log

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

run_exp r1_reinforce_stab_s0 \
    --configs dmc_vision r1_reinforce_stab \
    --task dmc_walker_walk --seed 0 \
    --logdir logs/r1_reinforce_stab_s0

echo "=== [$(ts)] ALL DONE ==="
