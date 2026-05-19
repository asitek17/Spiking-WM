#!/usr/bin/env bash
# Timing probe: runs a short experiment to measure fps and estimate fast_exp duration.
#
# Usage:
#   bash scripts/run_timing.sh
#   tail -f logs/timing_test.log

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

echo "=== [$(ts)] START timing_test ==="
start=$SECONDS
python dreamer.py \
    --configs dmc_vision timing_test \
    --task dmc_walker_walk --seed 0 \
    --logdir logs/timing_test \
    > logs/timing_test.log 2>&1
exit_code=$?
elapsed=$((SECONDS - start))

if [ $exit_code -ne 0 ]; then
    echo "=== [$(ts)] FAILED timing_test (exit $exit_code, wall time: ${elapsed}s) — see logs/timing_test.log ==="
    exit $exit_code
fi
echo "=== [$(ts)] OK  timing_test (wall time: ${elapsed}s) ==="

fps=$(grep -oP 'fps \K[0-9.]+' logs/timing_test.log | tail -1)
if [ -z "$fps" ]; then
    echo "Could not parse fps from log — check logs/timing_test.log"
    exit 1
fi

echo ""
echo "--- Timing results ---"
echo "Wall time (training + eval + startup): ${elapsed}s ($(echo "scale=1; $elapsed/60" | bc) min)"
echo "Training fps (last reading):           $fps logger-steps/sec"
echo ""
echo "Estimated training time per run (eval overhead not included):"
for steps in 200000 500000; do
    hours=$(echo "scale=1; $steps / $fps / 3600" | bc)
    echo "  steps=$steps  ->  ~${hours}h  (x2 for baseline+ET pair)"
done
echo ""
echo "Check logs/timing_test.log for NaN or instability before launching fast runs."
