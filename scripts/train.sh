# Original paper run (acrobot_swingup, 1M steps, spike_times=8)
# python dreamer.py --configs dmc_vision --task dmc_acrobot_swingup --spike_times 8 --seed 0 --logdir ./logs

# Smoke test on laptop (~15-30 min): verify code runs end-to-end
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python dreamer.py --configs dmc_vision --task dmc_walker_walk --spike_times 5 --steps 5000 --compile False --seed 0 --logdir ./logs/walker_walk_smoke

# Full experiment run for cluster (walker_walk, 500k steps)
# python dreamer.py --configs dmc_vision --task dmc_walker_walk --spike_times 5 --steps 500000 --seed 0 --logdir ./logs/walker_walk_500k
