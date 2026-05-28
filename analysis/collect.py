#!/usr/bin/env python3
"""Collect DreamerV3 metrics.jsonl from several runs into one tidy CSV.

Each run's logdir contains a `metrics.jsonl` written by tools.Logger
(one JSON object per line: {"step": int, "<scalar>": value, ...}). Not every
line carries every key (e.g. eval_return only appears on eval rounds), so we
emit one (condition, seed, step, metric, value) row per metric actually present.

Output: analysis/out/metrics_long.csv  (long/tidy format, seed-aware so it
scales to a multi-seed comparison without changes here).

Usage:
    python analysis/collect.py                 # uses <repo>/logs
    python analysis/collect.py --logs-dir DIR  # override logs root
"""
import argparse
import csv
import json
import sys
from pathlib import Path

# Runs to collect: (logdir basename under logs root, condition label, seed).
# The dynamics baseline keeps its irregular legacy name (logs/fast_baseline).
MANIFEST = [
    ("fast_baseline", "dynamics", 0),
    ("r1_reinforce_s0", "reinforce", 0),
    ("r1_reinforce_back_et_s0", "reinforce_back_et", 0),
    ("r1_reinforce_stab_s0", "reinforce_stab", 0),
    ("r1_reinforce_stab2_s0", "reinforce_stab2", 0),
    ("r1_ppo_s0", "ppo", 0),
    ("r1_ppo_back_et_s0", "ppo_back_et", 0),
    ("r1_ppo_stab_s0", "ppo_stab", 0),
    ("r1_ppo_stab2_s0", "ppo_stab2", 0),
    ("dynamics_li_s0", "dynamics_li", 0),
    ("ppo_stab2_li_s0", "ppo_stab2_li", 0),
]

# Primary metric is eval_return / train_return; the rest are diagnostics.
METRICS = [
    "eval_return",
    "train_return",
    "value_mean",
    "actor_entropy",
    "actor_et_max",
    "actor_grad_norm",
    "model_loss",
    "actor_ppo_r_mean",
    "actor_ppo_clip_frac",
]


def iter_rows(jsonl_path, condition, seed):
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = rec.get("step")
            if step is None:
                continue
            for metric in METRICS:
                if metric in rec and rec[metric] is not None:
                    yield condition, seed, step, metric, rec[metric]


def main():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=str(repo_root / "logs"),
                        help="root directory holding the run logdirs")
    parser.add_argument("--out", default=str(repo_root / "analysis" / "out" / "metrics_long.csv"))
    args = parser.parse_args()

    logs_root = Path(args.logs_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for logdir, condition, seed in MANIFEST:
        jsonl = logs_root / logdir / "metrics.jsonl"
        if not jsonl.exists():
            print(f"WARN: missing {jsonl} (skipping {condition} s{seed})", file=sys.stderr)
            continue
        run_rows = list(iter_rows(jsonl, condition, seed))
        rows.extend(run_rows)
        print(f"OK: {condition} s{seed}: {len(run_rows)} rows from {jsonl}")

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "seed", "step", "metric", "value"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
