#!/usr/bin/env python3
"""Summarise collected Rung-1 metrics into a results table (+ optional curves).

Reads analysis/out/metrics_long.csv (from collect.py) and prints a per-condition
summary. Primary metric = windowed train_return over the trailing `--window`
steps (matches how the dynamics baseline ~363 was computed; eval_return is sparse
so it is reported as a secondary, final-checkpoint value). Diagnostics
(value_mean, actor_entropy, actor_et_max, actor_grad_norm, model_loss) are taken
at the last available step.

When a condition has >1 seed, the primary metric is aggregated as mean with a
bootstrap CI (rliable-style, for small n); with a single seed (the pilot) the
raw value is shown. No "bold the best cell" — one primary metric, the rest is
diagnostics.

Output: stdout table + analysis/out/summary.csv; analysis/out/curves.png if
matplotlib is available.

Usage:
    python analysis/plot_table.py [--window 8000]
"""
import argparse
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path

PRIMARY = "train_return"
SECONDARY = "eval_return"
DIAGNOSTICS = ["eval_return", "value_mean", "actor_entropy", "actor_et_max",
               "actor_grad_norm", "model_loss"]
CURVE_METRICS = ["eval_return", "train_return", "value_mean"]
CONDITION_ORDER = ["dynamics", "reinforce", "reinforce_back_et"]


def load(csv_path):
    # series[condition][seed][metric] -> list of (step, value), sorted by step.
    series = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            series[row["condition"]][int(row["seed"])][row["metric"]].append(
                (float(row["step"]), float(row["value"]))
            )
    for cond in series.values():
        for seed in cond.values():
            for metric in seed:
                seed[metric].sort(key=lambda sv: sv[0])
    return series


def final_value(pairs):
    return pairs[-1][1] if pairs else None


def window_mean(pairs, window):
    if not pairs:
        return None
    max_step = pairs[-1][0]
    vals = [v for s, v in pairs if s >= max_step - window]
    return statistics.mean(vals) if vals else None


def bootstrap_ci(values, n=10000, alpha=0.05):
    if len(values) < 2:
        return (values[0], values[0]) if values else (None, None)
    means = []
    for _ in range(n):
        sample = [random.choice(values) for _ in values]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int(alpha / 2 * n)]
    hi = means[int((1 - alpha / 2) * n)]
    return lo, hi


def fmt(x):
    return "  --  " if x is None else f"{x:7.2f}"


def main():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(repo_root / "analysis" / "out" / "metrics_long.csv"))
    parser.add_argument("--window", type=float, default=8000,
                        help="trailing step window for the primary train_return metric")
    parser.add_argument("--out-dir", default=str(repo_root / "analysis" / "out"))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"missing {csv_path}; run collect.py first")
    series = load(csv_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = [c for c in CONDITION_ORDER if c in series] + \
                 [c for c in series if c not in CONDITION_ORDER]

    header = (f"{'condition':<20}{'n':>3}  {'train_ret(win)':>16}  "
              f"{'eval_ret':>9}  {'value_mean':>10}  {'act_ent':>8}  "
              f"{'et_max':>7}  {'model_loss':>10}")
    print(header)
    print("-" * len(header))

    summary_rows = []
    for cond in conditions:
        seeds = sorted(series[cond])
        primary_vals = [window_mean(series[cond][s][PRIMARY], args.window) for s in seeds]
        primary_vals = [v for v in primary_vals if v is not None]
        primary_mean = statistics.mean(primary_vals) if primary_vals else None

        # diagnostics: mean across seeds of the per-seed final value
        def diag(metric):
            vs = [final_value(series[cond][s][metric]) for s in seeds]
            vs = [v for v in vs if v is not None]
            return statistics.mean(vs) if vs else None

        primary_str = fmt(primary_mean)
        if len(primary_vals) > 1:
            lo, hi = bootstrap_ci(primary_vals)
            primary_str = f"{primary_mean:6.1f}[{lo:.0f},{hi:.0f}]"

        print(f"{cond:<20}{len(seeds):>3}  {primary_str:>16}  "
              f"{fmt(diag(SECONDARY))}  {fmt(diag('value_mean'))}  "
              f"{fmt(diag('actor_entropy'))}  {fmt(diag('actor_et_max'))}  "
              f"{fmt(diag('model_loss'))}")

        summary_rows.append({
            "condition": cond, "n_seeds": len(seeds),
            "train_return_window": primary_mean,
            **{f"{m}_final": diag(m) for m in DIAGNOSTICS},
        })

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        cols = ["condition", "n_seeds", "train_return_window"] + \
               [f"{m}_final" for m in DIAGNOSTICS]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nWrote {summary_path}")

    # Optional curves (bonus): one panel per metric, one line per condition.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -> skipping curves.png")
        return

    fig, axes = plt.subplots(1, len(CURVE_METRICS), figsize=(5 * len(CURVE_METRICS), 4))
    if len(CURVE_METRICS) == 1:
        axes = [axes]
    for ax, metric in zip(axes, CURVE_METRICS):
        plotted = False
        for cond in conditions:
            # mean over seeds at each step (single-seed pilot -> just that seed)
            by_step = defaultdict(list)
            for s in series[cond]:
                for step, val in series[cond][s][metric]:
                    by_step[step].append(val)
            if not by_step:
                continue
            steps = sorted(by_step)
            means = [statistics.mean(by_step[st]) for st in steps]
            ax.plot(steps, means, marker="o", ms=3, label=cond)
            plotted = True
        ax.set_title(metric)
        ax.set_xlabel("logger step")
        ax.grid(alpha=0.3)
        if plotted:
            ax.legend(fontsize=8)
    fig.tight_layout()
    curves_path = out_dir / "curves.png"
    fig.savefig(curves_path, dpi=120)
    print(f"Wrote {curves_path}")


if __name__ == "__main__":
    main()
