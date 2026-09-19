#!/usr/bin/env python3
"""Render the maze-size scaling and hyperparameter sensitivity figures.

Reads outputs/scaling/scaling.csv and outputs/sensitivity/sensitivity.csv
(produced by run_scaling_study.py / run_sensitivity_study.py) and renders two
print-ready PNGs for the thesis:

    outputs/scaling/ga_scaling.png       - why GA doesn't scale to bigger mazes
    outputs/sensitivity/ga_sensitivity.png - effect of GA hyperparameters

Usage:
    python scripts/plot_scaling_study.py
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# -- validated palette (dataviz skill, light mode) --------------------------
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"     # categorical slot 1 - success rate
ORANGE = "#eb6834"   # categorical slot 2 - mean progress toward goal
GOOD = "#0ca30c"     # status: solved
CRITICAL = "#d03b3b"  # status: not solved

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "text.color": INK_PRIMARY,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "grid.color": GRID,
})


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _style_ax(ax):
    ax.grid(True, axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
        ax.spines[spine].set_linewidth(1)


# -- scaling study ------------------------------------------------------
def plot_scaling(rows, out_path):
    by_size = defaultdict(list)
    for r in rows:
        by_size[int(r["size"])].append(r)
    sizes = sorted(by_size)

    success_rate = []
    mean_progress = []
    n_seeds = []
    for s in sizes:
        trials = by_size[s]
        solved = [int(r["solved"]) for r in trials]
        success_rate.append(100.0 * sum(solved) / len(solved))
        mean_progress.append(100.0 * np.mean([float(r["final_progress_frac"]) for r in trials]))
        n_seeds.append(len(trials))

    # Bucketed success rate by TRUE task difficulty (optimal path length) — the
    # cleanest, most quotable form of "works up to X, fails past Y".
    bucket_edges = [(0, 60, "<60"), (60, 80, "60-79"), (80, 100, "80-99"),
                    (100, 140, "100-139"), (140, 10**9, "140+")]
    bucket_labels, bucket_rates, bucket_counts = [], [], []
    for lo, hi, label in bucket_edges:
        sub = [r for r in rows if lo <= int(r["optimal_path_length"]) < hi]
        if not sub:
            continue
        bucket_labels.append(label)
        bucket_rates.append(100.0 * sum(int(r["solved"]) for r in sub) / len(sub))
        bucket_counts.append(len(sub))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "GA reliably solves mazes up to ~60 cells of optimal path — and fails past ~100",
        fontsize=15, fontweight="bold", color=INK_PRIMARY, x=0.01, ha="left",
    )
    fig.text(
        0.01, 0.925,
        "Fixed budget throughout: population 300, generations 200, max 300 steps/episode — thesis config",
        fontsize=10, color=INK_SECONDARY, ha="left",
    )

    # Panel A: success rate + mean progress vs size, with a shaded reliable zone
    ax = axes[0]
    _style_ax(ax)
    ax.axhspan(70, 108, color=GOOD, alpha=0.06, zorder=0)
    ax.axhspan(0, 30, color=CRITICAL, alpha=0.06, zorder=0)
    ax.plot(sizes, success_rate, color=BLUE, linewidth=2, marker="o", markersize=8,
             markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
             label="Success rate")
    ax.plot(sizes, mean_progress, color=ORANGE, linewidth=2, marker="o", markersize=8,
             markerfacecolor=ORANGE, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
             label="Mean progress toward goal")
    ax.text(sizes[-1], success_rate[-1], f"  {success_rate[-1]:.0f}%", color=BLUE,
            fontsize=10, va="center", fontweight="bold")
    ax.text(sizes[-1], mean_progress[-1], f"  {mean_progress[-1]:.0f}%", color=ORANGE,
            fontsize=10, va="center", fontweight="bold")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(sizes, n_seeds)], fontsize=8.5)
    ax.set_xlabel("Maze size (N×N)", labelpad=10)
    ax.set_ylabel("%")
    ax.set_ylim(0, 108)
    ax.set_title("Reliable on small/medium mazes, drops off on large ones", fontsize=11, color=INK_PRIMARY, loc="left")
    ax.legend(frameon=False, loc="center left", fontsize=9)
    ax.set_xlim(sizes[0] - 1.5, sizes[-1] + 1.5)

    # Panel B: the headline number — success rate binned by TRUE difficulty
    ax = axes[1]
    _style_ax(ax)
    bar_colors = [GOOD if r >= 70 else (ORANGE if r >= 30 else CRITICAL) for r in bucket_rates]
    bars = ax.bar(bucket_labels, bucket_rates, color=bar_colors, width=0.62, zorder=3)
    for bar, rate in zip(bars, bucket_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 3, f"{rate:.0f}%",
                ha="center", fontsize=10, fontweight="bold", color=INK_PRIMARY)
    ax.set_xticks(range(len(bucket_labels)))
    ax.set_xticklabels([f"{lab}\n(n={n})" for lab, n in zip(bucket_labels, bucket_counts)], fontsize=8.5)
    ax.set_xlabel("Optimal path length (cells) — true task difficulty", labelpad=10)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 112)
    ax.set_title("The threshold: ~100% below 60 cells, 0% past 100", fontsize=11, color=INK_PRIMARY, loc="left")

    # Panel C: generations-to-solve distribution vs size (solved trials only) —
    # "when it works, it works fast"
    ax = axes[2]
    _style_ax(ax)
    box_data, positions, counts = [], [], []
    for s in sizes:
        gens = [int(r["generations_to_solve"]) for r in by_size[s]
                if r["solved"] == "1" and r["generations_to_solve"] != ""]
        box_data.append(gens)
        positions.append(s)
        counts.append(len(gens))
    bp = ax.boxplot(
        box_data, positions=positions, widths=[min(3, (sizes[1]-sizes[0])*0.5) if len(sizes) > 1 else 2] * len(sizes),
        patch_artist=True, showmeans=False, manage_ticks=False,
        boxprops=dict(facecolor=GOOD + "33", edgecolor=GOOD, linewidth=1.5),
        medianprops=dict(color=GOOD, linewidth=2),
        whiskerprops=dict(color=GOOD, linewidth=1.5),
        capprops=dict(color=GOOD, linewidth=1.5),
        flierprops=dict(markerfacecolor=GOOD, markeredgecolor=SURFACE, marker="o", markersize=5),
    )
    ax.set_xticks(sizes)
    ax.set_xticklabels(
        [f"{s}\n({c}/{n})" for s, c, n in zip(sizes, counts, n_seeds)], fontsize=8.5
    )
    ax.set_xlim(sizes[0] - 1.5, sizes[-1] + 1.5)
    ax.set_xlabel("Maze size (N×N) — (solved/seeds)", labelpad=10)
    ax.set_ylabel("Generations to first solve")
    ax.margins(y=0.15)
    ax.set_title("When GA does solve it, it converges fast (<30 generations)", fontsize=11, color=INK_PRIMARY, loc="left")

    fig.text(
        0.01, -0.04,
        "GA's genome is a fixed, open-loop move sequence (it never senses the maze while replaying it): every open "
        "junction on the path must be guessed correctly in advance, so the odds of a random or mutated sequence doing "
        "that collapse once the path needs to thread more than ~60-100 cells' worth of junctions — independent of "
        "population size or generation budget (see the hyperparameter sweep).",
        fontsize=9, color=INK_SECONDARY, ha="left", wrap=True,
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# -- sensitivity study ----------------------------------------------------
SWEEP_TITLES = {
    "population_size": "Population size (generations=200, mutation=0.03)",
    "mutation_rate": "Mutation rate (population=300, generations=200)",
    "generations": "Generation budget (population=300, mutation=0.03)",
}
SWEEP_XLABEL = {
    "population_size": "Population size",
    "mutation_rate": "Mutation rate",
    "generations": "Generations",
}
# Thesis-default value for each sweep (GAConfig defaults / thesis_config()).
SWEEP_DEFAULT = {
    "population_size": 300.0,
    "mutation_rate": 0.03,
    "generations": 200.0,
}


def plot_sensitivity(rows, out_path, maze_size):
    by_sweep = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_sweep[r["sweep"]][r["value"]].append(r)

    sweep_order = [s for s in SWEEP_TITLES if s in by_sweep]
    fig, axes = plt.subplots(1, len(sweep_order), figsize=(16, 5), sharey=True)
    if len(sweep_order) == 1:
        axes = [axes]
    fig.suptitle(
        f"GA hyperparameters: diminishing (and non-monotonic) returns on a {maze_size}×{maze_size} maze",
        fontsize=15, fontweight="bold", color=INK_PRIMARY, x=0.01, ha="left",
    )
    fig.text(
        0.01, 0.925,
        "One-at-a-time sweeps — each panel varies one hyperparameter, others held at the thesis default (bold tick)",
        fontsize=10, color=INK_SECONDARY, ha="left",
    )

    for ax, sweep in zip(axes, sweep_order):
        _style_ax(ax)
        values = sorted(by_sweep[sweep], key=lambda v: float(v))
        success_rate = []
        mean_progress = []
        for v in values:
            trials = by_sweep[sweep][v]
            solved = [int(r["solved"]) for r in trials]
            success_rate.append(100.0 * sum(solved) / len(solved))
            mean_progress.append(100.0 * np.mean([float(r["final_progress_frac"]) for r in trials]))

        x = np.arange(len(values))
        ax.plot(x, success_rate, color=BLUE, linewidth=2, marker="o", markersize=8,
                markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
                label="Success rate")
        ax.plot(x, mean_progress, color=ORANGE, linewidth=2, marker="o", markersize=8,
                markerfacecolor=ORANGE, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
                label="Mean progress toward goal")
        ax.set_xticks(x)
        labels = [str(v) for v in values]
        ax.set_xticklabels(labels)
        default_val = SWEEP_DEFAULT.get(sweep)
        if default_val is not None:
            for tick, v in zip(ax.get_xticklabels(), values):
                if abs(float(v) - default_val) < 1e-9:
                    tick.set_fontweight("bold")
                    tick.set_color(INK_PRIMARY)
        ax.set_xlabel(SWEEP_XLABEL[sweep])
        ax.set_title(SWEEP_TITLES[sweep], fontsize=10.5, color=INK_PRIMARY, loc="left")
        ax.set_ylim(0, 108)

    axes[0].set_ylabel("%")
    handles = [
        Line2D([0], [0], color=BLUE, linewidth=2, marker="o", markersize=7, label="Success rate"),
        Line2D([0], [0], color=ORANGE, linewidth=2, marker="o", markersize=7, label="Mean progress toward goal"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=9)

    fig.tight_layout(rect=[0, 0.06, 1, 0.90])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    scaling_csv = "outputs/scaling/scaling.csv"
    sensitivity_csv = "outputs/sensitivity/sensitivity.csv"

    if os.path.exists(scaling_csv):
        rows = _read_csv(scaling_csv)
        plot_scaling(rows, "outputs/scaling/ga_scaling.png")
    else:
        print(f"skip: {scaling_csv} not found")

    if os.path.exists(sensitivity_csv):
        rows = _read_csv(sensitivity_csv)
        maze_size = rows[0].get("size", 21) if rows and "size" in rows[0] else 21
        plot_sensitivity(rows, "outputs/sensitivity/ga_sensitivity.png", maze_size=21)
    else:
        print(f"skip: {sensitivity_csv} not found")


if __name__ == "__main__":
    main()
