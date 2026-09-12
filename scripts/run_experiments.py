#!/usr/bin/env python3
"""Run the headless multi-method comparison study and produce CSVs + plots.

Compares A*, GA and Q-Learning on the same mazes.

Usage:
    python scripts/run_experiments.py [--preset demo|thesis] [--seeds 5]
        [--generations 80] [--episodes 4000] [--width 21] [--height 21]
        [--ga-pop 200] [--max-steps 300]
        [--methods A*,GA,Q-Learning] [--output outputs]

Any explicit flag overrides the chosen preset's value for that setting.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_preset, seed_everything  # noqa: E402
from src.experiments.metrics import write_history_csv, write_summary_csv  # noqa: E402
from src.experiments.plots import generate_all_plots  # noqa: E402
from src.experiments.runner import DEFAULT_METHODS, run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="A* vs GA vs Q-Learning maze study")
    parser.add_argument("--preset", choices=("demo", "thesis"), default="thesis")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--ga-pop", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    seed_everything(0)
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())

    cfg = get_preset(args.preset)
    if args.seeds is not None:
        cfg.n_seeds = args.seeds
    if args.output is not None:
        cfg.output_dir = args.output
    if args.width is not None:
        cfg.maze.width = args.width
    if args.height is not None:
        cfg.maze.height = args.height
    if args.max_steps is not None:
        cfg.simulation.max_steps = args.max_steps
        cfg.qlearning.max_steps = args.max_steps
    if args.ga_pop is not None:
        cfg.ga.population_size = args.ga_pop
    if args.generations is not None:
        cfg.ga.generations = args.generations
    if args.episodes is not None:
        cfg.qlearning.episodes = args.episodes

    print(
        f"preset={args.preset}  seeds={cfg.n_seeds}  maze={cfg.maze.width}x{cfg.maze.height}  "
        f"GA={cfg.ga.population_size}x{cfg.ga.generations}  QL={cfg.qlearning.episodes}ep"
    )
    print(f"Running study: {cfg.n_seeds} seeds, methods = {', '.join(methods)}")
    start = time.time()
    trials = run_experiment(cfg, methods=methods, progress=lambda m: print("  " + m, flush=True))
    elapsed = time.time() - start

    os.makedirs(cfg.output_dir, exist_ok=True)
    history_csv = os.path.join(cfg.output_dir, "history.csv")
    summary_csv = os.path.join(cfg.output_dir, "summary.csv")
    write_history_csv(trials, history_csv)
    write_summary_csv(trials, summary_csv)
    plot_paths = generate_all_plots(trials, cfg.output_dir)

    _print_summary(trials)
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  CSV   : {history_csv}, {summary_csv}")
    for p in plot_paths:
        print(f"  plot  : {p}")


def _print_summary(trials) -> None:
    grouped = defaultdict(list)
    for t in trials:
        grouped[t.algorithm].append(t)

    print("\n=== Summary ===")
    print(f"{'method':>11}  {'success':>8}  {'avg iters':>10}  {'path/opt':>9}  {'time(s)':>9}")
    for algo in sorted(grouped):
        ts = grouped[algo]
        solved = [t for t in ts if t.solved]
        success = 100.0 * len(solved) / len(ts)
        iters = [t.iterations_to_solve for t in solved if t.iterations_to_solve is not None]
        avg_iters = sum(iters) / len(iters) if iters else float("nan")
        ratios = [t.optimality_ratio for t in solved if t.optimality_ratio]
        avg_ratio = sum(ratios) / len(ratios) if ratios else float("nan")
        avg_time = sum(t.wall_time_s for t in ts) / len(ts)
        print(
            f"{algo:>11}  {success:7.1f}%  {avg_iters:10.2f}  {avg_ratio:9.2f}  {avg_time:9.3f}"
        )


if __name__ == "__main__":
    main()
