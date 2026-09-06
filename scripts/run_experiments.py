#!/usr/bin/env python3
"""Run the headless multi-method comparison study and produce CSVs + plots.

Compares A*, GA and Q-Learning on the same mazes.

Usage:
    python scripts/run_experiments.py [--seeds 5] [--generations 80]
        [--episodes 4000] [--width 21] [--height 21]
        [--ga-pop 200] [--max-steps 300]
        [--methods A*,GA,Q-Learning] [--output outputs]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (  # noqa: E402
    ExperimentConfig,
    GAConfig,
    MazeConfig,
    QLearningConfig,
    SimulationConfig,
    seed_everything,
)
from src.experiments.metrics import write_history_csv, write_summary_csv  # noqa: E402
from src.experiments.plots import generate_all_plots  # noqa: E402
from src.experiments.runner import DEFAULT_METHODS, run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="A* vs GA vs Q-Learning maze study")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--width", type=int, default=21)
    parser.add_argument("--height", type=int, default=21)
    parser.add_argument("--ga-pop", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--output", type=str, default="outputs")
    args = parser.parse_args()

    seed_everything(0)
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())

    cfg = ExperimentConfig(
        n_seeds=args.seeds,
        output_dir=args.output,
        maze=MazeConfig(width=args.width, height=args.height),
        simulation=SimulationConfig(max_steps=args.max_steps),
        ga=GAConfig(population_size=args.ga_pop, generations=args.generations),
        qlearning=QLearningConfig(episodes=args.episodes, max_steps=args.max_steps),
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
