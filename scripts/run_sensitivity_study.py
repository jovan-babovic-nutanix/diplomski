#!/usr/bin/env python3
"""Hyperparameter sensitivity study, for GA or Q-Learning, on a fixed "hard" maze size.

Answers the other flagged CLAUDE.md item: "Sensitivity analysis on key
hyperparameters (Q-Learning alpha/gamma, GA mutation_rate)". Holds maze size
at the thesis default (21x21) and sweeps one hyperparameter at a time
(others held at the thesis default) to show how each affects success rate on
a maze that's already hard for the method in question. A* has no tunable
hyperparameters in the current implementation (the heuristic is fixed to
Manhattan distance), so it has no sweep here.

Usage:
    python scripts/run_sensitivity_study.py [--method GA|Q-Learning] [--seeds 6]
        [--size 21] [--output outputs/sensitivity]

Writes <output>/sensitivity.csv with columns:
    sweep, value, seed, optimal_path_length, solved, generations_to_solve,
    evaluations_to_solve, final_progress, final_progress_frac, wall_time_s
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import thesis_config  # noqa: E402
from src.maze.distance import bfs_distance_field, optimal_path_length  # noqa: E402
from src.maze.generator import generate_maze  # noqa: E402
from src.experiments.runner import build_solver, run_solver  # noqa: E402

# One-at-a-time sweeps; every other hyperparameter stays at the thesis default.
GA_SWEEPS = {
    "population_size": [100, 200, 300, 450, 600],
    "mutation_rate": [0.01, 0.03, 0.06, 0.10, 0.20],
    "generations": [50, 100, 200, 300, 400],
}
# QLearningConfig thesis defaults: alpha=0.2, gamma=0.95, episodes=4000.
QL_SWEEPS = {
    "alpha": [0.05, 0.1, 0.2, 0.4, 0.8],
    "gamma": [0.7, 0.8, 0.9, 0.95, 0.99],
    "episodes": [1000, 2000, 4000, 6000, 8000],
}
SWEEPS_BY_METHOD = {"GA": GA_SWEEPS, "Q-Learning": QL_SWEEPS}
DEFAULT_OUTPUT = {"GA": "outputs/sensitivity", "Q-Learning": "outputs/sensitivity_ql"}


def _final_progress(method: str, solver, optimal: int) -> int:
    if method == "GA":
        best = solver.evolver.best()
        return best.result.progress if best.result else 0
    best_result = getattr(solver, "_best_result", None)
    return best_result.progress if best_result is not None else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter sensitivity study")
    parser.add_argument("--method", type=str, default="GA", choices=["GA", "Q-Learning"])
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--size", type=int, default=21)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    method = args.method
    output = args.output or DEFAULT_OUTPUT[method]
    sweeps = SWEEPS_BY_METHOD[method]

    os.makedirs(output, exist_ok=True)
    out_path = os.path.join(output, "sensitivity.csv")

    # Pre-generate the mazes once so every sweep/value combination for a given
    # seed is evaluated on exactly the same maze (fair comparison).
    mazes = {}
    for i in range(args.seeds):
        seed = args.base_seed + i
        maze = generate_maze(args.size, args.size, seed=seed)
        dist_field = bfs_distance_field(maze, maze.goal)
        optimal = optimal_path_length(maze)
        mazes[seed] = (maze, dist_field, optimal)

    rows = []
    start_all = time.time()
    for sweep_name, values in sweeps.items():
        for value in values:
            for i in range(args.seeds):
                seed = args.base_seed + i
                maze, dist_field, optimal = mazes[seed]

                cfg = thesis_config()
                cfg.maze.width = args.size
                cfg.maze.height = args.size
                target_cfg = cfg.ga if method == "GA" else cfg.qlearning
                setattr(target_cfg, sweep_name, value)

                t0 = time.time()
                solver, max_iters = build_solver(method, maze, dist_field, cfg, seed)
                history, iters, evals, wall = run_solver(solver, max_iters)

                solved = any(st.reached for st in history)
                final_progress = _final_progress(method, solver, optimal)
                frac = (final_progress / optimal) if optimal > 0 else float("nan")

                row = dict(
                    sweep=sweep_name,
                    value=value,
                    seed=seed,
                    optimal_path_length=optimal,
                    solved=int(solved),
                    generations_to_solve=iters if iters is not None else "",
                    evaluations_to_solve=evals if evals is not None else "",
                    final_progress=final_progress,
                    final_progress_frac=f"{frac:.4f}",
                    wall_time_s=f"{wall:.3f}",
                )
                rows.append(row)
                print(
                    f"[{time.time()-start_all:7.1f}s] method={method:10s} sweep={sweep_name:15s} value={value!s:6s} "
                    f"seed={seed} solved={solved!s:5} progress={frac*100:5.1f}% "
                    f"iters_to_solve={iters} ({time.time()-t0:.2f}s)",
                    flush=True,
                )

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_path} in {time.time()-start_all:.1f}s total")


if __name__ == "__main__":
    main()
