#!/usr/bin/env python3
"""GA hyperparameter sensitivity study, on a fixed "hard" maze size.

Answers the other flagged CLAUDE.md item: "Sensitivity analysis on key
hyperparameters (... GA mutation_rate)". Holds maze size at the thesis
default (21x21) and sweeps one GA hyperparameter at a time (others held at
the thesis default) to show how population size, mutation rate, and the
generation budget each affect success rate on a maze that's already hard for
the open-loop encoding.

Usage:
    python scripts/run_sensitivity_study.py [--seeds 6] [--size 21]
        [--output outputs/sensitivity]

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
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import thesis_config  # noqa: E402
from src.maze.distance import bfs_distance_field, optimal_path_length  # noqa: E402
from src.maze.generator import generate_maze  # noqa: E402
from src.experiments.runner import build_solver, run_solver  # noqa: E402

# One-at-a-time sweeps; every other GA hyperparameter stays at the thesis
# default (population_size=300, generations=200, mutation_rate=0.03).
SWEEPS = {
    "population_size": [100, 200, 300, 450, 600],
    "mutation_rate": [0.01, 0.03, 0.06, 0.10, 0.20],
    "generations": [50, 100, 200, 300, 400],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="GA hyperparameter sensitivity study")
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--size", type=int, default=21)
    parser.add_argument("--output", type=str, default="outputs/sensitivity")
    args = parser.parse_args()

    base_cfg = thesis_config()
    base_cfg.maze.width = args.size
    base_cfg.maze.height = args.size

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "sensitivity.csv")

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
    for sweep_name, values in SWEEPS.items():
        for value in values:
            for i in range(args.seeds):
                seed = args.base_seed + i
                maze, dist_field, optimal = mazes[seed]

                cfg = thesis_config()
                cfg.maze.width = args.size
                cfg.maze.height = args.size
                if sweep_name == "generations":
                    cfg.ga.generations = value
                else:
                    setattr(cfg.ga, sweep_name, value)

                t0 = time.time()
                solver, max_iters = build_solver("GA", maze, dist_field, cfg, seed)
                history, iters, evals, wall = run_solver(solver, max_iters)

                best = solver.evolver.best()
                final_progress = best.result.progress if best.result else 0
                solved = any(st.reached for st in history)
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
                    f"[{time.time()-start_all:7.1f}s] sweep={sweep_name:15s} value={value!s:6s} "
                    f"seed={seed} solved={solved!s:5} progress={frac*100:5.1f}% "
                    f"gen_to_solve={iters} ({time.time()-t0:.2f}s)",
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
