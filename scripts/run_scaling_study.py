#!/usr/bin/env python3
"""Maze-size scaling study for the GA (open-loop encoding).

Answers the question flagged in CLAUDE.md's "ideas discussed but not yet
implemented": how does GA's success rate / convergence speed scale with maze
size? Holds the GA's computational budget fixed (thesis config: population
300, generations 200, max_steps 300) and only varies maze width/height, so
any drop in success rate is attributable to the search problem getting
harder, not to a bigger budget.

Usage:
    python scripts/run_scaling_study.py [--sizes 11,15,19,21,25,31] [--seeds 8]
        [--output outputs/scaling]

Writes one row per (size, seed) trial to <output>/scaling.csv with columns:
    size, seed, optimal_path_length, solved, generations_to_solve,
    evaluations_to_solve, best_path_length, final_progress, final_progress_frac,
    wall_time_s
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


def main() -> None:
    parser = argparse.ArgumentParser(description="GA maze-size scaling study")
    parser.add_argument("--sizes", type=str, default="11,15,19,21,25,31")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--output", type=str, default="outputs/scaling")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    cfg = thesis_config()  # population=300, generations=200, max_steps=300 fixed

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "scaling.csv")

    rows = []
    start_all = time.time()
    for size in sizes:
        cfg.maze.width = size
        cfg.maze.height = size
        for i in range(args.seeds):
            seed = args.base_seed + i
            t0 = time.time()
            maze = generate_maze(size, size, seed=seed)
            dist_field = bfs_distance_field(maze, maze.goal)
            optimal = optimal_path_length(maze)

            solver, max_iters = build_solver("GA", maze, dist_field, cfg, seed)
            history, iters, evals, wall = run_solver(solver, max_iters)

            best = solver.evolver.best()
            final_progress = best.result.progress if best.result else 0
            solved = any(st.reached for st in history)
            best_len = None
            if solved:
                path = solver.best_path()
                best_len = (len(path) - 1) if path else None

            frac = (final_progress / optimal) if optimal > 0 else float("nan")
            row = dict(
                size=size,
                seed=seed,
                optimal_path_length=optimal,
                solved=int(solved),
                generations_to_solve=iters if iters is not None else "",
                evaluations_to_solve=evals if evals is not None else "",
                best_path_length=best_len if best_len is not None else "",
                final_progress=final_progress,
                final_progress_frac=f"{frac:.4f}",
                wall_time_s=f"{wall:.3f}",
            )
            rows.append(row)
            print(
                f"[{time.time()-start_all:7.1f}s] size={size:3d} seed={seed} "
                f"optimal={optimal:4d} solved={solved!s:5} "
                f"final_progress={final_progress:4d}/{optimal:4d} ({frac*100:5.1f}%) "
                f"gen_to_solve={iters} wall={wall:.2f}s ({time.time()-t0:.2f}s)",
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
