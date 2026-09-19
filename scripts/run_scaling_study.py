#!/usr/bin/env python3
"""Maze-size scaling study, for any of the three methods (GA / Q-Learning / A*).

Answers the question flagged in CLAUDE.md's "ideas discussed but not yet
implemented": how does each method's success rate / convergence speed scale
with maze size? Holds the method's computational budget fixed (thesis
config) and only varies maze width/height, so any drop in success rate is
attributable to the search problem getting harder, not to a bigger budget.

Usage:
    python scripts/run_scaling_study.py [--method GA|Q-Learning|A*]
        [--sizes 11,15,19,21,25,31] [--seeds 8] [--output outputs/scaling]

Writes one row per (size, seed) trial to <output>/scaling.csv with columns:
    size, seed, optimal_path_length, solved, generations_to_solve,
    evaluations_to_solve, best_path_length, final_progress, final_progress_frac,
    nodes_expanded, wall_time_s

``generations_to_solve`` is the solver's native iteration index (generation
for GA, greedy-rollout round for Q-Learning, always 0 for A*) — see
``StepStats.iteration`` in src/solver.py. ``nodes_expanded`` is populated only
for A* (from ``StepStats.extra``), blank otherwise.
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

DEFAULT_OUTPUT = {
    "GA": "outputs/scaling",
    "Q-Learning": "outputs/scaling_ql",
    "A*": "outputs/scaling_astar",
}


def _final_progress(method: str, solver, optimal: int, solved: bool) -> int:
    """Cells of net progress toward the goal at the end of the run, for any method."""
    if method == "GA":
        best = solver.evolver.best()
        return best.result.progress if best.result else 0
    if method == "Q-Learning":
        best_result = getattr(solver, "_best_result", None)
        return best_result.progress if best_result is not None else 0
    if method == "A*":
        # A* either finds the (optimal) path or the maze is disconnected; the
        # generator guarantees connectivity, so a run either fully solves
        # (progress == optimal) or, in principle, makes none.
        return optimal if solved else 0
    raise ValueError(f"Unknown method {method!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Maze-size scaling study")
    parser.add_argument("--method", type=str, default="GA", choices=["GA", "Q-Learning", "A*"])
    parser.add_argument("--sizes", type=str, default="11,15,19,21,25,31")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    method = args.method
    output = args.output or DEFAULT_OUTPUT[method]
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    cfg = thesis_config()  # population=300/generations=200 (GA), episodes=4000 (QL), fixed

    os.makedirs(output, exist_ok=True)
    out_path = os.path.join(output, "scaling.csv")

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

            solver, max_iters = build_solver(method, maze, dist_field, cfg, seed)
            history, iters, evals, wall = run_solver(solver, max_iters)

            solved = any(st.reached for st in history)
            final_progress = _final_progress(method, solver, optimal, solved)
            best_len = None
            nodes_expanded = ""
            if solved:
                path = solver.best_path()
                best_len = (len(path) - 1) if path else None
            if method == "A*" and history:
                nodes_expanded = int(history[-1].extra.get("nodes_expanded", 0))

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
                nodes_expanded=nodes_expanded,
                wall_time_s=f"{wall:.3f}",
            )
            rows.append(row)
            print(
                f"[{time.time()-start_all:7.1f}s] method={method:10s} size={size:3d} seed={seed} "
                f"optimal={optimal:4d} solved={solved!s:5} "
                f"final_progress={final_progress:4d}/{optimal:4d} ({frac*100:5.1f}%) "
                f"iters_to_solve={iters} wall={wall:.2f}s ({time.time()-t0:.2f}s)",
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
