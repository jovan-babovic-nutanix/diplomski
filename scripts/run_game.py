#!/usr/bin/env python3
"""Launch the interactive maze comparison visualizer.

Shows all three methods (GA, Q-Learning, A*) side by side,
running in lock-step on the same maze.

Usage:
    python scripts/run_game.py [--preset demo|thesis] [--width 15] [--height 15]
                               [--seed 42] [--max-steps 200] [--fps 60]
                               [--population N] [--generations N] [--episodes N]
                               [--episodes-per-step N]

Any explicit flag overrides the chosen preset's value for that setting.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_preset, seed_everything  # noqa: E402
from src.visualization.app import MazeApp  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Maze Evolution visualizer")
    parser.add_argument("--preset", choices=("demo", "thesis"), default="demo")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--episodes-per-step", type=int, default=None)
    args = parser.parse_args()

    cfg = get_preset(args.preset)
    if args.width is not None:
        cfg.maze.width = args.width
    if args.height is not None:
        cfg.maze.height = args.height
    if args.seed is not None:
        cfg.maze.seed = args.seed
    if args.max_steps is not None:
        cfg.simulation.max_steps = args.max_steps
        cfg.qlearning.max_steps = args.max_steps
    if args.population is not None:
        cfg.ga.population_size = args.population
    if args.generations is not None:
        cfg.ga.generations = args.generations
    if args.episodes is not None:
        cfg.qlearning.episodes = args.episodes
    if args.episodes_per_step is not None:
        cfg.qlearning.episodes_per_step = args.episodes_per_step

    seed_everything(cfg.maze.seed)
    app = MazeApp(
        maze_cfg=cfg.maze,
        sim_cfg=cfg.simulation,
        fit_cfg=cfg.fitness,
        ga_cfg=cfg.ga,
        ql_cfg=cfg.qlearning,
        fps=args.fps,
    )
    app.run()


if __name__ == "__main__":
    main()
