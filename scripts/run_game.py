#!/usr/bin/env python3
"""Launch the interactive maze-evolution visualizer.

Usage:
    python scripts/run_game.py [--width 21] [--height 21] [--seed 42]
                               [--max-steps 300] [--fps 30]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MazeConfig, SimulationConfig, seed_everything  # noqa: E402
from src.visualization.app import MazeApp  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Maze Evolution visualizer")
    parser.add_argument("--width", type=int, default=15)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()

    seed_everything(args.seed)
    app = MazeApp(
        maze_cfg=MazeConfig(width=args.width, height=args.height, seed=args.seed),
        sim_cfg=SimulationConfig(max_steps=args.max_steps),
        fps=args.fps,
    )
    app.run()


if __name__ == "__main__":
    main()
