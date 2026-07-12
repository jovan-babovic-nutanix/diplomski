"""Hand-designed mazes for deterministic regression-style testing.

Legend: '#' wall, '.' open, 'S' start, 'G' goal.
These are useful as fixed benchmarks alongside the procedurally generated ones.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .maze import Maze, OPEN, WALL

_EMPTY = [
    "S........",
    ".........",
    ".........",
    ".........",
    "........G",
]

_SIMPLE = [
    "S........",
    "########.",
    ".........",
    ".########",
    "........G",
]

_SPIRAL = [
    "S............",
    ".###########.",
    ".#.........#.",
    ".#.#######.#.",
    ".#.#.....#.#.",
    ".#.#.###.#.#.",
    ".#.#...#.#.#.",
    ".#.#####.#.#.",
    ".#.......#.#.",
    ".#########.#.",
    "...........#.",
    "###########..",
    "..........G..",
]

_RAW: Dict[str, List[str]] = {
    "empty": _EMPTY,
    "simple": _SIMPLE,
    "spiral": _SPIRAL,
}


def _from_strings(rows: List[str]) -> Maze:
    height = len(rows)
    width = len(rows[0])
    grid = np.full((height, width), OPEN, dtype=np.int8)
    start = goal = None
    for r, row in enumerate(rows):
        if len(row) != width:
            raise ValueError("All maze rows must have equal length")
        for c, ch in enumerate(row):
            if ch == "#":
                grid[r, c] = WALL
            elif ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
    if start is None or goal is None:
        raise ValueError("Fixed maze must define both 'S' and 'G'")
    return Maze(grid, start, goal)


FIXED_MAZES: Dict[str, Maze] = {name: _from_strings(rows) for name, rows in _RAW.items()}


def get_fixed_maze(name: str) -> Maze:
    if name not in FIXED_MAZES:
        raise KeyError(f"Unknown fixed maze {name!r}; options: {sorted(FIXED_MAZES)}")
    return FIXED_MAZES[name].copy()
