"""Procedural maze generation using the recursive-backtracker algorithm.

The generator is fully deterministic given a seed, which is important for
reproducible thesis experiments. Walls occupy even rows/columns; passages are
carved between odd-indexed cells, producing a "perfect" maze (exactly one path
between any two cells).
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np

from .maze import Maze, OPEN, WALL


def _force_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def generate_maze(width: int, height: int, seed: Optional[int] = None) -> Maze:
    """Generate a perfect maze of (approximately) the requested size.

    Args:
        width: Desired grid width (forced odd, minimum 5).
        height: Desired grid height (forced odd, minimum 5).
        seed: RNG seed for reproducibility.
    """
    width = max(5, _force_odd(width))
    height = max(5, _force_odd(height))
    rng = random.Random(seed)

    grid = np.full((height, width), WALL, dtype=np.int8)

    start_cell = (1, 1)
    grid[start_cell] = OPEN
    stack = [start_cell]

    while stack:
        r, c = stack[-1]
        unvisited = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr, nc = r + dr, c + dc
            if 1 <= nr < height - 1 and 1 <= nc < width - 1 and grid[nr, nc] == WALL:
                unvisited.append((nr, nc, dr, dc))

        if unvisited:
            nr, nc, dr, dc = rng.choice(unvisited)
            # Carve the wall between current cell and the chosen neighbor.
            grid[r + dr // 2, c + dc // 2] = OPEN
            grid[nr, nc] = OPEN
            stack.append((nr, nc))
        else:
            stack.pop()

    start = (1, 1)
    goal = (height - 2, width - 2)
    return Maze(grid, start, goal)
