"""Grid maze representation.

A maze is a 2D grid where each cell is either OPEN (walkable) or WALL.
Coordinates are ``(row, col)`` with the origin at the top-left.
"""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np

WALL = 1
OPEN = 0

Cell = Tuple[int, int]

# Cardinal moves in clockwise order in (row, col) deltas:
# Up, Right, Down, Left.
MOVES: Tuple[Cell, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
MOVE_NAMES = ("U", "R", "D", "L")


class Maze:
    """A grid maze with a fixed start and goal cell."""

    def __init__(self, grid: np.ndarray, start: Cell, goal: Cell):
        self.grid = grid.astype(np.int8)
        self.height, self.width = self.grid.shape
        self.start = start
        self.goal = goal

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def is_wall(self, row: int, col: int) -> bool:
        return self.grid[row, col] == WALL

    def passable(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and self.grid[row, col] == OPEN

    def neighbors(self, row: int, col: int) -> Iterator[Cell]:
        for dr, dc in MOVES:
            nr, nc = row + dr, col + dc
            if self.passable(nr, nc):
                yield nr, nc

    def open_cells(self) -> int:
        return int(np.count_nonzero(self.grid == OPEN))

    def copy(self) -> "Maze":
        return Maze(self.grid.copy(), self.start, self.goal)

    def __repr__(self) -> str:
        return (
            f"Maze({self.height}x{self.width}, start={self.start}, "
            f"goal={self.goal}, open={self.open_cells()})"
        )

    def to_ascii(self) -> str:
        chars = []
        for r in range(self.height):
            row = []
            for c in range(self.width):
                if (r, c) == self.start:
                    row.append("S")
                elif (r, c) == self.goal:
                    row.append("G")
                elif self.grid[r, c] == WALL:
                    row.append("#")
                else:
                    row.append(".")
            chars.append("".join(row))
        return "\n".join(chars)
