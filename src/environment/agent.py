"""The agent ("formula") that moves through a maze one cell at a time."""
from __future__ import annotations

from ..maze.maze import MOVES, Cell


class Agent:
    """Minimal grid agent. The brain (GA genome) is external;
    this class only tracks position and applies validated moves."""

    def __init__(self, start: Cell):
        self.start = start
        self.row, self.col = start

    @property
    def pos(self) -> Cell:
        return self.row, self.col

    def reset(self) -> None:
        self.row, self.col = self.start

    def try_move(self, action: int, maze) -> bool:
        """Attempt a move. Returns True if it moved, False if it bumped a wall."""
        dr, dc = MOVES[action]
        nr, nc = self.row + dr, self.col + dc
        if maze.passable(nr, nc):
            self.row, self.col = nr, nc
            return True
        return False
