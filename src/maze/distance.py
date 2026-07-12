"""Breadth-first distance fields over a maze.

The BFS distance field from the goal is the backbone of the fitness function:
it turns "did the agent reach the goal?" (a sparse, hard-to-optimize signal)
into a smooth gradient "how many cells away from the goal did it get?".
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .maze import Maze, OPEN

UNREACHABLE = -1


def bfs_distance_field(maze: Maze, source) -> np.ndarray:
    """Return an int grid of shortest-path distances (in cells) from ``source``.

    Cells that are walls or unreachable from ``source`` hold ``UNREACHABLE``.
    """
    dist = np.full(maze.grid.shape, UNREACHABLE, dtype=np.int32)
    sr, sc = source
    if maze.grid[sr, sc] != OPEN:
        return dist

    dist[sr, sc] = 0
    queue = deque([(sr, sc)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (
                maze.in_bounds(nr, nc)
                and maze.grid[nr, nc] == OPEN
                and dist[nr, nc] == UNREACHABLE
            ):
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def optimal_path_length(maze: Maze) -> int:
    """Length of the shortest start-to-goal path, or ``UNREACHABLE`` (-1)."""
    dist = bfs_distance_field(maze, maze.goal)
    return int(dist[maze.start])
