"""Sensor inputs for network-controlled agents (NEAT).

The GA sees nothing and just replays a move sequence, so to make NEAT a genuine
*reactive policy* learner it is given a richer local observation. None of these
sensors reveal the global solution (no BFS field is exposed), so solving still
requires learning real navigation behavior.

Layout (14 values), in move order Up, Down, Left, Right:
    [0:4]   adjacent wall?            (1.0 / 0.0)   -- immediate obstacles
    [4:8]   corridor ray distance     (open cells until a wall, normalized)
    [8:10]  normalized goal offset    (dx = col->goal / width, dy = row / height)
    [10:14] previous action, one-hot  (minimal memory; all zeros on step 0)

A constant bias input (1.0) is appended by the controller, giving NUM_INPUTS.
"""
from __future__ import annotations

import numpy as np

from ..maze.maze import MOVES, Maze, Cell

NUM_SENSORS = 14
NUM_OUTPUTS = 4          # one score per move (U, D, L, R)
NUM_INPUTS = NUM_SENSORS + 1   # + bias
NUM_STATIC = 10          # position-only part: walls(4) + rays(4) + goal dx,dy


def precompute_static_sensors(maze: Maze) -> np.ndarray:
    """Precompute the position-only sensor part for every open cell.

    Walls, corridor rays and the goal offset depend only on the cell (not on
    time), so they are computed once per maze instead of every simulation step.
    Returns an array of shape ``(height, width, NUM_STATIC)``.
    """
    from ..maze.maze import OPEN

    h, w = maze.height, maze.width
    max_dim = max(w, h)
    static = np.zeros((h, w, NUM_STATIC), dtype=np.float64)
    gr, gc = maze.goal
    for r in range(h):
        for c in range(w):
            if maze.grid[r, c] != OPEN:
                continue
            for i, (dr, dc) in enumerate(MOVES):
                static[r, c, i] = 0.0 if maze.passable(r + dr, c + dc) else 1.0
                dist = 0
                rr, rc = r + dr, c + dc
                while maze.passable(rr, rc):
                    dist += 1
                    rr, rc = rr + dr, rc + dc
                static[r, c, 4 + i] = dist / max_dim
            static[r, c, 8] = (gc - c) / w
            static[r, c, 9] = (gr - r) / h
    return static


def build_sensor(static_row: np.ndarray, last_action: int) -> np.ndarray:
    """Combine the cached position part with the runtime previous-action one-hot."""
    vec = np.empty(NUM_SENSORS, dtype=np.float64)
    vec[:NUM_STATIC] = static_row
    vec[NUM_STATIC:] = 0.0
    if 0 <= last_action < 4:
        vec[NUM_STATIC + last_action] = 1.0
    return vec


def sense(maze: Maze, pos: Cell, last_action: int = -1) -> np.ndarray:
    r, c = pos
    max_dim = max(maze.width, maze.height)

    walls = np.zeros(4, dtype=np.float64)
    rays = np.zeros(4, dtype=np.float64)
    for i, (dr, dc) in enumerate(MOVES):
        nr, nc = r + dr, c + dc
        walls[i] = 0.0 if maze.passable(nr, nc) else 1.0
        # March along the direction counting open cells until blocked.
        dist = 0
        rr, rc = r + dr, c + dc
        while maze.passable(rr, rc):
            dist += 1
            rr, rc = rr + dr, rc + dc
        rays[i] = dist / max_dim

    gr, gc = maze.goal
    goal = np.array([(gc - c) / maze.width, (gr - r) / maze.height], dtype=np.float64)

    last = np.zeros(4, dtype=np.float64)
    if 0 <= last_action < 4:
        last[last_action] = 1.0

    return np.concatenate([walls, rays, goal, last])
