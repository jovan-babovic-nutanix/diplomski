"""Runs a single agent (controller) through a maze and records what happened.

This module is shared by both evolutionary engines, which is what makes the
GA-vs-NEAT comparison fair: identical environment, identical episode dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol

import numpy as np

import numpy as np

from ..maze.distance import bfs_distance_field
from ..maze.maze import Cell, Maze, MOVES
from .sensors import NUM_SENSORS, build_sensor, precompute_static_sensors

_ZERO_SENSORS = np.zeros(NUM_SENSORS, dtype=np.float64)


class Controller(Protocol):
    """Anything that can pick an action each step.

    ``step`` is the time index (used by the GA's fixed move sequence) and
    ``sensors`` is the local observation (used by NEAT networks).
    """

    def act(self, step: int, sensors: np.ndarray) -> int:
        ...


@dataclass
class SimulationResult:
    trajectory: List[Cell]
    reached: bool
    steps: int
    closest_distance: int      # smallest BFS-distance-to-goal reached
    start_distance: int        # BFS distance of the start cell (optimal length)
    bumps: int

    @property
    def progress(self) -> int:
        """Cells of net progress toward the goal (0 = no progress)."""
        return max(0, self.start_distance - self.closest_distance)


def simulate(
    maze: Maze,
    controller: Controller,
    max_steps: int,
    dist_field: Optional[np.ndarray] = None,
    static_sensors: Optional[np.ndarray] = None,
) -> SimulationResult:
    if dist_field is None:
        dist_field = bfs_distance_field(maze, maze.goal)

    # Sensors are static per maze; precompute unless the controller ignores them.
    needs_sensors = getattr(controller, "uses_sensors", True)
    if needs_sensors and static_sensors is None:
        static_sensors = precompute_static_sensors(maze)

    r, c = maze.start
    start_distance = int(dist_field[r, c])
    closest = start_distance
    trajectory: List[Cell] = [(r, c)]
    bumps = 0
    reached = False
    steps = 0
    last_action = -1

    for step in range(max_steps):
        if needs_sensors:
            sensors = build_sensor(static_sensors[r, c], last_action)
        else:
            sensors = _ZERO_SENSORS
        action = controller.act(step, sensors)
        last_action = action
        dr, dc = MOVES[action]
        nr, nc = r + dr, c + dc
        if maze.passable(nr, nc):
            r, c = nr, nc
        else:
            bumps += 1
        trajectory.append((r, c))
        steps += 1

        d = int(dist_field[r, c])
        if d >= 0 and d < closest:
            closest = d
        if (r, c) == maze.goal:
            reached = True
            break

    return SimulationResult(
        trajectory=trajectory,
        reached=reached,
        steps=steps,
        closest_distance=closest,
        start_distance=start_distance,
        bumps=bumps,
    )
