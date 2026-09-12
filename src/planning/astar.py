"""A* pathfinding on the grid - the reference (optimal) method.

A* is the classic informed search: from the start it expands the cell that
minimises ``g + h`` where ``g`` is the cost so far and ``h`` is the Manhattan
distance to the goal (an admissible heuristic on a 4-connected grid). Because
``h`` never overestimates, A* is guaranteed to return a shortest path. In the
thesis it serves as the "correct answer" the learning methods are compared to.
"""
from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

from ..maze.maze import Cell, Maze, MOVES
from ..solver import Solver, StepStats


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(maze: Maze) -> Tuple[Optional[List[Cell]], int]:
    """Return ``(path, nodes_expanded)``; ``path`` is None if the goal is
    unreachable. ``path`` includes both the start and goal cells."""
    start, goal = maze.start, maze.goal
    open_heap: List[Tuple[int, int, Cell]] = [(manhattan(start, goal), 0, start)]
    came_from: Dict[Cell, Optional[Cell]] = {start: None}
    g_score: Dict[Cell, int] = {start: 0}
    closed: set = set()
    nodes_expanded = 0

    while open_heap:
        _f, g, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        nodes_expanded += 1

        if current == goal:
            return _reconstruct(came_from, goal), nodes_expanded

        r, c = current
        for dr, dc in MOVES:
            neighbor = (r + dr, c + dc)
            if not maze.passable(*neighbor):
                continue
            tentative = g + 1
            if tentative < g_score.get(neighbor, 1 << 30):
                g_score[neighbor] = tentative
                came_from[neighbor] = current
                heapq.heappush(
                    open_heap,
                    (tentative + manhattan(neighbor, goal), tentative, neighbor),
                )

    return None, nodes_expanded


def _reconstruct(came_from: Dict[Cell, Optional[Cell]], goal: Cell) -> List[Cell]:
    path = [goal]
    node: Optional[Cell] = goal
    while came_from[node] is not None:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path


class AStarSolver(Solver):
    """Wraps A* as a single-shot ``Solver`` (converges after one step)."""

    name = "A*"

    def __init__(self, maze: Maze):
        self.maze = maze
        self._path: Optional[List[Cell]] = None
        self._nodes_expanded = 0
        self._done = False

    def step(self) -> StepStats:
        if not self._done:
            self._path, self._nodes_expanded = astar(self.maze)
            self._done = True
        reached = self._path is not None
        # Edge count of the optimal path - the denominator of optimality_ratio.
        # See environment/simulation.py for why GA/Q-Learning "steps" are not
        # exactly comparable in kind (they can include wasted wall-bump steps).
        path_len = (len(self._path) - 1) if self._path else None
        return StepStats(
            iteration=0,
            reached=reached,
            best_path_length=path_len,
            best_fitness=float(path_len or 0),
            evaluations=1,
            extra={"nodes_expanded": float(self._nodes_expanded)},
        )

    def best_path(self) -> Optional[List[Cell]]:
        return self._path

    def is_converged(self) -> bool:
        return self._done
