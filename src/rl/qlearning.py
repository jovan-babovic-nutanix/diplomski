"""Tabular Q-Learning - the reinforcement-learning baseline.

Unlike GA (which searches over whole policies), Q-Learning learns a value
``Q[state, action]`` by trial and error, one transition at a time:

    Q[s,a] <- Q[s,a] + alpha * (reward + gamma * max_a' Q[s',a'] - Q[s,a])

State = grid cell ``(row, col)``, actions = the four moves. Exploration uses an
epsilon-greedy policy whose epsilon decays from ``epsilon_start`` to
``epsilon_end``. Optional potential-based reward shaping (using the BFS distance
field) speeds learning without changing the optimal policy.

The greedy rollout (always taking ``argmax`` Q) is reused both to measure
progress with the *shared* fitness function (so the convergence curve is
comparable to GA) and as the trajectory drawn by the visualizer.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from config import FitnessConfig, QLearningConfig
from ..environment.simulation import SimulationResult
from ..fitness import evaluate_fitness
from ..maze.maze import Cell, Maze, MOVES
from ..solver import Solver, StepStats


class GreedyQController:
    """Position-aware greedy controller (argmax over learned Q-values)."""

    def __init__(self, q: np.ndarray):
        self.q = q

    def action(self, pos: Cell) -> int:
        return int(np.argmax(self.q[pos[0], pos[1]]))


def train_episode(
    q: np.ndarray,
    maze: Maze,
    dist_field: np.ndarray,
    cfg: QLearningConfig,
    rng: np.random.Generator,
    epsilon: float,
) -> None:
    r, c = maze.start
    for _ in range(cfg.max_steps):
        if rng.random() < epsilon:
            a = int(rng.integers(0, 4))
        else:
            a = int(np.argmax(q[r, c]))

        dr, dc = MOVES[a]
        nr, nc = r + dr, c + dc
        if maze.passable(nr, nc):
            ns = (nr, nc)
        else:
            ns = (r, c)  # bumped a wall: stay in place

        if ns == maze.goal:
            reward = cfg.goal_reward
        else:
            reward = cfg.step_penalty
            if cfg.reward_shaping:
                reward += int(dist_field[r, c]) - int(dist_field[ns[0], ns[1]])

        best_next = float(np.max(q[ns[0], ns[1]]))
        q[r, c, a] += cfg.alpha * (reward + cfg.gamma * best_next - q[r, c, a])

        r, c = ns
        if (r, c) == maze.goal:
            break


def greedy_rollout(
    q: np.ndarray, maze: Maze, dist_field: np.ndarray, max_steps: int
) -> SimulationResult:
    r, c = maze.start
    start_distance = int(dist_field[r, c])
    closest = start_distance
    trajectory: List[Cell] = [(r, c)]
    bumps = 0
    reached = False
    steps = 0

    for _ in range(max_steps):
        a = int(np.argmax(q[r, c]))
        dr, dc = MOVES[a]
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


class QLearningSolver(Solver):
    name = "Q-Learning"

    def __init__(
        self,
        cfg: QLearningConfig,
        maze: Maze,
        dist_field: np.ndarray,
        fit_cfg: FitnessConfig,
    ):
        self.cfg = cfg
        self.maze = maze
        self.dist_field = dist_field
        self.fit_cfg = fit_cfg
        self.q = np.zeros((maze.height, maze.width, 4), dtype=np.float64)
        self.rng = np.random.default_rng(cfg.seed)
        self._episode = 0
        self._solved = False
        self._best_path_len: Optional[int] = None
        self._best_fit = 0.0
        self._best_result: Optional[SimulationResult] = None
        self._last_stats: Optional[StepStats] = None

    def _epsilon(self) -> float:
        frac = min(1.0, self._episode / max(1, self.cfg.epsilon_decay_episodes))
        return self.cfg.epsilon_start + frac * (
            self.cfg.epsilon_end - self.cfg.epsilon_start
        )

    def step(self) -> StepStats:
        # Once the greedy policy has reached the goal, keep the successful
        # policy stable instead of continuing to train and changing the path
        # shown by the visualizer.
        if self._solved and self._last_stats is not None:
            return self._last_stats

        for _ in range(self.cfg.episodes_per_step):
            train_episode(
                self.q, self.maze, self.dist_field, self.cfg, self.rng, self._epsilon()
            )
            self._episode += 1

        result = greedy_rollout(self.q, self.maze, self.dist_field, self.cfg.max_steps)
        fitness = evaluate_fitness(result, self.fit_cfg)
        if fitness > self._best_fit or self._best_result is None:
            self._best_fit = max(self._best_fit, fitness)
            self._best_result = result
        if result.reached:
            self._solved = True
            if self._best_path_len is None or result.steps < self._best_path_len:
                self._best_path_len = result.steps

        coverage = float(np.count_nonzero(np.any(self.q != 0.0, axis=2)))
        self._last_stats = StepStats(
            iteration=self._episode // self.cfg.episodes_per_step,
            reached=self._solved,
            best_path_length=self._best_path_len,
            best_fitness=self._best_fit,
            evaluations=self._episode,
            extra={"epsilon": self._epsilon(), "q_coverage": coverage},
        )
        return self._last_stats

    def best_path(self) -> Optional[List[Cell]]:
        if self._best_result is not None:
            return self._best_result.trajectory
        return greedy_rollout(
            self.q, self.maze, self.dist_field, self.cfg.max_steps
        ).trajectory

    def is_converged(self) -> bool:
        """Stop training once the greedy policy has solved the maze."""
        return self._solved
