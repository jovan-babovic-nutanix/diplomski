"""Headless state machine driving the three-method comparison.

This module holds no pygame imports, so it can be constructed and driven
directly in tests without a display. ``MazeApp`` (see ``app.py``) owns pygame
initialization, rendering, and input, and delegates every round-advancing /
completion / metrics decision to a ``ComparisonSession``.

One "round" = one generation (GA), one training batch (Q-Learning) or one
planning step (A*). All three methods go through the same ``Solver`` interface
(see ``src/solver.py``); GA's population swarm is exposed for drawing via the
optional ``population_trajectories()``/``leader_trajectory()`` hooks on
``Solver``, so there is exactly one place (``EvolverSolver``) that steps a GA
generation - the interactive app no longer duplicates that logic.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import FitnessConfig, GAConfig, MazeConfig, QLearningConfig, SimulationConfig

from ..experiments.runner import build_ga, make_eval
from ..maze.distance import bfs_distance_field, optimal_path_length
from ..maze.generator import generate_maze
from ..maze.maze import Cell, Maze
from ..planning.astar import AStarSolver
from ..rl.qlearning import QLearningSolver
from ..solver import EvolverSolver, Solver

METHODS = ["GA", "Q-Learning", "A*"]


@dataclass
class PanelState:
    """Everything the renderer needs to draw one method's current round."""
    trajectories: List[List[Cell]] = field(default_factory=list)
    leader_traj: List[Cell] = field(default_factory=list)
    best_path: Optional[List[Cell]] = None
    info: Dict[str, object] = field(default_factory=dict)
    max_len: int = 1


class ComparisonSession:
    """Drives GA / Q-Learning / A* solvers through synchronized rounds."""

    def __init__(
        self,
        maze_cfg: MazeConfig,
        sim_cfg: SimulationConfig,
        fit_cfg: FitnessConfig,
        ga_cfg: GAConfig,
        ql_cfg: QLearningConfig,
    ):
        self.maze_cfg = maze_cfg
        self.sim_cfg = sim_cfg
        self.fit_cfg = fit_cfg
        self.ga_cfg = ga_cfg
        self.ql_cfg = ql_cfg

        self.seed = maze_cfg.seed
        self.ga_done = False
        self.qlearning_done = False
        self.astar_done = False
        self.finished = False
        self.global_max = 1
        self.states: Dict[str, PanelState] = {}
        self.solvers: Dict[str, Solver] = {}
        self._build_world(self.seed)
        self._reset_metrics()
        self._start_all_rounds()

    # -- world setup ----------------------------------------------------
    def _build_world(self, seed: int) -> None:
        self.maze: Maze = generate_maze(self.maze_cfg.width, self.maze_cfg.height, seed=seed)
        self.dist_field = bfs_distance_field(self.maze, self.maze.goal)
        self.optimal = optimal_path_length(self.maze)
        eval_fn = make_eval(self.maze, self.sim_cfg, self.fit_cfg, self.dist_field)

        ga = build_ga(self.ga_cfg, self.sim_cfg)
        self.solvers = {
            "GA": EvolverSolver(ga, eval_fn, self.ga_cfg.population_size),
            "Q-Learning": QLearningSolver(self.ql_cfg, self.maze, self.dist_field, self.fit_cfg),
            "A*": AStarSolver(self.maze),
        }

    def reset(self, seed: int) -> None:
        """Start a fresh maze/seed, clearing all round and completion state."""
        self.seed = seed
        self._build_world(seed)
        self.ga_done = False
        self.qlearning_done = False
        self.astar_done = False
        self.finished = False
        self._reset_metrics()
        self._start_all_rounds()

    # -- rounds -----------------------------------------------------------
    def _start_all_rounds(self) -> None:
        for method in METHODS:
            if method == "GA" and self.ga_done:
                continue
            if method == "Q-Learning" and self.qlearning_done:
                continue
            if method == "A*" and self.astar_done:
                continue
            self.states[method] = self._solver_round(method)
            if method == "GA":
                generation = int(self.states[method].info.get("iteration", 0))
                self.ga_done = bool(self.states[method].info.get("solved")) or (
                    generation + 1 >= self.ga_cfg.generations
                )
                self._record_completion("GA", self.states[method], self.ga_done)
            elif method == "Q-Learning":
                iteration = int(self.states[method].info.get("iteration", 0))
                # iteration is 0-based (matches GA), so round k is the
                # (k+1)-th batch.
                self.qlearning_done = bool(
                    self.states[method].info.get("solved")
                ) or iteration + 1 >= self._max_qlearning_rounds()
                self._record_completion(
                    "Q-Learning", self.states[method], self.qlearning_done
                )
            elif method == "A*":
                # A* is one-shot: it is done after its single planning run
                # whether or not a path exists. Deriving this from "solved"
                # would hang the session forever on an unreachable goal.
                self.astar_done = self.solvers["A*"].is_converged()
                self._record_completion("A*", self.states[method], self.astar_done)
        self.finished = self.ga_done and self.qlearning_done and self.astar_done
        self.global_max = max((s.max_len for s in self.states.values()), default=1)

    def _reset_metrics(self) -> None:
        now = time.perf_counter()
        self.method_started = {method: now for method in METHODS}
        self.metrics: Dict[str, Dict[str, object]] = {
            method: {"completed": False} for method in METHODS
        }

    def _record_completion(
        self, method: str, state: PanelState, completed: bool
    ) -> None:
        if not completed or self.metrics[method].get("completed"):
            return
        info = state.info
        solved = bool(info.get("solved"))
        path_length = info.get("best_steps")
        optimality = "-"
        if solved and path_length is not None and self.optimal > 0:
            optimality = f"{float(path_length) / self.optimal:.2f}x"
        elapsed = time.perf_counter() - self.method_started[method]
        self.metrics[method] = {
            "completed": True,
            "solved": solved,
            "iterations": info.get("iteration", "-"),
            "evaluations": info.get("evaluations", "-"),
            "path_length": path_length if path_length is not None else "-",
            "optimality": optimality,
            "time_s": f"{elapsed:.3f}s",
        }

    def _max_qlearning_rounds(self) -> int:
        """Maximum rounds allowed by the Q-Learning episode budget."""
        return max(
            1,
            math.ceil(self.ql_cfg.episodes / self.ql_cfg.episodes_per_step),
        )

    def advance_all_rounds(self) -> None:
        self._start_all_rounds()

    def advance_one_round(self) -> None:
        """Advance every still-active method by exactly one round."""
        if self.finished:
            return
        self.advance_all_rounds()

    def _solver_round(self, method: str) -> PanelState:
        solver = self.solvers[method]
        st = solver.step()
        path = solver.best_path() or [self.maze.start]

        trajectories = solver.population_trajectories()
        if not trajectories:
            trajectories = [path]
        leader = solver.leader_trajectory()
        if leader is None:
            leader = path

        state = PanelState(
            trajectories=trajectories,
            leader_traj=leader,
            best_path=path if st.reached else None,
            info={
                "iteration": st.iteration,
                "solved": st.reached,
                "best_steps": st.best_path_length,
                "evaluations": st.evaluations,
                "extra": dict(st.extra),
            },
        )
        state.max_len = max((len(t) for t in trajectories), default=1)
        return state
