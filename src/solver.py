"""Common ``Solver`` abstraction unifying the maze-navigation methods.

The original project was built around the population-based GA ``Evolver``.
A* (a one-shot planner) and Q-Learning (episodic reinforcement learning) do not
fit that mould, so this thin layer lets the experiment runner and the visualizer
treat all three methods identically:

    stats = solver.step()      # do one unit of work, return a progress snapshot
    path  = solver.best_path() # best start->goal trajectory found so far

A "unit of work" (one ``step``) is method-specific:
    * GA         -> one generation
    * Q-Learning -> a batch of training episodes + a greedy evaluation
    * A*         -> the single planning run (then ``is_converged`` is True)

``StepStats.evaluations`` is the cumulative number of environment episodes the
method has consumed. It is the *fair common cost axis* for convergence plots,
since the methods otherwise count progress in different native units.

Accounting convention: every full environment episode counts once.
    * GA         -> ``population_size`` per generation.
    * Q-Learning -> the ``episodes_per_step`` training episodes *plus* the one
      greedy-rollout episode run each ``step()`` to measure progress.
    * A*         -> 1 (the single planning run - not an episode, but the one
      unit of work A* ever does).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .evolution.base import Evolver, EvalFn
from .maze.maze import Cell


@dataclass
class StepStats:
    iteration: int                       # native unit index (generation/step)
    reached: bool                        # has the goal been reached by now (sticky)
    best_path_length: Optional[int]      # steps of best successful path, else None
    best_fitness: float                  # shared progress metric (monotonic)
    evaluations: int                     # cumulative environment episodes used
    extra: Dict[str, float] = field(default_factory=dict)


class Solver(ABC):
    name: str

    @abstractmethod
    def step(self) -> StepStats:
        """Perform one unit of work and return a progress snapshot."""

    @abstractmethod
    def best_path(self) -> Optional[List[Cell]]:
        """Best start->goal trajectory found so far (for drawing)."""

    def is_converged(self) -> bool:
        """If True, the runner may stop early (used by A*)."""
        return False


class EvolverSolver(Solver):
    """Adapts the GA ``Evolver`` to the ``Solver`` interface."""

    def __init__(self, evolver: Evolver, eval_fn: EvalFn, population_size: int):
        self.evolver = evolver
        self.eval_fn = eval_fn
        self.population_size = population_size
        self.name = evolver.name
        self._evaluations = 0
        self._solved = False
        self._best_path_len: Optional[int] = None

    def step(self) -> StepStats:
        ev = self.evolver
        ev.evaluate(self.eval_fn)
        gen_stats = ev.stats()
        self._evaluations += self.population_size

        best = ev.best()
        if best.result is not None and best.result.reached:
            self._solved = True
            steps = best.result.steps
            if self._best_path_len is None or steps < self._best_path_len:
                self._best_path_len = steps

        extra: Dict[str, float] = {}
        if gen_stats.num_species:
            extra = {
                "species": gen_stats.num_species,
                "nodes": gen_stats.best_nodes,
                "connections": gen_stats.best_connections,
            }

        stats = StepStats(
            iteration=gen_stats.generation,
            reached=self._solved,
            best_path_length=self._best_path_len,
            best_fitness=best.fitness,
            evaluations=self._evaluations,
            extra=extra,
        )
        ev.reproduce()
        return stats

    def best_path(self) -> Optional[List[Cell]]:
        best = self.evolver.best()
        return best.result.trajectory if (best and best.result) else None

    def is_converged(self) -> bool:
        """Stop early once any evaluated GA individual reaches the goal."""
        return self._solved
