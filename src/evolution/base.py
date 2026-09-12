"""Common interfaces shared by evolutionary algorithms.

A single ``Evolver`` abstraction lets the visualizer and the experiment runner
treat evolutionary algorithms identically:

    evolver.evaluate(eval_fn)   # score the current generation
    stats = evolver.stats()     # summarize it
    evolver.reproduce()         # build the next generation
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from ..environment.simulation import Controller, SimulationResult
from ..maze.maze import Cell

# An evaluation function maps a controller to (fitness, simulation result).
EvalFn = Callable[[Controller], Tuple[float, SimulationResult]]


class Individual(ABC):
    """One member of a population."""

    fitness: float
    result: Optional[SimulationResult]

    @abstractmethod
    def controller(self) -> Controller:
        """Return a controller that drives an agent through the maze."""


@dataclass
class GenerationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    best_reached: bool
    best_steps: int
    # Optional method-specific extras.
    num_species: int = 0
    best_nodes: int = 0
    best_connections: int = 0


class Evolver(ABC):
    """Abstract evolutionary engine over a population of individuals."""

    name: str
    generation: int
    population: List[Individual]

    @abstractmethod
    def evaluate(self, eval_fn: EvalFn) -> None:
        """Assign ``fitness`` and ``result`` to every individual."""

    @abstractmethod
    def reproduce(self) -> None:
        """Advance ``population`` to the next generation."""

    @abstractmethod
    def best(self) -> Individual:
        """Best individual seen so far (across all generations)."""

    @abstractmethod
    def stats(self) -> GenerationStats:
        """Summary statistics for the most recently evaluated generation."""

    def population_trajectories(self) -> List[List[Cell]]:
        """Trajectories of the most recently evaluated population (for drawing)."""
        return [ind.result.trajectory for ind in self.population if ind.result is not None]
