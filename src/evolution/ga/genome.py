"""GA genome: a fixed-length sequence of moves.

The "brain" of a GA individual is simply a pre-computed list of actions. The
controller replays them step by step (cycling if the episode outlasts the
genome), ignoring sensors entirely. This is the classic, easy-to-explain
    evolutionary baseline.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ...environment.simulation import SimulationResult
from ..base import Individual


class MoveSequenceController:
    uses_sensors = False  # GA replays a fixed sequence; it ignores observations

    def __init__(self, genome: np.ndarray):
        self.genome = genome
        self._length = len(genome)

    def act(self, step: int, sensors: np.ndarray) -> int:
        return int(self.genome[step % self._length])


class GAIndividual(Individual):
    def __init__(self, genome: np.ndarray):
        self.genome = genome
        self.fitness: float = 0.0
        self.result: Optional[SimulationResult] = None

    def controller(self) -> MoveSequenceController:
        return MoveSequenceController(self.genome)

    def copy(self) -> "GAIndividual":
        clone = GAIndividual(self.genome.copy())
        clone.fitness = self.fitness
        clone.result = self.result
        return clone
