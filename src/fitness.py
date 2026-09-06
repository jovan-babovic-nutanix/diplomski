"""Shared fitness function used by the evolutionary solver.

Keeping a single fitness definition is essential for a fair comparison: the two
algorithms differ only in *how* they search, not in *what* they optimize.

Design:
* Primary signal is BFS progress toward the goal (smooth gradient).
    * Reaching the goal grants a large bonus, then shorter solutions score
      higher via a per-step penalty.
    * Wall bumps are mildly penalized to discourage flailing.
* Output is clamped to a small positive floor for stable selection.
"""
from __future__ import annotations

from config import FitnessConfig
from .environment.simulation import SimulationResult


def evaluate_fitness(result: SimulationResult, cfg: FitnessConfig) -> float:
    fitness = cfg.distance_weight * result.progress

    if result.reached:
        fitness += cfg.goal_bonus
        fitness -= cfg.step_penalty * result.steps

    fitness -= cfg.bump_penalty * result.bumps
    return max(cfg.min_fitness, fitness)
