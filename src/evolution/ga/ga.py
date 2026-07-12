"""Classic generational genetic algorithm over fixed-length move sequences.

Operators: tournament selection, two-point crossover, per-gene reset mutation,
and elitism. All randomness flows through a single seeded NumPy Generator.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from config import GAConfig
from ..base import Evolver, EvalFn, GenerationStats
from .genome import GAIndividual


class GeneticAlgorithm(Evolver):
    name = "GA"

    def __init__(self, cfg: GAConfig, genome_length: int, num_actions: int = 4):
        if genome_length <= 0:
            raise ValueError("genome_length must be positive")
        self.cfg = cfg
        self.genome_length = genome_length
        self.num_actions = num_actions
        self.rng = np.random.default_rng(cfg.seed)
        self.generation = 0
        self.population: List[GAIndividual] = [
            self._random_individual() for _ in range(cfg.population_size)
        ]
        self._best: Optional[GAIndividual] = None

    # -- construction ---------------------------------------------------
    def _random_individual(self) -> GAIndividual:
        genome = self.rng.integers(
            0, self.num_actions, size=self.genome_length, dtype=np.int64
        )
        return GAIndividual(genome)

    # -- evaluation -----------------------------------------------------
    def evaluate(self, eval_fn: EvalFn) -> None:
        for ind in self.population:
            fitness, result = eval_fn(ind.controller())
            ind.fitness = fitness
            ind.result = result

        gen_best = max(self.population, key=lambda i: i.fitness)
        if self._best is None or gen_best.fitness > self._best.fitness:
            self._best = gen_best.copy()

    # -- selection & variation -----------------------------------------
    def _tournament(self) -> GAIndividual:
        contenders = self.rng.choice(
            len(self.population), size=self.cfg.tournament_size, replace=False
        )
        best = max((self.population[i] for i in contenders), key=lambda i: i.fitness)
        return best

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.rng.random() > self.cfg.crossover_rate or self.genome_length < 2:
            return a.copy()
        p1, p2 = sorted(self.rng.integers(0, self.genome_length, size=2))
        child = a.copy()
        child[p1:p2] = b[p1:p2]
        return child

    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        mask = self.rng.random(self.genome_length) < self.cfg.mutation_rate
        if mask.any():
            genome = genome.copy()
            genome[mask] = self.rng.integers(
                0, self.num_actions, size=int(mask.sum()), dtype=np.int64
            )
        return genome

    def reproduce(self) -> None:
        ranked = sorted(self.population, key=lambda i: i.fitness, reverse=True)
        next_pop: List[GAIndividual] = [ranked[i].copy() for i in range(self.cfg.elitism)]

        while len(next_pop) < self.cfg.population_size:
            parent_a = self._tournament()
            parent_b = self._tournament()
            child_genome = self._crossover(parent_a.genome, parent_b.genome)
            child_genome = self._mutate(child_genome)
            next_pop.append(GAIndividual(child_genome))

        self.population = next_pop[: self.cfg.population_size]
        self.generation += 1

    # -- reporting ------------------------------------------------------
    def best(self) -> GAIndividual:
        assert self._best is not None, "Call evaluate() before best()"
        return self._best

    def stats(self) -> GenerationStats:
        fits = np.array([i.fitness for i in self.population], dtype=np.float64)
        gen_best = max(self.population, key=lambda i: i.fitness)
        result = gen_best.result
        return GenerationStats(
            generation=self.generation,
            best_fitness=float(fits.max()),
            mean_fitness=float(fits.mean()),
            best_reached=bool(result.reached) if result else False,
            best_steps=int(result.steps) if result else 0,
        )
