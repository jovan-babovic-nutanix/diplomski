"""The NEAT evolutionary engine, tying genomes, speciation and reproduction
together behind the shared ``Evolver`` interface.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from config import NEATConfig
from ...environment.sensors import NUM_INPUTS, NUM_OUTPUTS
from ...environment.simulation import SimulationResult
from ..base import Evolver, EvalFn, GenerationStats, Individual
from .crossover import crossover
from .genome import Genome, InnovationTracker, create_initial_genome
from .mutation import mutate
from .network import FeedForwardNetwork
from .species import Species, speciate


class NeatController:
    """Drives an agent using a compiled feed-forward network. The constant bias
    input (1.0) is appended to the sensor vector each step."""

    def __init__(self, network: FeedForwardNetwork):
        self.network = network

    def act(self, step: int, sensors: np.ndarray) -> int:
        inputs = list(sensors) + [1.0]
        outputs = self.network.activate(inputs)
        return int(np.argmax(outputs))


class NeatIndividual(Individual):
    def __init__(self, genome: Genome):
        self.genome = genome
        self.fitness: float = 0.0
        self.result: Optional[SimulationResult] = None

    def controller(self) -> NeatController:
        return NeatController(FeedForwardNetwork.create(self.genome))

    def copy(self) -> "NeatIndividual":
        clone = NeatIndividual(self.genome.copy())
        clone.fitness = self.fitness
        clone.result = self.result
        return clone


class Neat(Evolver):
    name = "NEAT"

    def __init__(
        self,
        cfg: NEATConfig,
        num_inputs: int = NUM_INPUTS,
        num_outputs: int = NUM_OUTPUTS,
    ):
        self.cfg = cfg
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.rng = np.random.default_rng(cfg.seed)
        self.tracker = InnovationTracker(first_hidden_id=num_inputs + num_outputs)
        self.generation = 0
        self.species: List[Species] = []
        self._best: Optional[NeatIndividual] = None

        self.population: List[NeatIndividual] = [
            NeatIndividual(
                create_initial_genome(
                    num_inputs, num_outputs, self.tracker, self.rng, cfg.weight_init_std
                )
            )
            for _ in range(cfg.population_size)
        ]

    # -- evaluation -----------------------------------------------------
    def evaluate(self, eval_fn: EvalFn) -> None:
        for ind in self.population:
            fitness, result = eval_fn(ind.controller())
            ind.fitness = fitness
            ind.genome.fitness = fitness
            ind.result = result

        gen_best = max(self.population, key=lambda i: i.fitness)
        if self._best is None or gen_best.fitness > self._best.fitness:
            self._best = gen_best.copy()

    # -- reproduction ---------------------------------------------------
    def reproduce(self) -> None:
        genomes = [ind.genome for ind in self.population]
        self.species = speciate(genomes, self.species, self.cfg)
        self._update_stagnation()

        species_sums = []
        for sp in self.species:
            n = len(sp.members)
            for g in sp.members:
                g.adjusted_fitness = g.fitness / n
            species_sums.append(sum(g.adjusted_fitness for g in sp.members))

        total = sum(species_sums)
        if total <= 0:
            total = 1.0
            species_sums = [1.0 for _ in self.species]

        offspring_counts = self._allocate_offspring(species_sums, total)

        new_genomes: List[Genome] = []
        for sp, count in zip(self.species, offspring_counts):
            new_genomes.extend(self._reproduce_species(sp, count))

        new_genomes = self._fit_to_population_size(new_genomes)

        # Pick fresh representatives for next generation's speciation.
        for sp in self.species:
            sp.representative = sp.members[int(self.rng.integers(0, len(sp.members)))]

        self.population = [NeatIndividual(g) for g in new_genomes]
        self.generation += 1

    def _update_stagnation(self) -> None:
        for sp in self.species:
            best = max(g.fitness for g in sp.members)
            if best > sp.best_fitness:
                sp.best_fitness = best
                sp.staleness = 0
            else:
                sp.staleness += 1

        if len(self.species) <= self.cfg.species_elitism:
            return

        protected = set(
            id(sp)
            for sp in sorted(
                self.species, key=lambda s: s.best_fitness, reverse=True
            )[: self.cfg.species_elitism]
        )
        survivors = [
            sp
            for sp in self.species
            if sp.staleness <= self.cfg.stagnation or id(sp) in protected
        ]
        if survivors:
            self.species = survivors

    def _allocate_offspring(self, species_sums, total) -> List[int]:
        pop = self.cfg.population_size
        raw = [s / total * pop for s in species_sums]
        counts = [int(math.floor(x)) for x in raw]
        remainder = pop - sum(counts)
        order = sorted(
            range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True
        )
        for i in range(remainder):
            counts[order[i % len(order)]] += 1
        return counts

    def _reproduce_species(self, sp: Species, count: int) -> List[Genome]:
        if count <= 0:
            return []

        ranked = sorted(sp.members, key=lambda g: g.fitness, reverse=True)
        num_survivors = max(1, math.ceil(self.cfg.survival_threshold * len(ranked)))
        survivors = ranked[:num_survivors]

        children: List[Genome] = []
        # Elitism: large species copy their champion unchanged.
        if len(ranked) >= 5 and self.cfg.elitism > 0:
            children.append(ranked[0].copy())

        while len(children) < count:
            p1 = survivors[int(self.rng.integers(0, len(survivors)))]
            p2 = survivors[int(self.rng.integers(0, len(survivors)))]
            child = crossover(p1, p2, self.rng)
            mutate(child, self.cfg, self.tracker, self.rng)
            children.append(child)

        return children[:count]

    def _fit_to_population_size(self, genomes: List[Genome]) -> List[Genome]:
        pop = self.cfg.population_size
        if len(genomes) > pop:
            return genomes[:pop]
        while len(genomes) < pop:
            # Fall back to mutated copies of the best-known genome.
            base = self._best.genome if self._best else self.population[0].genome
            clone = base.copy()
            mutate(clone, self.cfg, self.tracker, self.rng)
            genomes.append(clone)
        return genomes

    # -- reporting ------------------------------------------------------
    def best(self) -> NeatIndividual:
        assert self._best is not None, "Call evaluate() before best()"
        return self._best

    def stats(self) -> GenerationStats:
        fits = np.array([i.fitness for i in self.population], dtype=np.float64)
        gen_best = max(self.population, key=lambda i: i.fitness)
        result = gen_best.result
        nodes, conns = gen_best.genome.size()
        return GenerationStats(
            generation=self.generation,
            best_fitness=float(fits.max()),
            mean_fitness=float(fits.mean()),
            best_reached=bool(result.reached) if result else False,
            best_steps=int(result.steps) if result else 0,
            num_species=len(self.species),
            best_nodes=nodes,
            best_connections=conns,
        )
