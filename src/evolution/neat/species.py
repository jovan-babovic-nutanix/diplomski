"""Speciation: group genomes by topological/weight similarity.

Speciation lets innovative-but-currently-weak topologies survive by only making
genomes compete within their own niche (via fitness sharing in ``neat.py``).
"""
from __future__ import annotations

from typing import List

from config import NEATConfig
from .genome import Genome


class Species:
    def __init__(self, representative: Genome):
        self.representative = representative
        self.members: List[Genome] = [representative]
        self.best_fitness: float = representative.fitness
        self.staleness: int = 0

    def reset(self) -> None:
        """Keep the representative, clear members for the next assignment pass."""
        self.members = []


def speciate(
    population: List[Genome], species: List[Species], cfg: NEATConfig
) -> List[Species]:
    """Assign every genome to a species, creating new species as needed.

    Representatives are carried over from the previous generation; empty species
    are dropped.
    """
    for sp in species:
        sp.reset()

    for genome in population:
        placed = False
        for sp in species:
            distance = genome.distance(
                sp.representative, cfg.c_disjoint, cfg.c_weight
            )
            if distance < cfg.compatibility_threshold:
                sp.members.append(genome)
                placed = True
                break
        if not placed:
            species.append(Species(genome))

    return [sp for sp in species if sp.members]
