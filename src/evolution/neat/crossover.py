"""NEAT crossover by innovation-number matching.

Matching genes (same innovation in both parents) are inherited randomly from
either parent. Disjoint/excess genes are inherited from the fitter parent only.
If a gene is disabled in either parent there is a chance it stays disabled in
the child, mirroring the original NEAT paper.
"""
from __future__ import annotations

import numpy as np

from .genome import ConnectionGene, Genome

_DISABLE_IF_EITHER_DISABLED = 0.75


def crossover(parent1: Genome, parent2: Genome, rng: np.random.Generator) -> Genome:
    """Cross two parents. ``parent1`` is assumed at least as fit as ``parent2``;
    if they are equal the caller may pass them in either order."""
    if parent2.fitness > parent1.fitness:
        parent1, parent2 = parent2, parent1

    child = Genome(parent1.num_inputs, parent1.num_outputs)

    keys1 = set(parent1.connections)
    keys2 = set(parent2.connections)

    for innovation in keys1:
        gene1 = parent1.connections[innovation]
        if innovation in keys2:
            # Matching gene: pick randomly from either parent.
            gene2 = parent2.connections[innovation]
            chosen = gene1 if rng.random() < 0.5 else gene2
            new_gene = chosen.copy()
            if (not gene1.enabled or not gene2.enabled):
                new_gene.enabled = rng.random() >= _DISABLE_IF_EITHER_DISABLED
        else:
            # Disjoint/excess gene from the fitter parent.
            new_gene = gene1.copy()
        child.add_connection(new_gene)

    return child
