"""NEAT genome: nodes + connection genes with historical markings.

This is a from-scratch implementation of the core data structures from
Stanley & Miikkulainen's NEAT. Key ideas implemented here:

* Each connection carries an *innovation number* assigned by a shared
  ``InnovationTracker``. The same structural mutation occurring in different
  genomes receives the same innovation number, which makes crossover and the
  compatibility distance well-defined.
* Node identifiers: inputs are ``0..num_inputs-1`` (the last input is a constant
  bias), outputs are the next ``num_outputs`` ids, and hidden nodes get fresh
  ids from the tracker.
* Only feed-forward (acyclic) topologies are produced; ``creates_cycle`` guards
  the add-connection mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np


@dataclass
class ConnectionGene:
    in_node: int
    out_node: int
    weight: float
    enabled: bool
    innovation: int

    def copy(self) -> "ConnectionGene":
        return ConnectionGene(
            self.in_node, self.out_node, self.weight, self.enabled, self.innovation
        )


class InnovationTracker:
    """Hands out globally consistent innovation numbers and hidden node ids."""

    def __init__(self, first_hidden_id: int):
        self._conn_innovations: Dict[Tuple[int, int], int] = {}
        self._next_innovation = 0
        self._next_node_id = first_hidden_id

    def connection_innovation(self, in_node: int, out_node: int) -> int:
        key = (in_node, out_node)
        if key not in self._conn_innovations:
            self._conn_innovations[key] = self._next_innovation
            self._next_innovation += 1
        return self._conn_innovations[key]

    def new_node_id(self) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id


class Genome:
    def __init__(self, num_inputs: int, num_outputs: int):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.input_keys: List[int] = list(range(num_inputs))
        self.output_keys: List[int] = list(range(num_inputs, num_inputs + num_outputs))
        self.hidden_keys: Set[int] = set()
        self.connections: Dict[int, ConnectionGene] = {}  # innovation -> gene
        self.fitness: float = 0.0
        self.adjusted_fitness: float = 0.0

    # -- structure ------------------------------------------------------
    @property
    def nodes(self) -> Set[int]:
        return set(self.input_keys) | set(self.output_keys) | self.hidden_keys

    def add_connection(self, gene: ConnectionGene) -> None:
        self.connections[gene.innovation] = gene
        for node in (gene.in_node, gene.out_node):
            if node not in self.input_keys and node not in self.output_keys:
                self.hidden_keys.add(node)

    def enabled_connections(self) -> List[ConnectionGene]:
        return [c for c in self.connections.values() if c.enabled]

    def copy(self) -> "Genome":
        clone = Genome(self.num_inputs, self.num_outputs)
        clone.hidden_keys = set(self.hidden_keys)
        clone.connections = {k: g.copy() for k, g in self.connections.items()}
        clone.fitness = self.fitness
        clone.adjusted_fitness = self.adjusted_fitness
        return clone

    def size(self) -> Tuple[int, int]:
        """(number of nodes, number of enabled connections)."""
        return len(self.nodes), len(self.enabled_connections())

    # -- compatibility distance ----------------------------------------
    def distance(self, other: "Genome", c_disjoint: float, c_weight: float) -> float:
        """Compatibility distance based on connection genes.

        Disjoint and excess genes are pooled (a common simplification) and
        normalized by the larger genome size.
        """
        keys_a = set(self.connections)
        keys_b = set(other.connections)
        matching = keys_a & keys_b
        non_matching = len(keys_a ^ keys_b)

        if matching:
            weight_diff = sum(
                abs(self.connections[k].weight - other.connections[k].weight)
                for k in matching
            ) / len(matching)
        else:
            weight_diff = 0.0

        n = max(len(keys_a), len(keys_b), 1)
        return c_disjoint * non_matching / n + c_weight * weight_diff


def creates_cycle(genome: Genome, in_node: int, out_node: int) -> bool:
    """Would adding edge ``in_node -> out_node`` create a directed cycle?"""
    if in_node == out_node:
        return True
    # Search forward from out_node; if we can reach in_node, the edge closes a loop.
    visited = {out_node}
    stack = [out_node]
    while stack:
        node = stack.pop()
        for gene in genome.connections.values():
            if not gene.enabled or gene.in_node != node:
                continue
            target = gene.out_node
            if target == in_node:
                return True
            if target not in visited:
                visited.add(target)
                stack.append(target)
    return False


def create_initial_genome(
    num_inputs: int,
    num_outputs: int,
    tracker: InnovationTracker,
    rng: np.random.Generator,
    weight_std: float,
) -> Genome:
    """Minimal genome: every input fully connected to every output."""
    genome = Genome(num_inputs, num_outputs)
    for in_node in genome.input_keys:
        for out_node in genome.output_keys:
            innovation = tracker.connection_innovation(in_node, out_node)
            weight = float(rng.normal(0.0, weight_std))
            genome.add_connection(
                ConnectionGene(in_node, out_node, weight, True, innovation)
            )
    return genome
