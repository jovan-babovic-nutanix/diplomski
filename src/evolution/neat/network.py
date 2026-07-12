"""Compile a NEAT genome into an executable feed-forward network.

Activation order is computed by layering the directed acyclic graph (the same
approach used by neat-python): repeatedly find nodes whose inputs are all
already available. Because the genome is guaranteed acyclic, this terminates.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, List, Set, Tuple

from .genome import Genome


def tanh(x: float) -> float:
    # Clamp to avoid overflow warnings on extreme pre-activations.
    if x < -20.0:
        return -1.0
    if x > 20.0:
        return 1.0
    return math.tanh(x)


def _required_for_output(
    inputs: Set[int], outputs: Set[int], connections: List[Tuple[int, int]]
) -> Set[int]:
    required = set(outputs)
    seen = set(outputs)
    while True:
        layer = {a for (a, b) in connections if b in seen and a not in seen}
        if not layer:
            break
        new_required = {n for n in layer if n not in inputs}
        if not new_required:
            break
        required |= new_required
        seen |= layer
    return required


def _feed_forward_layers(
    inputs: List[int], outputs: List[int], connections: List[Tuple[int, int]]
) -> List[Set[int]]:
    input_set, output_set = set(inputs), set(outputs)
    required = _required_for_output(input_set, output_set, connections)

    layers: List[Set[int]] = []
    seen = set(inputs)
    while True:
        candidates = {b for (a, b) in connections if a in seen and b not in seen}
        this_layer = set()
        for node in candidates:
            if node in required and all(
                a in seen for (a, b) in connections if b == node
            ):
                this_layer.add(node)
        if not this_layer:
            break
        layers.append(this_layer)
        seen |= this_layer
    return layers


class FeedForwardNetwork:
    def __init__(self, input_keys: List[int], output_keys: List[int], node_evals):
        self.input_keys = input_keys
        self.output_keys = output_keys
        # node_evals: list of (node_id, [(in_node, weight), ...])
        self.node_evals = node_evals

    @staticmethod
    def create(genome: Genome) -> "FeedForwardNetwork":
        enabled = genome.enabled_connections()
        conn_pairs = [(c.in_node, c.out_node) for c in enabled]

        incoming = defaultdict(list)
        for c in enabled:
            incoming[c.out_node].append((c.in_node, c.weight))

        layers = _feed_forward_layers(
            genome.input_keys, genome.output_keys, conn_pairs
        )

        node_evals = []
        for layer in layers:
            for node in layer:
                node_evals.append((node, incoming.get(node, [])))

        return FeedForwardNetwork(genome.input_keys, genome.output_keys, node_evals)

    def activate(self, inputs: Iterable[float]) -> List[float]:
        values = {k: 0.0 for k in self.input_keys}
        for k, v in zip(self.input_keys, inputs):
            values[k] = float(v)

        for node, links in self.node_evals:
            total = 0.0
            for in_node, weight in links:
                total += values.get(in_node, 0.0) * weight
            values[node] = tanh(total)

        return [values.get(o, 0.0) for o in self.output_keys]
