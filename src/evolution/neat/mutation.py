"""Structural and weight mutations for NEAT genomes."""
from __future__ import annotations

import numpy as np

from config import NEATConfig
from .genome import ConnectionGene, Genome, InnovationTracker, creates_cycle


def mutate(
    genome: Genome,
    cfg: NEATConfig,
    tracker: InnovationTracker,
    rng: np.random.Generator,
) -> None:
    if rng.random() < cfg.weight_mutate_rate:
        mutate_weights(genome, cfg, rng)
    if rng.random() < cfg.add_conn_rate:
        mutate_add_connection(genome, cfg, tracker, rng)
    if rng.random() < cfg.add_node_rate:
        mutate_add_node(genome, tracker, rng)


def mutate_weights(genome: Genome, cfg: NEATConfig, rng: np.random.Generator) -> None:
    for gene in genome.connections.values():
        if rng.random() < cfg.weight_perturb_rate:
            gene.weight += float(rng.normal(0.0, cfg.weight_perturb_std))
        else:
            gene.weight = float(rng.normal(0.0, cfg.weight_init_std))


def mutate_add_connection(
    genome: Genome,
    cfg: NEATConfig,
    tracker: InnovationTracker,
    rng: np.random.Generator,
) -> None:
    possible_inputs = list(genome.input_keys) + list(genome.hidden_keys)
    possible_outputs = list(genome.output_keys) + list(genome.hidden_keys)
    existing = {(c.in_node, c.out_node) for c in genome.connections.values()}

    for _ in range(20):  # bounded retries
        in_node = int(rng.choice(possible_inputs))
        out_node = int(rng.choice(possible_outputs))
        if in_node == out_node:
            continue
        if (in_node, out_node) in existing:
            continue
        if creates_cycle(genome, in_node, out_node):
            continue
        innovation = tracker.connection_innovation(in_node, out_node)
        weight = float(rng.normal(0.0, cfg.weight_init_std))
        genome.add_connection(
            ConnectionGene(in_node, out_node, weight, True, innovation)
        )
        return


def mutate_add_node(
    genome: Genome, tracker: InnovationTracker, rng: np.random.Generator
) -> None:
    enabled = genome.enabled_connections()
    if not enabled:
        return
    gene = enabled[int(rng.integers(0, len(enabled)))]
    gene.enabled = False

    new_node = tracker.new_node_id()
    # in -> new (weight 1.0) preserves signal magnitude into the new node.
    innov_in = tracker.connection_innovation(gene.in_node, new_node)
    genome.add_connection(
        ConnectionGene(gene.in_node, new_node, 1.0, True, innov_in)
    )
    # new -> out (original weight) preserves the original transformation.
    innov_out = tracker.connection_innovation(new_node, gene.out_node)
    genome.add_connection(
        ConnectionGene(new_node, gene.out_node, gene.weight, True, innov_out)
    )
