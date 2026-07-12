import numpy as np

from config import FitnessConfig, NEATConfig, SimulationConfig
from src.environment.sensors import NUM_INPUTS, NUM_OUTPUTS
from src.evolution.neat.crossover import crossover
from src.evolution.neat.genome import (
    InnovationTracker,
    create_initial_genome,
    creates_cycle,
)
from src.evolution.neat.network import FeedForwardNetwork
from src.experiments.runner import build_neat, make_eval, run_single
from src.maze.fixed_mazes import get_fixed_maze


def _fresh_genome(seed=0):
    rng = np.random.default_rng(seed)
    tracker = InnovationTracker(first_hidden_id=NUM_INPUTS + NUM_OUTPUTS)
    genome = create_initial_genome(NUM_INPUTS, NUM_OUTPUTS, tracker, rng, 1.0)
    return genome, tracker, rng


def test_initial_genome_fully_connects_inputs_to_outputs():
    genome, _, _ = _fresh_genome()
    assert len(genome.connections) == NUM_INPUTS * NUM_OUTPUTS


def test_network_activation_shape_and_range():
    genome, _, _ = _fresh_genome()
    net = FeedForwardNetwork.create(genome)
    out = net.activate([0.1] * NUM_INPUTS)
    assert len(out) == NUM_OUTPUTS
    assert all(-1.0 <= v <= 1.0 for v in out)


def test_distance_to_self_is_zero_and_symmetric():
    g1, _, _ = _fresh_genome(1)
    g2, _, _ = _fresh_genome(2)
    assert g1.distance(g1, 1.0, 0.5) == 0.0
    assert g1.distance(g2, 1.0, 0.5) == g2.distance(g1, 1.0, 0.5)


def test_creates_cycle_detects_back_edge():
    genome, tracker, rng = _fresh_genome()
    # Inputs -> outputs only, so an output -> input edge would form a cycle.
    out_node = genome.output_keys[0]
    in_node = genome.input_keys[0]
    assert creates_cycle(genome, out_node, in_node) is True
    # input -> output never creates a cycle in a feed-forward graph.
    assert creates_cycle(genome, in_node, out_node) is False


def test_crossover_produces_valid_genome():
    g1, _, _ = _fresh_genome(10)
    g2, _, _ = _fresh_genome(11)
    g1.fitness, g2.fitness = 5.0, 1.0
    child = crossover(g1, g2, np.random.default_rng(0))
    assert len(child.connections) > 0
    # Child should be runnable.
    net = FeedForwardNetwork.create(child)
    assert len(net.activate([0.0] * NUM_INPUTS)) == NUM_OUTPUTS


def test_neat_runs_and_improves():
    maze = get_fixed_maze("empty")
    sim, fit = SimulationConfig(max_steps=200), FitnessConfig()
    eval_fn = make_eval(maze, sim, fit)

    cfg = NEATConfig(population_size=60, generations=30, seed=3)
    neat = build_neat(cfg)
    history, _ = run_single(neat, eval_fn, cfg.generations)

    assert len(history) == cfg.generations
    assert history[-1].best_fitness >= history[0].best_fitness
    assert history[-1].num_species >= 1
