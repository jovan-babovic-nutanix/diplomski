import numpy as np

from config import FitnessConfig, GAConfig, SimulationConfig
from src.evolution.ga.ga import GeneticAlgorithm
from src.evolution.ga.genome import MoveSequenceController
from src.experiments.runner import build_ga, make_eval, run_single
from src.maze.fixed_mazes import get_fixed_maze


def test_controller_returns_valid_actions():
    genome = np.array([0, 1, 2, 3], dtype=np.int64)
    ctrl = MoveSequenceController(genome)
    actions = [ctrl.act(step, np.zeros(6)) for step in range(8)]
    assert actions == [0, 1, 2, 3, 0, 1, 2, 3]  # cycles
    assert all(0 <= a <= 3 for a in actions)


def test_ga_is_deterministic():
    maze = get_fixed_maze("empty")
    sim, fit = SimulationConfig(max_steps=120), FitnessConfig()
    eval_fn = make_eval(maze, sim, fit)

    cfg = GAConfig(population_size=40, generations=10, seed=5)
    h1, _ = run_single(build_ga(cfg, sim), eval_fn, cfg.generations)
    h2, _ = run_single(build_ga(cfg, sim), eval_fn, cfg.generations)
    assert [g.best_fitness for g in h1] == [g.best_fitness for g in h2]


def test_ga_improves_and_solves_empty_maze():
    maze = get_fixed_maze("empty")
    sim, fit = SimulationConfig(max_steps=200), FitnessConfig()
    eval_fn = make_eval(maze, sim, fit)

    cfg = GAConfig(population_size=150, generations=60, seed=1)
    ga = build_ga(cfg, sim)
    history, solved_gen = run_single(ga, eval_fn, cfg.generations)

    assert history[-1].best_fitness >= history[0].best_fitness
    assert solved_gen is not None  # the open maze should be solved
    assert ga.best().result.reached
