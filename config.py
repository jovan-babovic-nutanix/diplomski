"""Central configuration for the Maze Evolution project.

All tunable hyperparameters live here as dataclasses so that experiments are
reproducible and easy to document in the thesis. Every randomized component
takes an explicit seed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python and NumPy global RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))


@dataclass
class MazeConfig:
    """Maze size and generation seed. Dimensions are forced odd by the
    recursive-backtracker generator (walls live on even rows/cols)."""
    width: int = 21
    height: int = 21
    seed: int = 42


@dataclass
class SimulationConfig:
    """How long an agent is allowed to act in a single episode."""
    max_steps: int = 300


@dataclass
class FitnessConfig:
    """Weights for the shared fitness function (higher fitness = better)."""
    distance_weight: float = 10.0   # reward per cell of BFS progress toward goal
    goal_bonus: float = 1000.0      # bonus for reaching the goal
    step_penalty: float = 1.0       # penalty per step (only applied when solved)
    bump_penalty: float = 1.0       # penalty per wall collision
    min_fitness: float = 0.01       # floor keeps evolutionary fitness positive


@dataclass
class GAConfig:
    """Classic genetic algorithm hyperparameters."""
    population_size: int = 300
    generations: int = 200
    genome_length: int = 0          # 0 -> auto = SimulationConfig.max_steps
    tournament_size: int = 5
    crossover_rate: float = 0.85
    mutation_rate: float = 0.03     # per-gene probability of random reset
    elitism: int = 4
    seed: int = 0


@dataclass
class QLearningConfig:
    """Tabular Q-Learning hyperparameters (reinforcement learning baseline)."""
    episodes: int = 4000            # total training episodes
    episodes_per_step: int = 50     # episodes between greedy evaluations (curve points)
    alpha: float = 0.2              # learning rate
    gamma: float = 0.95             # discount factor
    epsilon_start: float = 1.0      # exploration at the beginning
    epsilon_end: float = 0.05       # exploration at the end
    epsilon_decay_episodes: int = 3000   # episodes over which epsilon decays
    max_steps: int = 300            # cap per episode
    goal_reward: float = 100.0
    step_penalty: float = -1.0      # living cost per step
    reward_shaping: bool = True     # potential-based shaping via BFS distance
    seed: int = 0


@dataclass
class ExperimentConfig:
    """Headless comparison study settings."""
    n_seeds: int = 5
    base_seed: int = 1000
    output_dir: str = "outputs"
    maze: MazeConfig = field(default_factory=MazeConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    qlearning: QLearningConfig = field(default_factory=QLearningConfig)


def resolve_genome_length(ga: GAConfig, sim: SimulationConfig) -> int:
    """GA genome length defaults to the episode length when set to 0."""
    return ga.genome_length if ga.genome_length > 0 else sim.max_steps


# Compute-budget note (thesis methodology):
#   GA        : population_size * generations        = 300 * 200 = 60_000 episodes
#   Q-Learning: episodes + episodes/episodes_per_step = 4000 + 80 =  4_080 episodes
# ~15x gap at current defaults. This is deliberate for now (each method uses its
# own conventional budget); an equal-episode-budget ablation is an open
# methodology question, not something this preset machinery resolves.
# StepStats.evaluations (see src/solver.py) is the common cost axis that makes
# the gap visible in convergence_evaluations.png.


def demo_config() -> ExperimentConfig:
    """Interactive-demo preset: small maze, small budgets, smooth framerate."""
    return ExperimentConfig(
        n_seeds=3,
        maze=MazeConfig(width=15, height=15),
        simulation=SimulationConfig(max_steps=200),
        ga=GAConfig(population_size=100, generations=80),
        qlearning=QLearningConfig(episodes=2000, episodes_per_step=30, max_steps=200),
    )


def thesis_config() -> ExperimentConfig:
    """Headless study preset used for the numbers reported in the thesis."""
    return ExperimentConfig(
        n_seeds=20,
        maze=MazeConfig(width=21, height=21),
        simulation=SimulationConfig(max_steps=300),
        ga=GAConfig(population_size=300, generations=200),
        qlearning=QLearningConfig(episodes=4000, episodes_per_step=50, max_steps=300),
    )


PRESETS = {"demo": demo_config, "thesis": thesis_config}


def get_preset(name: str) -> ExperimentConfig:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}; options: {sorted(PRESETS)}")
    return PRESETS[name]()
