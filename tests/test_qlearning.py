import numpy as np

from config import FitnessConfig, QLearningConfig
from src.maze.distance import bfs_distance_field, optimal_path_length
from src.maze.fixed_mazes import get_fixed_maze
from src.rl.qlearning import QLearningSolver, greedy_rollout


def _solver(maze, episodes=2500, seed=0):
    dist = bfs_distance_field(maze, maze.goal)
    cfg = QLearningConfig(
        episodes=episodes, episodes_per_step=100, max_steps=200, seed=seed
    )
    return QLearningSolver(cfg, maze, dist, FitnessConfig()), cfg


def test_qtable_shape():
    maze = get_fixed_maze("empty")
    solver, _ = _solver(maze)
    assert solver.q.shape == (maze.height, maze.width, 4)


def test_qlearning_solves_simple_maze():
    maze = get_fixed_maze("simple")
    solver, cfg = _solver(maze, episodes=4000)
    steps = cfg.episodes // cfg.episodes_per_step
    last = None
    for _ in range(steps):
        last = solver.step()
    assert last.reached
    # greedy policy should produce a valid solving path near the optimum
    dist = bfs_distance_field(maze, maze.goal)
    rollout = greedy_rollout(solver.q, maze, dist, cfg.max_steps)
    assert rollout.reached
    assert rollout.steps >= optimal_path_length(maze)


def test_qlearning_optimal_on_empty():
    maze = get_fixed_maze("empty")
    solver, cfg = _solver(maze, episodes=3000)
    for _ in range(cfg.episodes // cfg.episodes_per_step):
        solver.step()
    dist = bfs_distance_field(maze, maze.goal)
    rollout = greedy_rollout(solver.q, maze, dist, cfg.max_steps)
    assert rollout.reached
    assert rollout.steps == optimal_path_length(maze)  # RL learns the optimum


def test_qlearning_is_deterministic():
    maze = get_fixed_maze("empty")
    s1, cfg = _solver(maze, episodes=1000, seed=7)
    s2, _ = _solver(maze, episodes=1000, seed=7)
    for _ in range(cfg.episodes // cfg.episodes_per_step):
        s1.step()
        s2.step()
    assert np.allclose(s1.q, s2.q)


def test_epsilon_decays():
    maze = get_fixed_maze("empty")
    solver, cfg = _solver(maze)
    start_eps = solver._epsilon()
    for _ in range(cfg.episodes // cfg.episodes_per_step):
        solver.step()
    end_eps = solver._epsilon()
    assert start_eps > end_eps
    assert end_eps >= cfg.epsilon_end - 1e-9
