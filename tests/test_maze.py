import numpy as np

from src.maze.distance import bfs_distance_field, optimal_path_length
from src.maze.fixed_mazes import FIXED_MAZES, get_fixed_maze
from src.maze.generator import generate_maze
from src.maze.maze import OPEN


def test_generation_is_deterministic():
    a = generate_maze(21, 21, seed=123)
    b = generate_maze(21, 21, seed=123)
    assert np.array_equal(a.grid, b.grid)
    assert a.start == b.start and a.goal == b.goal


def test_dimensions_forced_odd_and_bounded():
    m = generate_maze(20, 16, seed=1)
    assert m.width % 2 == 1
    assert m.height % 2 == 1


def test_start_and_goal_are_open_and_distinct():
    m = generate_maze(21, 21, seed=7)
    assert m.grid[m.start] == OPEN
    assert m.grid[m.goal] == OPEN
    assert m.start != m.goal


def test_generated_maze_is_solvable():
    for seed in range(5):
        m = generate_maze(21, 21, seed=seed)
        assert optimal_path_length(m) > 0


def test_bfs_distance_field_matches_optimal():
    m = generate_maze(15, 15, seed=42)
    field = bfs_distance_field(m, m.goal)
    assert field[m.goal] == 0
    assert int(field[m.start]) == optimal_path_length(m)


def test_fixed_mazes_load_and_are_solvable():
    assert set(FIXED_MAZES) >= {"empty", "simple", "spiral"}
    for name in FIXED_MAZES:
        maze = get_fixed_maze(name)
        assert optimal_path_length(maze) > 0
