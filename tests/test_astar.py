from src.maze.distance import optimal_path_length
from src.maze.fixed_mazes import _from_strings, get_fixed_maze
from src.maze.generator import generate_maze
from src.planning.astar import AStarSolver, astar, manhattan


def test_astar_matches_bfs_optimal():
    for seed in range(5):
        m = generate_maze(21, 21, seed=seed)
        path, expanded = astar(m)
        assert path is not None
        assert path[0] == m.start and path[-1] == m.goal
        assert len(path) - 1 == optimal_path_length(m)  # A* is optimal
        assert expanded > 0


def test_astar_path_is_valid():
    m = get_fixed_maze("simple")
    path, _ = astar(m)
    assert path is not None
    for (r, c) in path:
        assert m.passable(r, c)            # never passes through a wall
    for a, b in zip(path, path[1:]):
        assert manhattan(a, b) == 1        # only single-cell steps


def test_astar_solver_converges_and_reports():
    m = get_fixed_maze("spiral")
    solver = AStarSolver(m)
    stats = solver.step()
    assert stats.reached
    assert solver.is_converged()
    assert stats.best_path_length == optimal_path_length(m)
    assert stats.extra["nodes_expanded"] > 0
    assert solver.best_path()[0] == m.start


def test_astar_unreachable_goal_marks_converged():
    # A wall fully separates start from goal - not registered in FIXED_MAZES
    # since test_fixed_mazes_load_and_are_solvable requires every registered
    # maze to be solvable.
    blocked = _from_strings(["S..#..", "...#..", "...#.G"])
    solver = AStarSolver(blocked)
    stats = solver.step()
    assert stats.reached is False
    assert stats.best_path_length is None
    assert solver.is_converged() is True
    assert solver.best_path() is None
    assert stats.extra["nodes_expanded"] > 0
