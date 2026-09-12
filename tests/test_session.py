"""Headless tests for ComparisonSession - no pygame display required.

These exercise exactly the state machine that used to live untested inside
MazeApp (see src/visualization/session.py's module docstring).
"""
from config import GAConfig, MazeConfig, QLearningConfig, SimulationConfig
from src.maze.fixed_mazes import _from_strings
from src.planning.astar import AStarSolver
from src.visualization.session import ComparisonSession


def _tiny_session(seed: int = 1) -> ComparisonSession:
    return ComparisonSession(
        maze_cfg=MazeConfig(width=11, height=11, seed=seed),
        sim_cfg=SimulationConfig(max_steps=60),
        fit_cfg=__import__("config").FitnessConfig(),
        ga_cfg=GAConfig(population_size=20, generations=5),
        ql_cfg=QLearningConfig(episodes=200, episodes_per_step=20, max_steps=60),
    )


def test_session_finishes_when_all_methods_done():
    session = _tiny_session()
    for _ in range(20):
        if session.finished:
            break
        session.advance_all_rounds()
    assert session.finished
    for method in ("GA", "Q-Learning", "A*"):
        assert session.metrics[method]["completed"] is True


def test_next_round_is_noop_after_finished():
    session = _tiny_session()
    for _ in range(20):
        if session.finished:
            break
        session.advance_all_rounds()
    assert session.finished

    metrics_before = dict(session.metrics)
    iterations_before = {m: session.states[m].info["iteration"] for m in session.states}

    session.advance_one_round()

    assert session.metrics == metrics_before
    assert {m: session.states[m].info["iteration"] for m in session.states} == iterations_before


def test_reset_clears_completion_state():
    session = _tiny_session(seed=1)
    for _ in range(20):
        if session.finished:
            break
        session.advance_all_rounds()
    assert session.finished

    old_grid = session.maze.grid.copy()
    session.reset(seed=2)

    assert session.ga_done is False
    assert session.qlearning_done is False
    assert session.finished is False
    assert session.metrics["GA"]["completed"] is False
    assert session.metrics["Q-Learning"]["completed"] is False
    assert session.seed == 2
    assert old_grid.shape != session.maze.grid.shape or not (old_grid == session.maze.grid).all()


def test_astar_unreachable_does_not_hang_session():
    # A maze with a wall fully separating start from goal - A* must still be
    # marked done immediately (see AStarSolver.is_converged()).
    session = _tiny_session()
    blocked = _from_strings(["S..#..", "...#..", "...#.G"])
    session.maze = blocked
    session.solvers["A*"] = AStarSolver(blocked)
    session.astar_done = False

    session._start_all_rounds()

    assert session.astar_done is True
    assert session.states["A*"].info["solved"] is False
