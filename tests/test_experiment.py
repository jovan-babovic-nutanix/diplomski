"""Integration smoke tests for the full multi-method comparison pipeline."""
import os

from config import (
    ExperimentConfig,
    GAConfig,
    MazeConfig,
    NEATConfig,
    QLearningConfig,
    SimulationConfig,
)
from src.experiments.metrics import write_history_csv, write_summary_csv
from src.experiments.plots import generate_all_plots
from src.experiments.runner import DEFAULT_METHODS, run_experiment


def _tiny_cfg() -> ExperimentConfig:
    return ExperimentConfig(
        n_seeds=2,
        maze=MazeConfig(width=11, height=11),
        simulation=SimulationConfig(max_steps=120),
        ga=GAConfig(population_size=40, generations=8),
        neat=NEATConfig(population_size=40, generations=8),
        qlearning=QLearningConfig(episodes=400, episodes_per_step=50, max_steps=120),
    )


def test_run_experiment_all_methods():
    cfg = _tiny_cfg()
    trials = run_experiment(cfg)
    assert len(trials) == len(DEFAULT_METHODS) * cfg.n_seeds
    for t in trials:
        assert t.history
        assert t.optimal_path_length > 0
    astar = [t for t in trials if t.algorithm == "A*"]
    assert astar and all(t.solved and t.optimality_ratio == 1.0 for t in astar)


def test_run_experiment_is_deterministic():
    cfg = _tiny_cfg()
    a = run_experiment(cfg)
    b = run_experiment(cfg)
    key = lambda ts: [(t.algorithm, t.seed, t.solved, t.best_path_length) for t in ts]
    assert key(a) == key(b)


def test_outputs_are_written(tmp_path):
    cfg = _tiny_cfg()
    trials = run_experiment(cfg)
    summary = str(tmp_path / "summary.csv")
    history = str(tmp_path / "history.csv")
    write_summary_csv(trials, summary)
    write_history_csv(trials, history)
    plots = generate_all_plots(trials, str(tmp_path))
    assert os.path.exists(summary) and os.path.exists(history)
    for p in plots:
        assert os.path.exists(p)
