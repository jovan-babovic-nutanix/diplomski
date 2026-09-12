"""Integration smoke tests for the full multi-method comparison pipeline."""
import os

from config import (
    ExperimentConfig,
    GAConfig,
    MazeConfig,
    QLearningConfig,
    SimulationConfig,
    demo_config,
    resolve_genome_length,
    thesis_config,
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


def test_iterations_to_solve_matches_history_row():
    cfg = _tiny_cfg()
    trials = run_experiment(cfg)
    for t in trials:
        if t.iterations_to_solve is None:
            continue
        row = t.history[t.iterations_to_solve]
        assert row.iteration == t.iterations_to_solve
        assert row.reached is True


def test_presets_are_self_consistent():
    for cfg in (demo_config(), thesis_config()):
        assert cfg.qlearning.max_steps == cfg.simulation.max_steps
        assert cfg.n_seeds >= 1
        assert cfg.ga.population_size > cfg.ga.elitism
        assert 0 < cfg.qlearning.episodes_per_step <= cfg.qlearning.episodes
        assert resolve_genome_length(cfg.ga, cfg.simulation) == cfg.simulation.max_steps


def test_demo_preset_runs_end_to_end():
    cfg = demo_config()
    cfg.n_seeds = 1
    cfg.ga.generations = 3
    cfg.qlearning.episodes = 60
    trials = run_experiment(cfg)
    assert len(trials) == len(DEFAULT_METHODS) * cfg.n_seeds


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
