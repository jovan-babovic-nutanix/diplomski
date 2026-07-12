"""matplotlib figures for the multi-method comparison study.

Covers the four thesis criteria:
    1. success rate            -> plot_success_rate
    2. path length vs optimum  -> plot_optimality
    3. convergence speed       -> plot_convergence_* and plot_iterations_to_solve
    4. execution time          -> plot_wall_time

Uses the non-interactive Agg backend so it runs headless.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .metrics import TrialResult  # noqa: E402

LEARNING_METHODS = ("GA", "NEAT", "Q-Learning")


def _by_algorithm(trials: List[TrialResult]) -> Dict[str, List[TrialResult]]:
    grouped: Dict[str, List[TrialResult]] = defaultdict(list)
    for t in trials:
        grouped[t.algorithm].append(t)
    return grouped


def _stack(trials: List[TrialResult], getter: Callable) -> np.ndarray:
    """Rows = seeds, cols = iterations (truncated to the shortest history)."""
    series = [[getter(st) for st in t.history] for t in trials]
    series = [s for s in series if s]
    if not series:
        return np.empty((0, 0))
    min_len = min(len(s) for s in series)
    return np.array([s[:min_len] for s in series], dtype=np.float64)


def _convergence(trials: List[TrialResult], x_getter, xlabel: str, title: str, path: str) -> None:
    grouped = _by_algorithm(trials)
    plt.figure(figsize=(8, 5))
    plotted = False
    for algo in LEARNING_METHODS:
        if algo not in grouped:
            continue
        data = _stack(grouped[algo], lambda st: st.best_fitness)
        if data.size == 0:
            continue
        x = _stack(grouped[algo], x_getter)[0]
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        plt.plot(x, mean, label=f"{algo} (n={data.shape[0]})")
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel(xlabel)
    plt.ylabel("Best fitness (shared metric)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_convergence_iterations(trials: List[TrialResult], path: str) -> None:
    _convergence(
        trials, lambda st: st.iteration, "Iteration (native unit)",
        "Convergence vs native iterations", path,
    )


def plot_convergence_evaluations(trials: List[TrialResult], path: str) -> None:
    _convergence(
        trials, lambda st: st.evaluations, "Environment episodes (shared cost)",
        "Convergence vs environment episodes", path,
    )


def plot_success_rate(trials: List[TrialResult], path: str) -> None:
    grouped = _by_algorithm(trials)
    algos = sorted(grouped)
    rates = [np.mean([t.solved for t in grouped[a]]) * 100 for a in algos]
    plt.figure(figsize=(7, 5))
    bars = plt.bar(algos, rates)
    for bar, rate in zip(bars, rates):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{rate:.0f}%", ha="center")
    plt.ylabel("Success rate (%)")
    plt.ylim(0, 105)
    plt.title("Maze solve success rate")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_optimality(trials: List[TrialResult], path: str) -> None:
    grouped = _by_algorithm(trials)
    algos = sorted(grouped)
    means, errs, labels = [], [], []
    for a in algos:
        ratios = [t.optimality_ratio for t in grouped[a] if t.optimality_ratio is not None]
        if ratios:
            means.append(np.mean(ratios))
            errs.append(np.std(ratios))
            labels.append(a)
    plt.figure(figsize=(7, 5))
    if labels:
        plt.bar(labels, means, yerr=errs, capsize=4)
        plt.axhline(1.0, linestyle="--", linewidth=1, label="optimal (A*)")
        plt.legend()
    plt.ylabel("Path length / optimal (1.0 = optimal)")
    plt.title("Path optimality of solved mazes")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_wall_time(trials: List[TrialResult], path: str) -> None:
    grouped = _by_algorithm(trials)
    algos = sorted(grouped)
    means = [np.mean([t.wall_time_s for t in grouped[a]]) for a in algos]
    errs = [np.std([t.wall_time_s for t in grouped[a]]) for a in algos]
    plt.figure(figsize=(7, 5))
    plt.bar(algos, means, yerr=errs, capsize=4)
    plt.ylabel("Wall-clock time (s)")
    plt.title("Execution time per run (lower is better)")
    plt.yscale("log")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_iterations_to_solve(trials: List[TrialResult], path: str) -> None:
    grouped = _by_algorithm(trials)
    plot_data, labels = [], []
    for algo in LEARNING_METHODS:
        if algo not in grouped:
            continue
        vals = [t.iterations_to_solve for t in grouped[algo] if t.iterations_to_solve is not None]
        if vals:
            plot_data.append(vals)
            labels.append(algo)
    plt.figure(figsize=(7, 5))
    if plot_data:
        plt.boxplot(plot_data, labels=labels, showmeans=True)
    plt.ylabel("Iterations to first solve")
    plt.title("Convergence speed (lower is better)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_neat_complexity(trials: List[TrialResult], path: str) -> None:
    neat = [t for t in trials if t.algorithm == "NEAT"]
    if not neat:
        return
    nodes = _stack(neat, lambda st: st.extra.get("nodes", 0.0))
    conns = _stack(neat, lambda st: st.extra.get("connections", 0.0))
    if nodes.size == 0:
        return
    gens = np.arange(nodes.shape[1])
    plt.figure(figsize=(8, 5))
    plt.plot(gens, nodes.mean(axis=0), label="nodes")
    plt.plot(gens, conns.mean(axis=0), label="enabled connections")
    plt.xlabel("Generation")
    plt.ylabel("Count (mean over seeds)")
    plt.title("NEAT champion network complexity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def generate_all_plots(trials: List[TrialResult], output_dir: str) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    figures = {
        "success_rate.png": plot_success_rate,
        "optimality.png": plot_optimality,
        "convergence_iterations.png": plot_convergence_iterations,
        "convergence_evaluations.png": plot_convergence_evaluations,
        "iterations_to_solve.png": plot_iterations_to_solve,
        "wall_time.png": plot_wall_time,
        "neat_complexity.png": plot_neat_complexity,
    }
    written = []
    for name, fn in figures.items():
        full = os.path.join(output_dir, name)
        fn(trials, full)
        written.append(full)
    return written
