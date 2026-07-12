"""Shared setup helpers and the headless multi-method comparison study.

``make_eval`` / ``build_*`` / ``run_single`` are reused by the interactive
visualizer too, so both entry points share identical setup. The comparison study
drives every method through the common ``Solver`` interface.
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional, Tuple

import numpy as np

from config import (
    ExperimentConfig,
    FitnessConfig,
    GAConfig,
    NEATConfig,
    QLearningConfig,
    SimulationConfig,
    resolve_genome_length,
)
from ..environment.sensors import precompute_static_sensors
from ..environment.simulation import Controller, SimulationResult, simulate
from ..evolution.base import Evolver, GenerationStats
from ..evolution.ga.ga import GeneticAlgorithm
from ..evolution.neat.neat import Neat
from ..fitness import evaluate_fitness
from ..maze.distance import bfs_distance_field, optimal_path_length
from ..maze.generator import generate_maze
from ..maze.maze import Maze
from ..planning.astar import AStarSolver
from ..rl.qlearning import QLearningSolver
from ..solver import EvolverSolver, Solver, StepStats
from .metrics import TrialResult


def make_eval(
    maze: Maze,
    sim_cfg: SimulationConfig,
    fit_cfg: FitnessConfig,
    dist_field: Optional[np.ndarray] = None,
) -> Callable[[Controller], Tuple[float, SimulationResult]]:
    """Build the evaluation function shared by the evolutionary engines."""
    if dist_field is None:
        dist_field = bfs_distance_field(maze, maze.goal)
    static_sensors = precompute_static_sensors(maze)

    def eval_fn(controller: Controller) -> Tuple[float, SimulationResult]:
        result = simulate(
            maze, controller, sim_cfg.max_steps, dist_field, static_sensors
        )
        return evaluate_fitness(result, fit_cfg), result

    return eval_fn


# -- evolver builders (also used by the visualizer) ---------------------
def build_ga(cfg: GAConfig, sim_cfg: SimulationConfig) -> GeneticAlgorithm:
    return GeneticAlgorithm(cfg, genome_length=resolve_genome_length(cfg, sim_cfg))


def build_neat(cfg: NEATConfig) -> Neat:
    return Neat(cfg)


def run_single(
    evolver: Evolver,
    eval_fn,
    generations: int,
    on_generation: Optional[Callable[[Evolver, GenerationStats], None]] = None,
) -> Tuple[List[GenerationStats], Optional[int]]:
    """Run a single evolver (kept for the GA/NEAT unit tests)."""
    history: List[GenerationStats] = []
    solved_gen: Optional[int] = None
    for _ in range(generations):
        evolver.evaluate(eval_fn)
        stats = evolver.stats()
        history.append(stats)
        if stats.best_reached and solved_gen is None:
            solved_gen = stats.generation
        if on_generation is not None:
            on_generation(evolver, stats)
        evolver.reproduce()
    return history, solved_gen


# -- solver builders ----------------------------------------------------
def build_solver(
    method: str,
    maze: Maze,
    dist_field: np.ndarray,
    cfg: ExperimentConfig,
    seed: int,
) -> Tuple[Solver, int]:
    """Construct a ``Solver`` for ``method`` plus its max number of steps."""
    eval_fn = make_eval(maze, cfg.simulation, cfg.fitness, dist_field)
    if method == "GA":
        ga_cfg = GAConfig(**{**cfg.ga.__dict__, "seed": seed})
        ga = build_ga(ga_cfg, cfg.simulation)
        return EvolverSolver(ga, eval_fn, ga_cfg.population_size), ga_cfg.generations
    if method == "NEAT":
        neat_cfg = NEATConfig(**{**cfg.neat.__dict__, "seed": seed})
        neat = build_neat(neat_cfg)
        return (
            EvolverSolver(neat, eval_fn, neat_cfg.population_size),
            neat_cfg.generations,
        )
    if method == "Q-Learning":
        ql_cfg = QLearningConfig(**{**cfg.qlearning.__dict__, "seed": seed})
        solver = QLearningSolver(ql_cfg, maze, dist_field, cfg.fitness)
        return solver, ql_cfg.episodes // ql_cfg.episodes_per_step
    if method == "A*":
        return AStarSolver(maze), 1
    raise ValueError(f"Unknown method {method!r}")


def run_solver(
    solver: Solver, max_iterations: int
) -> Tuple[List[StepStats], Optional[int], Optional[int], float]:
    """Run a solver and return (history, iters_to_solve, evals_to_solve, wall_s)."""
    history: List[StepStats] = []
    iters_to_solve: Optional[int] = None
    evals_to_solve: Optional[int] = None

    start = time.perf_counter()
    for it in range(max_iterations):
        st = solver.step()
        history.append(st)
        if st.reached and iters_to_solve is None:
            iters_to_solve = it
            evals_to_solve = st.evaluations
        if solver.is_converged():
            break
    wall = time.perf_counter() - start
    return history, iters_to_solve, evals_to_solve, wall


def _trial(method: str, solver: Solver, max_iterations: int, seed: int, optimal: int) -> TrialResult:
    history, iters, evals, wall = run_solver(solver, max_iterations)
    path = solver.best_path()
    solved = any(st.reached for st in history)
    best_len = (len(path) - 1) if (solved and path) else None
    return TrialResult(
        algorithm=method,
        seed=seed,
        solved=solved,
        iterations_to_solve=iters,
        evaluations_to_solve=evals,
        wall_time_s=wall,
        best_path_length=best_len,
        optimal_path_length=optimal,
        history=history,
    )


DEFAULT_METHODS = ("A*", "GA", "NEAT", "Q-Learning")


def run_experiment(
    cfg: ExperimentConfig,
    methods: Tuple[str, ...] = DEFAULT_METHODS,
    progress: Optional[Callable[[str], None]] = None,
) -> List[TrialResult]:
    """Run every method across ``n_seeds`` seeds. One freshly generated maze per
    seed is shared by all methods so the comparison is fair."""
    log = progress or (lambda _msg: None)
    trials: List[TrialResult] = []

    for i in range(cfg.n_seeds):
        seed = cfg.base_seed + i
        maze = generate_maze(cfg.maze.width, cfg.maze.height, seed=seed)
        dist_field = bfs_distance_field(maze, maze.goal)
        optimal = optimal_path_length(maze)

        for method in methods:
            log(f"[seed {seed}] {method} ...")
            solver, max_iters = build_solver(method, maze, dist_field, cfg, seed)
            trials.append(_trial(method, solver, max_iters, seed, optimal))

    return trials
