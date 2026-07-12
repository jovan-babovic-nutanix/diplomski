"""Interactive pygame application visualizing all four methods in real time.

Controls:
    SPACE  pause / resume
    + / -  faster / slower animation
    g      Genetic Algorithm        (population cloud + leader)
    n      NEAT                      (population cloud + leader)
    q      Q-Learning               (single agent following the greedy policy)
    a      A*                        (single agent following the optimal path)
    r      new maze (new seed), restart all methods
    v      toggle animation rendering (off = fast-forward)
    ESC/Q  quit

GA and NEAT are population-based, so we drive their ``Evolver`` directly to keep
the nice translucent crowd. Q-Learning and A* are single-policy methods, so we
drive them through the ``Solver`` interface and animate one agent.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pygame

from config import (
    FitnessConfig,
    GAConfig,
    MazeConfig,
    NEATConfig,
    QLearningConfig,
    SimulationConfig,
)
from ..experiments.runner import build_ga, build_neat, make_eval
from ..maze.distance import bfs_distance_field, optimal_path_length
from ..maze.generator import generate_maze
from ..maze.maze import Cell
from ..planning.astar import AStarSolver
from ..rl.qlearning import QLearningSolver

METHODS = ["GA", "NEAT", "Q-Learning", "A*"]
POPULATION_METHODS = {"GA", "NEAT"}


class MazeApp:
    def __init__(
        self,
        maze_cfg: Optional[MazeConfig] = None,
        sim_cfg: Optional[SimulationConfig] = None,
        fit_cfg: Optional[FitnessConfig] = None,
        ga_cfg: Optional[GAConfig] = None,
        neat_cfg: Optional[NEATConfig] = None,
        ql_cfg: Optional[QLearningConfig] = None,
        fps: int = 60,
    ):
        pygame.init()
        self.maze_cfg = maze_cfg or MazeConfig()
        self.sim_cfg = sim_cfg or SimulationConfig()
        self.fit_cfg = fit_cfg or FitnessConfig()
        self.ga_cfg = ga_cfg or GAConfig()
        self.neat_cfg = neat_cfg or NEATConfig()
        self.ql_cfg = ql_cfg or QLearningConfig(
            episodes_per_step=30, max_steps=self.sim_cfg.max_steps
        )
        self.fps = fps

        self.seed = self.maze_cfg.seed
        self._build_world(self.seed)

        from .renderer import Renderer

        self.renderer = Renderer(self.maze)

        self.current = "GA"
        self.paused = False
        self.speed = 0.5
        self.show_render = True
        self.clock = pygame.time.Clock()
        self._start_round()

    # -- world setup ----------------------------------------------------
    def _build_world(self, seed: int) -> None:
        self.maze = generate_maze(self.maze_cfg.width, self.maze_cfg.height, seed=seed)
        self.dist_field = bfs_distance_field(self.maze, self.maze.goal)
        self.optimal = optimal_path_length(self.maze)
        self.eval_fn = make_eval(self.maze, self.sim_cfg, self.fit_cfg, self.dist_field)

        self.evolvers = {
            "GA": build_ga(self.ga_cfg, self.sim_cfg),
            "NEAT": build_neat(self.neat_cfg),
        }
        self.solvers = {
            "Q-Learning": QLearningSolver(
                self.ql_cfg, self.maze, self.dist_field, self.fit_cfg
            ),
            "A*": AStarSolver(self.maze),
        }
        # Sticky "best path found so far" per method (for the gold overlay).
        self.best_paths: Dict[str, Optional[List[Cell]]] = {m: None for m in METHODS}
        self.info: Dict[str, object] = {}

    # -- round (one generation / training batch / planning) ------------
    def _start_round(self) -> None:
        if self.current in POPULATION_METHODS:
            self._start_population_round()
        else:
            self._start_solver_round()
        self.max_len = max((len(t) for t in self.trajectories), default=1)
        self.anim_pos = 0.0

    def _start_population_round(self) -> None:
        ev = self.evolvers[self.current]
        ev.evaluate(self.eval_fn)
        stats = ev.stats()
        results = [ind.result for ind in ev.population]
        self.trajectories = [r.trajectory for r in results]
        gen_best = max(ev.population, key=lambda i: i.fitness)
        self.gen_best_traj = gen_best.result.trajectory
        best = ev.best()
        if best.result:
            self.best_paths[self.current] = best.result.trajectory
        self.info = {
            "iteration": stats.generation,
            "best_fitness": stats.best_fitness,
            "mean_fitness": stats.mean_fitness,
            "solved": stats.best_reached,
            "best_steps": stats.best_steps if stats.best_reached else None,
            "species": stats.num_species,
            "nodes": stats.best_nodes,
            "connections": stats.best_connections,
        }

    def _start_solver_round(self) -> None:
        solver = self.solvers[self.current]
        st = solver.step()
        path = solver.best_path() or [self.maze.start]
        self.trajectories = [path]
        self.gen_best_traj = path
        self.best_paths[self.current] = path
        self.info = {
            "iteration": st.iteration,
            "best_fitness": st.best_fitness,
            "solved": st.reached,
            "best_steps": st.best_path_length,
            "extra": dict(st.extra),
        }

    def _advance_round(self) -> None:
        if self.current in POPULATION_METHODS:
            self.evolvers[self.current].reproduce()
        self._start_round()

    # -- animation helpers ---------------------------------------------
    @staticmethod
    def _interp(traj, pos: float) -> Tuple[float, float]:
        last = len(traj) - 1
        i = min(int(pos), last)
        frac = pos - i if i < last else 0.0
        r0, c0 = traj[i]
        r1, c1 = traj[min(i + 1, last)]
        return (r0 + (r1 - r0) * frac, c0 + (c1 - c0) * frac)

    def _positions_at(self, pos: float) -> List[Tuple[float, float]]:
        return [self._interp(t, pos) for t in self.trajectories]

    # -- main loop ------------------------------------------------------
    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)
            self._update()
            self._draw()
            self.clock.tick(self.fps)
        pygame.quit()

    def _handle_key(self, key: int) -> bool:
        if key in (pygame.K_ESCAPE,):
            return False
        if key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.speed = min(2000.0, self.speed * 1.5)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed = max(0.05, self.speed / 1.5)
        elif key == pygame.K_g:
            self._switch("GA")
        elif key == pygame.K_n:
            self._switch("NEAT")
        elif key == pygame.K_q:
            self._switch("Q-Learning")
        elif key == pygame.K_a:
            self._switch("A*")
        elif key == pygame.K_v:
            self.show_render = not self.show_render
        elif key == pygame.K_r:
            self.seed += 1
            self._build_world(self.seed)
            self.renderer.set_maze(self.maze)
            self._start_round()
        return True

    def _switch(self, name: str) -> None:
        if name != self.current:
            self.current = name
            self._start_round()

    def _update(self) -> None:
        if self.paused:
            return
        if not self.show_render:
            self._advance_round()
            self.anim_pos = self.max_len
            return
        self.anim_pos += self.speed
        if self.anim_pos >= self.max_len:
            self._advance_round()

    def _draw(self) -> None:
        positions = self._positions_at(self.anim_pos)
        leader = self._interp(self.gen_best_traj, self.anim_pos)
        best_path = self.best_paths.get(self.current)
        progress = self.anim_pos / self.max_len if self.max_len else 1.0
        self.renderer.draw(positions, leader, best_path, self._hud_lines(), progress)

    def _hud_lines(self) -> List[Tuple[str, bool]]:
        info = self.info
        solved = "YES" if info.get("solved") else "no"
        best_steps = info.get("best_steps")
        lines: List[Tuple[str, bool]] = [
            (f"Method: {self.current}", True),
            ("", False),
            (f"Iteration  : {info.get('iteration', 0)}", False),
            (f"Solved     : {solved}", False),
            (f"Best steps : {best_steps if best_steps is not None else '-'}", False),
            (f"Optimal    : {self.optimal}", False),
        ]
        bf = info.get("best_fitness")
        if bf is not None:
            lines.insert(4, (f"Best fit   : {bf:.1f}", False))

        if self.current == "NEAT":
            lines += [
                ("", False),
                ("NEAT internals", True),
                (f"Species    : {info.get('species', 0)}", False),
                (f"Nodes      : {info.get('nodes', 0)}", False),
                (f"Conns      : {info.get('connections', 0)}", False),
            ]
        elif self.current == "Q-Learning":
            extra = info.get("extra", {}) or {}
            lines += [
                ("", False),
                ("Q-Learning", True),
                (f"Epsilon    : {extra.get('epsilon', 0):.2f}", False),
                (f"Visited    : {int(extra.get('q_coverage', 0))} cells", False),
            ]
        elif self.current == "A*":
            extra = info.get("extra", {}) or {}
            lines += [
                ("", False),
                ("A* search", True),
                (f"Expanded   : {int(extra.get('nodes_expanded', 0))} nodes", False),
            ]

        lines += [
            ("", False),
            ("Controls", True),
            ("g/n/q/a methods", False),
            ("SPACE  pause", False),
            (f"+/-  speed {self.speed:.2f}", False),
            ("r    new maze", False),
            (f"v    render {'on' if self.show_render else 'off'}", False),
            ("ESC  quit", False),
        ]
        return lines
