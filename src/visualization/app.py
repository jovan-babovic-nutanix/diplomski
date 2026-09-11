"""Interactive pygame app: three methods side by side on the same maze.

Every method gets its own panel and advances in lock-step so you can visually
compare how GA, Q-Learning and A* explore the *same* maze:

    * GA             -> a translucent swarm of dots + a gold "best of generation".
    * Q-Learning     -> one agent following the current greedy policy.
    * A*             -> one agent tracing the optimal path (found immediately).

One "round" = one generation (GA), one training batch (Q-Learning) or one
planning step (A*). After the shared animation plays out, all three advance to
their next round together.

Controls:
    SPACE  pause / resume
    + / -  faster / slower animation
    N      next generation/training batch
    click  NEXT ROUND button for the same action
    r      new maze (new seed), restart all methods
    v      toggle animation (off = fast-forward rounds)
    ESC    quit
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pygame

from config import (
    FitnessConfig,
    GAConfig,
    MazeConfig,
    QLearningConfig,
    SimulationConfig,
)
from ..experiments.runner import build_ga, make_eval
from ..maze.distance import bfs_distance_field, optimal_path_length
from ..maze.generator import generate_maze
from ..maze.maze import Cell
from ..planning.astar import AStarSolver
from ..rl.qlearning import QLearningSolver

METHODS = ["GA", "Q-Learning", "A*"]
POPULATION_METHODS = {"GA"}
MAX_VISIBLE_STEP = 8.0

FCell = Tuple[float, float]


@dataclass
class PanelState:
    """Everything the renderer needs to draw one method's current round."""
    trajectories: List[List[Cell]] = field(default_factory=list)
    leader_traj: List[Cell] = field(default_factory=list)
    best_path: Optional[List[Cell]] = None
    info: Dict[str, object] = field(default_factory=dict)
    max_len: int = 1


class MazeApp:
    def __init__(
        self,
        maze_cfg: Optional[MazeConfig] = None,
        sim_cfg: Optional[SimulationConfig] = None,
        fit_cfg: Optional[FitnessConfig] = None,
        ga_cfg: Optional[GAConfig] = None,
        ql_cfg: Optional[QLearningConfig] = None,
        fps: int = 60,
    ):
        pygame.init()
        self.maze_cfg = maze_cfg or MazeConfig()
        self.sim_cfg = sim_cfg or SimulationConfig()
        self.fit_cfg = fit_cfg or FitnessConfig()
        self.ga_cfg = ga_cfg or GAConfig()
        self.ql_cfg = ql_cfg or QLearningConfig(
            episodes_per_step=30, max_steps=self.sim_cfg.max_steps
        )
        self.fps = fps

        self.seed = self.maze_cfg.seed
        self._build_world(self.seed)

        from .renderer import Renderer

        self.renderer = Renderer(self.maze, cols=3, rows=1, target_cell_area=300)

        self.paused = False
        self.speed = 0.4
        self.show_render = True
        self.ga_done = False
        self.qlearning_done = False
        self.astar_done = False
        self.finished = False
        self._reset_metrics()
        self.clock = pygame.time.Clock()

        self.states: Dict[str, PanelState] = {}
        self._start_all_rounds()

    # -- world setup --------------------------------------------------------
    def _build_world(self, seed: int) -> None:
        self.maze = generate_maze(self.maze_cfg.width, self.maze_cfg.height, seed=seed)
        self.dist_field = bfs_distance_field(self.maze, self.maze.goal)
        self.optimal = optimal_path_length(self.maze)
        self.eval_fn = make_eval(self.maze, self.sim_cfg, self.fit_cfg, self.dist_field)

        self.evolvers = {"GA": build_ga(self.ga_cfg, self.sim_cfg)}
        self.solvers = {
            "Q-Learning": QLearningSolver(self.ql_cfg, self.maze, self.dist_field, self.fit_cfg),
            "A*": AStarSolver(self.maze),
        }

    # -- rounds -------------------------------------------------------------
    def _start_all_rounds(self) -> None:
        for method in METHODS:
            if method == "GA" and self.ga_done:
                continue
            if method == "Q-Learning" and self.qlearning_done:
                continue
            if method == "A*" and self.astar_done:
                continue
            self.states[method] = self._start_round(method)
            if method == "GA":
                info = self.states[method].info
                generation = int(info.get("iteration", 0))
                self.ga_done = bool(info.get("solved")) or (
                    generation + 1 >= self.ga_cfg.generations
                )
                self._record_completion("GA", self.states[method], self.ga_done)
            elif method == "Q-Learning":
                iteration = int(self.states[method].info.get("iteration", 0))
                self.qlearning_done = bool(
                    self.states[method].info.get("solved")
                ) or iteration >= self._max_qlearning_rounds()
                self._record_completion(
                    "Q-Learning", self.states[method], self.qlearning_done
                )
            elif method == "A*":
                self.astar_done = bool(self.states[method].info.get("solved"))
                self._record_completion("A*", self.states[method], self.astar_done)
        self.finished = self.ga_done and self.qlearning_done and self.astar_done
        self.global_max = max((s.max_len for s in self.states.values()), default=1)
        self.anim_pos = 0.0

    def _reset_metrics(self) -> None:
        now = time.perf_counter()
        self.method_started = {method: now for method in METHODS}
        self.metrics: Dict[str, Dict[str, object]] = {
            method: {"completed": False} for method in METHODS
        }

    def _record_completion(
        self, method: str, state: PanelState, completed: bool
    ) -> None:
        if not completed or self.metrics[method].get("completed"):
            return
        info = state.info
        solved = bool(info.get("solved"))
        path_length = info.get("best_steps")
        optimality = "-"
        if solved and path_length is not None and self.optimal > 0:
            optimality = f"{float(path_length) / self.optimal:.2f}x"
        elapsed = time.perf_counter() - self.method_started[method]
        self.metrics[method] = {
            "completed": True,
            "solved": solved,
            "iterations": info.get("iteration", "-"),
            "evaluations": info.get("evaluations", "-"),
            "path_length": path_length if path_length is not None else "-",
            "optimality": optimality,
            "time_s": f"{elapsed:.3f}s",
        }

    def _max_qlearning_rounds(self) -> int:
        """Maximum UI training rounds allowed by the episode budget."""
        return max(
            1,
            math.ceil(self.ql_cfg.episodes / self.ql_cfg.episodes_per_step),
        )

    def _advance_all_rounds(self) -> None:
        for method in POPULATION_METHODS:
            if method == "GA" and self.ga_done:
                continue
            self.evolvers[method].reproduce()
        self._start_all_rounds()

    def _start_round(self, method: str) -> PanelState:
        if method in POPULATION_METHODS:
            return self._population_round(method)
        return self._solver_round(method)

    def _population_round(self, method: str) -> PanelState:
        ev = self.evolvers[method]
        ev.evaluate(self.eval_fn)
        stats = ev.stats()
        trajectories = [ind.result.trajectory for ind in ev.population]
        gen_best = max(ev.population, key=lambda i: i.fitness)
        best = ev.best()
        state = PanelState(
            trajectories=trajectories,
            leader_traj=gen_best.result.trajectory,
            best_path=best.result.trajectory if best.result else None,
            info={
                "iteration": stats.generation,
                "solved": stats.best_reached,
                "best_steps": stats.best_steps if stats.best_reached else None,
                "evaluations": (stats.generation + 1) * self.ga_cfg.population_size,
                "species": stats.num_species,
            },
        )
        state.max_len = max((len(t) for t in trajectories), default=1)
        return state

    def _solver_round(self, method: str) -> PanelState:
        solver = self.solvers[method]
        st = solver.step()
        path = solver.best_path() or [self.maze.start]
        state = PanelState(
            trajectories=[path],
            leader_traj=path,
            best_path=path if st.reached else None,
            info={
                "iteration": st.iteration,
                "solved": st.reached,
                "best_steps": st.best_path_length,
                "evaluations": st.evaluations,
                "extra": dict(st.extra),
            },
        )
        state.max_len = max(len(path), 1)
        return state

    # -- animation helpers --------------------------------------------------
    @staticmethod
    def _interp(traj: List[Cell], pos: float) -> FCell:
        if not traj:
            return (0.0, 0.0)
        last = len(traj) - 1
        i = min(int(pos), last)
        frac = pos - i if i < last else 0.0
        r0, c0 = traj[i]
        r1, c1 = traj[min(i + 1, last)]
        return (r0 + (r1 - r0) * frac, c0 + (c1 - c0) * frac)

    def _positions_at(self, state: PanelState, pos: float) -> List[FCell]:
        capped = min(pos, state.max_len)
        return [self._interp(t, capped) for t in state.trajectories]

    # -- main loop ----------------------------------------------------------
    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.renderer.next_button_rect().collidepoint(event.pos):
                        self._advance_one_round()
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)
            self._update()
            self._draw()
            self.clock.tick(self.fps)
        pygame.quit()

    def _handle_key(self, key: int) -> bool:
        if key == pygame.K_ESCAPE:
            return False
        if key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.speed = min(2000.0, self.speed * 1.5)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed = max(0.05, self.speed / 1.5)
        elif key == pygame.K_n:
            self._advance_one_round()
        elif key == pygame.K_v:
            self.show_render = not self.show_render
        elif key == pygame.K_r:
            self.seed += 1
            self._build_world(self.seed)
            self.renderer.set_maze(self.maze)
            self.ga_done = False
            self.qlearning_done = False
            self.astar_done = False
            self.finished = False
            self._reset_metrics()
            self._start_all_rounds()
        return True

    def _advance_one_round(self) -> None:
        """Skip the current playback and advance all active methods once."""
        if self.finished:
            return
        self._advance_all_rounds()

    def _update(self) -> None:
        if self.paused or self.finished:
            return
        if not self.show_render:
            self._advance_all_rounds()
            self.anim_pos = float(self.global_max)
            return
        # At very high speeds, jumping several hundred cells per frame makes
        # the dots appear to disappear. Keep the displayed movement visible;
        # the v key remains available for true fast-forward without drawing.
        self.anim_pos += min(self.speed, MAX_VISIBLE_STEP)
        if self.anim_pos >= self.global_max:
            self._advance_all_rounds()

    # -- drawing ------------------------------------------------------------
    def _draw(self) -> None:
        self.renderer.begin_frame()
        progress = self.anim_pos / self.global_max if self.global_max else 1.0
        self.renderer.draw_header(
            "Maze solving: 3 methods, one maze",
            [
                f"maze {self.maze.width}x{self.maze.height}",
                f"seed {self.seed}",
                f"optimal {self.optimal}",
                (
                    f"speed {self.speed:.0f}x"
                    + (" (visual cap)" if self.show_render and self.speed > MAX_VISIBLE_STEP else "")
                ),
                (
                    "FINISHED"
                    if self.finished
                    else (
                        "PAUSED"
                        if self.paused
                        else ("fast-forward" if not self.show_render else "running")
                    )
                ),
            ],
            progress,
        )
        for idx, method in enumerate(METHODS):
            state = self.states[method]
            positions = self._positions_at(state, self.anim_pos)
            leader = self._interp(state.leader_traj, min(self.anim_pos, len(state.leader_traj)))
            self.renderer.draw_panel(
                idx,
                method,
                positions,
                leader,
                state.best_path,
                self._panel_stats(method, state),
                bool(state.info.get("solved")),
            )
        self.renderer.draw_summary(self.metrics, self.finished)
        self.renderer.draw_footer(
            "SPACE pause   N / NEXT ROUND   +/- speed   r new maze   v fast-forward   ESC quit"
        )
        self.renderer.end_frame()

    def _panel_stats(self, method: str, state: PanelState) -> List[Tuple[str, str]]:
        info = state.info
        steps = info.get("best_steps")
        stats: List[Tuple[str, str]] = [
            ("gen" if method in POPULATION_METHODS else "iter", str(info.get("iteration", 0))),
            ("steps", str(steps) if steps is not None else "-"),
        ]
        if method == "GA" and self.ga_done:
            stats.append(("status", "done"))
        if method == "Q-Learning":
            extra = info.get("extra", {}) or {}
            stats.append(("eps", f"{extra.get('epsilon', 0):.2f}"))
        elif method == "A*":
            extra = info.get("extra", {}) or {}
            stats.append(("expanded", str(int(extra.get("nodes_expanded", 0)))))
        return stats
