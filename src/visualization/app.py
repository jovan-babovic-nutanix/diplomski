"""Interactive pygame app: three methods side by side on the same maze.

Every method gets its own panel and advances in lock-step so you can visually
compare how GA, Q-Learning and A* explore the *same* maze:

    * GA             -> a translucent swarm of dots + a gold "best of generation".
    * Q-Learning     -> one agent following the current greedy policy.
    * A*             -> one agent tracing the optimal path (found immediately).

One "round" = one generation (GA), one training batch (Q-Learning) or one
planning step (A*). After the shared animation plays out, all three advance to
their next round together.

This module owns only pygame concerns (window, input, frame timing,
rendering); the actual round-advancing/completion/metrics state machine lives
in ``session.ComparisonSession`` so it can be tested without a display.

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

from typing import List, Optional, Tuple

import pygame

from config import (
    FitnessConfig,
    GAConfig,
    MazeConfig,
    QLearningConfig,
    SimulationConfig,
)
from ..maze.maze import Cell
from .session import METHODS, ComparisonSession, PanelState

MAX_VISIBLE_STEP = 8.0

FCell = Tuple[float, float]


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
        maze_cfg = maze_cfg or MazeConfig()
        sim_cfg = sim_cfg or SimulationConfig()
        fit_cfg = fit_cfg or FitnessConfig()
        ga_cfg = ga_cfg or GAConfig()
        ql_cfg = ql_cfg or QLearningConfig(
            episodes_per_step=30, max_steps=sim_cfg.max_steps
        )
        self.fps = fps

        self.session = ComparisonSession(maze_cfg, sim_cfg, fit_cfg, ga_cfg, ql_cfg)

        from .renderer import Renderer

        self.renderer = Renderer(self.session.maze, cols=3, rows=1, target_cell_area=300)

        self.paused = False
        self.speed = 0.4
        self.show_render = True
        self.anim_pos = 0.0
        self.clock = pygame.time.Clock()

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
            self.session.reset(self.session.seed + 1)
            self.renderer.set_maze(self.session.maze)
            self.anim_pos = 0.0
        return True

    def _advance_one_round(self) -> None:
        """Skip the current playback and advance all active methods once."""
        if self.session.finished:
            return
        self.session.advance_one_round()
        self.anim_pos = 0.0

    def _update(self) -> None:
        if self.paused or self.session.finished:
            return
        if not self.show_render:
            self.session.advance_all_rounds()
            self.anim_pos = float(self.session.global_max)
            return
        # At very high speeds, jumping several hundred cells per frame makes
        # the dots appear to disappear. Keep the displayed movement visible;
        # the v key remains available for true fast-forward without drawing.
        self.anim_pos += min(self.speed, MAX_VISIBLE_STEP)
        if self.anim_pos >= self.session.global_max:
            self.session.advance_all_rounds()
            self.anim_pos = 0.0

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

    # -- drawing ------------------------------------------------------------
    def _draw(self) -> None:
        session = self.session
        self.renderer.begin_frame()
        progress = self.anim_pos / session.global_max if session.global_max else 1.0
        self.renderer.draw_header(
            "Maze solving: 3 methods, one maze",
            [
                f"maze {session.maze.width}x{session.maze.height}",
                f"seed {session.seed}",
                f"optimal {session.optimal}",
                (
                    f"speed {self.speed:.0f}x"
                    + (" (visual cap)" if self.show_render and self.speed > MAX_VISIBLE_STEP else "")
                ),
                (
                    "FINISHED"
                    if session.finished
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
            state = session.states[method]
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
        self.renderer.draw_summary(session.metrics, session.finished)
        self.renderer.draw_footer(
            "SPACE pause   N / NEXT ROUND   +/- speed   r new maze   v fast-forward   ESC quit",
            next_enabled=not session.finished,
        )
        self.renderer.end_frame()

    def _panel_stats(self, method: str, state: PanelState) -> List[Tuple[str, str]]:
        info = state.info
        steps = info.get("best_steps")
        stats: List[Tuple[str, str]] = [
            ("gen" if method == "GA" else "iter", str(info.get("iteration", 0))),
            ("steps", str(steps) if steps is not None else "-"),
        ]
        if method == "GA" and self.session.ga_done:
            stats.append(("status", "done"))
        if method == "Q-Learning":
            extra = info.get("extra", {}) or {}
            stats.append(("eps", f"{extra.get('epsilon', 0):.2f}"))
        elif method == "A*":
            extra = info.get("extra", {}) or {}
            stats.append(("expanded", str(int(extra.get("nodes_expanded", 0)))))
        return stats
