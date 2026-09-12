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
    P      toggle showing GA's population swarm
    click  the control-bar buttons for the same actions
    r      new maze (new seed), restart all methods
    v      toggle animation (off = fast-forward rounds)
    ESC    quit
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import pygame

from config import (
    FitnessConfig,
    GAConfig,
    MazeConfig,
    QLearningConfig,
    SimulationConfig,
)
from ..maze.maze import Cell
from .renderer import GOOD, INK_DIM, METHOD_ON_MAZE, Renderer
from .session import METHODS, ComparisonSession, PanelState

MAX_VISIBLE_STEP = 8.0

FCell = Tuple[float, float]

STATUS_RUNNING_TEXT = {
    "GA": "EVOLVING POPULATION",
    "Q-Learning": "REFINING GREEDY POLICY",
    "A*": "PLANNING",
}
STATUS_SOLVED_TEXT = {
    "GA": "BEST ROUTE LOCKED",
    "Q-Learning": "POLICY SOLVED",
    "A*": "OPTIMAL PATH LOCKED",
}
STATUS_FAILED_TEXT = "NO SOLUTION IN BUDGET"


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
        self._ql_total_rounds = max(
            1, math.ceil(ql_cfg.episodes / ql_cfg.episodes_per_step)
        )

        self.session = ComparisonSession(maze_cfg, sim_cfg, fit_cfg, ga_cfg, ql_cfg)
        self.renderer = Renderer(self.session.maze, cols=3, rows=1, target_cell_area=300)

        self.paused = False
        self.speed = 0.4
        self.show_render = True
        self.show_population = True
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
                    self._handle_click(event.pos)
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
        elif key == pygame.K_p:
            self.show_population = not self.show_population
        elif key == pygame.K_r:
            self._new_maze()
        return True

    def _handle_click(self, pos: Tuple[int, int]) -> None:
        rects = self.renderer.control_rects()
        for name, rect in rects.items():
            if rect.collidepoint(pos):
                self._dispatch_control(name)
                return

    def _dispatch_control(self, name: str) -> None:
        if name == "pause":
            self.paused = not self.paused
        elif name == "next":
            self._advance_one_round()
        elif name == "speed_down":
            self.speed = max(0.05, self.speed / 1.5)
        elif name == "speed_up":
            self.speed = min(2000.0, self.speed * 1.5)
        elif name == "new_maze":
            self._new_maze()
        elif name == "fastfwd":
            self.show_render = not self.show_render
        elif name == "show_pop":
            self.show_population = not self.show_population

    def _new_maze(self) -> None:
        self.session.reset(self.session.seed + 1)
        self.renderer.set_maze(self.session.maze)
        self.anim_pos = 0.0

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

        if session.finished:
            state_label, state_color = "FINISHED", GOOD
        elif self.paused:
            state_label, state_color = "PAUSED", INK_DIM
        elif not self.show_render:
            state_label, state_color = "FAST-FWD", METHOD_ON_MAZE["Q-Learning"]
        else:
            state_label, state_color = "LIVE", GOOD

        speed_value = f"{self.speed:.0f}×" + (
            " cap" if self.show_render and self.speed > MAX_VISIBLE_STEP else ""
        )
        self.renderer.draw_header(
            "MAZE ARENA — RACE CONTROL",
            "GA · Q-LEARNING · A* — SYNCHRONIZED ROUNDS",
            [
                ("MAZE", f"{session.maze.width}×{session.maze.height}"),
                ("SEED", str(session.seed)),
                ("OPTIMAL", str(session.optimal)),
                ("SPEED", speed_value),
            ],
            state_label,
            state_color,
            progress,
        )
        for idx, method in enumerate(METHODS):
            state = session.states[method]
            positions = self._positions_at(state, self.anim_pos)
            if method == "GA" and not self.show_population:
                positions = []
            leader = self._interp(state.leader_traj, min(self.anim_pos, len(state.leader_traj)))
            status, status_text = self._panel_status(method, state)
            self.renderer.draw_panel(
                idx,
                method,
                positions,
                leader,
                state.best_path,
                self._panel_stats(method, state),
                status,
                status_text,
            )
        self.renderer.draw_summary(session.metrics, session.finished, session.optimal)
        self.renderer.draw_footer(
            {
                "paused": self.paused,
                "fast_forward": not self.show_render,
                "speed": self.speed,
                "next_enabled": not session.finished,
                "show_population": self.show_population,
                "legend": "SPACE pause · N next round · +/- speed · "
                          "R new maze · V fast-forward · P population · ESC quit",
            }
        )
        self.renderer.end_frame()

    def _panel_status(self, method: str, state: PanelState) -> Tuple[str, str]:
        done = {
            "GA": self.session.ga_done,
            "Q-Learning": self.session.qlearning_done,
            "A*": self.session.astar_done,
        }[method]
        if state.info.get("solved"):
            return "solved", STATUS_SOLVED_TEXT[method]
        if done:
            return "failed", STATUS_FAILED_TEXT
        return "running", STATUS_RUNNING_TEXT[method]

    def _panel_stats(self, method: str, state: PanelState) -> List[Tuple[str, str]]:
        info = state.info
        steps = info.get("best_steps")
        steps_text = str(steps) if steps is not None else "—"
        it = int(info.get("iteration", 0))

        if method == "GA":
            return [
                ("GEN", f"{it + 1}/{self.session.ga_cfg.generations}"),
                ("POP", str(self.session.ga_cfg.population_size)),
                ("BEST", steps_text),
            ]
        if method == "Q-Learning":
            extra = info.get("extra", {}) or {}
            return [
                ("ITER", f"{it + 1}/{self._ql_total_rounds}"),
                ("EPS", f"{extra.get('epsilon', 0):.2f}"),
                ("STEPS", steps_text),
            ]
        # A*
        extra = info.get("extra", {}) or {}
        ratio = "—"
        if steps is not None and self.session.optimal > 0:
            ratio = f"{steps / self.session.optimal:.2f}×"
        return [
            ("EXPANDED", str(int(extra.get("nodes_expanded", 0)))),
            ("STEPS", steps_text),
            ("OPTIMAL", ratio),
        ]
