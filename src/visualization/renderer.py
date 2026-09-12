"""Pygame rendering for the side-by-side method comparison.

Visual style copied from the "Flying Dots" GA maze project
(https://github.com/pantela002/GENETIC-ALGORITHM-MAZE):

    * cyan background  (126, 247, 247)
    * pure-blue walls  (0, 0, 255)     -- no gridlines, no checkerboard
    * a red square target (goal)
    * a swarm of small black square dots (the population)

Each method runs in its own panel; GA shows the full black swarm, while
single-policy methods (Q-Learning/A*) animate one dot. The
best solution found so far is traced as a thin orange path so the comparison
stays informative for the thesis.

The renderer holds no algorithm logic: the app feeds it positions/stats, it draws.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pygame

from ..maze.maze import Maze, WALL

Color = Tuple[int, int, int]
FCell = Tuple[float, float]   # fractional (row, col)
Cell = Tuple[int, int]

# -- Flying Dots palette -----------------------------------------------------
# Colors below were tuned against a WCAG contrast check (see
# tests/test_visual_contrast.py): >=3:1 for graphical markers, >=4.5:1 for
# normal-size text. Values differ from the original "Flying Dots" reference
# where needed to clear those thresholds against the cyan maze floor / panel
# chrome; the overall look is kept as close to the reference as contrast
# allows.
CYAN: Color = (126, 247, 247)          # maze background / open corridors
WALL_BLUE: Color = (0, 0, 255)         # walls
GOAL_RED: Color = (255, 0, 0)          # target square
START_GREEN: Color = (0, 120, 60)      # start square
DOT_BLACK: Color = (0, 0, 0)           # population dots
LEADER_RING: Color = (170, 112, 0)     # ring around best-of-generation
PATH_COLOR: Color = (196, 78, 0)       # best path found so far
MARKER_EDGE: Color = (255, 255, 255)   # keeps start/goal legible under the
                                        # dense black swarm

# -- window chrome (kept light so the cyan mazes read cleanly) ---------------
WINDOW_BG: Color = (233, 237, 242)
PANEL_BG: Color = (247, 249, 252)
TITLE_BG: Color = (255, 255, 255)
TITLE_BG_SOLVED: Color = (214, 245, 224)
TITLE_TEXT: Color = (28, 32, 44)
TITLE_DIM: Color = (88, 95, 110)
CHIP_BG: Color = (223, 228, 235)
GOOD: Color = (0, 125, 66)
BUTTON_BLUE: Color = (21, 101, 160)
BUTTON_DISABLED: Color = (203, 208, 215)
BUTTON_DISABLED_TEXT: Color = (99, 105, 118)
BORDER_OK: Color = (23, 140, 75)

METHOD_ACCENT = {
    "GA": (31, 110, 170),
    "Q-Learning": (183, 88, 10),
    "A*": (23, 140, 75),
}

PAD = 14
HEADER_H = 74
FOOTER_H = 30
TITLE_H = 56
INNER = 8
NEXT_BUTTON_W = 150
NEXT_BUTTON_H = 24
SUMMARY_W = 270


class Renderer:
    """Draws a 2x2 grid of Flying-Dots maze panels plus a shared header/footer."""

    def __init__(self, maze: Maze, cols: int = 2, rows: int = 2, target_cell_area: int = 380):
        self.cols = cols
        self.rows = rows
        self.target = target_cell_area
        self._compute_geometry(maze)

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Flying Dots - GA vs Q-Learning vs A*")

        self.font = pygame.font.SysFont("menlo,consolas,monospace", 15)
        self.small = pygame.font.SysFont("menlo,consolas,monospace", 13)
        self.title_font = pygame.font.SysFont("menlo,consolas,monospace", 17, bold=True)
        self.header_font = pygame.font.SysFont("menlo,consolas,monospace", 20, bold=True)

    # -- geometry -----------------------------------------------------------
    def _compute_geometry(self, maze: Maze) -> None:
        self.maze = maze
        dim = max(maze.width, maze.height)
        self.cell = max(8, self.target // dim)
        self.grid_w = self.cell * maze.width
        self.grid_h = self.cell * maze.height

        self.panel_w = self.grid_w + 2 * INNER
        self.panel_h = TITLE_H + self.grid_h + 2 * INNER

        self.panels_width = self.cols * (self.panel_w + PAD)
        self.width = PAD + self.panels_width + SUMMARY_W + PAD
        self.height = HEADER_H + PAD + self.rows * (self.panel_h + PAD) + FOOTER_H

    def set_maze(self, maze: Maze) -> None:
        self._compute_geometry(maze)

    def panel_rect(self, index: int) -> pygame.Rect:
        col = index % self.cols
        row = index // self.cols
        x = PAD + col * (self.panel_w + PAD)
        y = HEADER_H + PAD + row * (self.panel_h + PAD)
        return pygame.Rect(x, y, self.panel_w, self.panel_h)

    def summary_rect(self) -> pygame.Rect:
        return pygame.Rect(
            PAD + self.panels_width,
            HEADER_H + PAD,
            SUMMARY_W,
            self.panel_h,
        )

    def _maze_origin(self, panel: pygame.Rect) -> Tuple[int, int]:
        return panel.x + INNER, panel.y + TITLE_H + INNER

    @staticmethod
    def _px(ox: int, oy: int, cell: int, fr: float, fc: float) -> Tuple[float, float]:
        return ox + (fc + 0.5) * cell, oy + (fr + 0.5) * cell

    # -- frame lifecycle ----------------------------------------------------
    def begin_frame(self) -> None:
        self.screen.fill(WINDOW_BG)

    def end_frame(self) -> None:
        pygame.display.flip()

    # -- header / footer ----------------------------------------------------
    def draw_header(self, title: str, chips: List[str], progress: float) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, pygame.Rect(0, 0, self.width, HEADER_H))
        pygame.draw.line(self.screen, (208, 214, 222), (0, HEADER_H), (self.width, HEADER_H), 1)

        surf = self.header_font.render(title, True, TITLE_TEXT)
        self.screen.blit(surf, (PAD + 2, 12))

        x, y = PAD + 2, 44
        for chip in chips:
            csurf = self.small.render(chip, True, TITLE_DIM)
            w = csurf.get_width() + 18
            pygame.draw.rect(self.screen, CHIP_BG, pygame.Rect(x, y, w, 22), border_radius=11)
            self.screen.blit(csurf, (x + 9, y + 4))
            x += w + 8

        bw, bx, by = 220, self.width - 220 - PAD, 26
        pygame.draw.rect(self.screen, CHIP_BG, pygame.Rect(bx, by, bw, 10), border_radius=5)
        fill = int(bw * max(0.0, min(1.0, progress)))
        if fill > 0:
            pygame.draw.rect(self.screen, METHOD_ACCENT["Q-Learning"], pygame.Rect(bx, by, fill, 10), border_radius=5)
        self.screen.blit(self.small.render("playback", True, TITLE_DIM), (bx, by - 16))

    def draw_footer(self, text: str, next_enabled: bool = True) -> None:
        surf = self.small.render(text, True, TITLE_DIM)
        self.screen.blit(surf, (PAD + 2, self.height - FOOTER_H + 6))

        button = self.next_button_rect()
        fill = BUTTON_BLUE if next_enabled else BUTTON_DISABLED
        text_color = (255, 255, 255) if next_enabled else BUTTON_DISABLED_TEXT
        pygame.draw.rect(self.screen, fill, button, border_radius=6)
        label = self.small.render("NEXT ROUND", True, text_color)
        self.screen.blit(label, label.get_rect(center=button.center))

    def draw_summary(
        self, metrics: Dict[str, Dict[str, object]], finished: bool
    ) -> None:
        """Draw final per-method metrics in the right-hand sidebar."""
        rect = self.summary_rect()
        pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=8)
        pygame.draw.rect(
            self.screen,
            BORDER_OK if finished else (180, 188, 200),
            rect,
            width=2,
            border_radius=8,
        )

        title = self.title_font.render("FINAL RESULTS", True, TITLE_TEXT)
        self.screen.blit(title, (rect.x + 14, rect.y + 12))
        subtitle = self.small.render(
            "all methods complete" if finished else "waiting for completion",
            True,
            GOOD if finished else TITLE_DIM,
        )
        self.screen.blit(subtitle, (rect.x + 14, rect.y + 36))

        y = rect.y + 66
        for method in ("GA", "Q-Learning", "A*"):
            data = metrics.get(method, {})
            accent = METHOD_ACCENT[method]
            block_h = 100
            pygame.draw.rect(
                self.screen,
                accent,
                pygame.Rect(rect.x + 14, y, 5, block_h - 8),
                border_radius=2,
            )
            name = self.font.render(method, True, TITLE_TEXT)
            self.screen.blit(name, (rect.x + 28, y + 2))

            solved = data.get("solved")
            status = "SOLVED" if solved else (
                "FAILED" if data.get("completed") else "running..."
            )
            status_color = GOOD if solved else TITLE_DIM
            self.screen.blit(
                self.small.render(status, True, status_color),
                (rect.x + 28, y + 23),
            )

            lines = [
                f"iterations: {data.get('iterations', '-')}",
                f"evaluations: {data.get('evaluations', '-')}",
                f"path: {data.get('path_length', '-')}",
                f"optimality: {data.get('optimality', '-')}",
                f"time: {data.get('time_s', '-')}",
            ]
            line_y = y + 40
            for line in lines:
                self.screen.blit(
                    self.small.render(line, True, TITLE_DIM),
                    (rect.x + 28, line_y),
                )
                line_y += 11
            y += block_h

    def next_button_rect(self) -> pygame.Rect:
        """Clickable rectangle for advancing one generation/training batch."""
        return pygame.Rect(
            self.width - PAD - NEXT_BUTTON_W,
            self.height - FOOTER_H + 3,
            NEXT_BUTTON_W,
            NEXT_BUTTON_H,
        )

    # -- panel --------------------------------------------------------------
    def draw_panel(
        self,
        index: int,
        method: str,
        agents: Sequence[FCell],
        leader: Optional[FCell],
        best_path: Optional[Sequence[Cell]],
        stats: List[Tuple[str, str]],
        solved: bool,
    ) -> None:
        panel = self.panel_rect(index)
        accent = METHOD_ACCENT.get(method, BUTTON_BLUE)

        pygame.draw.rect(self.screen, PANEL_BG, panel, border_radius=8)
        pygame.draw.rect(self.screen, accent, panel, width=2, border_radius=8)
        self._draw_title(panel, method, accent, stats, solved)

        ox, oy = self._maze_origin(panel)
        self._draw_maze(ox, oy)
        if best_path:
            self._draw_path(ox, oy, best_path)
        self._draw_swarm(ox, oy, agents)
        # Markers are drawn *after* the swarm: hundreds of agents start every
        # round on the start cell, and their stacked semi-transparent dots
        # would otherwise bury the marker underneath.
        self._draw_markers(ox, oy)
        if leader is not None:
            self._draw_leader(ox, oy, leader)

    def _draw_title(self, panel, method, accent, stats, solved) -> None:
        title_rect = pygame.Rect(panel.x, panel.y, panel.width, TITLE_H)
        bg = TITLE_BG_SOLVED if solved else TITLE_BG
        pygame.draw.rect(self.screen, bg, title_rect, border_top_left_radius=8, border_top_right_radius=8)
        pygame.draw.rect(self.screen, accent, pygame.Rect(panel.x, panel.y, 6, TITLE_H), border_top_left_radius=8)

        name = self.title_font.render(method, True, TITLE_TEXT)
        self.screen.blit(name, (panel.x + 16, panel.y + 6))

        if solved:
            cx = panel.x + 16 + name.get_width() + 14
            cy = panel.y + 16
            pygame.draw.circle(self.screen, GOOD, (cx, cy), 9)
            pygame.draw.lines(self.screen, (255, 255, 255), False,
                              [(cx - 4, cy), (cx - 1, cy + 3), (cx + 4, cy - 4)], 2)

        text = "  ".join(f"{k} {v}" for k, v in stats)
        surf = self.small.render(text, True, TITLE_DIM)
        self.screen.blit(surf, (panel.x + 16, panel.y + 34))

    # -- maze ---------------------------------------------------------------
    def _draw_maze(self, ox: int, oy: int) -> None:
        # Cyan corridors on a solid cyan field, pure-blue wall blocks on top.
        pygame.draw.rect(self.screen, CYAN, pygame.Rect(ox, oy, self.grid_w, self.grid_h))
        for r in range(self.maze.height):
            for c in range(self.maze.width):
                if self.maze.grid[r, c] == WALL:
                    rect = pygame.Rect(ox + c * self.cell, oy + r * self.cell, self.cell, self.cell)
                    pygame.draw.rect(self.screen, WALL_BLUE, rect)

    def _draw_markers(self, ox: int, oy: int) -> None:
        edge = max(1, self.cell // 8)

        sr, sc = self.maze.start
        srect = pygame.Rect(ox + sc * self.cell, oy + sr * self.cell, self.cell, self.cell)
        pygame.draw.rect(self.screen, START_GREEN, srect)
        pygame.draw.rect(self.screen, MARKER_EDGE, srect, width=edge)

        gr, gc = self.maze.goal
        grect = pygame.Rect(ox + gc * self.cell, oy + gr * self.cell, self.cell, self.cell)
        pygame.draw.rect(self.screen, GOAL_RED, grect)
        pygame.draw.rect(self.screen, MARKER_EDGE, grect, width=edge)

    # -- overlays -----------------------------------------------------------
    def _overlay(self) -> pygame.Surface:
        return pygame.Surface((self.grid_w, self.grid_h), pygame.SRCALPHA)

    def _draw_path(self, ox: int, oy: int, path: Sequence[Cell]) -> None:
        if len(path) < 2:
            return
        overlay = self._overlay()
        pts = [self._px(0, 0, self.cell, r, c) for (r, c) in path]
        pygame.draw.lines(overlay, (*PATH_COLOR, 170), False, pts, max(2, self.cell // 6))
        self.screen.blit(overlay, (ox, oy))

    def _draw_swarm(self, ox: int, oy: int, agents: Sequence[FCell]) -> None:
        if not agents:
            return
        overlay = self._overlay()
        size = max(3, int(self.cell * 0.42))
        half = size / 2
        alpha = 80 if len(agents) > 60 else (150 if len(agents) > 1 else 255)
        color = (*DOT_BLACK, alpha)
        for fr, fc in agents:
            x, y = self._px(0, 0, self.cell, fr, fc)
            pygame.draw.rect(overlay, color, pygame.Rect(x - half, y - half, size, size))
        self.screen.blit(overlay, (ox, oy))

    def _draw_leader(self, ox: int, oy: int, leader: FCell) -> None:
        x, y = self._px(ox, oy, self.cell, leader[0], leader[1])
        size = max(4, int(self.cell * 0.5))
        half = size / 2
        pygame.draw.circle(self.screen, LEADER_RING, (int(x), int(y)), int(half) + 3, 2)
        pygame.draw.rect(self.screen, DOT_BLACK, pygame.Rect(x - half, y - half, size, size))
