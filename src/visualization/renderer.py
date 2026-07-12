"""Pygame rendering of the maze, the evolving population, and a HUD.

Design goals (readability first):
* Big maze cells so corridors are obvious.
* The population is drawn as a translucent "cloud" of gliding dots; where many
  agents overlap the colour gets denser, so you can read crowd behaviour.
* The current generation's best agent (the "leader") is a large, outlined gold
  dot that is easy to follow.
* The best solution found so far is traced as a gold path.
* A clear side panel shows a colour legend, live stats and the controls.

Kept free of evolution logic so it can be reused/tested independently.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import pygame

from ..maze.maze import Maze, WALL

Color = Tuple[int, int, int]
FCell = Tuple[float, float]   # fractional (row, col)
Cell = Tuple[int, int]

# Palette
BG: Color = (15, 17, 26)
WALL_COLOR: Color = (33, 37, 54)
OPEN_COLOR: Color = (238, 240, 248)
GRID_COLOR: Color = (214, 217, 230)
START_COLOR: Color = (46, 204, 113)
GOAL_COLOR: Color = (231, 76, 60)
GOAL_RING: Color = (231, 76, 60)
AGENT_COLOR: Color = (52, 152, 219)
LEADER_COLOR: Color = (241, 196, 15)
LEADER_OUTLINE: Color = (255, 255, 255)
PATH_COLOR: Color = (241, 196, 15)
HUD_BG: Color = (22, 24, 35)
HUD_PANEL: Color = (28, 31, 45)
HUD_TEXT: Color = (232, 234, 242)
HUD_DIM: Color = (155, 159, 175)
HUD_ACCENT: Color = (241, 196, 15)


class Renderer:
    def __init__(self, maze: Maze, maze_area_px: int = 760, hud_width: int = 320):
        self.hud_width = hud_width
        self.maze_area_px = maze_area_px
        self._fit(maze)

        self.width = self.maze_px + hud_width
        self.height = max(self.maze_px, 560)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Maze Evolution - GA vs NEAT")

        self.font = pygame.font.SysFont("menlo,consolas,monospace", 16)
        self.small = pygame.font.SysFont("menlo,consolas,monospace", 14)
        self.big = pygame.font.SysFont("menlo,consolas,monospace", 26, bold=True)
        self._tick = 0

    # -- geometry -------------------------------------------------------
    def _fit(self, maze: Maze) -> None:
        self.maze = maze
        self.cell = max(6, self.maze_area_px // max(maze.width, maze.height))
        self.grid_w = self.cell * maze.width
        self.grid_h = self.cell * maze.height
        self.maze_px = self.cell * max(maze.width, maze.height)
        self.ox = (self.maze_px - self.grid_w) // 2
        self.oy = (self.maze_px - self.grid_h) // 2

    def set_maze(self, maze: Maze) -> None:
        self._fit(maze)

    def _px(self, fr: float, fc: float) -> Tuple[float, float]:
        """Pixel center for a fractional (row, col)."""
        return (
            self.ox + (fc + 0.5) * self.cell,
            self.oy + (fr + 0.5) * self.cell,
        )

    def _cell_rect(self, cell: Cell) -> pygame.Rect:
        r, c = cell
        return pygame.Rect(
            self.ox + c * self.cell, self.oy + r * self.cell, self.cell, self.cell
        )

    # -- public draw ----------------------------------------------------
    def draw(
        self,
        agents: Sequence[FCell],
        leader: Optional[FCell],
        best_path: Optional[Sequence[Cell]],
        hud_lines: List[Tuple[str, bool]],
        progress: float = 0.0,
    ) -> None:
        self._tick += 1
        self.screen.fill(BG)
        self._draw_maze()
        if best_path:
            self._draw_path(best_path)
        self._draw_markers()
        self._draw_agents(agents)
        if leader is not None:
            self._draw_leader(leader)
        self._draw_hud(hud_lines, progress)
        pygame.display.flip()

    # -- maze -----------------------------------------------------------
    def _draw_maze(self) -> None:
        # Walls form the dark background; open cells are carved out light.
        pygame.draw.rect(
            self.screen, WALL_COLOR, pygame.Rect(self.ox, self.oy, self.grid_w, self.grid_h)
        )
        for r in range(self.maze.height):
            for c in range(self.maze.width):
                if self.maze.grid[r, c] != WALL:
                    pygame.draw.rect(self.screen, OPEN_COLOR, self._cell_rect((r, c)))
        # Light grid lines on the open area (only if cells are big enough).
        if self.cell >= 14:
            for r in range(self.maze.height + 1):
                y = self.oy + r * self.cell
                pygame.draw.line(self.screen, GRID_COLOR, (self.ox, y), (self.ox + self.grid_w, y), 1)
            for c in range(self.maze.width + 1):
                x = self.ox + c * self.cell
                pygame.draw.line(self.screen, GRID_COLOR, (x, self.oy), (x, self.oy + self.grid_h), 1)

    def _draw_markers(self) -> None:
        # Start: rounded green tile with an "S".
        srect = self._cell_rect(self.maze.start).inflate(-self.cell // 6, -self.cell // 6)
        pygame.draw.rect(self.screen, START_COLOR, srect, border_radius=max(2, self.cell // 4))
        self._blit_centered("S", self.maze.start, self.cell)

        # Goal: red disc with a pulsing ring (the target).
        gx, gy = self._px(self.maze.goal[0], self.maze.goal[1])
        base = self.cell * 0.42
        pulse = base + math.sin(self._tick * 0.12) * (self.cell * 0.12)
        pygame.draw.circle(self.screen, GOAL_RING, (int(gx), int(gy)), int(pulse), 2)
        pygame.draw.circle(self.screen, GOAL_COLOR, (int(gx), int(gy)), int(base))
        self._blit_centered("G", self.maze.goal, self.cell, color=(255, 255, 255))

    def _blit_centered(self, text: str, cell: Cell, size: int, color: Color = (20, 40, 25)) -> None:
        if self.cell < 16:
            return
        font = pygame.font.SysFont("menlo,consolas,monospace", max(10, int(size * 0.5)), bold=True)
        surf = font.render(text, True, color)
        cx, cy = self._px(cell[0], cell[1])
        self.screen.blit(surf, surf.get_rect(center=(cx, cy)))

    # -- agents ---------------------------------------------------------
    def _draw_path(self, path: Sequence[Cell]) -> None:
        if len(path) < 2:
            return
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        points = [self._px(r, c) for (r, c) in path]
        pygame.draw.lines(overlay, (*PATH_COLOR, 150), False, points, max(2, self.cell // 6))
        self.screen.blit(overlay, (0, 0))

    def _draw_agents(self, agents: Sequence[FCell]) -> None:
        if not agents:
            return
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        radius = max(2, int(self.cell * 0.22))
        alpha = 70 if len(agents) > 60 else 130
        color = (*AGENT_COLOR, alpha)
        for fr, fc in agents:
            x, y = self._px(fr, fc)
            pygame.draw.circle(overlay, color, (int(x), int(y)), radius)
        self.screen.blit(overlay, (0, 0))

    def _draw_leader(self, leader: FCell) -> None:
        x, y = self._px(leader[0], leader[1])
        r = max(4, int(self.cell * 0.34))
        pygame.draw.circle(self.screen, LEADER_OUTLINE, (int(x), int(y)), r + 2)
        pygame.draw.circle(self.screen, LEADER_COLOR, (int(x), int(y)), r)

    # -- HUD ------------------------------------------------------------
    def _draw_hud(self, lines: List[Tuple[str, bool]], progress: float) -> None:
        x0 = self.maze_px
        pygame.draw.rect(self.screen, HUD_BG, pygame.Rect(x0, 0, self.hud_width, self.height))

        pad = 18
        y = 16

        # Legend swatches.
        legend = [
            (AGENT_COLOR, "population"),
            (LEADER_COLOR, "best of gen"),
            (START_COLOR, "start"),
            (GOAL_COLOR, "goal"),
        ]
        for color, label in legend:
            pygame.draw.circle(self.screen, color, (x0 + pad + 6, y + 8), 6)
            surf = self.small.render(label, True, HUD_DIM)
            self.screen.blit(surf, (x0 + pad + 22, y))
            y += 22
        y += 8

        for text, is_header in lines:
            if text == "":
                y += 10
                continue
            if is_header:
                surf = self.big.render(text, True, HUD_ACCENT) if y < 200 else self.font.render(text, True, HUD_TEXT)
                self.screen.blit(surf, (x0 + pad, y))
                y += surf.get_height() + 6
            else:
                surf = self.font.render(text, True, HUD_DIM)
                self.screen.blit(surf, (x0 + pad, y))
                y += surf.get_height() + 4

        # Generation animation progress bar at the bottom.
        bar_y = self.height - 34
        bar_w = self.hud_width - 2 * pad
        pygame.draw.rect(self.screen, HUD_PANEL, pygame.Rect(x0 + pad, bar_y, bar_w, 10), border_radius=5)
        fill = int(bar_w * max(0.0, min(1.0, progress)))
        if fill > 0:
            pygame.draw.rect(self.screen, HUD_ACCENT, pygame.Rect(x0 + pad, bar_y, fill, 10), border_radius=5)
        label = self.small.render("generation playback", True, HUD_DIM)
        self.screen.blit(label, (x0 + pad, bar_y - 18))
