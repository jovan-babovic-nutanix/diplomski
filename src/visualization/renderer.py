"""Pygame rendering for the side-by-side method comparison — "Race Control".

A dark motorsport/mission-control telemetry HUD: cut-corner panel cards, a
grid-textured background, glowing accent markers per method, and a live
progress rail. Visual reference: ``scratch/maze-arena-directions.html``,
``data-dir="a"`` (see the redesign audit for the two other directions that
were not chosen).

The renderer holds no algorithm logic: the app feeds it positions/stats, it
draws. Colors were chosen and then verified against WCAG contrast thresholds
(>=3:1 for graphical markers, >=4.5:1 for normal-size text) - see
``tests/test_visual_contrast.py`` for the actual asserted pairs. Two mockup
tokens were adjusted from their raw CSS values because a flat pygame fill
doesn't get the browser's ``color-mix()``/alpha compositing for free: the
faint-ink text color was lightened, and maze-wall contrast is carried by a
wall/floor boundary outline (``WALL_EDGE``) rather than by the fill colors
themselves, so the corridor can stay dark enough for the glow effects to read.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pygame

try:
    import pygame.gfxdraw as _gfxdraw
    _HAS_GFX = True
except ImportError:  # pragma: no cover - depends on the local SDL build
    _gfxdraw = None
    _HAS_GFX = False

from ..maze.maze import Maze, WALL

Color = Tuple[int, int, int]
FCell = Tuple[float, float]   # fractional (row, col)
Cell = Tuple[int, int]


def _blend(fg: Color, bg: Color, a: float) -> Color:
    """Flatten a translucent foreground over an opaque background (the pygame
    equivalent of the mockup's CSS ``color-mix()``), so the composited pixel
    color - not just the raw fg hex - is what gets contrast-checked."""
    return tuple(round(f * a + b * (1 - a)) for f, b in zip(fg, bg))  # type: ignore[return-value]


# -- Race Control palette -----------------------------------------------
WINDOW_BG: Color = (10, 14, 20)
PANEL_BG: Color = (18, 24, 31)
LINE: Color = (35, 43, 54)
TRACK: Color = (27, 34, 44)
INK: Color = (238, 243, 250)
INK_DIM: Color = (154, 166, 184)
INK_FAINT: Color = (124, 138, 158)   # raised from the mockup's #657286 to clear 4.5:1
GOOD: Color = (34, 224, 160)
GOAL_RED: Color = (255, 93, 93)
GRID_TEX: Color = (21, 27, 35)

METHOD_ACCENT: Dict[str, Color] = {
    "GA": (47, 140, 224),
    "Q-Learning": (194, 96, 15),
    "A*": (21, 148, 103),
}
# Brighter variants for drawing *on* the dark maze floor (dots, path, leader).
METHOD_ON_MAZE: Dict[str, Color] = {
    "GA": (143, 196, 242),
    "Q-Learning": (255, 157, 77),
    "A*": (63, 224, 173),
}

MAZE_FLOOR: Color = (24, 34, 48)
MAZE_WALL: Color = (5, 7, 9)
WALL_EDGE: Color = (106, 125, 150)   # carries maze structure; see module docstring
START_GREEN: Color = GOOD
MARKER_EDGE: Color = (5, 8, 12)      # dark ring: separates markers from the bright swarm
LEADER_RING: Color = INK

STATUS_RUNNING_BG: Color = _blend(INK_DIM, PANEL_BG, 0.12)
STATUS_SOLVED_BG: Color = _blend(GOOD, PANEL_BG, 0.16)
STATUS_FAILED_BG: Color = _blend(GOAL_RED, PANEL_BG, 0.16)
TAG_LEARNING_BG: Color = _blend(METHOD_ACCENT["Q-Learning"], PANEL_BG, 0.22)
TAG_LEARNING_LINE: Color = _blend(METHOD_ACCENT["Q-Learning"], PANEL_BG, 0.55)
TAG_REFERENCE_BG: Color = _blend(METHOD_ACCENT["A*"], PANEL_BG, 0.22)
TAG_REFERENCE_LINE: Color = _blend(METHOD_ACCENT["A*"], PANEL_BG, 0.55)
ON_ACCENT: Color = WINDOW_BG
RAIL_C0, RAIL_C1 = METHOD_ACCENT["GA"], METHOD_ACCENT["A*"]

SWARM_ALPHA_DENSE, SWARM_ALPHA_FEW, SWARM_ALPHA_ONE = 140, 190, 255
GLOW_STEPS: Tuple[Tuple[int, int], ...] = ((6, 28), (4, 52), (2, 84))

DISPLAY_NAME = {"GA": "GENETIC ALGORITHM", "Q-Learning": "Q-LEARNING", "A*": "A* SEARCH"}
SHORT_NAME = {"GA": "GA", "Q-Learning": "Q-LEARNING", "A*": "A* SEARCH"}
METHOD_TAG = {"GA": "LEARNING", "Q-Learning": "LEARNING", "A*": "REFERENCE"}

# -- geometry -------------------------------------------------------------
PAD = 16
GAP = 14
CORNER_CUT = 10
BTN_CUT = 6
GRID_TEX_STEP = 26

TOPBAR_TOP = 16
TOPBAR_H = 44
RAIL_GAP = 12
RAIL_H = 6
HEADER_H = TOPBAR_TOP + TOPBAR_H + RAIL_GAP + RAIL_H + 20

PANEL_PAD = 12
PANEL_GAP = 10
HEAD_H = 26
STATS_H = 36
STATUS_H = 26
MIN_PANEL_W = 300

SIDEBAR_W = 264
SIDEBAR_MIN_H = 400

FOOTER_GAP = 14
BTN_H = 30
LEGEND_H = 16
FOOTER_H = 1 + 13 + BTN_H + 10 + LEGEND_H + 12
CHIP_H = 24
SPEED_LABEL_W = 90

# -- fonts ------------------------------------------------------------------
FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"
_FONT_FILES = {
    ("display", 600): "ChakraPetch-SemiBold.ttf",
    ("display", 700): "ChakraPetch-Bold.ttf",
    ("body", 500): "IBMPlexSans-Medium.ttf",
    ("body", 600): "IBMPlexSans-SemiBold.ttf",
    ("mono", 500): "IBMPlexMono-Medium.ttf",
    ("mono", 600): "IBMPlexMono-SemiBold.ttf",
}
_SYS_FALLBACK = {
    "display": "bahnschrift,segoeui,dejavusans,arial",
    "body": "segoeui,dejavusans,arial",
    "mono": "consolas,menlo,dejavusansmono,monospace",
}


def load_font(family: str, weight: int, size: int) -> pygame.font.Font:
    """Bundled TTF (see ``assets/fonts/``) if present, else the previous
    SysFont stack - so tests and any environment without the asset files
    still render (just with a different font)."""
    filename = _FONT_FILES.get((family, weight))
    if filename:
        path = FONT_DIR / filename
        if path.is_file():
            try:
                return pygame.font.Font(str(path), size)
            except pygame.error:
                pass
    return pygame.font.SysFont(_SYS_FALLBACK[family], size, bold=weight >= 600)


def _cut_points(rect: pygame.Rect, cut: int) -> List[Tuple[int, int]]:
    """Vertices of a rect with its top-right and bottom-left corners cut off
    (the mockup's ``clip-path: polygon(...)``)."""
    cut = max(0, min(cut, rect.width // 2, rect.height // 2))
    x, y = rect.x, rect.y
    r, b = rect.right - 1, rect.bottom - 1
    return [(x, y), (r - cut, y), (r, y + cut), (r, b), (x + cut, b), (x, b - cut)]


def _draw_glow(surface: pygame.Surface, center: Tuple[int, int], radius: float,
               color: Color, steps: Tuple[Tuple[int, int], ...] = GLOW_STEPS) -> None:
    """Fake a CSS ``box-shadow`` glow with concentric alpha rings on an
    SRCALPHA surface. Order matters: largest/dimmest ring first - drawing a
    circle on an SRCALPHA surface *replaces* pixel alpha rather than
    accumulating it, so a smaller/brighter ring painted afterwards is what
    produces the falloff."""
    for extra, alpha in steps:
        pygame.draw.circle(surface, (*color, alpha), center, int(radius + extra))


def _abbrev(n: object) -> str:
    if not isinstance(n, (int, float)):
        return "-"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(int(n))


class Renderer:
    """Draws the Race Control HUD: a header/rail, N method panels, a
    leaderboard sidebar, and a control-bar footer."""

    def __init__(self, maze: Maze, cols: int = 2, rows: int = 2, target_cell_area: int = 380):
        self.cols = cols
        self.rows = rows
        self.target = target_cell_area
        self._text_cache: Dict[tuple, pygame.Surface] = {}
        self._overlay_surf: Optional[pygame.Surface] = None
        self._compute_geometry(maze)

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("MAZE ARENA — Race Control | GA vs Q-Learning vs A*")

        self._load_fonts()

    def _load_fonts(self) -> None:
        self.f_brand = load_font("display", 700, 22)
        self.f_brand_sub = load_font("mono", 500, 11)
        self.f_chip_k = load_font("mono", 500, 10)
        self.f_chip_v = load_font("mono", 600, 13)
        self.f_live = load_font("mono", 600, 11)
        self.f_panel_name = load_font("display", 600, 16)
        self._f_panel_name_sm = load_font("display", 600, 13)
        self.f_tag = load_font("mono", 600, 10)
        self.f_stat_k = load_font("mono", 500, 10)
        self.f_stat_v = load_font("mono", 600, 15)
        self.f_status = load_font("body", 500, 13)
        self.f_status_b = load_font("body", 600, 13)
        self.f_side_h = load_font("display", 600, 14)
        self.f_side_note = load_font("mono", 500, 10)
        self.f_metric_h = load_font("mono", 500, 10)
        self.f_bar = load_font("mono", 500, 11)
        self.f_btn = load_font("mono", 600, 12)
        self.f_legend = load_font("mono", 500, 11)
        self.f_detail = load_font("mono", 500, 10)

    # -- tracked (letter-spaced) text, cached ------------------------------
    def _tracked_size(self, font: pygame.font.Font, s: str, tracking: int = 0) -> Tuple[int, int]:
        if tracking <= 0 or len(s) <= 1:
            return font.size(s)
        widths = [font.size(ch)[0] for ch in s]
        return (sum(widths) + tracking * (len(s) - 1), font.get_height())

    def _text(self, font: pygame.font.Font, s: str, color: Color, tracking: int = 0) -> pygame.Surface:
        key = (id(font), s, color, tracking)
        surf = self._text_cache.get(key)
        if surf is not None:
            return surf
        if tracking <= 0 or len(s) <= 1:
            surf = font.render(s, True, color)
        else:
            glyphs = [font.render(ch, True, color) for ch in s]
            w, h = self._tracked_size(font, s, tracking)
            surf = pygame.Surface((max(1, w), h), pygame.SRCALPHA)
            x = 0
            for g in glyphs:
                surf.blit(g, (x, 0))
                x += g.get_width() + tracking
        if len(self._text_cache) > 512:
            self._text_cache.clear()
        self._text_cache[key] = surf
        return surf

    # -- geometry -----------------------------------------------------------
    def _compute_geometry(self, maze: Maze) -> None:
        self.maze = maze
        dim = max(maze.width, maze.height)
        self.cell = max(8, self.target // dim)
        self.grid_w = self.cell * maze.width
        self.grid_h = self.cell * maze.height

        self.panel_w = max(MIN_PANEL_W, self.grid_w + 2 * PANEL_PAD)
        self.panel_h = (
            self.grid_h + 2 * PANEL_PAD + HEAD_H + STATS_H + STATUS_H + 3 * PANEL_GAP
        )
        self.body_h = max(self.panel_h, SIDEBAR_MIN_H)

        self.panels_width = self.cols * (self.panel_w + GAP)
        self.width = PAD + self.panels_width + SIDEBAR_W + PAD
        self.height = HEADER_H + self.body_h + FOOTER_GAP + FOOTER_H

        # Invalidate every cached surface/measurement - safe to call before
        # self.screen exists (they're built lazily, after set_mode).
        self._background: Optional[pygame.Surface] = None
        self._maze_surface: Optional[pygame.Surface] = None
        self._rail_gradient: Optional[pygame.Surface] = None
        self._control_rects: Optional[Dict[str, pygame.Rect]] = None
        self._overlay_surf = None
        self._text_cache.clear()

    def set_maze(self, maze: Maze) -> None:
        self._compute_geometry(maze)

    def panel_rect(self, index: int) -> pygame.Rect:
        col = index % self.cols
        row = index // self.cols
        x = PAD + col * (self.panel_w + GAP)
        y = HEADER_H + row * (self.panel_h + GAP)
        return pygame.Rect(x, y, self.panel_w, self.panel_h)

    def summary_rect(self) -> pygame.Rect:
        return pygame.Rect(PAD + self.panels_width, HEADER_H, SIDEBAR_W, self.body_h)

    def _maze_origin(self, panel: pygame.Rect) -> Tuple[int, int]:
        ox = panel.x + (self.panel_w - self.grid_w) // 2
        oy = panel.y + PANEL_PAD + HEAD_H + PANEL_GAP
        return ox, oy

    @staticmethod
    def _px(ox: int, oy: int, cell: int, fr: float, fc: float) -> Tuple[float, float]:
        return ox + (fc + 0.5) * cell, oy + (fr + 0.5) * cell

    # -- cached surfaces (built lazily: geometry runs before set_mode(), and
    # Surface.convert() needs a display to already exist) --------------------
    def _build_background(self) -> pygame.Surface:
        surf = pygame.Surface((self.width, self.height))
        surf.fill(WINDOW_BG)
        for x in range(0, self.width, GRID_TEX_STEP):
            pygame.draw.line(surf, GRID_TEX, (x, 0), (x, self.height))
        for y in range(0, self.height, GRID_TEX_STEP):
            pygame.draw.line(surf, GRID_TEX, (0, y), (self.width, y))
        return surf.convert()

    def _rail_gradient_surface(self) -> pygame.Surface:
        if self._rail_gradient is None:
            w = max(1, self.width - 2 * PAD)
            surf = pygame.Surface((w, RAIL_H))
            for x in range(w):
                t = x / max(1, w - 1)
                color = tuple(round(RAIL_C0[i] + (RAIL_C1[i] - RAIL_C0[i]) * t) for i in range(3))
                pygame.draw.line(surf, color, (x, 0), (x, RAIL_H - 1))
            self._rail_gradient = surf.convert()
        return self._rail_gradient

    def _build_maze_surface(self) -> pygame.Surface:
        s = pygame.Surface((self.grid_w, self.grid_h))
        s.fill(MAZE_FLOOR)
        cell, grid = self.cell, self.maze.grid
        for r in range(self.maze.height):
            for c in range(self.maze.width):
                if grid[r, c] == WALL:
                    pygame.draw.rect(s, MAZE_WALL, pygame.Rect(c * cell, r * cell, cell, cell))
        # Outline only the wall/open boundary - this is the graphical object
        # that actually carries maze structure at readable contrast; see the
        # module docstring for why the fills themselves stay low-contrast.
        ew = 1 if cell < 14 else 2
        for r in range(self.maze.height):
            for c in range(self.maze.width):
                if grid[r, c] != WALL:
                    continue
                x, y = c * cell, r * cell
                if r == 0 or grid[r - 1, c] != WALL:
                    pygame.draw.line(s, WALL_EDGE, (x, y), (x + cell - 1, y), ew)
                if r == self.maze.height - 1 or grid[r + 1, c] != WALL:
                    pygame.draw.line(s, WALL_EDGE, (x, y + cell - 1), (x + cell - 1, y + cell - 1), ew)
                if c == 0 or grid[r, c - 1] != WALL:
                    pygame.draw.line(s, WALL_EDGE, (x, y), (x, y + cell - 1), ew)
                if c == self.maze.width - 1 or grid[r, c + 1] != WALL:
                    pygame.draw.line(s, WALL_EDGE, (x + cell - 1, y), (x + cell - 1, y + cell - 1), ew)
        return s.convert()

    def _overlay(self) -> pygame.Surface:
        """A reusable SRCALPHA scratch surface sized to one maze grid, cleared
        on every request. Safe to call repeatedly within one draw_panel() -
        each caller clears, draws, and blits before the next caller asks."""
        if self._overlay_surf is None or self._overlay_surf.get_size() != (self.grid_w, self.grid_h):
            self._overlay_surf = pygame.Surface((self.grid_w, self.grid_h), pygame.SRCALPHA)
        else:
            self._overlay_surf.fill((0, 0, 0, 0))
        return self._overlay_surf

    def _draw_cut_panel(self, rect: pygame.Rect, fill: Color, border: Color = LINE,
                         cut: int = CORNER_CUT, accent: Optional[Color] = None) -> None:
        pts = _cut_points(rect, cut)
        drew = False
        if _HAS_GFX:
            try:
                _gfxdraw.filled_polygon(self.screen, pts, fill)
                _gfxdraw.aapolygon(self.screen, pts, border)
                drew = True
            except Exception:
                drew = False
        if not drew:
            pygame.draw.polygon(self.screen, fill, pts)
            pygame.draw.polygon(self.screen, border, pts, 1)
        if accent:
            cut2 = max(0, min(cut, rect.width // 2, rect.height // 2))
            r, x, y = rect.right - 1, rect.x, rect.y
            pygame.draw.line(self.screen, accent, (x, y + 1), (r - cut2, y + 1), 2)
            pygame.draw.line(self.screen, accent, (r - cut2, y + 1), (r - 1, y + cut2), 2)

    # -- frame lifecycle ----------------------------------------------------
    def begin_frame(self) -> None:
        if self._background is None:
            self._background = self._build_background()
        self.screen.blit(self._background, (0, 0))

    def end_frame(self) -> None:
        pygame.display.flip()

    # -- header / rail --------------------------------------------------------
    def _draw_chip(self, x: int, y: int, key: str, value: str, width: int) -> None:
        r = pygame.Rect(x, y, width, CHIP_H)
        pygame.draw.rect(self.screen, PANEL_BG, r, border_radius=3)
        pygame.draw.rect(self.screen, LINE, r, width=1, border_radius=3)
        ks = self._text(self.f_chip_k, key.upper(), INK_FAINT, tracking=1)
        vs = self._text(self.f_chip_v, value, INK)
        self.screen.blit(ks, (x + 11, y + (CHIP_H - ks.get_height()) // 2))
        self.screen.blit(vs, (x + 11 + ks.get_width() + 6, y + (CHIP_H - vs.get_height()) // 2))

    def _chip_width(self, key: str, value: str) -> int:
        kw = self._tracked_size(self.f_chip_k, key.upper(), 1)[0]
        vw = self.f_chip_v.size(value)[0]
        return 11 + kw + 6 + vw + 11

    def draw_header(
        self,
        brand: str,
        subtitle: str,
        chips: List[Tuple[str, str]],
        state_label: str,
        state_color: Color,
        progress: float,
    ) -> None:
        self.screen.blit(self._text(self.f_brand, brand, INK, tracking=1), (PAD, TOPBAR_TOP))
        sub = self._text(self.f_brand_sub, subtitle.upper(), INK_DIM, tracking=1)
        self.screen.blit(sub, (PAD, TOPBAR_TOP + 26))

        state_text = self._text(self.f_live, state_label.upper(), state_color, tracking=1)
        pill_w = 24 + state_text.get_width()
        px = self.width - PAD - pill_w
        cy = TOPBAR_TOP + TOPBAR_H // 2
        cx = px + 7
        halo = _blend(state_color, WINDOW_BG, 0.25)
        pygame.draw.circle(self.screen, halo, (cx, cy), 7)
        pygame.draw.circle(self.screen, state_color, (cx, cy), 4)
        self.screen.blit(state_text, (px + 18, cy - state_text.get_height() // 2))

        widths = [self._chip_width(k, v) for k, v in chips]
        total_w = sum(widths) + 8 * max(0, len(chips) - 1)
        x = px - 20 - total_w
        y = TOPBAR_TOP + (TOPBAR_H - CHIP_H) // 2
        for (k, v), w in zip(chips, widths):
            self._draw_chip(x, y, k, v, w)
            x += w + 8

        ry = TOPBAR_TOP + TOPBAR_H + RAIL_GAP
        rail_w = self.width - 2 * PAD
        track = pygame.Rect(PAD, ry, rail_w, RAIL_H)
        pygame.draw.rect(self.screen, TRACK, track, border_radius=3)
        pygame.draw.rect(self.screen, LINE, track, width=1, border_radius=3)
        fill_w = int(rail_w * max(0.0, min(1.0, progress)))
        if fill_w > 2:
            grad = self._rail_gradient_surface()
            self.screen.blit(grad.subsurface((0, 0, min(fill_w, grad.get_width()), RAIL_H)), (PAD, ry))

    # -- control bar (footer) ------------------------------------------------
    def control_rects(self) -> Dict[str, pygame.Rect]:
        if self._control_rects is not None:
            return self._control_rects
        y = self.height - FOOTER_H + 1 + 13
        x = PAD
        rects: Dict[str, pygame.Rect] = {}

        def place(key: str, w: int) -> None:
            nonlocal x
            rects[key] = pygame.Rect(x, y, w, BTN_H)
            x += w + 10

        place("pause", 100)
        place("next", 140)
        place("speed_down", 28)
        x += SPEED_LABEL_W + 6
        place("speed_up", 28)
        place("new_maze", 110)
        place("fastfwd", 100)

        label = self._text(self.f_btn, "SHOW POPULATION", INK_DIM, tracking=1)
        show_w = 14 + 8 + label.get_width()
        rects["show_pop"] = pygame.Rect(self.width - PAD - show_w, y, show_w, BTN_H)

        self._control_rects = rects
        return rects

    def next_button_rect(self) -> pygame.Rect:
        return self.control_rects()["next"]

    def _draw_ctrl_button(self, rect: pygame.Rect, label: str, primary: bool = False,
                           enabled: bool = True, active: bool = False) -> None:
        if not enabled:
            fill, text_color, border = TRACK, INK_FAINT, LINE
        elif primary:
            fill, text_color, border = METHOD_ACCENT["GA"], ON_ACCENT, METHOD_ACCENT["GA"]
        elif active:
            fill, text_color, border = STATUS_SOLVED_BG, GOOD, GOOD
        else:
            fill, text_color, border = PANEL_BG, INK, LINE
        self._draw_cut_panel(rect, fill, border, cut=BTN_CUT)
        surf = self._text(self.f_btn, label, text_color, tracking=1)
        self.screen.blit(surf, surf.get_rect(center=rect.center))

    def _draw_checkbox(self, rect: pygame.Rect, checked: bool, label: str) -> None:
        box = pygame.Rect(rect.x, rect.y + (rect.height - 14) // 2, 14, 14)
        self._draw_cut_panel(box, GOOD if checked else PANEL_BG, LINE, cut=3)
        if checked:
            pts = [(box.x + 3, box.y + 7), (box.x + 6, box.y + 10), (box.x + 11, box.y + 3)]
            pygame.draw.lines(self.screen, WINDOW_BG, False, pts, 2)
        text = self._text(self.f_btn, label, INK_DIM, tracking=1)
        self.screen.blit(text, (box.right + 8, rect.y + (rect.height - text.get_height()) // 2))

    def draw_footer(self, state: Dict[str, object]) -> None:
        rects = self.control_rects()
        top = self.height - FOOTER_H
        pygame.draw.line(self.screen, LINE, (PAD, top), (self.width - PAD, top), 1)

        paused = bool(state.get("paused"))
        fast_forward = bool(state.get("fast_forward"))
        speed = float(state.get("speed", 1.0))
        next_enabled = bool(state.get("next_enabled", True))
        show_population = bool(state.get("show_population", True))
        legend = str(state.get("legend", ""))

        self._draw_ctrl_button(rects["pause"], "RESUME" if paused else "PAUSE")
        self._draw_ctrl_button(rects["next"], "NEXT ROUND", primary=True, enabled=next_enabled)
        self._draw_ctrl_button(rects["speed_down"], "-")
        self._draw_ctrl_button(rects["speed_up"], "+")

        speed_label = self._text(self.f_btn, f"SPEED {speed:.0f}×", INK_DIM, tracking=1)
        sx = rects["speed_down"].right + 6
        self.screen.blit(
            speed_label,
            (sx + max(0, (SPEED_LABEL_W - speed_label.get_width()) // 2),
             rects["speed_down"].y + (BTN_H - speed_label.get_height()) // 2),
        )

        self._draw_ctrl_button(rects["new_maze"], "NEW MAZE")
        self._draw_ctrl_button(rects["fastfwd"], "FAST-FWD", active=fast_forward)
        self._draw_checkbox(rects["show_pop"], show_population, "SHOW POPULATION")

        legend_y = rects["pause"].bottom + 10
        self.screen.blit(self._text(self.f_legend, legend, INK_FAINT), (PAD, legend_y))

    # -- sidebar / leaderboard ------------------------------------------------
    def _draw_bar_row(self, x: int, y: int, width: int, method: str, value_text: str, frac: float) -> int:
        label_w, gap, val_w = 64, 8, 40
        label = self._text(self.f_bar, {"GA": "GA", "Q-Learning": "Q-LEARN", "A*": "A*"}[method], INK_DIM)
        self.screen.blit(label, (x, y + 1))
        track_x = x + label_w
        track_w = max(1, width - label_w - val_w - gap * 2)
        track = pygame.Rect(track_x, y + 3, track_w, 8)
        pygame.draw.rect(self.screen, TRACK, track, border_radius=4)
        fill_w = int(track_w * max(0.0, min(1.0, frac)))
        if fill_w > 0:
            pygame.draw.rect(self.screen, METHOD_ACCENT[method], pygame.Rect(track.x, track.y, fill_w, 8), border_radius=4)
        val = self._text(self.f_bar, str(value_text), INK)
        self.screen.blit(val, (x + width - val.get_width(), y))
        return y + 15

    def draw_summary(self, metrics: Dict[str, Dict[str, object]], finished: bool, optimal: int) -> None:
        """Draw the leaderboard sidebar. Deliberately shows real data only:
        a method's row fills in the moment that method's metrics are
        finalized (``metrics[m]["completed"]``), never a live/partial guess -
        see CLAUDE.md's "LEADERBOARD side panel" note."""
        rect = self.summary_rect()
        self._draw_cut_panel(rect, PANEL_BG, LINE, CORNER_CUT, accent=GOOD if finished else None)

        x = rect.x + 14
        y = rect.y + 14
        title = self._text(self.f_side_h, "LEADERBOARD", INK, tracking=1)
        self.screen.blit(title, (x, y))
        y += title.get_height() + 4
        note_text = "ALL METHODS COMPLETE" if finished else "WAITING FOR COMPLETION"
        note = self._text(self.f_side_note, note_text, GOOD if finished else INK_FAINT, tracking=1)
        self.screen.blit(note, (x, y))
        y += note.get_height() + 14

        completed = {m: d for m, d in metrics.items() if d.get("completed")}
        max_time = max((d["time_s_val"] for d in completed.values() if isinstance(d.get("time_s_val"), (int, float))), default=0.0)
        max_eval = max((d["evaluations"] for d in completed.values() if isinstance(d.get("evaluations"), (int, float))), default=0)

        def optimality_frac(m: str) -> float:
            d = metrics.get(m, {})
            pl = d.get("path_length")
            if d.get("completed") and isinstance(pl, (int, float)) and pl and optimal > 0:
                return min(1.0, optimal / pl)
            return 0.0

        def time_frac(m: str) -> float:
            d = metrics.get(m, {})
            v = d.get("time_s_val")
            return (v / max_time) if (m in completed and max_time > 0 and isinstance(v, (int, float))) else 0.0

        def eval_frac(m: str) -> float:
            d = metrics.get(m, {})
            v = d.get("evaluations")
            return (v / max_eval) if (m in completed and max_eval and isinstance(v, (int, float))) else 0.0

        groups: List[Tuple[str, Callable[[str], str], Callable[[str], float]]] = [
            ("PATH / OPTIMAL", lambda m: str(metrics.get(m, {}).get("optimality", "-")), optimality_frac),
            ("TIME TO SOLVE", lambda m: str(metrics.get(m, {}).get("time_s", "-")), time_frac),
            ("EVALUATIONS", lambda m: _abbrev(metrics.get(m, {}).get("evaluations")), eval_frac),
        ]
        for heading, value_fn, frac_fn in groups:
            h = self._text(self.f_metric_h, heading, INK_FAINT, tracking=1)
            self.screen.blit(h, (x, y))
            y += h.get_height() + 8
            for method in ("GA", "Q-Learning", "A*"):
                y = self._draw_bar_row(x, y, rect.width - 28, method, value_fn(method), frac_fn(method))
            y += 6

        y += 4
        pygame.draw.line(self.screen, LINE, (x, y), (rect.right - 14, y), 1)
        y += 10
        for method in ("GA", "Q-Learning", "A*"):
            d = metrics.get(method, {})
            name = self._text(self.f_bar, method.upper(), INK, tracking=1)
            self.screen.blit(name, (x, y))
            if d.get("completed"):
                status_txt, status_col = (("SOLVED", GOOD) if d.get("solved") else ("FAILED", GOAL_RED))
            else:
                status_txt, status_col = ("RUNNING", INK_FAINT)
            st = self._text(self.f_bar, status_txt, status_col, tracking=1)
            self.screen.blit(st, (rect.right - 14 - st.get_width(), y))
            y += name.get_height() + 3
            detail = f"it {d.get('iterations', '-')} · ev {d.get('evaluations', '-')} · len {d.get('path_length', '-')}"
            ds = self._text(self.f_detail, detail, INK_DIM)
            self.screen.blit(ds, (x, y))
            y += ds.get_height() + 10

    # -- panel ----------------------------------------------------------------
    def draw_panel(
        self,
        index: int,
        method: str,
        agents: Sequence[FCell],
        leader: Optional[FCell],
        best_path: Optional[Sequence[Cell]],
        stats: List[Tuple[str, str]],
        status: str,
        status_text: str,
    ) -> None:
        panel = self.panel_rect(index)
        accent = METHOD_ACCENT.get(method, METHOD_ACCENT["GA"])
        self._draw_cut_panel(panel, PANEL_BG, LINE, CORNER_CUT, accent=accent)
        self._draw_head(panel, method, accent)

        ox, oy = self._maze_origin(panel)
        self._draw_maze(ox, oy)
        if best_path:
            self._draw_path(ox, oy, best_path, method)
        self._draw_swarm(ox, oy, agents, method)
        # Markers are drawn *after* the swarm: hundreds of agents start every
        # round on the start cell, and their stacked semi-transparent dots
        # would otherwise bury the marker underneath.
        self._draw_markers(ox, oy)
        if leader is not None:
            self._draw_leader(ox, oy, leader, method)

        content_bottom = oy + self.grid_h
        self._draw_stats(panel, content_bottom, stats)
        self._draw_status(panel, content_bottom, status, status_text)

    def _panel_name_surface(self, method: str, max_w: int) -> pygame.Surface:
        for font in (self.f_panel_name, self._f_panel_name_sm):
            surf = self._text(font, DISPLAY_NAME[method], INK, tracking=1)
            if surf.get_width() <= max_w:
                return surf
        return self._text(self._f_panel_name_sm, SHORT_NAME[method], INK, tracking=1)

    def _draw_head(self, panel: pygame.Rect, method: str, accent: Color) -> None:
        x = panel.x + PANEL_PAD
        y = panel.y + PANEL_PAD
        cy = y + HEAD_H // 2

        tag = METHOD_TAG[method]
        is_ref = tag == "REFERENCE"
        bg = TAG_REFERENCE_BG if is_ref else TAG_LEARNING_BG
        line_c = TAG_REFERENCE_LINE if is_ref else TAG_LEARNING_LINE
        tag_text = self._text(self.f_tag, tag, INK, tracking=1)
        tag_w = tag_text.get_width() + 16
        tag_rect = pygame.Rect(panel.right - PANEL_PAD - tag_w, y + (HEAD_H - 18) // 2, tag_w, 18)

        glow_r = 5
        glow_size = glow_r * 2 + 24
        glow = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        _draw_glow(glow, (glow_size // 2, glow_size // 2), glow_r, accent)
        dot_cx = x + glow_r
        self.screen.blit(glow, (dot_cx - glow_size // 2, cy - glow_size // 2))
        pygame.draw.circle(self.screen, accent, (dot_cx, cy), glow_r)

        name_x = x + 18
        max_name_w = max(20, tag_rect.x - 8 - name_x)
        name = self._panel_name_surface(method, max_name_w)
        self.screen.blit(name, (name_x, cy - name.get_height() // 2))

        pygame.draw.rect(self.screen, bg, tag_rect, border_radius=2)
        pygame.draw.rect(self.screen, line_c, tag_rect, width=1, border_radius=2)
        self.screen.blit(tag_text, tag_text.get_rect(center=tag_rect.center))

    def _draw_stats(self, panel: pygame.Rect, maze_bottom: int, stats: List[Tuple[str, str]]) -> None:
        y = maze_bottom + PANEL_GAP
        inner_w = panel.width - 2 * PANEL_PAD
        col_w = (inner_w - 2 * 8) // 3
        x = panel.x + PANEL_PAD
        for key, value in stats:
            k = self._text(self.f_stat_k, key.upper(), INK_FAINT, tracking=1)
            v = self._text(self.f_stat_v, str(value), INK)
            self.screen.blit(k, (x, y))
            self.screen.blit(v, (x, y + k.get_height() + 3))
            x += col_w + 8

    def _draw_status(self, panel: pygame.Rect, maze_bottom: int, status: str, text: str) -> None:
        y = maze_bottom + PANEL_GAP + STATS_H + PANEL_GAP
        rect = pygame.Rect(panel.x + PANEL_PAD, y, panel.width - 2 * PANEL_PAD, STATUS_H)
        if status == "solved":
            bg, color, font = STATUS_SOLVED_BG, GOOD, self.f_status_b
        elif status == "failed":
            bg, color, font = STATUS_FAILED_BG, GOAL_RED, self.f_status_b
        else:
            bg, color, font = STATUS_RUNNING_BG, INK_DIM, self.f_status
        pygame.draw.rect(self.screen, bg, rect, border_radius=5)
        tx = rect.x + 10
        if status == "solved":
            cx, cy = tx + 5, rect.centery
            pygame.draw.lines(self.screen, color, False,
                               [(cx - 4, cy), (cx - 1, cy + 3), (cx + 4, cy - 4)], 2)
            tx += 16
        surf = self._text(font, text, color)
        self.screen.blit(surf, (tx, rect.centery - surf.get_height() // 2))

    # -- maze -----------------------------------------------------------------
    def _draw_maze(self, ox: int, oy: int) -> None:
        if self._maze_surface is None:
            self._maze_surface = self._build_maze_surface()
        self.screen.blit(self._maze_surface, (ox, oy))
        pygame.draw.rect(self.screen, LINE, pygame.Rect(ox, oy, self.grid_w, self.grid_h), width=1)

    def _draw_markers(self, ox: int, oy: int) -> None:
        edge = max(1, self.cell // 8)

        sr, sc = self.maze.start
        srect = pygame.Rect(ox + sc * self.cell, oy + sr * self.cell, self.cell, self.cell)
        pygame.draw.rect(self.screen, START_GREEN, srect)
        pygame.draw.rect(self.screen, MARKER_EDGE, srect, width=edge)

        gr, gc = self.maze.goal
        grect = pygame.Rect(ox + gc * self.cell, oy + gr * self.cell, self.cell, self.cell)
        glow_size = self.cell + 20
        glow = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        _draw_glow(glow, (glow_size // 2, glow_size // 2), self.cell // 2, GOAL_RED)
        self.screen.blit(glow, (grect.centerx - glow_size // 2, grect.centery - glow_size // 2))
        pygame.draw.rect(self.screen, GOAL_RED, grect)
        pygame.draw.rect(self.screen, MARKER_EDGE, grect, width=edge)

    # -- overlays (population / best path / leader) ---------------------------
    def _draw_path(self, ox: int, oy: int, path: Sequence[Cell], method: str) -> None:
        if len(path) < 2:
            return
        color = METHOD_ON_MAZE[method]
        overlay = self._overlay()
        pts = [self._px(0, 0, self.cell, r, c) for (r, c) in path]
        width = max(2, self.cell // 6)
        glow_w = width * 2 + 4
        pygame.draw.lines(overlay, (*color, 45), False, pts, glow_w)
        for p in pts:
            pygame.draw.circle(overlay, (*color, 45), (int(p[0]), int(p[1])), glow_w // 2)
        pygame.draw.lines(overlay, (*color, 220), False, pts, width)
        for p in pts:
            pygame.draw.circle(overlay, (*color, 220), (int(p[0]), int(p[1])), width // 2)
        self.screen.blit(overlay, (ox, oy))

    def _draw_swarm(self, ox: int, oy: int, agents: Sequence[FCell], method: str) -> None:
        if not agents:
            return
        color = METHOD_ON_MAZE[method]
        overlay = self._overlay()
        if len(agents) > 1:
            size = max(3, int(self.cell * 0.42))
            half = size / 2
            alpha = SWARM_ALPHA_DENSE if len(agents) > 60 else SWARM_ALPHA_FEW
            rgba = (*color, alpha)
            for fr, fc in agents:
                x, y = self._px(0, 0, self.cell, fr, fc)
                pygame.draw.rect(overlay, rgba, pygame.Rect(x - half, y - half, size, size))
        else:
            # Single-policy methods (Q-Learning/A*) draw their one agent as a
            # filled circle, matching the mockup's single-dot head.
            fr, fc = agents[0]
            x, y = self._px(0, 0, self.cell, fr, fc)
            radius = max(3, int(self.cell * 0.24))
            pygame.draw.circle(overlay, (*color, SWARM_ALPHA_ONE), (int(x), int(y)), radius)
        self.screen.blit(overlay, (ox, oy))

    def _draw_leader(self, ox: int, oy: int, leader: FCell, method: str) -> None:
        x, y = self._px(ox, oy, self.cell, leader[0], leader[1])
        accent = METHOD_ACCENT[method]
        glow_color = METHOD_ON_MAZE[method]
        size = max(4, int(self.cell * 0.5))
        radius = size // 2
        glow_size = radius * 2 + 24
        glow = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        _draw_glow(glow, (glow_size // 2, glow_size // 2), radius, glow_color)
        self.screen.blit(glow, (int(x) - glow_size // 2, int(y) - glow_size // 2))
        pygame.draw.circle(self.screen, accent, (int(x), int(y)), radius)
        pygame.draw.circle(self.screen, LEADER_RING, (int(x), int(y)), radius + 3, 2)
