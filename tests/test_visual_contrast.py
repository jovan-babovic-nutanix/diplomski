"""WCAG contrast regression guard for the Race Control palette.

Guards against a future palette tweak silently making a marker or a piece of
text illegible again (see CLAUDE.md's UI history and the 2026 redesign audit).
Thresholds are fixed: >=3.0 for graphical markers, >=4.5 for normal-size text.
If a color fails, fix the color - never lower these numbers.
"""
import pytest

pytest.importorskip("pygame")

from src.visualization.renderer import (  # noqa: E402
    GOAL_RED,
    GOOD,
    GRID_TEX,
    INK,
    INK_DIM,
    INK_FAINT,
    LEADER_RING,
    MAZE_FLOOR,
    MAZE_WALL,
    MARKER_EDGE,
    METHOD_ACCENT,
    METHOD_ON_MAZE,
    ON_ACCENT,
    PANEL_BG,
    START_GREEN,
    STATUS_FAILED_BG,
    STATUS_RUNNING_BG,
    STATUS_SOLVED_BG,
    SWARM_ALPHA_DENSE,
    TAG_LEARNING_BG,
    TAG_REFERENCE_BG,
    TRACK,
    WALL_EDGE,
    WINDOW_BG,
    _blend,
)

def _linearize(channel: float) -> float:
    c = channel / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _relative_luminance(rgb) -> float:
    r, g, b = rgb
    return 0.2126 * _linearize(r) + 0.7152 * _linearize(g) + 0.0722 * _linearize(b)


def _contrast(a, b) -> float:
    la, lb = _relative_luminance(a), _relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def test_marker_colors_are_visible_on_maze():
    pairs = [
        (START_GREEN, MAZE_FLOOR),
        (GOAL_RED, MAZE_FLOOR),
        (LEADER_RING, METHOD_ACCENT["GA"]),
        # The maze floor/wall fills are deliberately low-contrast (a dark HUD
        # backdrop for the glow effects) - the boundary outline is what
        # actually has to carry maze structure at readable contrast.
        (WALL_EDGE, MAZE_FLOOR),
        (WALL_EDGE, MAZE_WALL),
        (MARKER_EDGE, START_GREEN),
        (MARKER_EDGE, GOAL_RED),
    ]
    for a, b in pairs:
        assert _contrast(a, b) >= 3.0, f"{a} vs {b}"
    for color in METHOD_ON_MAZE.values():
        assert _contrast(color, MAZE_FLOOR) >= 3.0, f"{color} vs floor"
        assert _contrast(MARKER_EDGE, color) >= 3.0, f"marker edge vs {color}"


def test_swarm_stays_visible_when_alpha_blended():
    # The population dots are alpha-blended over the maze floor, not drawn at
    # full opacity - the *composited* pixel is what must clear 3:1, not the
    # raw dot color. This is what catches an alpha regression (e.g. the old
    # alpha=80 tier, which composited to ~2.07:1 and was invisible).
    for color in METHOD_ON_MAZE.values():
        composited = _blend(color, MAZE_FLOOR, SWARM_ALPHA_DENSE / 255)
        assert _contrast(composited, MAZE_FLOOR) >= 3.0, f"{color} dense swarm vs floor"


def test_accent_colors_are_visible_on_chrome():
    backgrounds = (PANEL_BG, WINDOW_BG, TRACK, GRID_TEX)
    for accent in METHOD_ACCENT.values():
        for bg in backgrounds:
            assert _contrast(accent, bg) >= 3.0, f"{accent} vs {bg}"
    assert _contrast(METHOD_ACCENT["Q-Learning"], TRACK) >= 3.0  # progress bar fill
    # The "solved" checkmark is drawn in GOOD directly on STATUS_SOLVED_BG (no
    # separate colored circle badge in this design) - see test_text_colors_meet_aa
    # for that pair at the stricter 4.5 text threshold.
    assert _contrast(GOOD, STATUS_SOLVED_BG) >= 3.0


def test_text_colors_meet_aa():
    backgrounds = (
        WINDOW_BG,
        PANEL_BG,
        TRACK,
        GRID_TEX,
        STATUS_RUNNING_BG,
        STATUS_SOLVED_BG,
        STATUS_FAILED_BG,
        TAG_LEARNING_BG,
        TAG_REFERENCE_BG,
    )
    for bg in backgrounds:
        assert _contrast(INK, bg) >= 4.5, f"INK vs {bg}"
        assert _contrast(INK_DIM, bg) >= 4.5, f"INK_DIM vs {bg}"
    for bg in (WINDOW_BG, PANEL_BG, TRACK, GRID_TEX):
        assert _contrast(INK_FAINT, bg) >= 4.5, f"INK_FAINT vs {bg}"
    assert _contrast(GOOD, STATUS_SOLVED_BG) >= 4.5
    assert _contrast(GOAL_RED, STATUS_FAILED_BG) >= 4.5
    assert _contrast(ON_ACCENT, GOOD) >= 4.5
    assert _contrast(ON_ACCENT, METHOD_ACCENT["GA"]) >= 4.5
