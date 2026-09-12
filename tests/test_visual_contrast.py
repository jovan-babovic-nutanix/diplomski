"""WCAG contrast regression guard for the Flying Dots palette.

Guards against a future palette tweak silently making a marker or a piece of
text illegible again (see CLAUDE.md's UI history and the 2026 audit that found
several palette constants below WCAG thresholds).
"""
import pytest

pytest.importorskip("pygame")

from src.visualization.renderer import (  # noqa: E402
    BUTTON_BLUE,
    CHIP_BG,
    CYAN,
    DOT_BLACK,
    GOAL_RED,
    GOOD,
    LEADER_RING,
    METHOD_ACCENT,
    PANEL_BG,
    PATH_COLOR,
    START_GREEN,
    TITLE_BG,
    TITLE_BG_SOLVED,
    TITLE_DIM,
    TITLE_TEXT,
    WALL_BLUE,
    WINDOW_BG,
)

WHITE = (255, 255, 255)


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
        (START_GREEN, CYAN),
        (GOAL_RED, CYAN),
        (PATH_COLOR, CYAN),
        (LEADER_RING, CYAN),
        (LEADER_RING, DOT_BLACK),
        (WALL_BLUE, CYAN),
        (DOT_BLACK, CYAN),
    ]
    for a, b in pairs:
        assert _contrast(a, b) >= 3.0, f"{a} vs {b}"


def test_accent_colors_are_visible_on_chrome():
    backgrounds = (PANEL_BG, WINDOW_BG, TITLE_BG)
    for accent in METHOD_ACCENT.values():
        for bg in backgrounds:
            assert _contrast(accent, bg) >= 3.0, f"{accent} vs {bg}"
    assert _contrast(METHOD_ACCENT["Q-Learning"], CHIP_BG) >= 3.0  # progress bar
    assert _contrast(GOOD, TITLE_BG_SOLVED) >= 3.0
    assert _contrast(WHITE, GOOD) >= 3.0  # check glyph on the badge


def test_text_colors_meet_aa():
    backgrounds = (TITLE_BG, TITLE_BG_SOLVED, PANEL_BG, CHIP_BG, WINDOW_BG)
    for bg in backgrounds:
        assert _contrast(TITLE_DIM, bg) >= 4.5, f"TITLE_DIM vs {bg}"
        assert _contrast(TITLE_TEXT, bg) >= 4.5, f"TITLE_TEXT vs {bg}"
    assert _contrast(GOOD, PANEL_BG) >= 4.5
    assert _contrast(WHITE, BUTTON_BLUE) >= 4.5
