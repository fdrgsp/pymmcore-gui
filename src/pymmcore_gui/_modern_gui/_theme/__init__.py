from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_gui._qt.QtWidgets import QApplication

from ._dark import DARK_THEME
from ._fonts import mono_font, ui_font
from ._qt import color_to_qcolor, to_qpalette
from ._style import RADIUS, MicroscopeStyle, Sp
from ._types import (
    Brush,
    Color,
    ColorGroup,
    GradientSpread,
    GradientStop,
    LinearGradient,
    Palette,
    Theme,
)

if TYPE_CHECKING:
    from pymmcore_gui._qt.QtGui import QColor, QPalette

__all__ = [
    "DARK_THEME",
    "RADIUS",
    "Brush",
    "Color",
    "ColorGroup",
    "GradientSpread",
    "GradientStop",
    "LinearGradient",
    "MicroscopeStyle",
    "Palette",
    "Sp",
    "Theme",
    "make_dark_palette",
    "mono_font",
    "qcolor",
    "set_theme",
    "theme",
    "ui_font",
]

# ═══════════════════════════════════════════════════════════════════
# Theme accessor
# ═══════════════════════════════════════════════════════════════════

_current_theme: Theme = DARK_THEME


def theme() -> Theme:
    """Return the active theme."""
    return _current_theme


def set_theme(t: Theme) -> None:
    """Set the active theme and push its QPalette to the application."""
    global _current_theme
    _current_theme = t
    app = QApplication.instance()
    if app is not None:
        app.setPalette(to_qpalette(t.palette))


def qcolor(c: Color) -> QColor:
    """Convenience: convert a theme Color to QColor."""
    return color_to_qcolor(c)


def make_dark_palette() -> QPalette:
    """Build a dark QPalette matching the mockup color tokens."""
    return to_qpalette(DARK_THEME.palette)
