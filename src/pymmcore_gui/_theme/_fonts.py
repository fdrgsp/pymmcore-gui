"""Zoom-aware font helpers.

Use ONLY for widgets that need a non-default font (custom size, weight,
or family). Widgets using the app default font are scaled automatically
by QApplication.setFont() — do NOT call setFont(ui_font()) on them, as
that opts them out of the app font cascade.
"""

from __future__ import annotations

from pymmcore_gui._qt.QtGui import QFont

UI_FONT_SIZE_PT = 11.0
UI_FONT_WEIGHT = QFont.Weight.Medium


def _zoom() -> float:
    """Get current zoom factor (lazy import avoids circular ref)."""
    from pymmcore_gui._theme import theme

    return theme().zoom_factor


def ui_font(
    size_pt: float = UI_FONT_SIZE_PT,
    weight: int = UI_FONT_WEIGHT,
) -> QFont:
    """System UI font, zoom-scaled."""
    f = QFont()
    f.setPointSizeF(size_pt * _zoom())
    f.setWeight(weight)
    return f
