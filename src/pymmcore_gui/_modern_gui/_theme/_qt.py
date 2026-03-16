from __future__ import annotations

from pymmcore_gui._qt.QtCore import QPointF
from pymmcore_gui._qt.QtGui import QBrush, QColor, QGradient, QLinearGradient, QPalette

from ._types import Brush, Color, LinearGradient, Palette

ColorRole = QPalette.ColorRole

ROLE_TO_FIELD: dict[ColorRole, str] = {
    ColorRole.WindowText: "window_text",
    ColorRole.Button: "button",
    ColorRole.Light: "light",
    ColorRole.Midlight: "midlight",
    ColorRole.Dark: "dark",
    ColorRole.Mid: "mid",
    ColorRole.Text: "text",
    ColorRole.BrightText: "bright_text",
    ColorRole.ButtonText: "button_text",
    ColorRole.Base: "base",
    ColorRole.Window: "window",
    ColorRole.Shadow: "shadow",
    ColorRole.Highlight: "highlight",
    ColorRole.HighlightedText: "highlighted_text",
    ColorRole.Link: "link",
    ColorRole.LinkVisited: "link_visited",
    ColorRole.AlternateBase: "alternate_base",
    ColorRole.ToolTipBase: "tooltip_base",
    ColorRole.ToolTipText: "tooltip_text",
    ColorRole.PlaceholderText: "placeholder_text",
    ColorRole.Accent: "accent",
}


def color_to_qcolor(c: Color) -> QColor:
    """Convert to ``QColor``."""
    return QColor(c.red, c.green, c.blue, c.alpha)


def _brush_to_qbrush(brush: Brush) -> QBrush:
    """Convert a model Brush to a QBrush."""
    if isinstance(brush, Color):
        return QBrush(color_to_qcolor(brush))
    if isinstance(brush, LinearGradient):
        g = QLinearGradient(
            QPointF(brush.x1, brush.y1),
            QPointF(brush.x2, brush.y2),
        )
        for stop in brush.stops:
            g.setColorAt(stop.position, color_to_qcolor(stop.color))
        g.setSpread(QGradient.Spread(brush.spread.value))
        g.setCoordinateMode(QGradient.CoordinateMode.ObjectBoundingMode)
        return QBrush(g)
    raise TypeError(f"Unknown brush type: {type(brush)}")


def to_qpalette(palette: Palette) -> QPalette:
    """Convert a Palette to a QPalette."""
    qpal = QPalette()
    groups = [
        (palette.active, QPalette.ColorGroup.Active),
        (palette.inactive, QPalette.ColorGroup.Inactive),
        (palette.disabled, QPalette.ColorGroup.Disabled),
    ]
    for group, qg in groups:
        for role, fname in ROLE_TO_FIELD.items():
            brush = getattr(group, fname)
            if brush is None:
                continue
            qr = QPalette.ColorRole(role.value)
            if isinstance(brush, Color):
                qpal.setColor(qg, qr, color_to_qcolor(brush))
            else:
                qpal.setBrush(qg, qr, _brush_to_qbrush(brush))
    return qpal
