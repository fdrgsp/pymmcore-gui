from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_gui._qt.QtGui import QColor, QPalette

if TYPE_CHECKING:
    from pymmcore_gui._theme._types import Color, Palette

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
                # An unset role means "inherit" (see ColorGroup's docstring),
                # but leaving it unset on the QPalette doesn't actually
                # inherit anything -- Qt fills the gap with its own built-in
                # default, which matches neither theme. E.g. both themes'
                # `disabled` group only defines muted *text* roles, relying
                # on this fallback for every background role (Base, Window,
                # Button, ...) -- without it, a widget that's ever disabled
                # (even transiently, e.g. an unchecked tab's content in
                # useq_widgets) shows Qt's default light gray regardless of
                # theme. Fall back to the *active* group, which always
                # defines the complete set.
                brush = getattr(palette.active, fname)
                if brush is None:
                    continue
            qpal.setColor(qg, role, color_to_qcolor(brush))
    return qpal
