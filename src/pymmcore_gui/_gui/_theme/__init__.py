from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_gui._array_viewer import ensure_visible_icon
from pymmcore_gui._qt.QtCore import QCoreApplication, QEvent, QSize
from pymmcore_gui._qt.QtGui import QFont, QGuiApplication
from pymmcore_gui._qt.QtWidgets import QAbstractButton, QApplication, QStyle, QToolBar

from ._dark import DARK_THEME
from ._fonts import mono_font, ui_font
from ._light import LIGHT_THEME
from ._qt import color_to_qcolor, to_qpalette
from ._scaled_view import ScaledThemeView
from ._style import MicroscopeStyle
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
    "LIGHT_THEME",
    "ZOOM_STEPS",
    "Brush",
    "Color",
    "ColorGroup",
    "GradientSpread",
    "GradientStop",
    "LinearGradient",
    "MicroscopeStyle",
    "Palette",
    "ScaledThemeView",
    "Theme",
    "make_dark_palette",
    "mono_font",
    "qcolor",
    "raw_theme",
    "reset_zoom",
    "set_style",
    "set_theme",
    "set_zoom",
    "theme",
    "ui_font",
    "zoom_in",
    "zoom_out",
]

# ═══════════════════════════════════════════════════════════════════
# State
# ═══════════════════════════════════════════════════════════════════

_current_theme: Theme = DARK_THEME
_current_style: MicroscopeStyle | None = None
_view: ScaledThemeView | None = None
_base_font_pt: float = 0.0

ZOOM_STEPS = (0.5, 0.67, 0.75, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
_DEFAULT_ZOOM = 1.25
_zoom_index: int = ZOOM_STEPS.index(_DEFAULT_ZOOM)

# ═══════════════════════════════════════════════════════════════════
# Theme accessors
# ═══════════════════════════════════════════════════════════════════


def theme() -> ScaledThemeView:
    """Return a live zoom-scaled view of the active theme."""
    if _view is None:
        raise RuntimeError("call set_style() during app init")
    return _view


def raw_theme() -> Theme:
    """Return the underlying unscaled Theme dataclass."""
    return _current_theme


def set_theme(t: Theme) -> None:
    """Set the active theme and push its QPalette to the application."""
    global _current_theme, _view

    # App-level style + theme setup
    if isinstance(qapp := QApplication.instance(), QApplication):
        if style := qapp.style():
            if not isinstance(style, MicroscopeStyle):
                style = MicroscopeStyle()
                qapp.setStyle(style)
                set_style(style)

    _current_theme = t
    if _current_style is not None:
        _view = ScaledThemeView(t, _current_style)
    app = QApplication.instance()
    if isinstance(app, QGuiApplication):
        app.setPalette(to_qpalette(t.palette))

    if isinstance(app, QApplication) and _current_style is not None:
        # set_zoom() sends every widget a StyleChange event and forces a
        # relayout -- set_style() (first-ever call only) already relies on
        # this to make the *initial* theme take effect. A *later* toggle
        # (this branch) needs the exact same forced refresh: without it,
        # QApplication.setPalette() above updates the application-wide
        # palette immediately, but an already-constructed widget's own
        # .palette() only catches up once Qt actually re-polishes it --
        # which, empirically, doesn't reliably happen from setPalette()
        # alone for widgets nested a few layers deep (e.g. a useq_widgets
        # table's viewport keeps showing the *previous* theme's colors
        # indefinitely otherwise). Re-applying the same zoom factor is a
        # no-op for sizing and just piggybacks on the refresh it already does.
        set_zoom(_current_style.zoom_factor)

        # Re-evaluate every button's icon contrast against the new palette.
        # ensure_visible_icon always re-derives from the button's stashed
        # *original* icon, so this is safe to call repeatedly across any
        # number of light/dark toggles -- without it, a dark-theme recolor
        # would stay baked in (and turn invisible) after switching to light.
        for w in app.allWidgets():
            if isinstance(w, QAbstractButton):
                ensure_visible_icon(w)


def set_style(style: MicroscopeStyle) -> None:
    """Register the app style and capture the base font size."""
    global _current_style, _view, _base_font_pt
    _current_style = style
    app = QApplication.instance()
    if isinstance(app, QApplication):
        _base_font_pt = app.font().pointSizeF()
    _view = ScaledThemeView(_current_theme, style)
    set_zoom(_DEFAULT_ZOOM)


# ═══════════════════════════════════════════════════════════════════
# Zoom
# ═══════════════════════════════════════════════════════════════════


def set_zoom(factor: float) -> None:
    """Set zoom factor and refresh the entire UI."""
    if _current_style is None:
        raise RuntimeError("call set_style() during app init")

    # no view rebuild needed; ScaledThemeView reads zoom live.
    _current_style.zoom_factor = factor

    app = QApplication.instance()
    if not isinstance(app, QApplication):
        return

    # Pillar 2: scale app font
    font = QFont(app.font())
    font.setPointSizeF(_base_font_pt * factor)
    app.setFont(font)

    # Pillar 3: icon sizes on caching widgets
    icon_sz = _current_style.pixelMetric(QStyle.PixelMetric.PM_ToolBarIconSize)
    for w in app.allWidgets():
        if isinstance(w, QToolBar):
            w.setIconSize(QSize(icon_sz, icon_sz))

    # Force complete relayout
    for w in app.allWidgets():
        if layout := w.layout():
            layout.invalidate()
        QCoreApplication.sendEvent(w, QEvent(QEvent.Type.StyleChange))
        w.updateGeometry()
    for win in app.topLevelWidgets():
        if layout := win.layout():
            layout.activate()

    # TODO: persist zoom to QSettings


def zoom_in() -> None:
    """Step zoom up to the next level."""
    global _zoom_index
    if _zoom_index < len(ZOOM_STEPS) - 1:
        _zoom_index += 1
        set_zoom(ZOOM_STEPS[_zoom_index])


def zoom_out() -> None:
    """Step zoom down to the previous level."""
    global _zoom_index
    if _zoom_index > 0:
        _zoom_index -= 1
        set_zoom(ZOOM_STEPS[_zoom_index])


def reset_zoom() -> None:
    """Reset zoom to the default level."""
    global _zoom_index
    _zoom_index = ZOOM_STEPS.index(_DEFAULT_ZOOM)
    set_zoom(_DEFAULT_ZOOM)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def qcolor(c: Color) -> QColor:
    """Convenience: convert a theme Color to QColor."""
    return color_to_qcolor(c)


def make_dark_palette() -> QPalette:
    """Build a dark QPalette matching the mockup color tokens."""
    return to_qpalette(DARK_THEME.palette)
