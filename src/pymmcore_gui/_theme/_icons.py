"""Keep icons in dynamically created dialogs visible against the theme."""

from pymmcore_gui._array_viewer import ensure_visible_icon
from pymmcore_gui._qt.QtCore import QEvent
from pymmcore_gui._qt.QtWidgets import QAbstractButton, QWidget
from pymmcore_gui._theme._events import add_event_hook


def _on_show(widget: QWidget, _event: QEvent) -> bool:
    if isinstance(widget, QAbstractButton):
        ensure_visible_icon(widget)
    return False


def install_icon_contrast() -> None:
    """Check new buttons on show, including those created after app startup."""
    add_event_hook(_on_show, QEvent.Type.Show)
