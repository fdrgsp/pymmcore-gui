"""Draw the app's subtle action buttons in the "primary" variant."""

from __future__ import annotations

from pymmcore_gui._qt.QtCore import QDynamicPropertyChangeEvent, QEvent
from pymmcore_gui._qt.QtWidgets import QPushButton, QTabBar, QToolButton, QWidget
from pymmcore_gui._theme._events import add_event_hook

_KEEP_SUBTLE = "_pymmcore_gui_keep_subtle_buttons"


def keep_subtle_buttons(widget: QWidget) -> None:
    """Leave the buttons under `widget` in the "subtle" variant."""
    widget.setProperty(_KEEP_SUBTLE, True)


def _use_primary(widget: QWidget, event: QEvent) -> bool:
    """Make a subtle button primary once it is shown, or set subtle while shown.

    Not before: a button is often made subtle before it is placed under a widget
    that keeps its buttons subtle. Checkable buttons keep their look, as primary
    has no checked state.
    """
    if isinstance(event, QDynamicPropertyChangeEvent) and (
        event.propertyName() != b"variant" or not widget.isVisible()
    ):
        return False
    if (
        isinstance(widget, (QPushButton, QToolButton))
        and widget.property("variant") == "subtle"
        and not widget.isCheckable()
        and not isinstance(widget.parentWidget(), QTabBar)
    ):
        ancestor: QWidget | None = widget
        while ancestor is not None:
            if ancestor.property(_KEEP_SUBTLE):
                return False
            ancestor = ancestor.parentWidget()
        widget.setProperty("variant", "primary")
        widget.update()
    return False


def install_primary_buttons() -> None:
    """Install the GUI-wide primary button handler once per application."""
    add_event_hook(_use_primary, QEvent.Type.Show, QEvent.Type.DynamicPropertyChange)
