"""Keep icons in dynamically created dialogs visible against the theme."""

from pymmcore_gui._array_viewer import ensure_visible_icon
from pymmcore_gui._qt.QtCore import QEvent, QObject
from pymmcore_gui._qt.QtWidgets import QAbstractButton, QApplication


class _ButtonIconFilter(QObject):
    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if (
            isinstance(a0, QAbstractButton)
            and a1 is not None
            and a1.type() == QEvent.Type.Show
        ):
            ensure_visible_icon(a0)
        return False


_button_icon_filter: _ButtonIconFilter | None = None


def install_icon_contrast() -> None:
    """Check new buttons on show, including those created after app startup."""
    global _button_icon_filter
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        return
    if _button_icon_filter is not None and _button_icon_filter.parent() is app:
        return
    _button_icon_filter = _ButtonIconFilter(app)
    app.installEventFilter(_button_icon_filter)
