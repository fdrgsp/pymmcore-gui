"""One application-wide event filter shared by the app's widget hooks.

PyQt calls a Python event filter for every event in the application, so the
hooks share one filter, dispatched by event type, rather than each paying that
cost (noticeable when many widgets are created, e.g. a large position table).
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from pymmcore_gui._qt.QtCore import QEvent, QObject
from pymmcore_gui._qt.QtWidgets import QApplication, QWidget

if TYPE_CHECKING:
    from collections.abc import Callable

    EventHook = Callable[[QWidget, QEvent], bool]

_hooks: defaultdict[QEvent.Type, list[EventHook]] = defaultdict(list)


class _AppEventFilter(QObject):
    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if a1 is None or not isinstance(a0, QWidget):
            return False
        if hooks := _hooks.get(a1.type()):
            return any(hook(a0, a1) for hook in hooks)
        return False


_filter: _AppEventFilter | None = None


def add_event_hook(hook: EventHook, *types: QEvent.Type) -> None:
    """Call `hook(widget, event)` for widget events of `types`.

    A hook returning True consumes the event. Adding a hook twice is a no-op.
    """
    global _filter
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        return
    if _filter is None or _filter.parent() is not app:
        _filter = _AppEventFilter(app)
        app.installEventFilter(_filter)
    for event_type in types:
        if hook not in (hooks := _hooks[event_type]):
            hooks.append(hook)
