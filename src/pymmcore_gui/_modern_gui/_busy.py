"""A translucent "working…" overlay for blocking operations.

Device-adapter scanning and configuration loading run synchronously on the GUI
thread, so the overlay is shown and the event loop flushed *before* the blocking
call starts — otherwise nothing would ever paint.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from pymmcore_gui._qt.QtCore import QEventLoop, Qt
from pymmcore_gui._qt.QtGui import QPainter, QPaintEvent
from pymmcore_gui._qt.QtWidgets import QApplication, QWidget

from ._theme import qcolor, theme, ui_font

if TYPE_CHECKING:
    from collections.abc import Iterator


class BusyOverlay(QWidget):
    """Dim the parent widget and show a centered message."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._message = ""
        self.hide()

    def start(self, message: str) -> None:
        """Cover the parent with `message` and repaint immediately."""
        self._message = message
        if (parent := self.parentWidget()) is None:  # pragma: no cover
            return
        self.setGeometry(parent.rect())
        self.raise_()
        self.show()
        # force a paint now: the caller is about to block the event loop
        self.repaint()
        if app := QApplication.instance():
            # Paint/layout work only -- NOT queued user input. This runs from
            # inside the very handler that is about to block, so delivering a
            # click here would re-enter it: a second click on a "Save" button
            # (which sits in a page toolbar, outside the widget this overlay
            # covers) would start a nested rewrite of the same core while the
            # first is still running.
            app.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def stop(self) -> None:
        self.hide()

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        p = QPainter(self)
        t = theme()
        backdrop = qcolor(t.bg_deepest)
        backdrop.setAlpha(200)
        p.fillRect(self.rect(), backdrop)
        p.setFont(ui_font())
        p.setPen(qcolor(t.text_primary))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message)
        p.end()


@contextmanager
def busy(overlay: BusyOverlay | None, message: str) -> Iterator[None]:
    """Show `overlay` for the duration of a blocking operation.

    Does nothing if the overlay's parent isn't visible yet (e.g. during
    construction), where painting would be pointless and processEvents unsafe.
    """
    parent = overlay.parentWidget() if overlay is not None else None
    active = overlay is not None and parent is not None and parent.isVisible()
    if active:
        assert overlay is not None
        overlay.start(message)
    try:
        yield
    finally:
        if active:
            assert overlay is not None
            overlay.stop()
