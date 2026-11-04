"""Reusable left dock panel for a tab page."""

from __future__ import annotations

from pymmcore_gui._qt.QtCore import QEvent, QSize, Qt
from pymmcore_gui._qt.QtGui import QPainter, QPaintEvent
from pymmcore_gui._qt.QtWidgets import QSizePolicy, QVBoxLayout, QWidget
from pymmcore_gui._theme import qcolor, theme


class Sidebar(QWidget):
    """Themed dock panel for the left edge of a tab page.

    Background and edge separator are painted live from the current theme.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        # Without this, a QVBoxLayout with no item claiming the extra
        # space spreads it evenly across every margin/gap instead of
        # packing widgets to the top.
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        self._apply_metrics()

    # ── public API ────────────────────────────────────────────────

    def add_widget(self, widget: QWidget, stretch: int = 0) -> None:
        """Append a widget; pass ``stretch=1`` for content that should fill the dock."""
        self._layout.addWidget(widget, stretch)

    # ── sizing ────────────────────────────────────────────────────

    def sizeHint(self) -> QSize:
        width = theme().sidebar_width
        return QSize(width, width)

    def _apply_metrics(self) -> None:
        t = theme()
        self._layout.setContentsMargins(t.sp_sm, t.sp_sm, t.sp_sm, t.sp_sm)
        self._layout.setSpacing(t.sp_sm)

    # ── painting ──────────────────────────────────────────────────

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        p = QPainter(self)
        t = theme()
        p.fillRect(self.rect(), qcolor(t.bg_base))
        p.setPen(qcolor(t.border_subtle))
        w, h = self.width(), self.height()
        p.drawLine(w - 1, 0, w - 1, h)
        p.end()

    def changeEvent(self, a0: QEvent | None) -> None:
        if a0 is not None and a0.type() == QEvent.Type.StyleChange:
            self._apply_metrics()
            self.updateGeometry()
            self.update()
        super().changeEvent(a0)
