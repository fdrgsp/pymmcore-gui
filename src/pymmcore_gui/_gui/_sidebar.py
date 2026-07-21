"""Reusable dock panels (left / right / bottom) for a tab page.

Empty placeholders for now; each exposes :meth:`Sidebar.add_widget` to be
populated in later iterations.
"""

from __future__ import annotations

from enum import Enum, auto

from pymmcore_gui._qt.QtCore import QEvent, QSize
from pymmcore_gui._qt.QtGui import QPainter, QPaintEvent
from pymmcore_gui._qt.QtWidgets import (
    QBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ._theme import qcolor, theme


class SidebarPosition(Enum):
    """Where a :class:`Sidebar` sits within a tab page."""

    LEFT = auto()
    RIGHT = auto()
    BOTTOM = auto()


class Sidebar(QWidget):
    """Themed dock panel for one edge of a tab page (empty for now).

    LEFT/RIGHT stack their content vertically; BOTTOM lays it out
    horizontally. Background and edge separator are painted live from the
    current theme.
    """

    _BASE_HEIGHT = 160

    def __init__(
        self, position: SidebarPosition, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._position = position

        self._layout: QBoxLayout
        if position is SidebarPosition.BOTTOM:
            self._layout = QHBoxLayout(self)
            self.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )
        else:
            self._layout = QVBoxLayout(self)
            self.setSizePolicy(
                QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
            )

        self._apply_metrics()

    # ── public API ────────────────────────────────────────────────

    @property
    def position(self) -> SidebarPosition:
        return self._position

    def add_widget(self, widget: QWidget, stretch: int = 0) -> None:
        """Append a widget.

        Pass ``stretch=1`` for content that should fill the dock; leave it at 0
        and finish with :meth:`add_stretch` to stack items against the edge.
        """
        self._layout.addWidget(widget, stretch)

    def add_stretch(self, stretch: int = 1) -> None:
        """Append an expanding spacer, pinning prior widgets to the edge."""
        self._layout.addStretch(stretch)

    # ── sizing ────────────────────────────────────────────────────

    def sizeHint(self) -> QSize:
        t = theme()
        if self._position is SidebarPosition.BOTTOM:
            return QSize(t.sidebar_width, t.scaled(self._BASE_HEIGHT))
        return QSize(t.sidebar_width, t.sidebar_width)

    def _apply_metrics(self) -> None:
        t = theme()
        self._layout.setContentsMargins(t.sp_sm, t.sp_sm, t.sp_sm, t.sp_sm)
        self._layout.setSpacing(t.sp_sm)

    # ── painting ──────────────────────────────────────────────────

    def paintEvent(self, event: QPaintEvent | None) -> None:
        p = QPainter(self)
        t = theme()
        p.fillRect(self.rect(), qcolor(t.bg_base))
        p.setPen(qcolor(t.border_subtle))
        w, h = self.width(), self.height()
        if self._position is SidebarPosition.LEFT:
            p.drawLine(w - 1, 0, w - 1, h)
        elif self._position is SidebarPosition.RIGHT:
            p.drawLine(0, 0, 0, h)
        else:  # BOTTOM
            p.drawLine(0, 0, w, 0)
        p.end()

    def changeEvent(self, event: QEvent | None) -> None:
        if event is not None and event.type() == QEvent.Type.StyleChange:
            self._apply_metrics()
            self.updateGeometry()
            self.update()
        super().changeEvent(event)
