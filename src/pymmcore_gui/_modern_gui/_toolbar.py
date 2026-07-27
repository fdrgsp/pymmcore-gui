"""Reusable, per-tab toolbar strip (empty placeholder for now)."""

from __future__ import annotations

from pymmcore_gui._qt.QtCore import QEvent
from pymmcore_gui._qt.QtGui import QPainter, QPaintEvent
from pymmcore_gui._qt.QtWidgets import QHBoxLayout, QSizePolicy, QWidget

from ._theme import qcolor, theme


class TabToolBar(QWidget):
    """Horizontal toolbar strip for a tab page.

    Empty by default; populate it later with :meth:`add_widget` /
    :meth:`add_stretch`. Background and separator are painted live from the
    current theme, so it tracks theme and zoom changes automatically.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._layout = QHBoxLayout(self)
        self._apply_metrics()

    # ── public API ────────────────────────────────────────────────

    def add_widget(self, widget: QWidget) -> None:
        """Append a widget to the toolbar."""
        self._layout.addWidget(widget)

    def add_stretch(self, stretch: int = 1) -> None:
        """Append an expanding spacer."""
        self._layout.addStretch(stretch)

    # ── sizing ────────────────────────────────────────────────────

    def _apply_metrics(self) -> None:
        t = theme()
        self.setFixedHeight(t.row_height)
        self._layout.setContentsMargins(t.sp_sm, t.sp_xxs, t.sp_sm, t.sp_xxs)
        self._layout.setSpacing(t.sp_xs)

    # ── painting ──────────────────────────────────────────────────

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        p = QPainter(self)
        t = theme()
        p.fillRect(self.rect(), qcolor(t.bg_raised))
        p.setPen(qcolor(t.border_subtle))
        y = self.height() - 1
        p.drawLine(0, y, self.width(), y)
        p.end()

    def changeEvent(self, a0: QEvent | None) -> None:
        if a0 is not None and a0.type() == QEvent.Type.StyleChange:
            self._apply_metrics()
            self.update()
        super().changeEvent(a0)
