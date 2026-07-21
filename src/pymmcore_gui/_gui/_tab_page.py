"""Reusable tab page shell: toolbar + left/right/bottom docks + center."""

from __future__ import annotations

from pymmcore_gui._qt.QtCore import Qt
from pymmcore_gui._qt.QtWidgets import (
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ._sidebar import Sidebar, SidebarPosition
from ._theme import theme
from ._toolbar import TabToolBar


class TabPage(QWidget):
    """Empty structured page used for each tab.

    Layout::

        toolbar
        ┌──────┬────────────┬───────┐
        │ left │  content   │ right │
        ├──────┴────────────┴───────┤
        │          bottom           │
        └───────────────────────────┘

    All regions are empty placeholders exposed as attributes (``toolbar``,
    ``left``, ``right``, ``bottom``, ``content``) so later iterations can
    populate any of them. The docks are resizable via nested splitters.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.toolbar = TabToolBar()
        self.left = Sidebar(SidebarPosition.LEFT)
        self.right = Sidebar(SidebarPosition.RIGHT)
        self.bottom = Sidebar(SidebarPosition.BOTTOM)
        self.content = QWidget()
        self.content.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._content_layout = QVBoxLayout(self.content)
        m = theme().sp_sm
        self._content_layout.setContentsMargins(m, m, m, m)
        self._content_layout.setSpacing(m)

        # left | content | right
        self._h_split = QSplitter(Qt.Orientation.Horizontal)
        self._h_split.addWidget(self.left)
        self._h_split.addWidget(self.content)
        self._h_split.addWidget(self.right)
        self._h_split.setStretchFactor(0, 0)
        self._h_split.setStretchFactor(1, 1)
        self._h_split.setStretchFactor(2, 0)

        # (row above) / bottom
        self._v_split = QSplitter(Qt.Orientation.Vertical)
        self._v_split.addWidget(self._h_split)
        self._v_split.addWidget(self.bottom)
        self._v_split.setStretchFactor(0, 1)
        self._v_split.setStretchFactor(1, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self._v_split)

        self._seed_sizes()

    def add_content_widget(self, widget: QWidget) -> None:
        """Add a widget to the central content area."""
        self._content_layout.addWidget(widget)

    def _seed_sizes(self) -> None:
        """Seed initial splitter sizes from the theme."""
        side = theme().sidebar_width
        self._h_split.setSizes([side, max(side, 600), side])
        self._v_split.setSizes([600, self.bottom.sizeHint().height()])
