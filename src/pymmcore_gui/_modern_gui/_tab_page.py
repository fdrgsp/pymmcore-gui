"""Reusable tab page shell: toolbar + left dock + center."""

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
        ┌──────┬────────────┐
        │ left │  content   │
        └──────┴────────────┘

    All regions are empty placeholders exposed as attributes (``toolbar``,
    ``left``, ``content``) so each page can populate the ones it needs. The
    left dock is resizable via a splitter.

    ``AcquirePage`` uses only ``toolbar`` and ``content`` -- it fills the
    latter with a QtAds ``CDockManager``, which supplies its own docking on
    all four sides.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.toolbar = TabToolBar()
        self.left = Sidebar(SidebarPosition.LEFT)
        self.content = QWidget()
        self.content.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._content_layout = QVBoxLayout(self.content)
        m = theme().sp_sm
        self._content_layout.setContentsMargins(m, m, m, m)
        self._content_layout.setSpacing(m)

        # left | content
        self._h_split = QSplitter(Qt.Orientation.Horizontal)
        self._h_split.addWidget(self.left)
        self._h_split.addWidget(self.content)
        self._h_split.setStretchFactor(0, 0)
        self._h_split.setStretchFactor(1, 1)

        self._page_layout = QVBoxLayout(self)
        self._page_layout.setContentsMargins(0, 0, 0, 0)
        self._page_layout.setSpacing(0)
        self._page_layout.addWidget(self.toolbar)
        self._page_layout.addWidget(self._h_split)

        self._seed_sizes()

    def add_content_widget(self, widget: QWidget) -> None:
        """Add a widget to the central content area."""
        self._content_layout.addWidget(widget)

    def add_toolbar_row(self, widget: QWidget) -> None:
        """Insert *widget* as a full-width row directly below ``toolbar``."""
        self._page_layout.insertWidget(self._page_layout.indexOf(self._h_split), widget)

    def _seed_sizes(self) -> None:
        """Seed initial splitter sizes from the theme."""
        side = theme().sidebar_width
        self._h_split.setSizes([side, max(side, 600)])
