from __future__ import annotations

from pymmcore_gui._modern_gui._theme import qcolor, theme
from pymmcore_gui._qt.QtCore import QSize, Qt
from pymmcore_gui._qt.QtGui import QPainter, QPaintEvent, QPen
from pymmcore_gui._qt.QtWidgets import (
    QFrame,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ._camera_panel import CollapsibleCameraPanel
from ._collapsible_panel import CollapsiblePanel, PlaceholderContent, SidebarContent
from ._objectives_panel import CollapsibleObjectivesPanel
from ._xy_stage_panel import CollapsibleXYStagePanel


class Sidebar(QWidget):
    """Scrollable sidebar containing collapsible panels."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        # Scroll content
        content = SidebarContent()
        self._layout = QVBoxLayout(content)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)

        # Outer layout (1px right margin for border)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 1, 0)
        outer.setSpacing(0)
        outer.addWidget(scroll)

        # Panels
        self._layout.addWidget(CollapsibleXYStagePanel(self))
        self._layout.addWidget(CollapsibleObjectivesPanel(self))
        self._layout.addWidget(CollapsibleCameraPanel(self))

        ch = CollapsiblePanel(title="Channels", summary="DAPI \u00b7 GFP \u00b7 Cy5")
        ch.body_layout.addWidget(
            PlaceholderContent(
                [
                    "\u25a0 DAPI    50 ms   \ud83d\udc41",
                    "\u25a0 GFP    100 ms   \ud83d\udc41",
                    "\u25a0 Cy5    200 ms   \ud83d\udc41",
                ]
            )
        )
        self._layout.addWidget(ch)

        hist = CollapsiblePanel(
            title="Histogram", summary="0 - 4095", show_status_dot=False
        )
        hist.body_layout.addWidget(
            PlaceholderContent(
                [
                    "\u250c\u2500\u2500\u2500 histogram \u2500\u2500\u2500\u2510",
                    "\u2502\u2593\u2593\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2502",
                    "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500-\u2518",
                    "[\u26a1 Auto] [\u21ba Reset] [\u33d2] [\u25d0]",
                ]
            )
        )
        self._layout.addWidget(hist)

        acq = CollapsiblePanel(
            title="Acquisition", summary="Zx50 \u00b7 Tx100", show_status_dot=False
        )
        acq.body_layout.addWidget(
            PlaceholderContent(
                [
                    "\u2611 Z-Stack      50 sl \u00b7 0.5 \u03bcm",
                    "\u2611 Time Series  100 x 30 s",
                    "\u2610 Tile Scan    \u2014",
                    "\u2610 Positions    \u2014",
                    "",
                    "Frames:    15,000",
                    "Duration:  50 min",
                    "Storage:   11.2 GB",
                ]
            )
        )
        self._layout.addWidget(acq)

    def sizeHint(self) -> QSize:
        return QSize(theme().sidebar_width, super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        p = QPainter(self)
        p.setPen(QPen(qcolor(theme().border_subtle), 1))
        p.drawLine(self.width() - 1, 0, self.width() - 1, self.height())
        p.end()
