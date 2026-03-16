from __future__ import annotations

from pymmcore_plus import CMMCorePlus

from pymmcore_gui._modern_gui._sidebar import Sidebar
from pymmcore_gui._modern_gui._viewport import ImageViewport
from pymmcore_gui._qt.QtCore import QRectF, Qt, pyqtSignal
from pymmcore_gui._qt.QtGui import (
    QFont,
    QFontMetricsF,
    QMouseEvent,
    QPainter,
)
from pymmcore_gui._qt.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QSizePolicy,
    QToolBar,
    QWidget,
)

from ._theme import MicroscopeStyle, Sp, make_dark_palette, qcolor, theme, ui_font

UNDERLINE_H = 3
TAB_FONT_SIZE = 11


class ModeTab(QWidget):
    """Single mode tab, custom-painted with optional active underline."""

    clicked = pyqtSignal()

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = label
        self._active = False
        self._hovered = False

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

        fm = QFontMetricsF(ui_font(TAB_FONT_SIZE, QFont.Weight.Medium))
        w = int(fm.horizontalAdvance(label)) + Sp.LG * 2
        self.setFixedSize(w, 40)

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, val: bool) -> None:
        self._active = val
        self.update()

    def paintEvent(self, event: object) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        t = theme()
        w, h = self.width(), self.height()

        # Text
        if self._active:
            text_color = qcolor(t.accent)
        elif self._hovered:
            text_color = qcolor(t.text_primary)
        else:
            text_color = qcolor(t.text_secondary)

        p.setFont(ui_font(TAB_FONT_SIZE, QFont.Weight.Medium))
        p.setPen(text_color)
        p.drawText(
            QRectF(0, 0, w, h - UNDERLINE_H),
            Qt.AlignmentFlag.AlignCenter,
            self._label,
        )

        # Active underline
        if self._active:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qcolor(t.accent))
            bar_w = w - Sp.SM * 2
            p.drawRoundedRect(
                QRectF((w - bar_w) / 2, h - UNDERLINE_H, bar_w, UNDERLINE_H),
                1.5,
                1.5,
            )

        p.end()

    def enterEvent(self, event: object) -> None:
        self._hovered = True
        self.update()

    def leaveEvent(self, event: object) -> None:
        self._hovered = False
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()


class ModeTabBar(QWidget):
    """Horizontal bar of mode tabs (Acquire / Process / Analyze)."""

    mode_changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        labels = ["Configure", "Acquire", "Process", "Analyze"]
        layout = QHBoxLayout(self)
        layout.setContentsMargins(Sp.SM, 0, 0, 0)
        layout.setSpacing(0)

        self._tabs: list[ModeTab] = []
        for label in labels:
            tab = ModeTab(label)
            tab.clicked.connect(lambda _l=label: self._select(_l))
            layout.addWidget(tab)
            self._tabs.append(tab)

        layout.addStretch()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        if self._tabs:
            self._tabs[1].active = True

    def _select(self, label: str) -> None:
        for tab in self._tabs:
            tab.active = tab._label == label
        self.mode_changed.emit(label)


class MainWindow(QMainWindow):
    """Top-level window with sidebar + viewport."""

    def __init__(self, *, mmcore: CMMCorePlus | None = None) -> None:
        super().__init__()

        # MOVE ME
        if isinstance(qapp := QApplication.instance(), QApplication):
            qapp.setStyle(MicroscopeStyle())
            qapp.setPalette(make_dark_palette())
            qapp.setFont(ui_font(10))

        self._mmc = mmcore or CMMCorePlus.instance()
        self.setWindowTitle("Microscope Control — Panel Mockup")
        self.resize(900, 700)

        self._toolbar = QToolBar()
        self._toolbar.setMovable(False)
        self._toolbar.setFloatable(False)
        self._toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self._mode_tabs = ModeTabBar()
        self._toolbar.addWidget(self._mode_tabs)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self._toolbar)

        central = QWidget()
        self.setCentralWidget(central)

        self._viewport = ImageViewport()

        if status_bar := self.statusBar():
            status_bar.showMessage("Ready")

        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(Sidebar())
        layout.addWidget(self._viewport)

    @property
    def mmcore(self) -> CMMCorePlus | None:
        """Access to the microscope core, if provided."""
        return self._mmc
