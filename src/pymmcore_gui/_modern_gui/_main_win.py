from __future__ import annotations

from pymmcore_plus import CMMCorePlus

from pymmcore_gui._modern_gui._sidebar import Sidebar
from pymmcore_gui._modern_gui._viewport import ImageViewport
from pymmcore_gui._qt.QtCore import QEvent, QRectF, QSize, Qt, pyqtSignal
from pymmcore_gui._qt.QtGui import (
    QEnterEvent,
    QFont,
    QFontMetricsF,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QShortcut,
)
from pymmcore_gui._qt.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QSizePolicy,
    QToolBar,
    QWidget,
)

from ._theme import (
    MicroscopeStyle,
    qcolor,
    reset_zoom,
    set_style,
    set_theme,
    theme,
    ui_font,
    zoom_in,
    zoom_out,
)
from ._theme._dark import DARK_THEME


class ModeTab(QWidget):
    """Single mode tab, custom-painted with optional active underline."""

    _BASE_HEIGHT = 40

    clicked = pyqtSignal()

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = label
        self._active = False
        self._hovered = False

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        t = theme()
        fm = QFontMetricsF(ui_font(11, QFont.Weight.Medium))
        w = int(fm.horizontalAdvance(self._label)) + t.sp_lg * 2
        return QSize(w, t.scaled(self._BASE_HEIGHT))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, val: bool) -> None:
        self._active = val
        self.update()

    def paintEvent(self, event: QPaintEvent | None) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        t = theme()
        w, h = self.width(), self.height()
        underline_h = t.scaled(3)

        # Text
        if self._active:
            text_color = qcolor(t.accent)
        elif self._hovered:
            text_color = qcolor(t.text_primary)
        else:
            text_color = qcolor(t.text_secondary)

        p.setFont(ui_font(11, QFont.Weight.Medium))
        p.setPen(text_color)
        p.drawText(
            QRectF(0, 0, w, h - underline_h),
            Qt.AlignmentFlag.AlignCenter,
            self._label,
        )

        # Active underline
        if self._active:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qcolor(t.accent))
            bar_w = w - t.sp_sm * 2
            p.drawRoundedRect(
                QRectF((w - bar_w) / 2, h - underline_h, bar_w, underline_h),
                1.5,
                1.5,
            )

        p.end()

    def enterEvent(self, event: QEnterEvent | None) -> None:
        self._hovered = True
        self.update()

    def leaveEvent(self, event: QEvent | None) -> None:
        self._hovered = False
        self.update()

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()


class ModeTabBar(QWidget):
    """Horizontal bar of mode tabs (Acquire / Process / Analyze)."""

    mode_changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        labels = ["Configure", "Acquire", "Process", "Analyze"]
        layout = QHBoxLayout(self)
        layout.setContentsMargins(theme().sp_sm, 0, 0, 0)
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

    def changeEvent(self, event: QEvent | None) -> None:
        if event is not None and event.type() == QEvent.Type.StyleChange:
            t = theme()
            if lay := self.layout():
                lay.setContentsMargins(t.sp_sm, 0, 0, 0)
        super().changeEvent(event)


class MainWindow(QMainWindow):
    """Top-level window with sidebar + viewport."""

    def __init__(self, *, mmcore: CMMCorePlus | None = None) -> None:
        super().__init__()

        # App-level style + theme setup
        if isinstance(qapp := QApplication.instance(), QApplication):
            style = MicroscopeStyle()
            qapp.setStyle(style)
            set_style(style)
            set_theme(DARK_THEME)

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

        # Zoom shortcuts
        mods = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
        QShortcut(QKeySequence(mods | Qt.Key.Key_Equal), self, zoom_in)  # type: ignore
        QShortcut(QKeySequence(mods | Qt.Key.Key_Plus), self, zoom_in)  # type: ignore
        QShortcut(QKeySequence(mods | Qt.Key.Key_Minus), self, zoom_out)  # type: ignore
        QShortcut(QKeySequence(mods | Qt.Key.Key_0), self, reset_zoom)  # type: ignore

    @property
    def mmcore(self) -> CMMCorePlus | None:
        """Access to the microscope core, if provided."""
        return self._mmc
