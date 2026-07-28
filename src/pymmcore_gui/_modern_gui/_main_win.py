"""Top-level window for the simplified GUI iteration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_plus import CMMCorePlus

from pymmcore_gui._qt.QtCore import QEvent, QRectF, QSize, Qt, QTimer, pyqtSignal
from pymmcore_gui._qt.QtGui import (
    QCloseEvent,
    QEnterEvent,
    QFontMetricsF,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QShortcut,
)
from pymmcore_gui._qt.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QToolBar,
    QWidget,
)

from ._acquire import AcquirePage
from ._configurations import ConfigurationsPage
from ._hardware import HardwareSetupPage
from ._theme import (
    qcolor,
    reset_zoom,
    set_theme,
    theme,
    ui_font,
    zoom_in,
    zoom_out,
)
from ._theme._dark import DARK_THEME
from ._theme._light import LIGHT_THEME

if TYPE_CHECKING:
    from collections.abc import Sequence


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
        fm = QFontMetricsF(ui_font())
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

    def paintEvent(self, a0: QPaintEvent | None) -> None:
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

        p.setFont(ui_font())
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

    def leaveEvent(self, a0: QEvent | None) -> None:
        self._hovered = False
        self.update()

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:
        if a0 is not None and a0.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()


class ModeTabBar(QWidget):
    """Horizontal bar of mode tabs; emits the selected index on click."""

    current_changed = pyqtSignal(int)

    def __init__(self, labels: Sequence[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(theme().sp_sm, 0, 0, 0)
        layout.setSpacing(0)

        self._tabs: list[ModeTab] = []
        for index, label in enumerate(labels):
            tab = ModeTab(label)
            tab.clicked.connect(lambda _i=index: self._select(_i))
            layout.addWidget(tab)
            self._tabs.append(tab)

        layout.addStretch()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        if self._tabs:
            self._tabs[0].active = True

    def _select(self, index: int) -> None:
        for i, tab in enumerate(self._tabs):
            tab.active = i == index
        self.current_changed.emit(index)

    def changeEvent(self, a0: QEvent | None) -> None:
        if a0 is not None and a0.type() == QEvent.Type.StyleChange:
            t = theme()
            if lay := self.layout():
                lay.setContentsMargins(t.sp_sm, 0, 0, 0)
        super().changeEvent(a0)


class MainWindow(QMainWindow):
    """Top-level window: mode tabs over a stack of (empty) tab pages."""

    TAB_LABELS = ("Hardware Setup", "Configurations", "Acquire")

    def __init__(self, *, mmcore: CMMCorePlus | None = None) -> None:
        super().__init__()

        set_theme(DARK_THEME)

        self._mmc = mmcore or CMMCorePlus.instance()
        self.setObjectName("MicroManagerGUI")
        self.setWindowTitle("pymmcore-gui")
        self.resize(1800, 1200)

        # ── top toolbar: mode tabs + theme toggle ─────────────────
        self._toolbar = QToolBar()
        self._toolbar.setMovable(False)
        self._toolbar.setFloatable(False)
        self._toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)

        self._mode_tabs = ModeTabBar(self.TAB_LABELS)
        self._toolbar.addWidget(self._mode_tabs)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._toolbar.addWidget(spacer)

        self._theme_btn = QPushButton("☀")
        self._theme_btn.setFixedSize(32, 32)
        self._theme_btn.setToolTip("Toggle light/dark theme")
        self._theme_btn.clicked.connect(self._toggle_theme)
        self._is_dark = True
        self._toolbar.addWidget(self._theme_btn)

        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self._toolbar)

        # ── central stack: one page per tab ───────────────────────
        self._stack = QStackedWidget()
        self._hardware = HardwareSetupPage(self._mmc)
        self._configurations = ConfigurationsPage(self._mmc)
        self._acquire = AcquirePage(self._mmc)
        self._stack.addWidget(self._hardware)
        self._stack.addWidget(self._configurations)
        self._stack.addWidget(self._acquire)
        self.setCentralWidget(self._stack)

        # The toolbar action commits its selected editor, then saves the whole
        # configuration. Closing with unsaved changes still commits both.
        self._configurations.saveToFileRequested.connect(
            self._save_current_configuration
        )

        self._mode_tabs.current_changed.connect(self._stack.setCurrentIndex)
        self._stack.setCurrentIndex(0)

        if status_bar := self.statusBar():
            status_bar.showMessage("Ready")

        # ── zoom shortcuts ────────────────────────────────────────
        mods = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
        QShortcut(QKeySequence(mods | Qt.Key.Key_Equal), self, zoom_in)  # type: ignore
        QShortcut(QKeySequence(mods | Qt.Key.Key_Plus), self, zoom_in)  # type: ignore
        QShortcut(QKeySequence(mods | Qt.Key.Key_Minus), self, zoom_out)  # type: ignore
        QShortcut(QKeySequence(mods | Qt.Key.Key_0), self, reset_zoom)  # type: ignore

    def on_startup_configuration_loaded(self) -> None:
        """Land on Acquire after the application loads its initial config."""
        self._activate_acquire()
        # Explicit -c loads finish before app.exec(). Repeat once the event loop
        # starts so platform-specific window initialization cannot restore the
        # initial Hardware selection afterward.
        QTimer.singleShot(0, self._activate_acquire)

    def _activate_acquire(self) -> None:
        """Keep the mode tab and its stacked page on Acquire."""
        idx = self._stack.indexOf(self._acquire)
        if idx >= 0:
            self._mode_tabs._select(idx)
            self._stack.setCurrentIndex(idx)

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Offer to save hardware / group / pixel edits before closing."""
        if self._hardware.is_dirty() or self._configurations.is_dirty():
            choice = QMessageBox.question(
                self,
                "Unsaved changes",
                "There are unsaved changes to the configuration "
                "(hardware, groups or pixel sizes).\n\n"
                "Save them to a .cfg file before closing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if choice == QMessageBox.StandardButton.Cancel:
                if a0 is not None:
                    a0.ignore()
                return
            if choice == QMessageBox.StandardButton.Save and not self._save_all():
                # save was cancelled or failed — don't close
                if a0 is not None:
                    a0.ignore()
                return
        super().closeEvent(a0)

    def _save_all(self) -> bool:
        """Commit group/pixel edits to the core, then save everything to a .cfg.

        Returns True if a file was written, False if cancelled or on error.
        """
        self._configurations.commit_to_core()
        if self._hardware.save_config():
            self._configurations.mark_saved()
            return True
        return False

    def _save_current_configuration(self) -> bool:
        """Commit the selected configuration editor, then write the full .cfg."""
        self._configurations.commit_current_to_core()
        if self._hardware.save_config():
            self._configurations.mark_current_saved()
            return True
        return False

    def _toggle_theme(self) -> None:
        self._is_dark = not self._is_dark
        set_theme(DARK_THEME if self._is_dark else LIGHT_THEME)
        self._theme_btn.setText("☀" if self._is_dark else "🌙")

    @property
    def mmcore(self) -> CMMCorePlus | None:
        """Access to the microscope core, if provided."""
        return self._mmc
