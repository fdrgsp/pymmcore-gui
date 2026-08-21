"""Top-level window for the simplified GUI iteration."""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from pymmcore_plus import CMMCorePlus, Keyword, find_micromanager
from superqt.iconify import QIconifyIcon

from pymmcore_gui._array_viewer import set_source_icon
from pymmcore_gui._layouts import (
    LAST_SESSION_LAYOUT_NAME,
    AcquireLayout,
    store_session_layout,
)
from pymmcore_gui._notification_manager import NotificationManager
from pymmcore_gui._qt.QtCore import QEvent, QRectF, QSize, Qt, QTimer, Signal
from pymmcore_gui._qt.QtGui import (
    QAction,
    QCloseEvent,
    QEnterEvent,
    QFontMetricsF,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QShortcut,
)
from pymmcore_gui._qt.QtOpenGLWidgets import QOpenGLWidget
from pymmcore_gui._qt.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QToolBar,
    QWidget,
)
from pymmcore_gui._settings import Settings

from ._acquire import AcquirePage
from ._configurations import ConfigurationsPage
from ._hardware import HardwareSetupPage
from ._installation import InstallationPage
from ._panels import PanelKey
from ._startup import StartupChoice, StartupDialog
from ._theme import (
    qcolor,
    reset_zoom,
    set_theme,
    set_zoom_step,
    theme,
    ui_font,
    zoom_factor,
    zoom_in,
    zoom_out,
)
from ._theme._dark import DARK_THEME
from ._theme._light import LIGHT_THEME

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymmcore_gui._app import MMQApplication
    from pymmcore_gui._notification_manager import Notification
    from pymmcore_gui.widgets._exception_log import ExceptionLog


def apply_saved_appearance() -> bool:
    """Apply the user's saved theme/zoom, returning whether the theme is dark.

    ``_app.create_mmgui`` already calls ``set_theme(DARK_THEME)`` before any
    window exists (installing ``MicroscopeStyle``, which the classic GUI also
    depends on) -- that call stays. This applies the user's actual preference
    before the first widget is constructed and before the window is ever
    shown (``show()`` happens in ``restore_state``), so there's no flash and
    no risk of unconditionally clobbering a restored light theme.

    Module-level rather than a ``MainWindow`` method because the startup
    dialog is themed too and runs before any window exists. Calling it twice
    (dialog, then window) is harmless -- it only ever re-applies the same
    stored values.
    """
    prefs = Settings.instance().modern_window
    is_dark = prefs.theme != "light"
    set_theme(DARK_THEME if is_dark else LIGHT_THEME)
    if prefs.zoom is not None:
        set_zoom_step(prefs.zoom)
    return is_dark


class ModeTab(QWidget):
    """Single mode tab, custom-painted with optional active underline."""

    _BASE_HEIGHT = 40

    clicked = Signal()

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
        if not self.isEnabled():
            text_color = qcolor(t.text_disabled)
        elif self._active:
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
        if (
            self.isEnabled()
            and a0 is not None
            and a0.button() == Qt.MouseButton.LeftButton
        ):
            self.clicked.emit()


class ModeTabBar(QWidget):
    """Horizontal bar of mode tabs; emits the selected index on click."""

    current_changed = Signal(int)

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
        if not 0 <= index < len(self._tabs) or not self._tabs[index].isEnabled():
            return
        for i, tab in enumerate(self._tabs):
            tab.active = i == index
        self.current_changed.emit(index)

    def setTabEnabled(self, index: int, enabled: bool) -> None:
        """Enable or disable a mode tab."""
        if 0 <= index < len(self._tabs):
            self._tabs[index].setEnabled(enabled)

    def changeEvent(self, a0: QEvent | None) -> None:
        if a0 is not None and a0.type() == QEvent.Type.StyleChange:
            t = theme()
            if lay := self.layout():
                lay.setContentsMargins(t.sp_sm, 0, 0, 0)
        super().changeEvent(a0)


class NotificationBellButton(QPushButton):
    """Status-bar bell that pops up recent notification history.

    Chrome, not a state indicator -- same "text_secondary, rebuild on
    StyleChange" treatment as ``LayoutMenuButton`` in ``_acquire_toolbar``,
    except it turns ``status_red`` while notifications are waiting to be
    looked at, resetting the moment the bell is clicked open.
    """

    _ICON = "codicon:bell"
    _MAX_HISTORY = 20
    _SEVERITY_ICON: ClassVar[dict[str, tuple[str, str]]] = {
        "error": ("codicon:error", "status_red"),
        "warning": ("codicon:warning", "status_amber"),
        "info": ("codicon:info", "accent"),
    }

    def __init__(
        self, manager: NotificationManager, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._unread = 0
        self.setFlat(True)
        self.setProperty("variant", "subtle")
        self.setFixedSize(24, 24)
        self.setToolTip("Notifications")
        self.clicked.connect(self._popup)
        manager.notificationAdded.connect(self._on_notification_added)
        self._apply_icon()

    def _on_notification_added(self, _notification: Notification) -> None:
        self._unread += 1
        self._apply_icon()

    def _popup(self) -> None:
        self._unread = 0
        self._apply_icon()
        self._build_menu().exec(self.mapToGlobal(self.rect().bottomLeft()))

    def _build_menu(self) -> QMenu:
        menu = QMenu(self)
        history = list(self._manager.notifications())[-self._MAX_HISTORY :]
        if not history:
            empty = QAction("No notifications", menu)
            empty.setEnabled(False)
            menu.addAction(empty)
            return menu
        for notification in reversed(history):
            icon_name, color_attr = self._SEVERITY_ICON.get(
                notification.severity, self._SEVERITY_ICON["info"]
            )
            icon = QIconifyIcon(
                icon_name, color=qcolor(getattr(theme(), color_attr)).name()
            )
            when = datetime.fromtimestamp(notification.timestamp).strftime("%H:%M:%S")
            text = (notification.message.splitlines() or [""])[0]
            if len(text) > 80:
                text = text[:77] + "…"
            action = QAction(icon, f"{when}   {text}", menu)
            # Only notifications with a primary action (e.g. the exception
            # toast's "See traceback") are actionable from here; plain info
            # entries are just a read-only record of what happened.
            if notification.actions and notification.on_action is not None:
                action.triggered.connect(
                    partial(notification.on_action, notification.actions[0])
                )
            else:
                action.setEnabled(False)
            menu.addAction(action)
        return menu

    def _apply_icon(self) -> None:
        color = theme().status_red if self._unread else theme().text_secondary
        set_source_icon(self, QIconifyIcon(self._ICON, color=qcolor(color).name()))
        size = theme().scaled(16)
        self.setIconSize(QSize(size, size))

    def changeEvent(self, e: QEvent | None) -> None:
        if e is not None and e.type() == QEvent.Type.StyleChange:
            self._apply_icon()
        super().changeEvent(e)


class MainWindow(QMainWindow):
    """Top-level window: mode tabs over a stack of (empty) tab pages."""

    TAB_LABELS = ("Installation", "Hardware Setup", "Configurations", "Acquire")

    def __init__(self, *, mmcore: CMMCorePlus | None = None) -> None:
        super().__init__()

        self._apply_saved_appearance()

        self._mmc = mmcore or CMMCorePlus.instance()
        self.setObjectName("pyMMGUI")
        self.setWindowTitle("pyMM")
        self.setWindowState(Qt.WindowState.WindowMaximized)

        self._notification_manager = NotificationManager(self)
        self._bell_button = NotificationBellButton(self._notification_manager, self)
        if app := QApplication.instance():
            if hasattr(app, "exceptionRaised"):
                cast("MMQApplication", app).exceptionRaised.connect(self._on_exception)

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

        self._theme_btn = QPushButton("☀" if self._is_dark else "🌙")
        self._theme_btn.setFixedSize(32, 32)
        self._theme_btn.setToolTip("Toggle light/dark theme")
        self._theme_btn.clicked.connect(self._toggle_theme)
        self._toolbar.addWidget(self._theme_btn)

        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self._toolbar)

        # ── central stack: one page per tab ───────────────────────
        self._stack = QStackedWidget()
        self._installation = InstallationPage()
        self._installation.aboutToUninstall.connect(self._prepare_uninstall)
        self._hardware = HardwareSetupPage(self._mmc)
        self._configurations = ConfigurationsPage(self._mmc)
        self._acquire = AcquirePage(self._mmc)
        self._stack.addWidget(self._installation)
        self._stack.addWidget(self._hardware)
        self._stack.addWidget(self._configurations)
        self._stack.addWidget(self._acquire)
        self.setCentralWidget(self._stack)

        # Adding a QOpenGLWidget (e.g. ndv canvas) to a window that uses raster
        # rendering forces Qt to destroy and recreate the native window with an
        # OpenGL-compatible surface, causing a visible flash. Adding a zero-size
        # QOpenGLWidget before the first show() ensures the window is born with
        # the right surface type, avoiding the flash. Without this, the first
        # snap/MDA run -- whichever creates the first viewer canvas -- flickers.
        _gl = QOpenGLWidget(self)
        _gl.setFixedSize(0, 0)
        _gl.close()

        # The toolbar action commits its selected editor, then saves the whole
        # configuration. Closing with unsaved changes still commits both.
        self._configurations.saveToFileRequested.connect(
            self._save_current_configuration
        )
        self._configurations.calibrationRunningChanged.connect(
            self._on_pixel_calibration_running
        )
        self._configurations.pixelConfigurationsApplied.connect(
            self._acquire.refresh_stage_explorer_pixel_geometry
        )

        self._acquire.layoutReset.connect(self._on_acquire_layout_reset)
        self._acquire.layoutNameChanged.connect(self._on_layout_name_changed)
        self._mmc.events.systemConfigurationLoaded.connect(self._on_config_loaded)
        self._installation.activeInstallChanged.connect(self._on_active_install_changed)

        self._mode_tabs.current_changed.connect(self._on_mode_tab_changed)
        self._select_startup_tab()

        if status_bar := self.statusBar():
            status_bar.showMessage("Ready")
            status_bar.addPermanentWidget(self._bell_button)

        # ── zoom shortcuts ────────────────────────────────────────
        mods = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
        QShortcut(QKeySequence(mods | Qt.Key.Key_Equal), self, zoom_in)  # type: ignore
        QShortcut(QKeySequence(mods | Qt.Key.Key_Plus), self, zoom_in)  # type: ignore
        QShortcut(QKeySequence(mods | Qt.Key.Key_Minus), self, zoom_out)  # type: ignore
        QShortcut(QKeySequence(mods | Qt.Key.Key_0), self, reset_zoom)  # type: ignore

    def _apply_saved_appearance(self) -> None:
        """Apply the saved theme/zoom before any widget exists, so nothing flashes."""
        self._is_dark = apply_saved_appearance()

    @staticmethod
    def prompt_startup_choices(layout: str | None = None) -> StartupChoice | None:
        """Ask which layout and configuration to launch with.

        Called by ``_app.create_mmgui`` (via ``getattr``, like
        :meth:`restore_state`) when no config was given on the command line,
        *before* any window exists -- hence the staticmethod. *layout* is the
        ``-l`` flag, which preselects the layout field rather than skipping
        the dialog: it answers only half of what the dialog asks. Returns
        None if the user chose to quit.
        """
        # The dialog is themed, and it's shown before MainWindow.__init__ has
        # had a chance to apply the user's saved theme, so do that here.
        apply_saved_appearance()
        dialog = StartupDialog(preselect_layout=layout)
        try:
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return None
            choice = dialog.value()
        finally:
            dialog.deleteLater()

        settings = Settings.instance()
        settings.modern_window.last_layout = choice.layout
        settings.flush()
        return choice

    def restore_state(self, *, show: bool = False, layout: str | None = None) -> None:
        """Restore window geometry and an Acquire dock layout, then optionally show.

        Detected and called by ``_app.create_mmgui`` via ``hasattr`` --
        adding this method means the app no longer calls ``show()`` directly,
        so *this* method must, when ``show`` is True.

        *layout* is the name chosen in the startup dialog. When it's None --
        no dialog ran, e.g. a ``-c`` launch or an embedding caller -- the
        last-session arrangement is restored, which is what this always did.
        """
        prefs = Settings.instance().modern_window
        if geo := prefs.geometry:
            self.restoreGeometry(geo)
        name = layout or LAST_SESSION_LAYOUT_NAME
        self._acquire.select_layout(name)
        if show:
            self.show()
            self._notification_manager.reposition_notifications()

    def _on_acquire_layout_reset(self) -> None:
        """Drop the persisted "Last session" layout so a crash can't resurrect it.

        ``_save_state`` would write the freshly-reset arrangement on close
        anyway; clearing now just means the reset also survives an abnormal
        exit. Scoped to the layout keys only -- geometry, theme and zoom are
        preferences, not layout. Named layouts are untouched: resetting the
        page is not deleting anything the user saved.
        """
        settings = Settings.instance()
        store_session_layout(AcquireLayout())
        settings.flush()

    def _on_layout_name_changed(self, name: str) -> None:
        """Remember which layout to preselect in the next startup dialog."""
        settings = Settings.instance()
        settings.modern_window.last_layout = name
        settings.flush()

    def _save_state(self) -> None:
        """Persist geometry, the "Last session" layout, theme, and zoom.

        The live arrangement is always written to the reserved session slot,
        never back into whichever named layout it came from -- a named layout
        changes only when the user explicitly saves it.
        """
        settings = Settings.instance()
        prefs = settings.modern_window
        prefs.geometry = self.saveGeometry().data()
        store_session_layout(self._acquire.current_layout())
        prefs.theme = "dark" if self._is_dark else "light"
        prefs.zoom = zoom_factor()
        settings.flush(timeout=5000)

    def _on_config_loaded(self, *_: object) -> None:
        """Offer whatever config the core just loaded in the next startup dialog.

        Covers every route into the core -- the startup dialog, ``-c``, and
        the Hardware page's own Load button -- since they all end in
        ``loadSystemConfiguration``.
        """
        if cfg := self._mmc.systemConfigurationFile():
            Settings.instance().remember_config(cfg)

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

    def _select_startup_tab(self) -> None:
        """Open on Hardware Setup — or on Installation if there's nothing to run.

        Installation leads the tab order because it leads the workflow, but
        it's a once-in-a-while errand: landing there every launch would put a
        page nobody asked for (and the network fetch that fills it) in front of
        the actual work. Without a Micro-Manager install, though, every other
        tab is a dead end, so that's where the session starts.
        """
        found = find_micromanager(return_first=True)
        index = self._stack.indexOf(self._hardware if found else self._installation)
        self._mode_tabs._select(index)
        self._stack.setCurrentIndex(index)

    def _prepare_uninstall(self, paths: set[str]) -> None:
        """Release a device adapter DLL before its install directory is deleted.

        Windows keeps a still-loaded ``mmgr_dal_*.dll`` locked against deletion
        until the process that loaded it lets go -- so uninstalling whichever
        install is actively driving the connected hardware failed with
        ``PermissionError: [WinError 5] Access is denied`` on that DLL, even
        though the user had already confirmed the delete. Only unloads when
        one of *paths* is actually the core's current adapter search path
        (comparing resolved paths, since Windows paths are case-insensitive);
        an unrelated, unused old install never touches the live session.
        Best-effort, matching ``HardwareSetupPage._start_over``'s own
        unconditional ``unloadAllDevices()`` before a search-path change.
        """
        core_device = Keyword.CoreDevice.value
        if not [d for d in self._mmc.getLoadedDevices() if d != core_device]:
            return
        current = {Path(p).resolve() for p in self._mmc.getDeviceAdapterSearchPaths()}
        if not any(Path(p).resolve() in current for p in paths):
            return
        with suppress(Exception):
            self._mmc.unloadAllDevices()

    def _on_active_install_changed(self, path: str) -> None:
        """Follow a switch of the active Micro-Manager install, or defer it.

        ``use_micromanager`` (behind the Installation page's "Set Active")
        writes a preference that every *future* session reads, but the running
        core took its adapter search path at construction and keeps it. Rather
        than let the two disagree silently, offer to restart this session's
        hardware on the newly chosen install right away.
        """
        if not path:
            # the last install was removed; there's nothing to point the core at
            self._status("No Micro-Manager installation left.")
            return
        core_device = Keyword.CoreDevice.value
        loaded = [d for d in self._mmc.getLoadedDevices() if d != core_device]
        at_stake = (
            bool(loaded) or self._hardware.is_dirty() or self._configurations.is_dirty()
        )
        if at_stake and not self._confirm_install_switch(path):
            self._status(f"{Path(path).name} will be used the next time pyMM starts.")
            return
        self._hardware.use_adapter_path(path)
        self._status(f"Now using the device adapters in {path}")

    def _confirm_install_switch(self, path: str) -> bool:
        """Ask before tearing down a live session to change installs."""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Switch Micro-Manager installation")
        msg.setText(
            f"Switching to {Path(path).name} unloads every device and discards "
            "the current configuration, including any unsaved changes — the "
            "loaded devices come from the installation being replaced.\n\n"
            "Switch now, or keep this session and use it from the next launch?"
        )
        switch_btn = msg.addButton("Switch now", QMessageBox.ButtonRole.DestructiveRole)
        keep_btn = msg.addButton("Keep this session", QMessageBox.ButtonRole.RejectRole)
        for button, variant in ((switch_btn, "danger"), (keep_btn, "subtle")):
            if button is not None:
                button.setProperty("variant", variant)
        msg.setDefaultButton(keep_btn)
        msg.exec()
        return msg.clickedButton() is switch_btn

    def _status(self, message: str) -> None:
        """Show a transient message in the status bar."""
        if status_bar := self.statusBar():
            status_bar.showMessage(message, 5000)

    def _on_mode_tab_changed(self, index: int) -> None:
        """Gate leaving Configurations with unsaved group/pixel edits.

        ``ModeTabBar._select`` already flipped the clicked tab's visual state
        (and emitted this signal) before this runs, so a cancelled switch has
        to explicitly flip it back -- calling it again with the *current*
        stack index does that and re-emits this signal, which is a no-op
        next time since index == the (unchanged) current index by then.
        """
        current_index = self._stack.currentIndex()
        if index == current_index:
            return
        configuration_index = self._stack.indexOf(self._configurations)
        if current_index == configuration_index and self._configurations.is_dirty():
            choice = self._prompt_unsaved_configuration_changes()
            if choice == "cancel":
                self._mode_tabs._select(current_index)
                return
            if choice == "save_core":
                self._configurations.commit_current_to_core()
            elif choice == "save_file" and not self._save_current_configuration():
                # cancelled or failed (e.g. the file dialog was dismissed) —
                # stay on Configurations rather than navigate away silently
                self._mode_tabs._select(current_index)
                return
            elif choice == "discard":
                self._configurations.discard_changes()
        self._stack.setCurrentIndex(index)

    def _prompt_unsaved_configuration_changes(self) -> str:
        """Ask how to handle unsaved group/pixel edits before leaving the page.

        Returns "save_core", "save_file", "discard", or "cancel".
        """
        dirty_parts = self._configurations.dirty_parts()
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Unsaved changes")
        msg.setText(
            f"There are unsaved changes in {' and '.join(dirty_parts)}.\n\n"
            '"Save to core" applies them to the running session only. '
            '"Save to file" also writes them to the .cfg file.'
        )
        save_core_btn = msg.addButton("Save to core", QMessageBox.ButtonRole.AcceptRole)
        save_file_btn = msg.addButton(
            "Save to file…", QMessageBox.ButtonRole.AcceptRole
        )
        discard_btn = msg.addButton(
            "Discard changes and continue", QMessageBox.ButtonRole.DestructiveRole
        )
        cancel_btn = msg.addButton(QMessageBox.StandardButton.Cancel)
        for button, variant in (
            (save_core_btn, "subtle"),
            (save_file_btn, "primary"),
            (discard_btn, "danger"),
            (cancel_btn, "subtle"),
        ):
            if button is not None:
                button.setProperty("variant", variant)
        msg.setDefaultButton(save_file_btn)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked is save_core_btn:
            return "save_core"
        if clicked is save_file_btn:
            return "save_file"
        if clicked is discard_btn:
            return "discard"
        return "cancel"

    def _on_pixel_calibration_running(self, running: bool) -> None:
        """Keep other microscope workflows unavailable during stage calibration."""
        configuration_index = self._stack.indexOf(self._configurations)
        for page in (self._installation, self._hardware, self._acquire):
            self._mode_tabs.setTabEnabled(self._stack.indexOf(page), not running)
        if running and configuration_index >= 0:
            self._mode_tabs._select(configuration_index)
            self._stack.setCurrentIndex(configuration_index)
        if status_bar := self.statusBar():
            status_bar.showMessage(
                "Pixel calibration is controlling the camera and XY stage"
                if running
                else "Ready"
            )

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Offer to save hardware / group / pixel edits before closing."""
        # Restoration is part of the calibration transaction.  Do not destroy
        # its worker (or ask the user to save) until that transaction has ended.
        self._configurations.shutdownCalibration()
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
        self._save_state()
        self._acquire.shutdown()
        super().closeEvent(a0)

    def _save_all(self) -> bool:
        """Commit group/pixel edits to the core, then save everything to a .cfg.

        Returns True if a file was written, False if cancelled or on error.
        """
        # Where to save is asked first: the commit below blocks the GUI thread,
        # and on real hardware there is no reason to spend that time before
        # learning the user meant to cancel.
        if not (path := self._hardware.prompt_save_path()):
            return False
        self._configurations.commit_to_core()
        if self._hardware.save_to(path):
            self._configurations.mark_saved()
            return True
        return False

    def _save_current_configuration(self) -> bool:
        """Commit the selected configuration editor, then write the full .cfg."""
        if not (path := self._hardware.prompt_save_path()):
            return False
        self._configurations.commit_current_to_core()
        if self._hardware.save_to(path):
            self._configurations.mark_current_saved()
            return True
        return False

    def _toggle_theme(self) -> None:
        self._is_dark = not self._is_dark
        set_theme(DARK_THEME if self._is_dark else LIGHT_THEME)
        self._theme_btn.setText("☀" if self._is_dark else "🌙")

    def _on_exception(self, exc: BaseException) -> None:
        """Show a toast notification when an unhandled exception is raised."""
        see_tb = "See traceback"

        def _open_traceback(choice: str | None) -> None:
            if choice != see_tb:
                return
            self._activate_acquire()
            self._acquire.panel_button(PanelKey.EXCEPTION_LOG).setChecked(True)
            log = cast(
                "ExceptionLog", self._acquire.panel_widget(PanelKey.EXCEPTION_LOG)
            )
            log.show_exception(exc)

        self._notification_manager.show_error_message(
            str(exc), see_tb, on_action=_open_traceback
        )

    @property
    def nm(self) -> NotificationManager:
        """Toast notification manager for this window."""
        return self._notification_manager

    @property
    def mmcore(self) -> CMMCorePlus | None:
        """Access to the microscope core, if provided."""
        return self._mmc
