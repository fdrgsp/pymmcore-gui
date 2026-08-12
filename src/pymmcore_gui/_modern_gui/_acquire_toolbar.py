"""Acquire-tab toolbar pieces: snap/live and shutters.

Snap and Live are built in-house rather than wrapping ``pymmcore_widgets``'
``SnapButton``/``LiveButton`` directly: those hardcode their own text,
30px icon size, and text-swapping behaviour in ways that fought this app's
"icon-only, persistently-boxed" toolbar style. The core-facing logic they
wrap (snap-with-shutter, live start/stop) is a handful of lines, so owning it
directly gives full control over appearance without post-hoc patching.
"""

from __future__ import annotations

from contextlib import suppress
from functools import partial
from typing import TYPE_CHECKING

from pymmcore_plus import CMMCorePlus, DeviceType
from pymmcore_widgets import ShuttersWidget
from superqt.iconify import QIconifyIcon
from superqt.utils import create_worker

from pymmcore_gui._array_viewer import ensure_visible_icon, set_source_icon
from pymmcore_gui._qt.QtCore import QEvent, QPoint, QSize, Signal
from pymmcore_gui._qt.QtGui import QAction
from pymmcore_gui._qt.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QMenu,
    QPushButton,
    QWidget,
)

from ._theme import qcolor, theme

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pymmcore_gui._qt.QtWidgets import QLayout

    from ._panels import PanelInfo


def _icon_size() -> QSize:
    """The app's compact action-icon size, scaled with the current zoom."""
    size = theme().scaled(20)
    return QSize(size, size)


def toolbar_separator() -> QFrame:
    """A thin vertical divider for grouping toolbar sections."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setFrameShadow(QFrame.Shadow.Plain)
    return line


def _clear(layout: QLayout) -> None:
    while layout.count():
        if (item := layout.takeAt(0)) and (w := item.widget()):
            w.deleteLater()


class SnapButton(QPushButton):
    """Acquire-style icon-only snap button, optionally controlling MMCore."""

    snapRequested = Signal()

    def __init__(
        self,
        mmcore: CMMCorePlus | None = None,
        parent: QWidget | None = None,
        *,
        control_core: bool = True,
    ) -> None:
        super().__init__(parent)
        self._core = mmcore or CMMCorePlus.instance()

        self._apply_icon()
        self.setIconSize(_icon_size())
        self.setToolTip("Snap")
        self.setProperty("variant", "subtle")
        if control_core:
            self.clicked.connect(self._snap)

        self._core.events.systemConfigurationLoaded.connect(self._on_config_loaded)
        self.destroyed.connect(self._disconnect)
        self._on_config_loaded()

    def _apply_icon(self) -> None:
        color = qcolor(theme().status_green).name()
        self.setIcon(QIconifyIcon("mdi:camera-outline", color=color))

    def changeEvent(self, e: QEvent | None) -> None:
        # status_green differs between light/dark themes -- a static icon
        # set once at construction would go stale after a theme toggle. The
        # icon size is re-applied here too since it's zoom-scaled and this
        # button (a QPushButton, not a QToolBar) isn't touched by the app's
        # zoom pass over QToolBar instances.
        if e is not None and e.type() == QEvent.Type.StyleChange:
            self._apply_icon()
            self.setIconSize(_icon_size())
        super().changeEvent(e)

    def _on_config_loaded(self, *_: object) -> None:
        self.setEnabled(bool(self._core.getCameraDevice()))

    def _snap(self) -> None:
        core = self._core
        if core.isSequenceRunning():
            core.stopSequenceAcquisition()

        # Emitted synchronously after stopping any live sequence so listeners can
        # safely apply the active channel's capture settings and a lazy preview can
        # subscribe before the worker performs the first snap.
        self.snapRequested.emit()

        def snap_with_shutter() -> None:
            # Not all shutter devices reliably send their own open/close
            # signals -- emit them explicitly so listeners stay in sync.
            autoshutter = core.getAutoShutter()
            if autoshutter:
                core.events.propertyChanged.emit(core.getShutterDevice(), "State", True)
            core.snap()
            if autoshutter:
                core.events.propertyChanged.emit(
                    core.getShutterDevice(), "State", False
                )

        create_worker(snap_with_shutter, _start_thread=True)

    def _disconnect(self) -> None:
        with suppress(RuntimeError, TypeError):
            self._core.events.systemConfigurationLoaded.disconnect(
                self._on_config_loaded
            )


class LiveButton(QPushButton):
    """Acquire-style live toggle, optionally controlling MMCore directly."""

    liveStartedRequested = Signal()

    def __init__(
        self,
        mmcore: CMMCorePlus | None = None,
        parent: QWidget | None = None,
        *,
        control_core: bool = True,
    ) -> None:
        super().__init__(parent)
        self._core = mmcore or CMMCorePlus.instance()

        self.setCheckable(True)
        self.setIconSize(_icon_size())
        self.setProperty("variant", "subtle")
        self._set_running(False)
        if control_core:
            self.clicked.connect(self._toggle)

        ev = self._core.events
        ev.systemConfigurationLoaded.connect(self._on_config_loaded)
        ev.continuousSequenceAcquisitionStarted.connect(self._on_started)
        ev.sequenceAcquisitionStopped.connect(self._on_stopped)
        self.destroyed.connect(self._disconnect)
        self._on_config_loaded()

    def _on_config_loaded(self, *_: object) -> None:
        self.setEnabled(bool(self._core.getCameraDevice()))

    def _toggle(self) -> None:
        if self._core.isSequenceRunning():
            self._core.stopSequenceAcquisition()
        else:
            self.ensure_live()

    def ensure_live(self) -> None:
        """Start live mode if needed without toggling an existing stream off."""
        if self._core.isSequenceRunning():
            return
        # Give a lazy preview time to attach to the streaming signals before the
        # core starts emitting frames.
        self.liveStartedRequested.emit()
        self._core.startContinuousSequenceAcquisition()

    def _on_started(self, *_: object) -> None:
        self._set_running(True)

    def _on_stopped(self, *_: object) -> None:
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        with suppress(RuntimeError):
            self.setChecked(running)
        if running:
            self.setIcon(QIconifyIcon("mdi:video-off-outline", color="magenta"))
            self.setToolTip("Stop")
        else:
            color = qcolor(theme().status_green).name()
            self.setIcon(QIconifyIcon("mdi:video-outline", color=color))
            self.setToolTip("Live")

    def changeEvent(self, e: QEvent | None) -> None:
        # status_green differs between light/dark themes -- re-derive the
        # idle icon's color from whichever theme is now active. The running
        # (magenta) icon isn't theme-derived, so no re-render needed there.
        # The icon size is re-applied too since it's zoom-scaled and this
        # button (a QPushButton, not a QToolBar) isn't touched by the app's
        # zoom pass over QToolBar instances.
        is_style_change = e is not None and e.type() == QEvent.Type.StyleChange
        if is_style_change:
            if not self.isChecked():
                self._set_running(False)
            self.setIconSize(_icon_size())
        super().changeEvent(e)

    def _disconnect(self) -> None:
        with suppress(RuntimeError, TypeError):
            ev = self._core.events
            ev.systemConfigurationLoaded.disconnect(self._on_config_loaded)
            ev.continuousSequenceAcquisitionStarted.disconnect(self._on_started)
            ev.sequenceAcquisitionStopped.disconnect(self._on_stopped)


class ShuttersBar(QWidget):
    """Row of :class:`ShuttersWidget` for every loaded shutter device.

    Mirrors the legacy ``ShuttersToolbar``; the autoshutter toggle lives on the
    last shutter, matching the reference behaviour.
    """

    def __init__(
        self, mmcore: CMMCorePlus | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._core = mmcore or CMMCorePlus.instance()

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(theme().sp_xxs)

        self._core.events.systemConfigurationLoaded.connect(self._refresh)
        self._refresh()

    def refresh(self) -> None:
        """Re-scan the core (e.g. after devices change on another tab)."""
        self._refresh()

    def changeEvent(self, a0: QEvent | None) -> None:
        # ShuttersWidget bakes its open-shutter icon's color in at
        # construction time (upstream), so a mere StyleChange event can't
        # refresh it in place -- rebuild from scratch, picking up the now-
        # active theme's status_green the same way a real device-list change
        # already does.
        if a0 is not None and a0.type() == QEvent.Type.StyleChange:
            self._refresh()
        super().changeEvent(a0)

    def _refresh(self, *_: object) -> None:
        _clear(self._layout)
        shutters = self._core.getLoadedDevicesOfType(DeviceType.ShutterDevice)
        if not shutters:
            return
        # devices exposing a "Physical Shutter" property come first
        ordered = sorted(
            shutters,
            key=lambda d: any(
                "Physical Shutter" in p for p in self._core.getDevicePropertyNames(d)
            ),
            reverse=True,
        )
        open_color = qcolor(theme().status_green).name()
        for idx, shutter in enumerate(ordered):
            widget = ShuttersWidget(
                shutter,
                autoshutter=idx == len(ordered) - 1,
                button_text_open=shutter,
                button_text_closed=shutter,
                icon_color_open=open_color,
                icon_size=theme().scaled(20),
                mmcore=self._core,
            )
            # a persistently visible box, not just on hover — matches Snap/Live
            widget.shutter_button.setProperty("variant", "subtle")
            ensure_visible_icon(widget.shutter_button)
            self._layout.addWidget(widget)


class PanelButtonBar(QWidget):
    """Icon-only toggle buttons for the registry panels (see ``_panels.py``).

    Two independent levels of control:

    * each button toggles whether its panel's dock is *open*;
    * the trailing ``⋯`` menu (also reachable by right-clicking the toolbar
      row that hosts this bar) toggles whether that button is *present at
      all*, so users can pare the bar down to the tools they actually use.

    A self-contained content strip: no background painting, no assumptions
    about its parent. That's what makes it relocatable -- today it shares the
    Acquire toolbar row (see ``AcquirePage._place_panel_bar``), but it could
    just as easily be dropped onto a second row or into ``MainWindow`` as its
    own ``QToolBar`` without changing anything here.
    """

    _MENU_ICON = "mdi:dots-horizontal"

    panelVisibilityChanged = Signal(str, bool)
    """Emitted with (key, visible) when the customize menu shows/hides a button."""

    resetLayoutRequested = Signal()
    """Emitted when the customize menu's "Reset Layout" entry is chosen."""

    def __init__(
        self, panels: Iterable[PanelInfo], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._panels = list(panels)

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(theme().sp_xxs)

        self._buttons: dict[str, QPushButton] = {}
        for info in self._panels:
            btn = QPushButton(self)
            btn.setCheckable(True)
            btn.setProperty("variant", "subtle")
            btn.setToolTip(info.tooltip)
            btn.setIconSize(_icon_size())
            self._layout.addWidget(btn)
            self._buttons[info.key] = btn

        self._menu_btn = QPushButton(self)
        self._menu_btn.setProperty("variant", "subtle")
        self._menu_btn.setIconSize(_icon_size())
        self._menu_btn.setToolTip("Choose which tool buttons to show")
        self._menu_btn.clicked.connect(self._popup_at_menu_button)
        self._layout.addWidget(self._menu_btn)

        self._apply_icons()

    # ── buttons ───────────────────────────────────────────────────

    def button_for(self, key: str) -> QPushButton:
        """Return the toggle button for the panel registered under *key*."""
        return self._buttons[key]

    def set_button_visible(self, key: str, visible: bool) -> None:
        """Show or hide *key*'s button. Always-visible panels ignore hiding."""
        info = self._info_for(key)
        if info.always_visible and not visible:
            return
        self._buttons[key].setVisible(visible)

    def hidden_keys(self) -> set[str]:
        """Return the keys whose buttons are currently hidden.

        The *hidden* set (rather than the visible one) is what gets persisted:
        a panel added to the registry in a future release is then visible by
        default for existing users, instead of silently missing because it
        wasn't in their saved visible set.
        """
        return {
            info.key
            for info in self._panels
            if not info.always_visible and self._buttons[info.key].isHidden()
        }

    def _info_for(self, key: str) -> PanelInfo:
        return next(info for info in self._panels if info.key == key)

    # ── customize menu ────────────────────────────────────────────

    def build_menu(self) -> QMenu:
        """Build the customize menu: one checkable entry per hideable panel.

        Rebuilt per invocation so the check states can't drift out of sync
        with the buttons; it's a handful of actions, so cheap.
        """
        menu = QMenu(self)
        for info in self._panels:
            if info.always_visible:
                continue
            action = QAction(info.title, menu)
            action.setCheckable(True)
            action.setChecked(not self._buttons[info.key].isHidden())
            action.toggled.connect(partial(self.panelVisibilityChanged.emit, info.key))
            menu.addAction(action)
        menu.addSeparator()
        reset = QAction("Reset Layout", menu)
        reset.setToolTip("Restore the default panel arrangement")
        reset.triggered.connect(self.resetLayoutRequested.emit)
        menu.addAction(reset)
        return menu

    def popup_menu(self, global_pos: QPoint) -> None:
        """Pop the customize menu at *global_pos* (screen coordinates)."""
        self.build_menu().exec(global_pos)

    def _popup_at_menu_button(self) -> None:
        self.popup_menu(self._menu_btn.mapToGlobal(self._menu_btn.rect().bottomLeft()))

    # ── theming ───────────────────────────────────────────────────

    def _apply_icons(self) -> None:
        # QIconifyIcon bakes its color in at construction, so the icon must
        # be rebuilt (not just resized) whenever the theme changes.
        panel_color = qcolor(theme().status_green).name()
        for info in self._panels:
            btn = self._buttons[info.key]
            # set_source_icon (not setIcon) refreshes the stash that the
            # app-wide theme-change sweep (ensure_visible_icon) re-derives
            # from -- a bare setIcon would leave that sweep recoloring from
            # the *previous* theme's icon. See the "Theme-aware icons" note
            # in the panels-toolbar design doc.
            set_source_icon(btn, QIconifyIcon(info.icon, color=panel_color))
            btn.setIconSize(_icon_size())
        menu_color = qcolor(theme().text_primary).name()
        set_source_icon(
            self._menu_btn,
            QIconifyIcon(self._MENU_ICON, color=menu_color),
        )
        self._menu_btn.setIconSize(_icon_size())

    def changeEvent(self, a0: QEvent | None) -> None:
        if a0 is not None and a0.type() == QEvent.Type.StyleChange:
            self._apply_icons()
        super().changeEvent(a0)
