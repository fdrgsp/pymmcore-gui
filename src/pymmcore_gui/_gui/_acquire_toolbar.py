"""Acquire-tab toolbar pieces: snap/live, optical-config presets, shutters.

Snap and Live are built in-house rather than wrapping ``pymmcore_widgets``'
``SnapButton``/``LiveButton`` directly: those hardcode their own text,
30px icon size, and text-swapping behaviour in ways that fought this app's
"icon-only, persistently-boxed" toolbar style. The core-facing logic they
wrap (snap-with-shutter, live start/stop) is a handful of lines, so owning it
directly gives full control over appearance without post-hoc patching.
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from pymmcore_plus import CMMCorePlus, DeviceType
from pymmcore_widgets import ShuttersWidget
from superqt.iconify import QIconifyIcon
from superqt.utils import create_worker

from pymmcore_gui._array_viewer import ensure_visible_icon
from pymmcore_gui._qt.QtCore import QEvent, QSize
from pymmcore_gui._qt.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QWidget,
)

from ._theme import qcolor, theme

if TYPE_CHECKING:
    from pymmcore_gui._qt.QtWidgets import QLayout

_ICON_SIZE = QSize(20, 20)


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
    """Icon-only snap button, wired directly to ``core.snap()``."""

    def __init__(
        self, mmcore: CMMCorePlus | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._core = mmcore or CMMCorePlus.instance()

        self._apply_icon()
        self.setIconSize(_ICON_SIZE)
        self.setToolTip("Snap")
        self.setProperty("variant", "subtle")
        self.clicked.connect(self._snap)

        self._core.events.systemConfigurationLoaded.connect(self._on_config_loaded)
        self.destroyed.connect(self._disconnect)
        self._on_config_loaded()

    def _apply_icon(self) -> None:
        color = qcolor(theme().status_green).name()
        self.setIcon(QIconifyIcon("mdi:camera-outline", color=color))

    def changeEvent(self, e: QEvent | None) -> None:
        # status_green differs between light/dark themes -- a static icon
        # set once at construction would go stale after a theme toggle.
        if e is not None and e.type() == QEvent.Type.StyleChange:
            self._apply_icon()
        super().changeEvent(e)

    def _on_config_loaded(self, *_: object) -> None:
        self.setEnabled(bool(self._core.getCameraDevice()))

    def _snap(self) -> None:
        core = self._core
        if core.isSequenceRunning():
            core.stopSequenceAcquisition()

        def snap_with_shutter() -> None:
            # Not all shutter devices reliably send their own open/close
            # signals -- emit them explicitly so listeners stay in sync.
            autoshutter = core.getAutoShutter()
            if autoshutter:
                core.events.propertyChanged.emit(
                    core.getShutterDevice(), "State", True
                )
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
    """Icon-only live-toggle button, wired directly to core sequence control."""

    def __init__(
        self, mmcore: CMMCorePlus | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._core = mmcore or CMMCorePlus.instance()

        self.setCheckable(True)
        self.setIconSize(_ICON_SIZE)
        self.setProperty("variant", "subtle")
        self._set_running(False)
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
        is_style_change = e is not None and e.type() == QEvent.Type.StyleChange
        if is_style_change and not self.isChecked():
            self._set_running(False)
        super().changeEvent(e)

    def _disconnect(self) -> None:
        with suppress(RuntimeError, TypeError):
            ev = self._core.events
            ev.systemConfigurationLoaded.disconnect(self._on_config_loaded)
            ev.continuousSequenceAcquisitionStarted.disconnect(self._on_started)
            ev.sequenceAcquisitionStopped.disconnect(self._on_stopped)


class ChannelPresetsBar(QWidget):
    """Row of checkable buttons for the presets of the current channel group.

    Mirrors the legacy ``OCToolBar`` behaviour with themed QPushButtons.
    """

    def __init__(
        self, mmcore: CMMCorePlus | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._core = mmcore or CMMCorePlus.instance()

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(theme().sp_xxs)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        ev = self._core.events
        ev.systemConfigurationLoaded.connect(self._refresh)
        ev.configGroupChanged.connect(self._refresh)
        ev.channelGroupChanged.connect(self._refresh)
        ev.configSet.connect(self._on_config_set)
        ev.propertyChanged.connect(self._on_property_changed)
        self._refresh()

    def refresh(self) -> None:
        """Re-scan the core (e.g. after devices change on another tab)."""
        self._refresh()

    def _refresh(self, *_: object) -> None:
        _clear(self._layout)
        if not (ch_group := self._core.getChannelGroup()):
            return
        current = self._core.getCurrentConfig(ch_group)
        for preset in self._core.getAvailableConfigs(ch_group):
            btn = QPushButton(preset)
            btn.setCheckable(True)
            btn.setChecked(preset == current)
            btn.clicked.connect(
                lambda _c=False, p=preset, g=ch_group: self._core.setConfig(g, p)
            )
            self._group.addButton(btn)
            self._layout.addWidget(btn)

    def _on_config_set(self, group: str, config: str) -> None:
        if group == self._core.getChannelGroup():
            for btn in self._group.buttons():
                btn.setChecked(btn.text() == config)

    def _on_property_changed(self, device: str, prop: str, _value: str) -> None:
        if device == "Core" and prop == "ChannelGroup":
            self._refresh()


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
                mmcore=self._core,
            )
            # a persistently visible box, not just on hover — matches Snap/Live
            widget.shutter_button.setProperty("variant", "subtle")
            ensure_visible_icon(widget.shutter_button)
            self._layout.addWidget(widget)
