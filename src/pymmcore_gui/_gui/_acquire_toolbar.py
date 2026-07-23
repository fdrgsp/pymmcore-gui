"""Acquire-tab toolbar pieces: optical-config presets and shutters.

Snap and Live come straight from ``pymmcore_widgets`` (``SnapButton`` /
``LiveButton``); these two widgets cover the parts that need a themed,
QPushButton-based look consistent with the rest of the GUI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_plus import CMMCorePlus, DeviceType
from pymmcore_widgets import ShuttersWidget

from pymmcore_gui._qt.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QWidget,
)

from ._theme import theme

if TYPE_CHECKING:
    from pymmcore_gui._qt.QtWidgets import QLayout


def toolbar_separator() -> QFrame:
    """A thin vertical divider for grouping toolbar sections."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setFrameShadow(QFrame.Shadow.Plain)
    return line


def icon_only(btn: QAbstractButton, tooltip: str) -> QAbstractButton:
    """Strip a button's text label down to just its icon.

    SnapButton/LiveButton set their own text (LiveButton even re-sets it
    internally via ``_button_text_on``/``_button_text_off`` on every state
    change, e.g. "Live" <-> "Stop") — patching those attributes too keeps the
    label empty across state changes, not just at construction time. The
    "subtle" variant gives the button a persistently visible box (rather than
    only on hover), since an icon with no label is otherwise easy to miss.
    """
    btn.setText("")
    for attr in ("_button_text_on", "_button_text_off"):
        if hasattr(btn, attr):
            setattr(btn, attr, "")
    btn.setToolTip(tooltip)
    btn.setProperty("variant", "subtle")
    return btn


def _clear(layout: QLayout) -> None:
    while layout.count():
        if (item := layout.takeAt(0)) and (w := item.widget()):
            w.deleteLater()


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
        for idx, shutter in enumerate(ordered):
            widget = ShuttersWidget(
                shutter,
                autoshutter=idx == len(ordered) - 1,
                button_text_open=shutter,
                button_text_closed=shutter,
                mmcore=self._core,
            )
            # a persistently visible box, not just on hover — matches Snap/Live
            widget.shutter_button.setProperty("variant", "subtle")
            self._layout.addWidget(widget)
