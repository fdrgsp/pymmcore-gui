"""The Channels panel: per-preset exposure and light source settings."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pymmcore_widgets import ChannelSettingsTable
from pymmcore_widgets._channel_settings import apply_channel_settings
from pymmcore_widgets._light_sources import find_light_sources
from superqt.iconify import QIconifyIcon

from pymmcore_gui._array_viewer import unstyle_widgets
from pymmcore_gui._channel_settings_store import channel_settings_store
from pymmcore_gui._light_sources import ChannelDeclaration, write_channel_comments
from pymmcore_gui._qt.QtCore import QEvent, QSize, QTimer
from pymmcore_gui._qt.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QStyle,
)
from pymmcore_gui._theme import qcolor, theme

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

    from pymmcore_gui._light_sources import ChannelDeclarations
    from pymmcore_gui._qt.QtWidgets import QBoxLayout, QCheckBox, QWidget


class SaveToCfgButton(QPushButton):
    """Icon button saving the channel settings into a cfg.

    Styled like the app's other icon buttons, the same as ``PreferencesButton``, but
    sized to go with the checkbox it sits next to: its icon is as big as a checkbox's
    indicator, in a box leaving the row's height as it is.
    """

    _ICON = "mdi:content-save-outline"

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setProperty("variant", "subtle")
        self.setToolTip(
            "Save to cfg: write each channel's exposure and light source into a "
            "Micro-Manager configuration file, so they are restored the next time "
            "that file is loaded."
        )
        self._apply_icon()

    def _apply_icon(self) -> None:
        color = qcolor(theme().text_secondary).name()
        self.setIcon(QIconifyIcon(self._ICON, color=color))
        style = cast("QStyle", self.style())
        size = style.pixelMetric(QStyle.PixelMetric.PM_IndicatorWidth)
        self.setIconSize(QSize(size, size))
        self.setFixedSize(size + 4, size + 4)

    def changeEvent(self, e: QEvent | None) -> None:
        if e is not None and e.type() == QEvent.Type.StyleChange:
            self._apply_icon()
        super().changeEvent(e)


def add_save_to_cfg_button(
    row: QBoxLayout, checkbox: QCheckBox, parent: QWidget
) -> SaveToCfgButton:
    """Put the button that saves the settings into a cfg right after ``checkbox``.

    Both go in a tighter layout of their own, at the checkbox's place in ``row``, so
    the button reads as belonging to it rather than as one more item of the row. It
    starts enabled as the checkbox is checked; the owner keeps it so, in the setter
    the checkbox calls, since a checkbox changed from elsewhere has its signals
    blocked.
    """
    button = SaveToCfgButton(parent)
    button.setEnabled(checkbox.isChecked())
    index = row.indexOf(checkbox)
    row.removeWidget(checkbox)
    pair = QHBoxLayout()
    pair.setSpacing(4)
    pair.addWidget(checkbox)
    pair.addWidget(button)
    row.insertLayout(index, pair)
    return button


def save_channel_settings(parent: QWidget, core: CMMCorePlus) -> None:
    """Save the whole cfg to a file, with the channel settings appended.

    The block is rewritten from scratch, so a setting the user cleared actually
    disappears. The configuration itself is dumped from the core, so uncommitted
    edits in the Hardware Setup page are not included.
    """
    # an editor still holding the focus commits its value when losing it
    parent.setFocus()
    if not (channel_group := core.getChannelGroup()):
        QMessageBox.warning(
            parent,
            "Save Channel Settings",
            "No channel group is set on the current configuration, so there is "
            "nothing to attach the channel settings to.",
        )
        return

    current = core.systemConfigurationFile() or ""
    path_str, _ = QFileDialog.getSaveFileName(
        parent, "Save Micro-Manager configuration file", current, "cfg(*.cfg)"
    )
    if not path_str:
        return
    path = Path(path_str)
    if path.suffix.lower() != ".cfg":
        path = path.with_suffix(".cfg")

    try:
        # dumps the configuration itself; comments do not survive this, which is
        # why the settings block is appended afterwards rather than merged
        core.saveSystemConfiguration(str(path))
        write_channel_comments(path, channel_group, _declarations(core, channel_group))
    except Exception as exc:
        QMessageBox.critical(
            parent, "Save Channel Settings", f"Could not save {path}:\n{exc}"
        )


def _declarations(core: CMMCorePlus, channel_group: str) -> ChannelDeclarations:
    """The cfg declarations of ``channel_group``'s preset settings."""
    stored = channel_settings_store(core).all().get(channel_group, {})
    light_sources = find_light_sources(core)
    declarations: ChannelDeclarations = {}
    for preset in core.getAvailableConfigs(channel_group):
        if (settings := stored.get(preset)) is None:
            continue
        declarations[preset] = ChannelDeclaration(
            exposure=settings["exposure"],
            light_source=[
                (device, prop, settings["intensity"])
                for device, prop in light_sources.get(settings["light_source"], ())
            ],
        )
    return declarations


class ChannelSettingsPanel(ChannelSettingsTable):
    """`ChannelSettingsTable` kept in sync with the app's channel settings store.

    What the user enters here is what a channel added to the MDA starts with, and
    an edit to an MDA channel shows up here (see ``MDAChannelTable``); the channel
    selected in either one follows the core, so it is selected in both. While live is
    running, an edit of the selected preset reaches the hardware right away. The
    settings can be saved into a cfg, so they are restored when it is next loaded
    (see ``pymmcore_gui._light_sources``).
    """

    def __init__(self, mmcore: CMMCorePlus, parent: QWidget | None = None) -> None:
        # before the base class's first refresh(), which reads it
        self._store = channel_settings_store(mmcore)
        super().__init__(mmcore, parent)
        self.setAllChannelSettings(self._store.all())
        # a config load changes every seeded preset at once: reload just once
        self._pull_timer = QTimer(self)
        self._pull_timer.setSingleShot(True)
        self._pull_timer.setInterval(0)
        self._pull_timer.timeout.connect(self._pull_from_store)
        self.valueChanged.connect(self._push_to_store)
        self._store.settingsChanged.connect(self._on_store_changed)
        self._save_btn = add_save_to_cfg_button(self._btn_row, self.light_sources, self)
        self._save_btn.clicked.connect(self._on_save_clicked)

        # the checkbox is shared with the MDA's channel table
        self.setLightSourcesVisible(self._store.lightSourcesVisible())
        self.light_sources.toggled.connect(self._store.setLightSourcesVisible)
        self._store.lightSourcesVisibleChanged.connect(self.setLightSourcesVisible)
        self.destroyed.connect(self._disconnect_store)

        self.currentPresetDelegate().setActiveColor(qcolor(theme().status_yellow))
        # rows are rebuilt on every group change, with fresh (stylesheet-carrying)
        # cell widgets; theme them like the MDA's channel rows
        if (model := self.table().model()) is not None:
            model.rowsInserted.connect(self._unstyle)

    def refresh(self) -> None:
        # Rebuilt for a new configuration: start from the store's settings, which
        # it re-seeds from the loaded cfg, rather than pushing this panel's older
        # ones over them once the rows are rebuilt.
        self.setAllChannelSettings(self._store.all())
        super().refresh()

    def _on_save_clicked(self) -> None:
        save_channel_settings(self, self._mmc)

    def setLightSourcesVisible(self, visible: bool) -> None:
        super().setLightSourcesVisible(visible)
        self._save_btn.setEnabled(visible)

    def _push_to_store(self) -> None:
        self._store.update(self.allChannelSettings(), source=self)

    def _on_store_changed(self, group: str, preset: str, source: Any) -> None:
        if source is not self:
            self._pull_timer.start()
        elif (
            self._store.is_live()
            and group == self.channelGroup()
            and preset == self.activePreset()
            and (settings := self._store.get(group, preset)) is not None
        ):
            apply_channel_settings(self._mmc, settings, self.lightSources())

    def _pull_from_store(self) -> None:
        self.setAllChannelSettings(self._store.all())

    def _unstyle(self, *_: object) -> None:
        unstyle_widgets(self)

    def _disconnect_store(self, *_: object) -> None:
        self._store.settingsChanged.disconnect(self._on_store_changed)
        self._store.lightSourcesVisibleChanged.disconnect(self.setLightSourcesVisible)
