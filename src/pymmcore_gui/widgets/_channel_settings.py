"""The Channels panel: per-preset exposure, light source and camera settings."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pymmcore_widgets import ChannelSettingsTable

from pymmcore_gui._array_viewer import unstyle_widgets
from pymmcore_gui._channel_settings_store import (
    CAMERA,
    LIGHT_SOURCES,
    channel_settings_store,
)
from pymmcore_gui._light_sources import ChannelDeclaration, write_channel_comments
from pymmcore_gui._qt.QtCore import QTimer
from pymmcore_gui._qt.QtWidgets import QFileDialog, QMessageBox, QPushButton
from pymmcore_gui._theme import qcolor, theme

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

    from pymmcore_gui._light_sources import ChannelDeclarations
    from pymmcore_gui._qt.QtWidgets import QWidget


class ChannelSettingsPanel(ChannelSettingsTable):
    """`ChannelSettingsTable` kept in sync with the app's channel settings store.

    What the user enters here is what a channel added to the MDA starts with, and
    an edit to an MDA channel shows up here (see ``MDAChannelTable``). The settings
    can be saved into a cfg, so they are restored when it is next loaded (see
    ``pymmcore_gui._light_sources``).
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
        # the column checkboxes are shared with the MDA's channel table
        self._on_columns_visible_changed(
            LIGHT_SOURCES, self._store.columnsVisible(LIGHT_SOURCES)
        )
        self._on_columns_visible_changed(CAMERA, self._store.columnsVisible(CAMERA))
        self.light_sources.toggled.connect(
            lambda on: self._store.setColumnsVisible(LIGHT_SOURCES, on)
        )
        self.camera.toggled.connect(
            lambda on: self._store.setColumnsVisible(CAMERA, on)
        )
        self._store.columnsVisibleChanged.connect(self._on_columns_visible_changed)
        self.destroyed.connect(self._disconnect_store)

        self._save_btn = QPushButton("Save Channel Settings to cfg")
        self._save_btn.setToolTip(
            "Write each channel's exposure, light source and camera properties into "
            "a Micro-Manager configuration file, so they are restored the next time "
            "that file is loaded."
        )
        self._save_btn.setProperty("variant", "subtle")
        self._save_btn.clicked.connect(self.save_channel_settings)
        self._btn_row.addWidget(self._save_btn)

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

    def save_channel_settings(self) -> None:
        """Save the whole cfg to a file, with the channel settings appended.

        The block is rewritten from scratch, so a setting the user cleared actually
        disappears. The configuration itself is dumped from the core, so
        uncommitted edits in the Hardware Setup page are not included.
        """
        if not (channel_group := self._mmc.getChannelGroup()):
            QMessageBox.warning(
                self,
                "Save Channel Settings",
                "No channel group is set on the current configuration, so there is "
                "nothing to attach the channel settings to.",
            )
            return

        current = self._mmc.systemConfigurationFile() or ""
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save Micro-Manager configuration file", current, "cfg(*.cfg)"
        )
        if not path_str:
            return
        path = Path(path_str)
        if path.suffix.lower() != ".cfg":
            path = path.with_suffix(".cfg")

        try:
            # dumps the configuration itself; comments do not survive this, which is
            # why the settings block is appended afterwards rather than merged
            self._mmc.saveSystemConfiguration(str(path))
            write_channel_comments(
                path, channel_group, self._declarations(channel_group)
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Save Channel Settings", f"Could not save {path}:\n{exc}"
            )

    def _declarations(self, channel_group: str) -> ChannelDeclarations:
        """The cfg declarations of `channel_group`'s preset settings."""
        stored = self._store.all().get(channel_group, {})
        light_sources = self.lightSources()
        camera = self._mmc.getCameraDevice()
        declarations: ChannelDeclarations = {}
        for preset in self._mmc.getAvailableConfigs(channel_group):
            if (settings := stored.get(preset)) is None:
                continue
            declarations[preset] = ChannelDeclaration(
                exposure=settings["exposure"],
                light_source=[
                    (device, prop, settings["intensity"])
                    for device, prop in light_sources.get(settings["light_source"], ())
                ],
                camera_properties=[
                    (camera, prop, str(value))
                    for prop, value in settings["camera_properties"].items()
                    if camera
                ],
            )
        return declarations

    def _push_to_store(self) -> None:
        self._store.update(self.allChannelSettings(), source=self)

    def _on_store_changed(self, group: str, preset: str, source: Any) -> None:
        if source is not self:
            self._pull_timer.start()

    def _on_columns_visible_changed(self, columns: str, visible: bool) -> None:
        if columns == LIGHT_SOURCES:
            self.setLightSourcesVisible(visible)
        elif columns == CAMERA:
            self.setCameraVisible(visible)

    def _pull_from_store(self) -> None:
        self.setAllChannelSettings(self._store.all())

    def _unstyle(self, *_: object) -> None:
        unstyle_widgets(self)

    def _disconnect_store(self, *_: object) -> None:
        self._store.settingsChanged.disconnect(self._on_store_changed)
        self._store.columnsVisibleChanged.disconnect(self._on_columns_visible_changed)
