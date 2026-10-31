"""The per-preset channel settings shared by the Channels panel and the MDA.

Panels are created only when opened, and the MDA needs a channel's settings
whether or not the Channels panel ever was, so they live here rather than in
either widget: one store per core. Both widgets read it and write their user's
edits back, and ``settingsChanged`` tells each of them about the other's.
"""

from __future__ import annotations

import copy
from contextlib import suppress
from typing import TYPE_CHECKING, Any
from weakref import WeakKeyDictionary

from pymmcore_plus import PropertyType
from pymmcore_widgets._channel_settings import apply_channel_settings
from pymmcore_widgets._light_sources import find_light_sources

from pymmcore_gui._light_sources import parse_channel_comments
from pymmcore_gui._qt.QtCore import QObject, Signal

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pymmcore_plus import CMMCorePlus
    from pymmcore_widgets._channel_settings import ChannelSettings


# the optional sets of columns of the channel tables (the exposure is always shown)
LIGHT_SOURCES = "light_sources"
CAMERA = "camera"


class ChannelSettingsStore(QObject):
    """``{group: {preset: settings}}`` for a core, seeded from its loaded cfg.

    The channel group's presets are seeded when a configuration is loaded: each
    gets the current exposure, or whatever the cfg records for it (see
    ``pymmcore_gui._light_sources``). So every widget reading the store starts from
    the same values, and the recorded settings reach a new MDA channel.
    """

    # group, preset, the object that made the change (or None)
    settingsChanged = Signal(str, str, object)
    # LIGHT_SOURCES or CAMERA, whether it is now shown
    columnsVisibleChanged = Signal(str, bool)

    def __init__(self, core: CMMCorePlus) -> None:
        super().__init__()
        self._core = core
        self._settings: dict[str, dict[str, ChannelSettings]] = {}
        # which optional columns both channel tables show, see setColumnsVisible
        self._columns_visible = {LIGHT_SOURCES: True, CAMERA: False}
        core.events.systemConfigurationLoaded.connect(self.seed)
        self.seed()

    def get(self, group: str, preset: str) -> ChannelSettings | None:
        """Return the settings of ``preset`` of ``group``, or None if unknown."""
        settings = self._settings.get(group, {}).get(preset)
        return copy.deepcopy(settings) if settings is not None else None

    def all(self) -> dict[str, dict[str, ChannelSettings]]:
        """Return ``{group: {preset: settings}}`` for every known preset."""
        return copy.deepcopy(self._settings)

    def set(
        self, group: str, preset: str, settings: ChannelSettings, source: Any = None
    ) -> None:
        """Store ``settings``, emitting ``settingsChanged`` if they differ."""
        if not group or not preset:
            return
        presets = self._settings.setdefault(group, {})
        if presets.get(preset) == settings:
            return
        presets[preset] = copy.deepcopy(settings)
        self.settingsChanged.emit(group, preset, source)

    def update(
        self, value: Mapping[str, Mapping[str, ChannelSettings]], source: Any = None
    ) -> None:
        """Store every entry of ``{group: {preset: settings}}``."""
        for group, presets in value.items():
            for preset, settings in presets.items():
                self.set(group, preset, settings, source)

    def columnsVisible(self, columns: str) -> bool:
        """Return whether both channel tables show `columns` (see setColumnsVisible)."""
        return self._columns_visible[columns]

    def setColumnsVisible(self, columns: str, visible: bool) -> None:
        """Show or hide `columns` (LIGHT_SOURCES or CAMERA) in both channel tables.

        The Channels panel and the MDA's channel table each have a checkbox per
        set of columns; checking one checks the other.
        """
        if self._columns_visible[columns] != visible:
            self._columns_visible[columns] = visible
            self.columnsVisibleChanged.emit(columns, visible)

    def seed(self) -> None:
        """Give each channel group preset a starting point from the loaded cfg.

        Presets that already have settings keep them, except for those the cfg
        records: the cfg is the authority on those.
        """
        core = self._core
        group = core.getChannelGroup()
        if not group:
            return
        declarations = parse_channel_comments(
            core.systemConfigurationFile() or "", group
        )
        # a multi-property declaration is one multi-slider light source: map the
        # whole pair set back to the label of the source driving all of them
        by_pairs = {
            frozenset(pairs): label for label, pairs in find_light_sources(core).items()
        }
        camera = core.getCameraDevice()
        exposure = core.getExposure() if camera else 100.0
        for preset in core.getAvailableConfigs(group):
            settings = self.get(group, preset) or {
                "exposure": exposure,
                "light_source": "",
                "intensity": 0.0,
                "camera_properties": {},
            }
            if (declared := declarations.get(preset)) is not None:
                if declared.exposure is not None:
                    settings["exposure"] = declared.exposure
                pairs = frozenset((dev, prop) for dev, prop, _ in declared.light_source)
                if label := by_pairs.get(pairs):
                    settings["light_source"] = label
                    settings["intensity"] = declared.light_source[0][2]
                for device, prop, raw in declared.camera_properties:
                    if device == camera and core.hasProperty(camera, prop):
                        settings["camera_properties"][prop] = _camera_value(
                            core, camera, prop, raw
                        )
            self.set(group, preset, settings)

    def apply_current_channel(self) -> bool:
        """Apply the settings of the channel preset currently set on the core.

        Called before a snap or live, so they image with the Channels panel's
        settings. Returns whether there were settings to apply.
        """
        core = self._core
        if not (group := core.getChannelGroup()):
            return False
        preset = core.getCurrentConfig(group)
        if not preset or (settings := self.get(group, preset)) is None:
            return False
        # any device delays of the preset must have completed before imaging
        core.waitForConfig(group, preset)
        apply_channel_settings(core, settings, find_light_sources(core))
        core.waitForSystem()
        return True


def _camera_value(
    core: CMMCorePlus, camera: str, prop: str, raw: str
) -> float | int | str:
    """Cast a camera property value read from the cfg, like the Channels panel.

    Properties with allowed values keep their string; numeric ones are numbers.
    """
    if core.getAllowedPropertyValues(camera, prop):
        return raw
    with suppress(ValueError):
        if core.getPropertyType(camera, prop) is PropertyType.Integer:
            return int(float(raw))
        if core.getPropertyType(camera, prop) is PropertyType.Float:
            return float(raw)
    return raw


_STORES: WeakKeyDictionary[CMMCorePlus, ChannelSettingsStore] = WeakKeyDictionary()


def channel_settings_store(core: CMMCorePlus) -> ChannelSettingsStore:
    """Return the channel settings store of ``core``, creating it on first use."""
    if (store := _STORES.get(core)) is None:
        store = _STORES[core] = ChannelSettingsStore(core)
    return store
