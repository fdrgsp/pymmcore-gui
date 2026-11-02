"""The per-preset channel settings shared by the Channels panel and the MDA.

Panels are created only when opened, and the MDA needs a channel's settings
whether or not the Channels panel ever was, so they live here rather than in
either widget: one store per core. Both widgets read it and write their user's
edits back, and ``settingsChanged`` tells each of them about the other's.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any
from weakref import WeakKeyDictionary, ref

from pymmcore_widgets._channel_settings import apply_channel_settings
from pymmcore_widgets._light_sources import find_light_sources

from pymmcore_gui._light_sources import parse_channel_comments
from pymmcore_gui._qt.QtCore import QObject, Signal

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pymmcore_plus import CMMCorePlus
    from pymmcore_widgets._channel_settings import ChannelSettings


class ChannelSettingsStore(QObject):
    """``{group: {preset: settings}}`` for a core, seeded from its loaded cfg.

    The channel group's presets are seeded when a configuration is loaded, and a
    group's the first time it becomes the channel group: each gets the current
    exposure, or whatever the cfg records for it (see
    ``pymmcore_gui._light_sources``). So every widget reading the store starts from
    the same values, and the recorded settings reach a new MDA channel.
    """

    # group, preset, the object that made the change (or None)
    settingsChanged = Signal(str, str, object)
    # whether both channel tables now show their light source columns
    lightSourcesVisibleChanged = Signal(bool)

    def __init__(self, core: CMMCorePlus) -> None:
        super().__init__()
        # The weak-key cache must not keep its own core alive through the value.
        self._core_ref = ref(core)
        self._settings: dict[str, dict[str, ChannelSettings]] = {}
        self._light_sources_visible = True
        # the groups seeded since the configuration was loaded
        self._seeded_groups: set[str] = set()
        core.events.systemConfigurationLoaded.connect(self._on_config_loaded)
        core.events.channelGroupChanged.connect(self._on_channel_group_changed)
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

    def lightSourcesVisible(self) -> bool:
        """Return whether both channel tables show their light source columns."""
        return self._light_sources_visible

    def setLightSourcesVisible(self, visible: bool) -> None:
        """Show or hide the light source columns in both channel tables.

        The Channels panel and the MDA's channel table each have a checkbox for
        them; checking one checks the other.
        """
        if self._light_sources_visible != visible:
            self._light_sources_visible = visible
            self.lightSourcesVisibleChanged.emit(visible)

    def _on_config_loaded(self) -> None:
        self._seeded_groups.clear()
        self.seed()

    def _on_channel_group_changed(self, group: str) -> None:
        # only once: coming back to a group must keep what was edited for it
        if group not in self._seeded_groups:
            self.seed()

    def seed(self) -> None:
        """Give each channel group preset a starting point from the loaded cfg.

        Presets that already have settings keep them, except for those the cfg
        records: the cfg is the authority on those.
        """
        if (core := self._core_ref()) is None:
            return
        group = core.getChannelGroup()
        if not group:
            return
        self._seeded_groups.add(group)
        declarations = parse_channel_comments(
            core.systemConfigurationFile() or "", group
        )
        # a multi-property declaration is one multi-slider light source: map the
        # whole pair set back to the label of the source driving all of them
        by_pairs = {
            frozenset(pairs): label for label, pairs in find_light_sources(core).items()
        }
        exposure = core.getExposure() if core.getCameraDevice() else 100.0
        for preset in core.getAvailableConfigs(group):
            settings = self.get(group, preset) or {
                "exposure": exposure,
                "light_source": "",
                "intensity": 0.0,
            }
            if (declared := declarations.get(preset)) is not None:
                if declared.exposure is not None:
                    settings["exposure"] = declared.exposure
                pairs = frozenset((dev, prop) for dev, prop, _ in declared.light_source)
                if label := by_pairs.get(pairs):
                    settings["light_source"] = label
                    settings["intensity"] = declared.light_source[0][2]
            self.set(group, preset, settings)

    def is_live(self) -> bool:
        """Return whether live is running, i.e. a sequence that is not an MDA."""
        if (core := self._core_ref()) is None:
            return False
        return bool(core.isSequenceRunning() and not core.mda.is_running())

    def apply_current_channel(self) -> bool:
        """Apply the settings of the channel preset currently set on the core.

        Called before a snap or live, so they image with the Channels panel's
        settings. Returns whether there were settings to apply.
        """
        if (core := self._core_ref()) is None:
            return False
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


_STORES: WeakKeyDictionary[CMMCorePlus, ChannelSettingsStore] = WeakKeyDictionary()


def channel_settings_store(core: CMMCorePlus) -> ChannelSettingsStore:
    """Return the channel settings store of ``core``, creating it on first use."""
    if (store := _STORES.get(core)) is None:
        store = _STORES[core] = ChannelSettingsStore(core)
    return store
