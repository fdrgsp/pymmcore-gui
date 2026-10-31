"""The MDA's channel table, linked to the Channels panel's per-preset settings."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any

from pymmcore_widgets.mda import CoreConnectedChannelTable
from superqt.utils import signals_blocked

from pymmcore_gui._channel_settings_store import CAMERA, LIGHT_SOURCES

if TYPE_CHECKING:
    from collections.abc import Iterable

    import useq
    from pymmcore_plus import CMMCorePlus
    from pymmcore_widgets._channel_settings import ChannelSettings

    from pymmcore_gui._channel_settings_store import ChannelSettingsStore
    from pymmcore_gui._qt.QtWidgets import QWidget

_ChannelKey = tuple[str, str]  # (group, config)


def _key_of(record: dict[str, Any]) -> _ChannelKey:
    """A channel table row's (group, config)."""
    return str(record.get("group") or ""), str(record.get("config") or "")


class MDAChannelTable(CoreConnectedChannelTable):
    """Core channel table whose rows share their presets' settings.

    With a settings store (see `setSettingsStore`), a channel added or switched
    to a preset the table has not seen starts from that preset's stored
    settings, a user's edit to a row is stored for its preset, and a change to
    the stored settings from elsewhere (the Channels panel) updates the rows of
    that preset. Its Light Sources and Camera checkboxes are shared with the
    Channels panel's, through the store.
    """

    def __init__(
        self,
        rows: int = 0,
        mmcore: CMMCorePlus | None = None,
        parent: QWidget | None = None,
    ) -> None:
        # The settings each (group, config) last had when it left the table --
        # its row was removed, or switched to another config with its Config
        # combo -- so bringing that channel back recalls its own exposure,
        # light source, intensity and camera properties. Used when there is no
        # settings store, or it does not know the channel.
        self._channel_settings: dict[_ChannelKey, ChannelSettings] = {}
        # Every row's (group, config) as of the last change, see _on_changed.
        self._row_keys: list[_ChannelKey] = []
        self._store: ChannelSettingsStore | None = None
        super().__init__(rows, mmcore, parent)
        self.show_light_source.toggled.connect(
            lambda on: self._set_store_columns_visible(LIGHT_SOURCES, on)
        )
        self.camera.toggled.connect(
            lambda on: self._set_store_columns_visible(CAMERA, on)
        )
        self.valueChanged.connect(self._on_changed)
        self._row_keys = self._current_keys()
        self.destroyed.connect(self._disconnect_store)

    def setSettingsStore(self, store: ChannelSettingsStore | None) -> None:
        """Link the rows to ``store``'s per-preset settings."""
        if self._store is not None:
            self._store.settingsChanged.disconnect(self._on_store_changed)
            self._store.columnsVisibleChanged.disconnect(
                self._on_columns_visible_changed
            )
        self._store = store
        if store is not None:
            store.settingsChanged.connect(self._on_store_changed)
            store.columnsVisibleChanged.connect(self._on_columns_visible_changed)
            for columns in (LIGHT_SOURCES, CAMERA):
                self._on_columns_visible_changed(columns, store.columnsVisible(columns))

    def _set_store_columns_visible(self, columns: str, visible: bool) -> None:
        if self._store is not None:
            self._store.setColumnsVisible(columns, visible)

    def _on_columns_visible_changed(self, columns: str, visible: bool) -> None:
        if columns == LIGHT_SOURCES and visible != self.lightSourceVisible():
            self.setLightSourceVisible(visible)
        elif columns == CAMERA:
            self.setCameraVisible(visible)

    def applyStoredSettings(self, rows: Iterable[int]) -> list[int]:
        """Give each of ``rows`` its preset's stored settings.

        Returns the rows the store has no settings for.
        """
        table = self.table()
        missing: list[int] = []
        applied = False
        for row in rows:
            key = _key_of(table.rowData(row))
            stored = self._store.get(*key) if self._store and key[1] else None
            if stored is None:
                missing.append(row)
                continue
            with signals_blocked(table):
                self.setRowSettings(row, stored)
            applied = True
        if applied:
            self.valueChanged.emit()
        return missing

    def storeRowSettings(self, row: int) -> None:
        """Store ``row``'s settings as those of its preset (a user edited it)."""
        key = _key_of(self.table().rowData(row))
        if self._store is not None and key[1]:
            self._store.set(*key, self.rowSettings(row), source=self)

    def _on_store_changed(self, group: str, preset: str, source: object) -> None:
        """Update the rows of a preset whose settings were changed elsewhere."""
        if source is self:
            return
        # what this table remembers for the preset is now out of date
        self._channel_settings.pop((group, preset), None)
        table = self.table()
        rows = [
            row
            for row in range(table.rowCount())
            if _key_of(table.rowData(row)) == (group, preset)
        ]
        if rows:
            self.applyStoredSettings(rows)

    def _disconnect_store(self, *_: object) -> None:
        if self._store is not None:
            with suppress(TypeError, RuntimeError):
                self._store.settingsChanged.disconnect(self._on_store_changed)
                self._store.columnsVisibleChanged.disconnect(
                    self._on_columns_visible_changed
                )

    def _current_keys(self) -> list[_ChannelKey]:
        table = self.table()
        return [_key_of(table.rowData(r)) for r in range(table.rowCount())]

    def _row_entry(self, row: int) -> tuple[_ChannelKey, ChannelSettings]:
        return _key_of(self.table().rowData(row)), self.rowSettings(row)

    def _on_changed(self) -> None:
        """Remember a channel's settings when its row switches to another config.

        A Config combo change alters only that row's key: until
        restoreChannelSettings() runs, the row still holds the settings it
        had under its previous config, so they are filed under that one.
        Anything else -- rows added, removed (see _remove_selected), or moved
        (which changes two rows' keys at once) -- just updates the keys.
        """
        keys = self._current_keys()
        if len(keys) == len(self._row_keys):
            switched = [
                row
                for row, (old, new) in enumerate(zip(self._row_keys, keys, strict=True))
                if old != new
            ]
            if len(switched) == 1:
                row = switched[0]
                self._remember([(self._row_keys[row], self.rowSettings(row))])
        self._row_keys = keys

    def _remember(self, rows: Iterable[tuple[_ChannelKey, ChannelSettings]]) -> None:
        for key, settings in rows:
            if key[1]:
                self._channel_settings[key] = settings

    def setValue(self, value: Iterable[useq.Channel]) -> None:
        # fills the table with signals blocked, i.e. without a valueChanged
        super().setValue(value)
        self._row_keys = self._current_keys()

    def _remove_selected(self) -> None:
        self._remember(self._row_entry(r) for r in self._selected_rows())
        super()._remove_selected()

    def _remove_all(self) -> None:
        self._remember(self._row_entry(r) for r in range(self.table().rowCount()))
        super()._remove_all()

    def _add_row(self) -> None:
        table = self.table()
        count = table.rowCount()
        super()._add_row()
        if table.rowCount() > count:
            self.restoreChannelSettings(count)

    def restoreChannelSettings(self, row: int) -> bool:
        """Give ``row`` its preset's settings.

        In order of preference: another row's (so duplicating a channel copies
        it), the ones its preset last had in this table, or the stored ones.

        Returns whether they were applied in full, i.e. including a light source
        this configuration offers.
        """
        # file the row's previous channel first, while it still holds its values
        self._on_changed()
        table = self.table()
        key = _key_of(table.rowData(row))
        if not key[1]:
            return False
        remembered = self._channel_settings.get(key)
        for other_row in reversed(range(table.rowCount())):
            if other_row == row:
                continue
            other_key, other_settings = self._row_entry(other_row)
            if other_key == key:
                remembered = other_settings
                break
        if remembered is None and self._store is not None:
            remembered = self._store.get(*key)
        if remembered is None:
            return False
        source = remembered["light_source"]
        source_available = source == "" or source in self.lightSources()
        with signals_blocked(table):
            self.setRowSettings(row, remembered)
        self.valueChanged.emit()
        return source_available
