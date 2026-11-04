"""The MDA's channel table, linked to the Channels panel's per-preset settings."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any

from pymmcore_widgets._channel_settings import apply_channel_settings
from pymmcore_widgets._light_sources import NO_LIGHT_SOURCE
from pymmcore_widgets.control._channel_settings_table import (
    CurrentPresetColumn,
    CurrentPresetDelegate,
)
from pymmcore_widgets.mda import CoreConnectedChannelTable
from superqt.utils import signals_blocked

from pymmcore_gui._qt.QtWidgets import QCheckBox
from pymmcore_gui._theme import qcolor, theme

from ._channel_settings import add_save_to_cfg_button, save_channel_settings

if TYPE_CHECKING:
    from collections.abc import Iterable

    import useq
    from pymmcore_plus import CMMCorePlus
    from pymmcore_widgets._channel_settings import ChannelSettings

    from pymmcore_gui._channel_settings_store import ChannelSettingsStore
    from pymmcore_gui._qt.QtWidgets import QWidget

_ChannelKey = tuple[str, str]  # (group, config)

# the first column: a radio-like indicator of the channel active on the microscope
CURRENT_COLUMN = CurrentPresetColumn()


def _key_of(record: dict[str, Any]) -> _ChannelKey:
    """A channel table row's (group, config)."""
    return str(record.get("group") or ""), str(record.get("config") or "")


class MDAChannelTable(CoreConnectedChannelTable):
    """Core channel table that selects channels and shares their presets' settings.

    The first column marks the row of the channel active on the microscope. Clicking
    a row's indicator applies that channel (see `applyRow`); nothing else in the
    table does. The indicator follows the core, so a channel selected in the Channels
    panel is selected here too, and vice versa. It is independent of the table's row
    selection, which the move and remove buttons act on. While live is running, an
    edit of the marked row reaches the hardware right away.

    The *Channel Control* checkbox turns all of that on or off (see
    `setChannelControlEnabled`). While off, the first column is hidden and the table
    only plans an acquisition: it never changes the microscope's channel or settings.

    With a settings store (see `setSettingsStore`), a channel added or switched
    to a preset the table has not seen starts from that preset's stored
    settings, a user's edit to a row is stored for its preset, and a change to
    the stored settings from elsewhere (the Channels panel) updates the rows of
    that preset. Its Light Source checkbox is shared with the Channels panel's,
    through the store.
    """

    def __init__(
        self,
        rows: int = 0,
        mmcore: CMMCorePlus | None = None,
        parent: QWidget | None = None,
    ) -> None:
        # Recall channels removed or switched since the channel group last changed.
        self._channel_settings: dict[_ChannelKey, ChannelSettings] = {}
        # Every row's (group, config) as of the last change, see _on_changed.
        self._row_keys: list[_ChannelKey] = []
        # The row whose config last changed, until a user's pick of it is handled
        # (see presetPicked).
        self._switched_row = -1
        self._store: ChannelSettingsStore | None = None
        # The row marked active, or -1.
        self._active_row = -1
        # That row, its (group, config) and settings, to tell when it is edited.
        self._active_state: tuple[int, _ChannelKey, ChannelSettings] | None = None
        super().__init__(rows, mmcore, parent)
        self.valueChanged.connect(self._on_changed)
        self._row_keys = self._current_keys()

        table = self.table()
        table.addColumn(CURRENT_COLUMN, 0)
        delegate = CurrentPresetDelegate(table)
        delegate.setActiveColor(qcolor(theme().status_yellow))
        CURRENT_COLUMN.setup(table, delegate)
        table.cellClicked.connect(self._on_cell_clicked)

        self.channel_control = QCheckBox("Channel Control")
        self.channel_control.setToolTip(
            "Select the microscope's channel from this table, with the first column.\n"
            "While unchecked that column is hidden, and this table only plans the "
            "acquisition: it never changes the microscope's channel or settings."
        )
        self.channel_control.setChecked(True)
        self.channel_control.toggled.connect(self.setChannelControlEnabled)
        self._btn_row.insertWidget(0, self.channel_control)
        # Light Source goes last, with its save button next to it
        self._btn_row.removeWidget(self.show_light_source)
        self._btn_row.insertWidget(
            self._btn_row.indexOf(self.advanced) + 1, self.show_light_source
        )
        self._save_btn = add_save_to_cfg_button(
            self._btn_row, self.show_light_source, self
        )
        self._save_btn.clicked.connect(self._on_save_clicked)

        self._mmc.events.configSet.connect(self._on_config_set)
        self._mmc.events.propertyChanged.connect(self._on_property_changed)
        # after the base class's connection, which clears (and so remembers) the rows
        self._mmc.events.channelGroupChanged.connect(self._forget_channel_settings)
        self.destroyed.connect(self._disconnect_store)
        self._update_active_row()

    def _on_save_clicked(self) -> None:
        save_channel_settings(self, self._mmc)

    def setLightSourceVisible(self, visible: bool) -> None:
        super().setLightSourceVisible(visible)
        self._save_btn.setEnabled(visible)
        # told here, as a loaded sequence turns it on without the checkbox's signal
        if self._store is not None:
            self._store.setLightSourcesVisible(visible)

    def setChannelControlEnabled(self, enabled: bool) -> None:
        """Show or hide the first column, and let this table drive the microscope."""
        with signals_blocked(self.channel_control):
            self.channel_control.setChecked(enabled)
        table = self.table()
        table.setColumnHidden(table.indexOf(CURRENT_COLUMN), not enabled)

    def channelControlEnabled(self) -> bool:
        """Return whether this table can change the microscope's channel."""
        return bool(self.channel_control.isChecked())

    def activeRow(self) -> int:
        """Return the row marked as the channel active on the microscope, or -1."""
        return self._active_row

    def applyRow(self, row: int, *, wait: bool = False) -> bool:
        """Select ``row``: set its channel, exposure and light source on the core.

        The row's own values are applied and left as they are, so that several rows
        of one preset each keep theirs. With ``wait``, returns once the devices are
        done, as imaging needs. Returns whether the row holds a channel to apply.
        """
        group, config = _key_of(self.table().rowData(row))
        core = self._mmc
        if group not in core.getAvailableConfigGroups():
            return False
        if config not in core.getAvailableConfigs(group):
            return False
        # keeps the mark on this row, rather than on another row of the same preset
        self._active_row = row
        if core.getCurrentConfig(group) != config:
            core.setConfig(group, config)
        if wait:
            core.waitForConfig(group, config)
        settings = self.rowSettings(row)
        if not self.lightSourceVisible():
            settings["light_source"] = NO_LIGHT_SOURCE
        apply_channel_settings(core, settings, self.lightSources())
        if wait:
            core.waitForSystem()
        self._update_active_row()
        return True

    def presetPicked(self, row: int) -> None:
        """Handle a user's pick in ``row``'s Config combo.

        Picking another preset gives the row that preset's settings; picking the one
        it already has keeps the row's own. This never changes the microscope's
        channel: only the first column does.
        """
        self._on_changed()
        if self._switched_row == row:
            self.restoreChannelSettings(row)
        self._switched_row = -1

    def refresh(self) -> None:
        super().refresh()
        self._update_active_row()

    def _on_cell_clicked(self, row: int, col: int) -> None:
        if col == self.table().indexOf(CURRENT_COLUMN):
            self.applyRow(row)

    def _on_config_set(self, group: str, preset: str) -> None:
        if group == self._group_combo.currentText():
            self._update_active_row()

    def _on_property_changed(self, device: str, prop: str, value: str) -> None:
        # a device in a preset may have been moved directly
        self._update_active_row()

    def _update_active_row(self) -> None:
        """Mark the row of the channel the core reports as current.

        The row already marked stays so as long as it fits, since several rows may
        hold one preset; otherwise the first checked row of it, or else the first.
        """
        group = self._group_combo.currentText()
        try:
            preset = self._mmc.getCurrentConfigFromCache(group) if group else ""
        except RuntimeError:  # pragma: no cover
            preset = ""
        rows = [
            row
            for row, key in enumerate(self._current_keys())
            if preset and key == (group, preset)
        ]
        if self._active_row not in rows:
            table = self.table()
            selector_col = table._get_selector_col()
            selector = table.columnInfo(selector_col)
            checked = [
                row
                for row in rows
                if selector is not None and selector.isChecked(table, row, selector_col)
            ]
            self._active_row = next(iter(checked or rows), -1)
        CURRENT_COLUMN.markActive(self.table(), self._active_row)
        row = self._active_row
        self._active_state = (
            None
            if row < 0
            else (row, _key_of(self.table().rowData(row)), self.rowSettings(row))
        )

    def _push_live_edit(
        self, before: tuple[int, _ChannelKey, ChannelSettings] | None
    ) -> None:
        """Apply the marked row if it was edited while live is running.

        Otherwise an edit takes effect at the next selection, snap or live.
        """
        after = self._active_state
        if (
            before is not None
            and after is not None
            and before[:2] == after[:2]
            and before[2] != after[2]
            and self.channelControlEnabled()
            and self._store is not None
            and self._store.is_live()
        ):
            self.applyRow(after[0])

    def _swap_rows(self, row: int, other: int) -> None:
        super()._swap_rows(row, other)
        # the mark is not part of a row's data, but it moves with the row
        if self._active_row in (row, other):
            self._active_row = other if self._active_row == row else row

    def _on_group_changed(self) -> None:
        super()._on_group_changed()
        self._update_active_row()

    def setSettingsStore(self, store: ChannelSettingsStore | None) -> None:
        """Link the rows to ``store``'s per-preset settings."""
        if self._store is not None:
            self._store.settingsChanged.disconnect(self._on_store_changed)
            self._store.lightSourcesVisibleChanged.disconnect(
                self._on_light_sources_visible_changed
            )
        self._store = store
        if store is not None:
            store.settingsChanged.connect(self._on_store_changed)
            store.lightSourcesVisibleChanged.connect(
                self._on_light_sources_visible_changed
            )
            self._on_light_sources_visible_changed(store.lightSourcesVisible())

    def _on_light_sources_visible_changed(self, visible: bool) -> None:
        if visible != self.lightSourceVisible():
            self.setLightSourceVisible(visible)

    def applyStoredSettings(self, rows: Iterable[int]) -> None:
        """Give each of ``rows`` its preset's stored settings."""
        table = self.table()
        applied = False
        for row in rows:
            key = _key_of(table.rowData(row))
            stored = self._store.get(*key) if self._store and key[1] else None
            if stored is None:
                continue
            with signals_blocked(table):
                self.setRowSettings(row, stored)
            applied = True
        if applied:
            self.valueChanged.emit()

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
                self._store.lightSourcesVisibleChanged.disconnect(
                    self._on_light_sources_visible_changed
                )
        with suppress(TypeError, RuntimeError):
            events = self._mmc.events
            events.configSet.disconnect(self._on_config_set)
            events.propertyChanged.disconnect(self._on_property_changed)
            events.channelGroupChanged.disconnect(self._forget_channel_settings)

    def _forget_channel_settings(self, *_: object) -> None:
        """Leave a group's settings to the store once its rows are gone.

        Otherwise a row that differs from the store (a restored sequence's, say) would
        override it on coming back to the group, unlike what the Channels panel shows.
        """
        self._channel_settings.clear()

    def _current_keys(self) -> list[_ChannelKey]:
        table = self.table()
        return [_key_of(table.rowData(r)) for r in range(table.rowCount())]

    def _row_entry(self, row: int) -> tuple[_ChannelKey, ChannelSettings]:
        return _key_of(self.table().rowData(row)), self.rowSettings(row)

    def _on_changed(self) -> None:
        """Remember settings before a config switch replaces them.

        A single changed key identifies a config switch; row moves change two keys.
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
                self._switched_row = row
        self._row_keys = keys
        before = self._active_state
        self._update_active_row()
        self._push_live_edit(before)

    def _remember(self, rows: Iterable[tuple[_ChannelKey, ChannelSettings]]) -> None:
        for key, settings in rows:
            if key[1]:
                self._channel_settings[key] = settings

    def setValue(self, value: Iterable[useq.Channel]) -> None:
        # fills the table with signals blocked, i.e. without a valueChanged
        super().setValue(value)
        self._row_keys = self._current_keys()
        self._update_active_row()

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
        it), the ones its preset last had in this table since the channel group last
        changed, or the stored ones.

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
