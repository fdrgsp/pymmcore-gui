"""The Channels panel and the MDA channel table share per-preset settings."""

from __future__ import annotations

import gc
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
import useq
from pymmcore_widgets._light_sources import find_light_sources

from pymmcore_gui._channel_settings_store import channel_settings_store
from pymmcore_gui._qt.QtCore import QEvent, QSize, Qt
from pymmcore_gui._qt.QtWidgets import QComboBox, QFileDialog, QHeaderView, QStyle
from pymmcore_gui._theme import qcolor, set_theme, theme
from pymmcore_gui._theme._dark import DARK_THEME
from pymmcore_gui._theme._light import LIGHT_THEME
from pymmcore_gui.widgets._acquire import AcquirePage
from pymmcore_gui.widgets._channel_settings import ChannelSettingsPanel
from pymmcore_gui.widgets._mda_channel_table import CURRENT_COLUMN
from pymmcore_gui.widgets._mda_widget import MemoryMDAWidget
from pymmcore_gui.widgets._preferences import PreferencesButton

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pymmcore_plus import CMMCorePlus
    from pymmcore_widgets._channel_settings import ChannelSettings
    from pytestqt.qtbot import QtBot

    from pymmcore_gui._channel_settings_store import ChannelSettingsStore


def _panel(mmcore: CMMCorePlus, qtbot: QtBot) -> ChannelSettingsPanel:
    panel = ChannelSettingsPanel(mmcore)
    qtbot.addWidget(panel)
    return panel


def _mda(mmcore: CMMCorePlus, qtbot: QtBot, *configs: str) -> MemoryMDAWidget:
    mda = MemoryMDAWidget(mmcore)
    qtbot.addWidget(mda)
    mda.setValue(
        useq.MDASequence(
            channels=tuple(
                useq.Channel(group="Channel", config=c, exposure=100) for c in configs
            )
        )
    )
    return mda


def _panel_cell(panel: ChannelSettingsPanel, preset: str, key: str) -> Any:
    table = panel.table()
    row = next(
        r
        for r in range(table.rowCount())
        if (item := table.item(r, 1)) is not None and item.text() == preset
    )
    col = next(
        c
        for c in range(table.columnCount())
        if (info := table.columnInfo(c)) is not None and info.key == key
    )
    return table.cellWidget(row, col)


def _mda_cell(mda: MemoryMDAWidget, row: int, column: Any) -> Any:
    table = mda.channels.table()
    return table.cellWidget(row, table.indexOf(column))


def _pick_preset(mda: MemoryMDAWidget, row: int, preset: str) -> None:
    """Pick ``preset`` in ``row``'s Config combo, as the user does."""
    cell = _mda_cell(mda, row, mda.channels._config_column)
    combo = cell if isinstance(cell, QComboBox) else cell.findChild(QComboBox)
    combo.setCurrentText(preset)
    combo.activated.emit(combo.currentIndex())


def _stored(mmcore: CMMCorePlus, preset: str) -> ChannelSettings:
    settings = channel_settings_store(mmcore).get("Channel", preset)
    assert settings is not None
    return settings


def _mda_row_exposures(mda: MemoryMDAWidget) -> list[tuple[str, float | None]]:
    return [(str(c.config), c.exposure) for c in mda.channels.value()]


def test_store_does_not_retain_unused_core(mmcore: CMMCorePlus) -> None:
    # Both of pytest's log handlers otherwise retain constructor arguments.
    with patch("pymmcore_plus._logger.logger.disabled", True):
        other = type(mmcore)()
    reference = weakref.ref(other)
    store = channel_settings_store(other)
    del other
    gc.collect()
    assert reference() is None
    assert store.all() == {}
    store.seed()
    assert not store.apply_current_channel()


def test_store_is_seeded_with_every_channel_preset(mmcore: CMMCorePlus) -> None:
    store = channel_settings_store(mmcore)
    assert store is channel_settings_store(mmcore)
    assert set(store.all()["Channel"]) == set(mmcore.getAvailableConfigs("Channel"))


def test_panel_settings_reach_a_new_mda_channel(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    panel = _panel(mmcore, qtbot)
    mda = _mda(mmcore, qtbot, "DAPI")
    _panel_cell(panel, "FITC", "exposure").setValue(42)
    assert _stored(mmcore, "FITC")["exposure"] == 42

    mda.channels.act_add_row.trigger()
    _pick_preset(mda, 1, "FITC")
    assert _mda_row_exposures(mda) == [("DAPI", 100), ("FITC", 42)]


def test_panel_edit_updates_the_mda_rows_of_that_preset(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    panel = _panel(mmcore, qtbot)
    mda = _mda(mmcore, qtbot, "DAPI", "FITC", "DAPI")
    _panel_cell(panel, "DAPI", "exposure").setValue(25)
    assert _mda_row_exposures(mda) == [("DAPI", 25), ("FITC", 100), ("DAPI", 25)]


def test_mda_edit_is_stored_and_shown_in_the_panel(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    panel = _panel(mmcore, qtbot)
    mda = _mda(mmcore, qtbot, "DAPI", "DAPI")
    spin = _mda_cell(mda, 0, mda.channels.EXPOSURE)
    spin.setValue(77)
    spin.editingFinished.emit()  # as the user's Enter / focus-out does

    assert _stored(mmcore, "DAPI")["exposure"] == 77
    qtbot.waitUntil(lambda: _panel_cell(panel, "DAPI", "exposure").value() == 77)
    # the MDA's other DAPI row is its own: it is not changed behind its back
    assert _mda_row_exposures(mda) == [("DAPI", 77), ("DAPI", 100)]


def test_a_restored_sequence_is_not_stored(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    store = channel_settings_store(mmcore)
    before = store.get("Channel", "DAPI")
    _mda(mmcore, qtbot, "DAPI")  # setValue with exposure 100
    assert store.get("Channel", "DAPI") == before


def _load_cfg(mmcore: CMMCorePlus, tmp_path: Path, *extra: str) -> Path:
    """Load a copy of the test cfg with ``extra`` lines appended."""
    dest = tmp_path / "declared.cfg"
    lines = Path(__file__).with_name("test_config.cfg").read_text().splitlines()
    dest.write_text("\n".join([*lines, *extra]) + "\n")
    mmcore.loadSystemConfiguration(dest)
    return dest


def _label_of(mmcore: CMMCorePlus, *pairs: tuple[str, str]) -> str:
    return next(
        label
        for label, dev_props in find_light_sources(mmcore).items()
        if dev_props == list(pairs)
    )


def test_cfg_settings_seed_the_store(mmcore: CMMCorePlus, tmp_path: Path) -> None:
    _load_cfg(
        mmcore,
        tmp_path,
        "#@Exposure,Channel,DAPI,12.5",
        "#@LightSource,Channel,DAPI,Camera,TestProperty1,0.05",
    )
    settings = channel_settings_store(mmcore).get("Channel", "DAPI")
    assert settings is not None
    assert settings["exposure"] == 12.5
    # whichever label offers exactly that property
    sources = find_light_sources(mmcore)
    assert sources[settings["light_source"]] == [("Camera", "TestProperty1")]
    assert settings["intensity"] == 0.05
    # FITC has no declaration, and nothing is inferred for it
    fitc = channel_settings_store(mmcore).get("Channel", "FITC")
    assert fitc is not None
    assert fitc["light_source"] == ""


def test_cfg_multi_property_light_source(mmcore: CMMCorePlus, tmp_path: Path) -> None:
    """Several properties declared together restore as their group's label.

    That is what makes one Intensity spin box drive every slider of a multi-channel
    illuminator, rather than landing on a single "Device · Property" entry.
    """
    pairs = [("Camera", "TestProperty2"), ("Camera", "TestProperty4")]
    _load_cfg(
        mmcore,
        tmp_path,
        *(f"#@LightSource,Channel,DAPI,{dev},{prop},12.0" for dev, prop in pairs),
    )
    for device, prop in pairs:
        mmcore.defineConfig("_lida", "levels", device, prop, "0.0")
    # seeded again once the light source exists, as on the next config load
    store = channel_settings_store(mmcore)
    store.seed()
    settings = store.get("Channel", "DAPI")
    assert settings is not None
    assert settings["light_source"] == "_lida"
    assert settings["intensity"] == pytest.approx(12.0)


def test_light_source_is_never_inferred_from_the_channel_preset(
    mmcore: CMMCorePlus,
) -> None:
    """A ranged property inside a channel preset is not treated as a declaration.

    Storing the light source there would break preset identity, so it is only ever
    read from the cfg's comment block -- see ``pymmcore_gui._light_sources``.
    """
    mmcore.defineConfig("Channel", "DAPI", "Camera", "TestProperty1", "0.05")
    store = channel_settings_store(mmcore)
    store.seed()
    settings = store.get("Channel", "DAPI")
    assert settings is not None
    assert settings["light_source"] == ""


def test_stale_and_malformed_declarations_are_ignored(
    mmcore: CMMCorePlus, tmp_path: Path
) -> None:
    """The block is hand-editable, so junk must be ignored rather than raise."""
    _load_cfg(
        mmcore,
        tmp_path,
        "#@LightSource,Channel,DAPI,Camera,TestProperty1",  # too few fields
        "#@LightSource,Channel,FITC,Camera,TestProperty1,not-a-number",
        "#@LightSource,OtherGroup,Cy5,Camera,TestProperty1,0.05",  # other group
        "#@LightSource,Channel,Cy5,NoSuchDevice,Level,5",  # not in this config
        "#@Exposure,Channel,Cy5,fast",
        "#@LightSource,Channel,Rhodamine,Camera,TestProperty1,0.05",  # the good one
    )
    store = channel_settings_store(mmcore)
    for preset in ("DAPI", "FITC", "Cy5"):
        settings = store.get("Channel", preset)
        assert settings is not None
        assert settings["light_source"] == ""
    assert store.get("Channel", "Cy5")["exposure"] == mmcore.getExposure()  # type: ignore[index]
    assert store.get("Channel", "Rhodamine")["intensity"] == pytest.approx(0.05)  # type: ignore[index]


def test_cfg_light_source_reaches_a_new_mda_channel(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    _load_cfg(mmcore, tmp_path, "#@LightSource,Channel,FITC,Camera,TestProperty1,0.1")
    mda = _mda(mmcore, qtbot, "DAPI")
    mda.channels.act_add_row.trigger()
    _pick_preset(mda, 1, "FITC")
    props = mda.channels.channelProperties(exclude_unchecked=False)
    assert [(p["channel_index"], p["property"], p["value"]) for p in props] == [
        (1, "TestProperty1", 0.1)
    ]


def test_save_channel_settings_writes_a_complete_cfg(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    panel = _panel(mmcore, qtbot)
    store = channel_settings_store(mmcore)
    label = _label_of(mmcore, ("Camera", "TestProperty1"))
    store.set(
        "Channel",
        "DAPI",
        {
            "exposure": 12.5,
            "light_source": label,
            "intensity": 0.05,
        },
    )

    dest = tmp_path / "with_settings.cfg"
    with patch.object(QFileDialog, "getSaveFileName", return_value=(str(dest), "")):
        panel._save_btn.click()

    lines = dest.read_text().splitlines()
    assert "#@Exposure,Channel,DAPI,12.5" in lines
    assert "#@LightSource,Channel,DAPI,Camera,TestProperty1,0.05" in lines
    # FITC has no light source, so it gets no light source declaration
    assert not any(line.startswith("#@LightSource,Channel,FITC,") for line in lines)
    # the whole configuration is written, not just the channel settings
    assert any(line.startswith("Device,") for line in lines)
    assert any(line.startswith("PixelSize_um,") for line in lines)
    # nothing was written as real configuration, so preset identity still works
    assert not any(
        line.startswith("ConfigGroup,Channel,DAPI,Camera,TestProperty1")
        for line in lines
    )
    mmcore.setConfig("Channel", "DAPI")
    assert mmcore.getCurrentConfig("Channel") == "DAPI"

    # and they come back when that cfg is loaded
    store.set(
        "Channel",
        "DAPI",
        {
            "exposure": 1.0,
            "light_source": "",
            "intensity": 0.0,
        },
    )
    mmcore.loadSystemConfiguration(dest)
    # the core reports the load once control returns to the event loop
    qtbot.waitUntil(lambda: store.get("Channel", "DAPI")["exposure"] == 12.5)  # type: ignore[index]
    reloaded = store.get("Channel", "DAPI")
    assert reloaded is not None
    assert find_light_sources(mmcore)[reloaded["light_source"]] == [
        ("Camera", "TestProperty1")
    ]
    # the panel shows them too
    qtbot.waitUntil(lambda: panel.channelSettings("DAPI") == reloaded)


def test_light_source_checkbox_is_shared(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    panel = _panel(mmcore, qtbot)
    mda = _mda(mmcore, qtbot, "DAPI")
    channels = mda.channels
    assert panel.lightSourcesVisible() and channels.lightSourceVisible()

    channels.show_light_source.setChecked(False)
    assert not panel.light_sources.isChecked()
    panel.light_sources.setChecked(True)
    assert channels.show_light_source.isChecked()
    channels.show_light_source.setChecked(False)
    # a panel opened later starts the same way
    assert not _panel(mmcore, qtbot).lightSourcesVisible()


def test_no_camera_columns_or_checkbox(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    panel = _panel(mmcore, qtbot)
    mda = _mda(mmcore, qtbot, "DAPI")
    channels = mda.channels
    assert not hasattr(panel, "camera")
    assert not hasattr(channels, "camera")
    for table in (panel.table(), channels.table()):
        keys = [table.columnInfo(c).key for c in range(table.columnCount())]  # type: ignore[union-attr]
        assert not [key for key in keys if key.startswith("camera")]
    assert "camera_properties" not in _stored(mmcore, "DAPI")


def _click(qtbot: QtBot, mda: MemoryMDAWidget, row: int, column: Any) -> None:
    """Click a cell through the viewport, as a real click reaches it."""
    table = mda.channels.table()
    model = table.model()
    assert model is not None
    rect = table.visualRect(model.index(row, table.indexOf(column)))
    qtbot.mouseClick(  # type: ignore[no-untyped-call]
        table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center()
    )


def _selected(mda: MemoryMDAWidget) -> list[int]:
    table = mda.channels.table()
    return sorted({index.row() for index in table.selectedIndexes()})


def test_mda_table_marks_the_active_channel(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    mda = _mda(mmcore, qtbot, "DAPI", "FITC")
    channels = mda.channels
    table = channels.table()
    assert table.indexOf(CURRENT_COLUMN) == 0
    header = table.horizontalHeader()
    assert header is not None
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Fixed

    # it follows the core, whatever changes the channel
    mmcore.setConfig("Channel", "FITC")
    assert channels.activeRow() == 1
    mmcore.setConfig("Channel", "DAPI")
    assert channels.activeRow() == 0
    # a channel without a row clears it
    mmcore.setConfig("Channel", "Cy5")
    assert channels.activeRow() == -1

    # following the core never moves the row's settings to the hardware
    mmcore.setExposure(7)
    mmcore.setConfig("Channel", "FITC")
    assert mmcore.getExposure() == 7


def test_clicking_the_indicator_selects_the_channel(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    label = _label_of(mmcore, ("Camera", "TestProperty2"))
    panel = _panel(mmcore, qtbot)
    mda = _mda(mmcore, qtbot, "DAPI", "FITC")
    mda.channels.setRowSettings(
        1, {"exposure": 33.0, "light_source": label, "intensity": 0.25}
    )
    mda.show()
    qtbot.waitExposed(mda)
    mda.revealAxis("c")

    _click(qtbot, mda, 1, CURRENT_COLUMN)
    assert mmcore.getCurrentConfig("Channel") == "FITC"
    assert mmcore.getExposure() == 33
    assert float(mmcore.getProperty("Camera", "TestProperty2")) == 0.25
    assert mda.channels.activeRow() == 1
    # the Channels panel shows the same channel
    assert panel.activePreset() == "FITC"


def test_only_the_indicator_changes_channel(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    mda = _mda(mmcore, qtbot, "DAPI", "FITC")
    channels = mda.channels
    mda.show()
    qtbot.waitExposed(mda)
    mda.revealAxis("c")
    mmcore.setConfig("Channel", "DAPI")

    # clicking another row's editor selects the row, for the move and remove buttons
    spin = _mda_cell(mda, 1, channels.EXPOSURE)
    _click(qtbot, mda, 1, channels.EXPOSURE)
    assert mda.focusWidget() is spin
    if mda.isActiveWindow():  # focus events need it, and CI cannot always make it so
        assert _selected(mda) == [1]
    assert mmcore.getCurrentConfig("Channel") == "DAPI"
    assert channels.activeRow() == 0

    # picking a preset in the Config combo gives the row its settings and selects it
    table = channels.table()
    table.selectRow(0)
    _pick_preset(mda, 1, "Cy5")
    assert _selected(mda) == [1]
    assert mmcore.getCurrentConfig("Channel") == "DAPI"
    assert channels.activeRow() == 0
    assert _mda_row_exposures(mda)[1] == ("Cy5", _stored(mmcore, "Cy5")["exposure"])

    # not even in the active row, which then no longer holds the active channel
    _pick_preset(mda, 0, "FITC")
    assert mmcore.getCurrentConfig("Channel") == "DAPI"
    assert channels.activeRow() == -1


def test_picking_a_rows_own_preset_keeps_its_settings(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    mda = _mda(mmcore, qtbot, "DAPI", "DAPI")
    _mda_cell(mda, 1, mda.channels.EXPOSURE).setValue(55)

    _pick_preset(mda, 1, "DAPI")
    assert _mda_row_exposures(mda) == [("DAPI", 100), ("DAPI", 55)]


def test_rows_of_one_preset_keep_their_own_settings(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    mda = _mda(mmcore, qtbot, "DAPI", "DAPI")
    channels = mda.channels
    _mda_cell(mda, 0, channels.EXPOSURE).setValue(11)
    _mda_cell(mda, 1, channels.EXPOSURE).setValue(22)
    stored = _stored(mmcore, "DAPI")

    # selecting one applies its values, and writes them nowhere
    assert channels.applyRow(1)
    assert channels.activeRow() == 1
    assert mmcore.getExposure() == 22
    assert _stored(mmcore, "DAPI") == stored
    assert _mda_row_exposures(mda) == [("DAPI", 11), ("DAPI", 22)]

    # a capture uses the selected row's, whatever the core was left at
    mmcore.setExposure(5)
    assert mda.apply_active_channel_for_capture()
    assert mmcore.getExposure() == 22
    channels.applyRow(0)
    mmcore.setExposure(5)
    assert mda.apply_active_channel_for_capture()
    assert mmcore.getExposure() == 11


def test_capture_without_a_marked_row_uses_the_stored_settings(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    mda = _mda(mmcore, qtbot, "DAPI")
    mmcore.setConfig("Channel", "FITC")
    assert mda.channels.activeRow() == -1
    stored = _stored(mmcore, "FITC")
    channel_settings_store(mmcore).set("Channel", "FITC", {**stored, "exposure": 66.0})
    assert mda.apply_active_channel_for_capture()
    assert mmcore.getExposure() == 66


def test_selection_is_shared_with_the_panel(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    panel = _panel(mmcore, qtbot)
    mda = _mda(mmcore, qtbot, "DAPI", "FITC")
    table = panel.table()

    row = next(
        r
        for r in range(table.rowCount())
        if (item := table.item(r, 1)) is not None and item.text() == "FITC"
    )
    table.cellClicked.emit(row, table.indexOf(panel.CURRENT))
    assert mda.channels.activeRow() == 1

    mda.channels.applyRow(0)
    assert panel.activePreset() == "DAPI"


def test_row_selection_is_independent_of_the_mark(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    mda = _mda(mmcore, qtbot, "DAPI", "FITC", "DAPI")
    channels = mda.channels
    table = channels.table()

    # of several rows of one preset, the one clicked stays marked
    channels.applyRow(2)
    assert channels.activeRow() == 2

    # the core moving does not move the selection
    table.selectRow(1)
    mmcore.setConfig("Channel", "Cy5")
    assert channels.activeRow() == -1
    assert _selected(mda) == [1]

    # the mark moves with its row
    channels.applyRow(2)
    table.selectRow(2)
    channels.act_move_up.trigger()
    assert [c.config for c in channels.value()] == ["DAPI", "DAPI", "FITC"]
    assert channels.activeRow() == 1
    assert _selected(mda) == [1]

    # moving another row does not take the mark along
    table.selectRow(2)
    channels.act_move_up.trigger()
    assert [c.config for c in channels.value()] == ["DAPI", "FITC", "DAPI"]
    assert channels.activeRow() == 2

    # the remove button acts on the selection, and the mark stays on its channel
    table.selectRow(0)
    channels.act_remove_row.trigger()
    assert [c.config for c in channels.value()] == ["FITC", "DAPI"]
    assert channels.activeRow() == 1

    # a restored sequence leaves the hardware alone, and marks what is active
    exposure = mmcore.getExposure()
    mda.setValue(
        useq.MDASequence(
            channels=(
                useq.Channel(group="Channel", config="DAPI", exposure=9),
                useq.Channel(group="Channel", config="FITC", exposure=8),
            )
        )
    )
    assert mmcore.getCurrentConfig("Channel") == "DAPI"
    assert mmcore.getExposure() == exposure
    assert channels.activeRow() == 0

    # a new row gets an indicator too
    channels.act_add_row.trigger()
    assert table.item(2, table.indexOf(CURRENT_COLUMN)) is not None


def _pick_group(combo: QComboBox, group: str) -> None:
    """Pick ``group`` in a Group combo, as the user does."""
    combo.setCurrentText(group)
    combo.activated.emit(combo.currentIndex())


def test_group_combos_set_the_channel_group(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    panel = _panel(mmcore, qtbot)
    mda = _mda(mmcore, qtbot, "DAPI")
    channels = mda.channels
    assert mmcore.getChannelGroup() == "Channel"

    # from the panel, which clears the MDA's rows: they held the old group's presets
    _pick_group(panel._group_combo, "Objective")
    assert mmcore.getChannelGroup() == "Objective"
    assert channels._group_combo.currentText() == "Objective"
    assert list(panel.value()) == list(mmcore.getAvailableConfigs("Objective"))
    assert channels.value() == ()
    assert channels.activeRow() == -1

    # ...and from the MDA's channel table
    _pick_group(channels._group_combo, "Channel")
    assert mmcore.getChannelGroup() == "Channel"
    assert panel.channelGroup() == "Channel"

    # the core's is followed too
    mmcore.setChannelGroup("Objective")
    assert panel.channelGroup() == "Objective"
    assert channels._group_combo.currentText() == "Objective"

    # the channel marked is the new group's
    mmcore.setChannelGroup("Channel")
    mmcore.setConfig("Channel", "FITC")
    assert panel.activePreset() == "FITC"

    # showing a group without the user picking it leaves the core alone
    panel.setChannelGroup("Objective")
    assert mmcore.getChannelGroup() == "Channel"


def _exposure_of(store: ChannelSettingsStore, preset: str) -> float:
    settings = store.get("Objective", preset)
    assert settings is not None
    return settings["exposure"]


def test_new_channel_group_is_seeded_once(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    objective = mmcore.getAvailableConfigs("Objective")[0]
    _load_cfg(
        mmcore,
        tmp_path,
        f"#@Exposure,Objective,{objective},12.5",
        "#@Exposure,Channel,DAPI,7",
    )
    store = channel_settings_store(mmcore)
    mda = _mda(mmcore, qtbot, "DAPI")
    assert store.get("Objective", objective) is None

    # the cfg's settings for it arrive when it becomes the channel group...
    mmcore.setChannelGroup("Objective")
    settings = store.get("Objective", objective)
    assert settings is not None
    assert settings["exposure"] == 12.5
    # ...and a channel added then gets them
    assert mda.channels.value() == ()
    mda.channels.act_add_row.trigger()
    assert [c.group for c in mda.channels.value()] == ["Objective"]
    config = str(mda.channels.value()[0].config)
    assert _mda_row_exposures(mda) == [(config, _exposure_of(store, config))]

    # edits survive coming back to a group
    store.set("Objective", objective, {**settings, "exposure": 3.0})
    mmcore.setChannelGroup("Channel")
    mmcore.setChannelGroup("Objective")
    edited = store.get("Objective", objective)
    assert edited is not None
    assert edited["exposure"] == 3


@contextmanager
def _live(mmcore: CMMCorePlus) -> Iterator[None]:
    mmcore.startContinuousSequenceAcquisition()
    try:
        yield
    finally:
        mmcore.stopSequenceAcquisition()


def test_mda_edit_of_the_marked_row_is_applied_during_live(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    label = _label_of(mmcore, ("Camera", "TestProperty2"))
    mda = _mda(mmcore, qtbot, "DAPI", "FITC", "DAPI")
    channels = mda.channels
    channels.applyRow(0)
    channels.setRowSettings(
        0, {"exposure": 100.0, "light_source": label, "intensity": 1.0}
    )
    channels.valueChanged.emit()

    # while idle, it waits for the next selection, snap or live
    _mda_cell(mda, 0, channels.EXPOSURE).setValue(40)
    assert mmcore.getExposure() != 40

    with _live(mmcore):
        _mda_cell(mda, 0, channels.EXPOSURE).setValue(30)
        assert mmcore.getExposure() == 30
        _mda_cell(mda, 0, channels.INTENSITY).setValue(2)
        assert float(mmcore.getProperty("Camera", "TestProperty2")) == 2

        # no other row reaches the hardware, not even one of the same preset
        _mda_cell(mda, 1, channels.EXPOSURE).setValue(20)
        _mda_cell(mda, 2, channels.EXPOSURE).setValue(10)
        assert mmcore.getExposure() == 30
        assert mmcore.getCurrentConfig("Channel") == "DAPI"


def test_panel_edit_of_the_selected_preset_is_applied_during_live(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    panel = _panel(mmcore, qtbot)
    mda = _mda(mmcore, qtbot, "FITC")
    mmcore.setConfig("Channel", "DAPI")
    mmcore.setExposure(5)

    # while idle, it waits for the next selection, snap or live
    _panel_cell(panel, "DAPI", "exposure").setValue(40)
    assert mmcore.getExposure() == 5

    with _live(mmcore):
        _panel_cell(panel, "DAPI", "exposure").setValue(30)
        assert mmcore.getExposure() == 30

        # another preset's edit does not reach the hardware...
        _panel_cell(panel, "FITC", "exposure").setValue(20)
        assert mmcore.getExposure() == 30
        assert mmcore.getCurrentConfig("Channel") == "DAPI"
        # ...but does reach its MDA row
        assert _mda_row_exposures(mda) == [("FITC", 20)]


def test_panel_edit_of_the_marked_rows_preset_is_applied_during_live(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    panel = _panel(mmcore, qtbot)
    mda = _mda(mmcore, qtbot, "DAPI")
    mda.channels.applyRow(0)
    with _live(mmcore):
        _panel_cell(panel, "DAPI", "exposure").setValue(33)
        assert mmcore.getExposure() == 33
        assert _mda_row_exposures(mda) == [("DAPI", 33)]


def test_channel_control_checkbox(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    mda = _mda(mmcore, qtbot, "DAPI", "FITC")
    channels = mda.channels
    table = channels.table()
    col = table.indexOf(CURRENT_COLUMN)
    assert channels.channel_control.text() == "Channel Control"
    assert channels.channelControlEnabled()
    assert not table.isColumnHidden(col)

    channels.applyRow(0)
    _mda_cell(mda, 0, channels.EXPOSURE).setValue(44)
    stored = _stored(mmcore, "DAPI")
    channel_settings_store(mmcore).set("Channel", "DAPI", {**stored, "exposure": 12.0})
    _mda_cell(mda, 0, channels.EXPOSURE).setValue(44)

    channels.channel_control.setChecked(False)
    assert not channels.channelControlEnabled()
    assert table.isColumnHidden(col)

    # the table no longer has a say: a capture uses the preset's stored settings...
    mmcore.setExposure(5)
    assert mda.apply_active_channel_for_capture()
    assert mmcore.getExposure() == 12
    # ...and an edit during live does not reach the hardware
    with _live(mmcore):
        _mda_cell(mda, 0, channels.EXPOSURE).setValue(30)
        assert mmcore.getExposure() == 12

    # still marked underneath, so turning it back on shows the right row
    mmcore.setConfig("Channel", "FITC")
    channels.setChannelControlEnabled(True)
    assert channels.channel_control.isChecked()
    assert not table.isColumnHidden(col)
    assert channels.activeRow() == 1


def test_mda_table_saves_channel_settings_like_the_panel(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    panel = _panel(mmcore, qtbot)
    mda = _mda(mmcore, qtbot, "DAPI")
    channels = mda.channels
    # an icon with a tooltip, right after the Light Source checkbox...
    panel_pair = panel._btn_row.itemAt(0).layout()  # type: ignore[union-attr]
    # ...which comes last in the MDA's channel table
    row = channels._btn_row
    assert row.indexOf(channels.channel_control) == 0
    assert row.indexOf(channels.advanced) == 1
    mda_pair = row.itemAt(2).layout()  # type: ignore[union-attr]
    for pair, checkbox, button in (
        (panel_pair, panel.light_sources, panel._save_btn),
        (mda_pair, channels.show_light_source, channels._save_btn),
    ):
        assert pair is not None
        assert pair.indexOf(checkbox) == 0
        assert pair.indexOf(button) == 1
        assert not button.text()
        assert not button.icon().isNull()
        assert "Save to cfg" in button.toolTip()

    label = _label_of(mmcore, ("Camera", "TestProperty1"))
    channels.setRowSettings(
        0, {"exposure": 12.5, "light_source": label, "intensity": 0.05}
    )
    channels.storeRowSettings(0)

    from_mda, from_panel = tmp_path / "mda.cfg", tmp_path / "panel.cfg"
    for button, dest in ((channels._save_btn, from_mda), (panel._save_btn, from_panel)):
        with patch.object(QFileDialog, "getSaveFileName", return_value=(str(dest), "")):
            button.click()

    lines = from_mda.read_text().splitlines()
    assert "#@Exposure,Channel,DAPI,12.5" in lines
    assert "#@LightSource,Channel,DAPI,Camera,TestProperty1,0.05" in lines
    assert any(line.startswith("Device,") for line in lines)
    assert from_mda.read_text() == from_panel.read_text()

    # cancelling the dialog writes nothing
    with (
        patch.object(QFileDialog, "getSaveFileName", return_value=("", "")),
        patch.object(mmcore, "saveSystemConfiguration") as save,
    ):
        channels._save_btn.click()
    save.assert_not_called()


def test_save_buttons_follow_the_light_source_checkbox(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    panel = _panel(mmcore, qtbot)
    mda = _mda(mmcore, qtbot, "DAPI")
    channels = mda.channels

    def enabled() -> tuple[bool, bool]:
        return panel._save_btn.isEnabled(), channels._save_btn.isEnabled()

    assert enabled() == (True, True)
    # from either widget, whose checkbox then changes with its signals blocked
    panel.light_sources.setChecked(False)
    assert enabled() == (False, False)
    channels.show_light_source.setChecked(True)
    assert enabled() == (True, True)
    channels.show_light_source.setChecked(False)
    assert enabled() == (False, False)

    # a panel opened while off starts disabled
    assert not _panel(mmcore, qtbot)._save_btn.isEnabled()

    # a sequence that uses light sources turns them back on
    channels.setLightSourceVisible(True)
    assert enabled() == (True, True)


def test_save_button_looks_like_the_preferences_gear(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    panel = _panel(mmcore, qtbot)
    button, gear = panel._save_btn, PreferencesButton(_page := AcquirePage(mmcore))
    qtbot.addWidget(_page)
    metric = QStyle.PixelMetric.PM_IndicatorWidth

    def color() -> str:
        image = button.icon().pixmap(button.iconSize()).toImage()
        colors = {
            str(image.pixelColor(x, y).name())
            for x in range(image.width())
            for y in range(image.height())
            if image.pixelColor(x, y).alpha() == 255
        }
        assert len(colors) == 1
        return next(iter(colors))

    assert button.property("variant") == gear.property("variant") == "subtle"
    # its icon is as big as the checkbox next to it
    style = panel.light_sources.style()
    assert style is not None
    size = style.pixelMetric(metric)
    assert button.iconSize() == QSize(size, size)
    assert button.height() <= size + 4
    assert not button.text()
    for scheme in (LIGHT_THEME, DARK_THEME):
        set_theme(scheme)
        button.changeEvent(QEvent(QEvent.Type.StyleChange))
        assert color() == qcolor(theme().text_secondary).name()


def test_run_takes_what_is_being_typed(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    """Without Enter: on macOS, clicking Run leaves the editor its focus."""
    mda = _mda(mmcore, qtbot, "DAPI")
    mda.show()
    mda.activateWindow()
    qtbot.waitExposed(mda)
    mda.revealAxis("c")
    qtbot.wait(100)
    if not mda.isActiveWindow():  # pragma: no cover
        pytest.skip("focus needs the window to be active, which CI cannot always do")

    spin = _mda_cell(mda, 0, mda.channels.EXPOSURE)
    spin.setFocus()
    edit = spin.lineEdit()
    qtbot.keyClick(edit, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)  # type: ignore[no-untyped-call]
    qtbot.keyClicks(edit, "77")  # type: ignore[no-untyped-call]
    assert _mda_row_exposures(mda) == [("DAPI", 100)]

    mda.prepare_mda()
    assert _mda_row_exposures(mda) == [("DAPI", 77)]
    # it counts as an edit, so the Channels panel's settings have it too
    assert _stored(mmcore, "DAPI")["exposure"] == 77


def _type_without_enter(qtbot: QtBot, spin: Any, text: str) -> None:
    spin.setFocus()
    edit = spin.lineEdit()
    qtbot.keyClick(edit, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)  # type: ignore[no-untyped-call]
    qtbot.keyClicks(edit, text)  # type: ignore[no-untyped-call]


def _activate(qtbot: QtBot, window: Any) -> None:
    window.show()
    window.activateWindow()
    qtbot.waitExposed(window)
    qtbot.wait(100)
    if not window.isActiveWindow():  # pragma: no cover
        pytest.skip("focus needs the window to be active, which CI cannot always do")


def test_capture_takes_what_is_being_typed_in_the_mda(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    mda = _mda(mmcore, qtbot, "DAPI")
    _activate(qtbot, mda)
    mda.revealAxis("c")
    mda.channels.applyRow(0)

    _type_without_enter(qtbot, _mda_cell(mda, 0, mda.channels.EXPOSURE), "77")
    assert _mda_row_exposures(mda) == [("DAPI", 100)]
    assert mda.apply_active_channel_for_capture()
    assert mmcore.getExposure() == 77


def test_capture_takes_what_is_being_typed_in_the_panel(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    panel = _panel(mmcore, qtbot)
    mda = _mda(mmcore, qtbot, "DAPI")
    _activate(qtbot, panel)
    mmcore.setConfig("Channel", "DAPI")

    # the MDA's row of that preset is the selected one, and gets it from the panel
    _type_without_enter(qtbot, _panel_cell(panel, "DAPI", "exposure"), "66")
    assert mda.apply_active_channel_for_capture()
    assert mmcore.getExposure() == 66
    assert _mda_row_exposures(mda) == [("DAPI", 66)]

    # with no row of it, the stored settings have it
    mmcore.setConfig("Channel", "FITC")
    _type_without_enter(qtbot, _panel_cell(panel, "FITC", "exposure"), "55")
    assert mda.apply_active_channel_for_capture()
    assert mmcore.getExposure() == 55
