"""The Channels panel and the MDA channel table share per-preset settings."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
import useq
from pymmcore_widgets._light_sources import find_light_sources

from pymmcore_gui._channel_settings_store import channel_settings_store
from pymmcore_gui._qt.QtWidgets import QComboBox, QFileDialog
from pymmcore_gui.widgets._channel_settings import ChannelSettingsPanel
from pymmcore_gui.widgets._mda_widget import MemoryMDAWidget

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pymmcore_widgets._channel_settings import ChannelSettings
    from pytestqt.qtbot import QtBot


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
        "#@CameraProperty,Channel,DAPI,Camera,Gain,2",
        "#@CameraProperty,Channel,DAPI,Camera,ScanMode,3",
    )
    settings = channel_settings_store(mmcore).get("Channel", "DAPI")
    assert settings is not None
    assert settings["exposure"] == 12.5
    # whichever label offers exactly that property
    sources = find_light_sources(mmcore)
    assert sources[settings["light_source"]] == [("Camera", "TestProperty1")]
    assert settings["intensity"] == 0.05
    # typed like the Channels panel's cells: a number, or an allowed value
    assert settings["camera_properties"] == {"Gain": 2, "ScanMode": "3"}
    # FITC has no declaration, and nothing is inferred for it
    fitc = channel_settings_store(mmcore).get("Channel", "FITC")
    assert fitc is not None
    assert fitc["light_source"] == ""
    assert fitc["camera_properties"] == {}


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
        "#@CameraProperty,Channel,Cy5,OtherCamera,Gain,2",  # not this camera
        "#@LightSource,Channel,Rhodamine,Camera,TestProperty1,0.05",  # the good one
    )
    store = channel_settings_store(mmcore)
    for preset in ("DAPI", "FITC", "Cy5"):
        settings = store.get("Channel", preset)
        assert settings is not None
        assert settings["light_source"] == ""
        assert settings["camera_properties"] == {}
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
            "camera_properties": {"Gain": 2, "ScanMode": "3"},
        },
    )

    dest = tmp_path / "with_settings.cfg"
    with patch.object(QFileDialog, "getSaveFileName", return_value=(str(dest), "")):
        panel._save_btn.click()

    lines = dest.read_text().splitlines()
    assert "#@Exposure,Channel,DAPI,12.5" in lines
    assert "#@LightSource,Channel,DAPI,Camera,TestProperty1,0.05" in lines
    assert "#@CameraProperty,Channel,DAPI,Camera,Gain,2" in lines
    assert "#@CameraProperty,Channel,DAPI,Camera,ScanMode,3" in lines
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
            "camera_properties": {},
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
    assert reloaded["camera_properties"] == {"Gain": 2, "ScanMode": "3"}
    # the panel shows them too
    qtbot.waitUntil(lambda: panel.channelSettings("DAPI") == reloaded)


def test_column_checkboxes_are_shared(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    panel = _panel(mmcore, qtbot)
    mda = _mda(mmcore, qtbot, "DAPI")
    channels = mda.channels
    assert panel.lightSourcesVisible() and channels.lightSourceVisible()
    assert not panel.cameraVisible() and not channels.cameraVisible()

    panel.camera.setChecked(True)
    assert channels.camera.isChecked()
    channels.show_light_source.setChecked(False)
    assert not panel.light_sources.isChecked()
    # a panel opened later starts the same way
    assert not _panel(mmcore, qtbot).lightSourcesVisible()
