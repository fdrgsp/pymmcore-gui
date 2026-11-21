"""Sequence compatibility with the loaded hardware configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
import useq
from pymmcore_widgets.mda._channel_properties import CHANNEL_PROPERTIES_KEY
from pymmcore_widgets.useq_widgets import PYMMCW_METADATA_KEY

from pymmcore_gui._qt.QtWidgets import QMessageBox
from pymmcore_gui._sequence_support import drop_unsupported
from pymmcore_gui.widgets._acquire import AcquirePage
from pymmcore_gui.widgets._mda_widget import MemoryMDAWidget, TopbarMemoryMDAWidget

if TYPE_CHECKING:
    from pathlib import Path

    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot

GRID = useq.GridRowsColumns(rows=2, columns=2, fov_width=10, fov_height=10)


def _channels(*pairs: tuple[str, str]) -> tuple[useq.Channel, ...]:
    return tuple(useq.Channel(group=g, config=c, exposure=10) for g, c in pairs)


def test_a_runnable_sequence_is_returned_as_is(mmcore: CMMCorePlus) -> None:
    seq = useq.MDASequence(
        channels=_channels(("Channel", "DAPI")),
        stage_positions=(useq.Position(x=0, y=0, z=0),),
        grid_plan=GRID,
        z_plan=useq.ZRangeAround(range=4, step=1),
        autofocus_plan=useq.AxesBasedAF(autofocus_device_name="Z", axes=("p",)),
    )
    assert drop_unsupported(seq, mmcore) == (seq, [])


def test_channels_of_a_missing_group_or_preset_are_dropped(
    mmcore: CMMCorePlus,
) -> None:
    seq = useq.MDASequence(
        channels=_channels(
            ("Channel", "DAPI"), ("Channels", "Cy5"), ("Channel", "Nope")
        )
    )
    out, dropped = drop_unsupported(seq, mmcore)
    assert [c.config for c in out.channels] == ["DAPI"]
    assert dropped == [
        "channel Cy5 (no group 'Channels')",
        "channel Nope (not in 'Channel')",
    ]


def test_channels_are_reset_to_every_z_plane_and_timepoint(
    mmcore: CMMCorePlus,
) -> None:
    """OME data is a regular array: a channel can't skip Z planes or timepoints."""
    seq = useq.MDASequence(
        channels=(
            useq.Channel(config="DAPI", exposure=10, do_stack=False, acquire_every=3),
            useq.Channel(config="FITC", exposure=10),
        ),
        z_plan=useq.ZRangeAround(range=4, step=1),
    )
    out, dropped = drop_unsupported(seq, mmcore)
    assert [(c.do_stack, c.acquire_every) for c in out.channels] == [(True, 1)] * 2
    assert dropped == ["channel DAPI: Do Stack off", "channel DAPI: Acquire Every 3"]
    # without a Z plan, Do Stack does nothing
    no_z = seq.replace(
        z_plan=None,
        channels=(useq.Channel(config="DAPI", exposure=10, do_stack=False),),
    )
    assert drop_unsupported(no_z, mmcore) == (no_z, [])


@pytest.mark.parametrize(
    "fields",
    [
        {"stage_positions": (useq.Position(x=100, y=200),)},
        {"grid_plan": GRID},
        {
            "stage_positions": (
                useq.Position(z=5, sequence=useq.MDASequence(grid_plan=GRID)),
            )
        },
    ],
)
def test_positions_and_grids_need_an_xy_stage(
    mmcore: CMMCorePlus, fields: dict[str, Any]
) -> None:
    mmcore.unloadDevice("XY")
    out, dropped = drop_unsupported(useq.MDASequence(**fields), mmcore)
    assert not out.stage_positions
    assert out.grid_plan is None
    assert dropped == ["stage positions and grid (no XY stage)"]


def test_z_only_positions_dont_need_an_xy_stage(mmcore: CMMCorePlus) -> None:
    mmcore.unloadDevice("XY")
    seq = useq.MDASequence(stage_positions=(useq.Position(z=1), useq.Position(z=2)))
    assert drop_unsupported(seq, mmcore) == (seq, [])


def test_z_plan_and_software_autofocus_need_a_focus_stage(
    mmcore: CMMCorePlus,
) -> None:
    mmcore.unloadDevice("Z")
    seq = useq.MDASequence(
        z_plan=useq.ZRangeAround(range=4, step=1),
        focus_surface=useq.FocusSurface(points=((0, 0, 0), (1, 0, 1), (0, 1, 1))),
        autofocus_plan=useq.SoftwareAxesBasedAF(method="Brent", axes=("p",)),
    )
    out, dropped = drop_unsupported(seq, mmcore)
    assert out.z_plan is None
    assert out.focus_surface is None
    assert out.autofocus_plan is None
    assert dropped == [
        "Z plan (no focus stage)",
        "focus surface (no focus stage)",
        "software autofocus (no focus stage)",
    ]


def test_hardware_autofocus_needs_an_autofocus_device(mmcore: CMMCorePlus) -> None:
    mmcore.setAutoFocusDevice("")
    seq = useq.MDASequence(
        autofocus_plan=useq.AxesBasedAF(autofocus_device_name="Z", axes=("p",))
    )
    out, dropped = drop_unsupported(seq, mmcore)
    assert out.autofocus_plan is None
    assert dropped == ["hardware autofocus (no autofocus device)"]


MISSING = useq.MDASequence(channels=_channels(("Channel", "DAPI"), ("Channels", "Cy5")))


@pytest.fixture(params=[MemoryMDAWidget, TopbarMemoryMDAWidget])
def mda(
    request: pytest.FixtureRequest, qtbot: QtBot, mmcore: CMMCorePlus
) -> MemoryMDAWidget | TopbarMemoryMDAWidget:
    widget: MemoryMDAWidget | TopbarMemoryMDAWidget = request.param(mmcore)
    qtbot.addWidget(widget)
    return widget


def test_loading_a_file_drops_what_cant_run(
    mda: MemoryMDAWidget, qtbot: QtBot, tmp_path: Path
) -> None:
    path = tmp_path / "mda.json"
    path.write_text(MISSING.model_dump_json())
    with qtbot.waitSignal(mda.settingsIgnored) as blocker:
        mda.load(path)
    assert [c.config for c in mda.value().channels] == ["DAPI"]
    assert blocker.args is not None
    assert "channel Cy5 (no group 'Channels')" in blocker.args[0]


def test_running_refuses_what_cant_run(
    mda: MemoryMDAWidget, monkeypatch: pytest.MonkeyPatch, mmcore: CMMCorePlus
) -> None:
    mda.setValue(MISSING)  # restoring the editor's state keeps everything
    shown: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: shown.append(a[2]))
    assert mda.prepare_mda() is False
    assert "channel Cy5 (no group 'Channels')" in shown[0]
    assert not mmcore.mda.is_running()


def test_reusing_an_mda_reports_what_was_left_out(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    with (
        patch.object(
            QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes
        ),
        qtbot.waitSignal(page.mdaSettingsIgnored) as blocker,
    ):
        page._on_reuse_mda_requested(MISSING, "z_005_stitched.ome.tiff")
    assert [c.config for c in page.mda_widget.value().channels] == ["DAPI"]
    assert blocker.args is not None
    assert "Cy5" in blocker.args[0]


def test_per_position_settings_are_checked(mmcore: CMMCorePlus) -> None:
    mmcore.unloadDevice("Z")
    mmcore.setAutoFocusDevice("")
    sub = useq.MDASequence(
        channels=_channels(("Channel", "DAPI"), ("Channel", "Nope")),
        z_plan=useq.ZRangeAround(range=4, step=1),
        autofocus_plan=useq.AxesBasedAF(axes=("p",)),
    )
    seq = useq.MDASequence(
        stage_positions=(useq.Position(x=1, y=2, z=3, sequence=sub),)
    )
    out, dropped = drop_unsupported(seq, mmcore)
    pos = out.stage_positions[0]
    assert (pos.x, pos.y, pos.z) == (1, 2, None)
    assert pos.sequence is not None
    assert [c.config for c in pos.sequence.channels] == ["DAPI"]
    assert pos.sequence.z_plan is None
    assert pos.sequence.autofocus_plan is None
    assert dropped == [
        "position 1: Z coordinate (no focus stage)",
        "position 1: channel Nope (not in 'Channel')",
        "position 1: Z plan (no focus stage)",
        "position 1: hardware autofocus (no autofocus device)",
    ]
    assert seq.stage_positions[0].z == 3
    assert seq.stage_positions[0].sequence is sub
    assert drop_unsupported(out, mmcore) == (out, [])


def test_presets_are_queried_once_per_group(mmcore: CMMCorePlus) -> None:
    seq = useq.MDASequence(
        channels=_channels(("Channel", "DAPI"), ("Channel", "FITC")),
        stage_positions=(
            useq.Position(
                x=1,
                y=2,
                sequence=useq.MDASequence(channels=_channels(("Channel", "DAPI"))),
            ),
        ),
    )
    with patch.object(
        mmcore, "getAvailableConfigs", wraps=mmcore.getAvailableConfigs
    ) as configs:
        out, dropped = drop_unsupported(seq, mmcore)
    assert out is seq
    assert not dropped
    configs.assert_called_once_with("Channel")


def test_dropping_channels_preserves_their_light_source_settings(
    mda: MemoryMDAWidget | TopbarMemoryMDAWidget, mmcore: CMMCorePlus
) -> None:
    configs = ("Nope", "DAPI", "FITC", "DAPI")
    properties = [
        {
            "channel_index": i,
            "config": config,
            "group": "Camera · TestProperty1",
            "device": "Camera",
            "property": "TestProperty1",
            "value": (i + 1) / 100,
        }
        for i, config in enumerate(configs)
    ]
    seq = useq.MDASequence(
        channels=_channels(*(("Channel", config) for config in configs)),
        metadata={
            "other": "preserved",
            PYMMCW_METADATA_KEY: {CHANNEL_PROPERTIES_KEY: properties, "extra": 42},
        },
    )
    out, _ = drop_unsupported(seq, mmcore)
    meta = out.metadata[PYMMCW_METADATA_KEY]
    assert out.metadata["other"] == "preserved"
    assert meta["extra"] == 42
    assert meta[CHANNEL_PROPERTIES_KEY] == [
        {**entry, "channel_index": i} for i, entry in enumerate(properties[1:])
    ]
    assert seq.metadata[PYMMCW_METADATA_KEY][CHANNEL_PROPERTIES_KEY] == properties

    mda.loadSequence(seq)
    assert [c.config for c in mda.value().channels] == ["DAPI", "FITC", "DAPI"]
    assert [
        (entry["channel_index"], entry["value"])
        for entry in mda.channels.channelProperties()
    ] == [(0, 0.02), (1, 0.03), (2, 0.04)]


def test_preset_cache_is_refreshed_on_each_call(mmcore: CMMCorePlus) -> None:
    seq = useq.MDASequence(channels=_channels(("Channel", "DAPI")))
    assert drop_unsupported(seq, mmcore) == (seq, [])
    mmcore.deleteConfig("Channel", "DAPI")
    out, dropped = drop_unsupported(seq, mmcore)
    assert out.channels == ()
    assert dropped == ["channel DAPI (not in 'Channel')"]
