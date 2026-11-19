"""Leaving out what the loaded configuration can't run (channels, stages, AF)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
import useq

from pymmcore_gui._qt.QtWidgets import QMessageBox
from pymmcore_gui._sequence_support import drop_unsupported
from pymmcore_gui.widgets._acquire import AcquirePage
from pymmcore_gui.widgets._mda_widget import MemoryMDAWidget

if TYPE_CHECKING:
    from pathlib import Path

    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot

GRID = useq.GridRowsColumns(rows=2, columns=2, fov_width=10, fov_height=10)


def _channels(*pairs: tuple[str, str]) -> tuple[useq.Channel, ...]:
    return tuple(useq.Channel(group=g, config=c, exposure=10) for g, c in pairs)


# ------------------------------- the rules -------------------------------


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
        autofocus_plan=useq.SoftwareAxesBasedAF(method="Brent", axes=("p",)),
    )
    out, dropped = drop_unsupported(seq, mmcore)
    assert out.z_plan is None
    assert out.autofocus_plan is None
    assert dropped == [
        "Z plan (no focus stage)",
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


# ------------------------------- the editor -------------------------------


MISSING = useq.MDASequence(channels=_channels(("Channel", "DAPI"), ("Channels", "Cy5")))


@pytest.fixture
def mda(qtbot: QtBot, mmcore: CMMCorePlus) -> MemoryMDAWidget:
    widget = MemoryMDAWidget(mmcore)
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
