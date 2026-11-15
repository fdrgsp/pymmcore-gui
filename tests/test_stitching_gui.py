"""Stitching in the GUI: the Saving row, the dialog, the session, the status bar.

The worker is the fake one from `_stitch_fakes`, so none of this needs
multiview-stitcher: availability is patched where the GUI checks it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import useq

from _stitch_fakes import fake_controller, write_fake_worker
from pymmcore_gui._qt.QtWidgets import QDialog, QWidget
from pymmcore_gui._stitching import _session as session_mod
from pymmcore_gui._stitching._controller import JobState, StitchingController
from pymmcore_gui._stitching._models import (
    Alignment,
    Blending,
    OutputFormat,
    SequenceStitchingV1,
    StitchSettingsV1,
    request_from,
    with_request,
)
from pymmcore_gui._stitching._session import FinishedRun, StitchingSession
from pymmcore_gui.widgets import _mda_widget
from pymmcore_gui.widgets._mda_widget import MemoryMDAWidget
from pymmcore_gui.widgets._stitching_dialog import StitchingSettingsDialog
from pymmcore_gui.widgets._stitching_status import StitchingStatusButton

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot

    from pymmcore_gui._settings import Settings

GRID = useq.GridRowsColumns(
    rows=2, columns=2, overlap=(10, 10), fov_width=256, fov_height=256
)
ON = SequenceStitchingV1(enabled=True)


@dataclass
class Toasts:
    """Stands in for the window's NotificationManager."""

    shown: list[tuple[str, str, tuple[str, ...]]] = field(default_factory=list)
    callbacks: list[Callable[[str | None], None] | None] = field(default_factory=list)

    def _add(
        self, severity: str, message: str, actions: tuple[str, ...], cb: Any
    ) -> None:
        self.shown.append((severity, message, actions))
        self.callbacks.append(cb)

    def show_info_message(
        self, message: str, *actions: str, on_action: Any = None
    ) -> None:
        self._add("info", message, actions, on_action)

    def show_warning_message(
        self, message: str, *actions: str, on_action: Any = None
    ) -> None:
        self._add("warning", message, actions, on_action)


@pytest.fixture
def available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_mda_widget, "stitching_available", lambda: True)


@pytest.fixture
def mda(qtbot: QtBot, mmcore: CMMCorePlus, available: None) -> MemoryMDAWidget:
    widget = MemoryMDAWidget(mmcore)
    qtbot.addWidget(widget)
    return widget


def _row_shown(widget: MemoryMDAWidget) -> bool:
    return widget._stitching_row.isVisibleTo(widget.save_info)


# ----------------------------- the Saving row -----------------------------


def test_row_only_with_a_grid(mda: MemoryMDAWidget) -> None:
    assert not _row_shown(mda)
    mda.setValue(useq.MDASequence(grid_plan=GRID))
    assert _row_shown(mda)
    one_tile = useq.GridRowsColumns(rows=1, columns=1, fov_width=256, fov_height=256)
    mda.setValue(useq.MDASequence(grid_plan=one_tile))
    assert not _row_shown(mda)


def test_row_for_per_position_grids(mda: MemoryMDAWidget) -> None:
    sub = useq.MDASequence(grid_plan=GRID)
    mda.setValue(
        useq.MDASequence(stage_positions=(useq.Position(x=0, y=0, sequence=sub),))
    )
    assert _row_shown(mda)


def test_no_row_without_the_package(
    qtbot: QtBot, mmcore: CMMCorePlus, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_mda_widget, "stitching_available", lambda: False)
    widget = MemoryMDAWidget(mmcore)
    qtbot.addWidget(widget)
    widget.setValue(useq.MDASequence(grid_plan=GRID))
    assert not _row_shown(widget)
    # nothing is added to the sequence either
    assert "pymmcore_gui" not in widget.value().metadata


def test_row_height_follows_visibility(mda: MemoryMDAWidget) -> None:
    hidden = mda.save_info.height()
    mda.setValue(useq.MDASequence(grid_plan=GRID))
    assert mda.save_info.height() > hidden


def test_request_rides_in_the_sequence(mda: MemoryMDAWidget) -> None:
    mda.setValue(useq.MDASequence(grid_plan=GRID))
    assert not request_from(mda.value().metadata).enabled
    mda._stitching_check.setChecked(True)
    assert request_from(mda.value().metadata).enabled
    assert mda.stitchingRequest().enabled

    # a saved sequence brings its request back...
    settings = StitchSettingsV1(alignment=Alignment.STAGE, blending=Blending.MAX)
    saved = useq.MDASequence(
        grid_plan=GRID,
        metadata=with_request({}, SequenceStitchingV1(enabled=True, settings=settings)),
    )
    mda._stitching_check.setChecked(False)
    mda.setValue(saved)
    assert mda._stitching_check.isChecked()
    assert request_from(mda.value().metadata).settings == settings

    # ...and one without it starts unchecked
    mda.setValue(useq.MDASequence(grid_plan=GRID))
    assert not mda._stitching_check.isChecked()


def test_new_sequences_start_from_saved_defaults(
    mda: MemoryMDAWidget, settings: Settings
) -> None:
    settings.stitching = StitchSettingsV1(blending=Blending.AVERAGE)
    mda.setValue(useq.MDASequence(grid_plan=GRID))
    assert mda.stitchingRequest().settings.blending == Blending.AVERAGE


def test_no_overlap_is_explained(mda: MemoryMDAWidget) -> None:
    mda.setValue(useq.MDASequence(grid_plan=GRID))
    assert "no overlap" not in mda._stitching_check.toolTip()
    flush = GRID.model_copy(update={"overlap": (0.0, 0.0)})
    mda.setValue(useq.MDASequence(grid_plan=flush))
    assert "no overlap" in mda._stitching_check.toolTip()


def test_settings_dialog_updates_request_and_defaults(
    mda: MemoryMDAWidget, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    chosen = StitchSettingsV1(alignment=Alignment.STAGE, open_when_done=False)

    def accept(dialog: StitchingSettingsDialog) -> int:
        dialog.setValue(chosen)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(StitchingSettingsDialog, "exec", accept)
    assert mda.edit_stitching_settings()
    assert mda.stitchingRequest().settings == chosen
    assert settings.stitching == chosen

    monkeypatch.setattr(
        StitchingSettingsDialog, "exec", lambda _: QDialog.DialogCode.Rejected
    )
    assert not mda.edit_stitching_settings()
    assert mda.stitchingRequest().settings == chosen


def test_request_survives_switching_presentation(
    qtbot: QtBot, mmcore: CMMCorePlus, available: None
) -> None:
    from pymmcore_gui.widgets._acquire import AcquirePage
    from pymmcore_gui.widgets._panels import MdaKind

    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page.mda_widget.setStitchingRequest(ON)
    page._set_mda_kind(MdaKind.TOPBAR)
    assert page.mda_widget.stitchingRequest() == ON


# ----------------------------- the dialog -----------------------------


def test_dialog_round_trip(qtbot: QtBot) -> None:
    settings = StitchSettingsV1(
        alignment=Alignment.REGISTER,
        reg_channel="DAPI",
        reg_binning=4,
        z_max_projection=False,
        register_every_timepoint=True,
        reject_poor_matches=False,
        quality_threshold=0.35,
        blending=Blending.MAX,
        blend_width_um=12.5,
        open_when_done=False,
        output_format=OutputFormat.OME_TIFF,
        worker_threads=3,
        low_priority=False,
    )
    dialog = StitchingSettingsDialog(settings, ["DAPI", "FITC"], max_threads=8)
    qtbot.addWidget(dialog)
    assert dialog.value() == settings
    dialog.setValue(StitchSettingsV1())
    assert dialog.value() == StitchSettingsV1()


def test_dialog_keeps_a_missing_channel(qtbot: QtBot) -> None:
    settings = StitchSettingsV1(reg_channel="Cy5")
    dialog = StitchingSettingsDialog(settings, ["DAPI"])
    qtbot.addWidget(dialog)
    assert "not in this sequence" in dialog.channel_combo.currentText()
    assert dialog.value().reg_channel == "Cy5"


def test_dialog_stage_only_disables_registration(qtbot: QtBot) -> None:
    dialog = StitchingSettingsDialog(StitchSettingsV1(), ["DAPI"])
    qtbot.addWidget(dialog)
    assert dialog.channel_combo.isEnabled()
    dialog.stage_radio.setChecked(True)
    assert not dialog.channel_combo.isEnabled()
    assert not dialog.threshold_spin.isEnabled()
    dialog.defaults_btn.click()
    assert dialog.register_radio.isChecked()


# ----------------------------- the session -----------------------------


@pytest.fixture
def toasts() -> Toasts:
    return Toasts()


@pytest.fixture
def opened() -> list[Path]:
    return []


def _session(
    mmcore: CMMCorePlus,
    toasts: Toasts,
    opened: list[Path],
    controller: StitchingController,
) -> StitchingSession:
    return StitchingSession(
        mmcore,
        toasts,
        opened.append,
        controller=controller,
    )


@pytest.fixture
def make_session(
    mmcore: CMMCorePlus, toasts: Toasts, opened: list[Path], tmp_path: Path
) -> Iterator[Callable[[str], StitchingSession]]:
    script = write_fake_worker(tmp_path)
    sessions: list[StitchingSession] = []

    def make(mode: str) -> StitchingSession:
        s = _session(mmcore, toasts, opened, fake_controller(tmp_path, script, mode))
        sessions.append(s)
        return s

    yield make
    for s in sessions:
        s.shutdown()


def _run(
    source: Path | None,
    reason: str = "completed",
    request: SequenceStitchingV1 = ON,
    **kw: Any,
) -> FinishedRun:
    seq = useq.MDASequence(grid_plan=GRID, metadata=with_request({}, request))
    return FinishedRun(
        sequence=seq,
        finish_reason=reason,
        source=source,
        tile_shape_yx=(256, 256),
        calibration=kw.pop("calibration", (1.0, np.eye(2))),
        **kw,
    )


def test_finished_job_opens_mosaic_and_toasts(
    qtbot: QtBot,
    make_session: Callable[[str], StitchingSession],
    toasts: Toasts,
    opened: list[Path],
    tmp_path: Path,
) -> None:
    session = make_session("ok")
    with qtbot.waitSignal(session.controller.jobFinished, timeout=20_000):
        handle = session._submit(_run(tmp_path / "run.ome.zarr"))
    assert handle is not None
    assert handle.title == "run"
    out = handle.job.groups[0].output
    assert out.name == "run_stitched.ome.zarr"
    assert opened == [out]
    severity, message, actions = toasts.shown[-1]
    assert severity == "info"
    assert out.name in message
    assert actions == (session_mod.SHOW_IN_FOLDER,)


def test_not_opened_when_asked_not_to(
    qtbot: QtBot,
    make_session: Callable[[str], StitchingSession],
    toasts: Toasts,
    opened: list[Path],
    tmp_path: Path,
) -> None:
    session = make_session("ok")
    request = SequenceStitchingV1(
        enabled=True, settings=StitchSettingsV1(open_when_done=False)
    )
    run = _run(tmp_path / "run.ome.zarr", request=request)
    with qtbot.waitSignal(session.controller.jobFinished, timeout=20_000):
        handle = session._submit(run)
    assert handle is not None
    assert opened == []
    _, _, actions = toasts.shown[-1]
    assert session_mod.OPEN in actions
    # the toast's Open button opens it
    callback = toasts.callbacks[-1]
    assert callback is not None
    callback(session_mod.OPEN)
    assert opened == [handle.job.groups[0].output]


def test_failed_job_warns_with_log(
    qtbot: QtBot,
    make_session: Callable[[str], StitchingSession],
    toasts: Toasts,
    tmp_path: Path,
) -> None:
    session = make_session("fail")
    with qtbot.waitSignal(session.controller.jobFailed, timeout=20_000):
        session._submit(_run(tmp_path / "run.ome.zarr"))
    severity, message, actions = toasts.shown[-1]
    assert severity == "warning"
    assert "no overlap at all" in message
    assert actions == (session_mod.SHOW_LOG,)


@pytest.mark.parametrize(
    ("reason", "word"), [("canceled", "cancelled"), ("errored", "failed")]
)
def test_unfinished_run_is_skipped(
    make_session: Callable[[str], StitchingSession],
    toasts: Toasts,
    tmp_path: Path,
    reason: str,
    word: str,
) -> None:
    session = make_session("ok")
    assert session._submit(_run(tmp_path / "run.ome.zarr", reason)) is None
    assert not session.controller.jobs
    assert toasts.shown == [
        ("info", f"run was not stitched: the acquisition {word}.", ())
    ]


def test_unsaved_or_unknown_runs_are_skipped(
    make_session: Callable[[str], StitchingSession],
    toasts: Toasts,
    tmp_path: Path,
) -> None:
    session = make_session("ok")
    assert session._submit(_run(None)) is None
    assert toasts.shown == []  # nothing saved, nothing to say
    assert session._submit(_run(tmp_path / "a.ome.zarr", calibration=None)) is None
    assert "no pixel size" in toasts.shown[-1][1]
    assert session._submit(_run(tmp_path / "b.ome.zarr", stores_events=True)) is None
    assert "tile" in toasts.shown[-1][1]
    assert not session.controller.jobs


def test_real_run_is_handed_over_after_the_store_closes(
    qtbot: QtBot,
    mmcore: CMMCorePlus,
    make_session: Callable[[str], StitchingSession],
    tmp_path: Path,
) -> None:
    """A run on the demo core becomes a job for its saved, closed store."""
    session = make_session("ok")
    submitted: list[Any] = []
    original = session._submit

    def spy(run: FinishedRun) -> Any:
        submitted.append(run)
        return original(run)

    session._runFinished.disconnect(original)
    session._runFinished.connect(spy)

    w, h = mmcore.getImageWidth(), mmcore.getImageHeight()
    p = mmcore.getPixelSizeUm()
    grid = useq.GridRowsColumns(
        rows=2, columns=2, overlap=(10, 10), fov_width=w * p, fov_height=h * p
    )
    seq = useq.MDASequence(grid_plan=grid, metadata=with_request({}, ON))
    out = tmp_path / "data" / "run.ome.zarr"
    out.parent.mkdir()
    with qtbot.waitSignal(session.controller.jobFinished, timeout=30_000):
        mmcore.run_mda(seq, output=str(out))

    run = submitted[0]
    assert run.finish_reason == "completed"
    assert run.source == out
    assert run.tile_shape_yx == (h, w)
    assert run.calibration is not None
    assert run.calibration[0] == p
    handle = session.controller.jobs[0]
    assert handle.state == JobState.FINISHED
    assert len(handle.job.groups[0].tiles) == 4


def test_runs_without_the_request_are_ignored(
    qtbot: QtBot,
    mmcore: CMMCorePlus,
    make_session: Callable[[str], StitchingSession],
    tmp_path: Path,
) -> None:
    session = make_session("ok")
    seq = useq.MDASequence(grid_plan=GRID)
    with qtbot.waitSignal(mmcore.mda.events.sequenceFinished, timeout=30_000):
        mmcore.run_mda(seq, output=str(tmp_path / "run.ome.zarr"))
    qtbot.wait(50)
    assert not session.controller.jobs


# ----------------------------- the status bar -----------------------------


def test_status_button_and_popover(
    qtbot: QtBot,
    make_session: Callable[[str], StitchingSession],
    tmp_path: Path,
) -> None:
    session = make_session("hang")
    controller = session.controller
    bar = QWidget()  # stands in for the status bar
    qtbot.addWidget(bar)
    button = StitchingStatusButton(controller, bar)
    bar.show()
    assert button.isHidden()

    first = session._submit(_run(tmp_path / "a.ome.zarr"))
    second = session._submit(_run(tmp_path / "b.ome.zarr"))
    assert first is not None
    assert second is not None
    qtbot.waitUntil(lambda: first.state == JobState.RUNNING, timeout=10_000)
    qtbot.waitUntil(lambda: first.progress.done == 1, timeout=10_000)
    assert not button.isHidden()
    assert button.text().startswith("Stitching 1/2 · ")
    assert button.text().endswith(" %")

    button.click()
    popup = button.popup
    assert popup.isVisible()
    rows = popup.rows()
    assert list(rows) == [first.id, second.id]
    assert rows[second.id].status.text() == "Waiting"
    assert rows[second.id].cancel_btn.text() == "Remove"
    assert rows[first.id].bar.value() > 0

    rows[second.id].cancel_btn.click()
    assert list(popup.rows()) == [first.id]
    assert button.text().startswith("Stitching · ")

    with qtbot.waitSignal(controller.jobCancelled, timeout=10_000):
        popup.rows()[first.id].cancel_btn.click()
    assert button.isHidden()
    assert not popup.isVisible()


# ----------------------------- the main window -----------------------------


@pytest.fixture
def gui_with_stitching(
    qtbot: QtBot,
    mmcore: CMMCorePlus,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    available: None,
) -> Iterator[Any]:
    import pymmcore_gui._stitching as stitching_pkg
    from pymmcore_gui import MicroManagerGUI

    script = write_fake_worker(tmp_path)
    monkeypatch.setattr(stitching_pkg, "is_available", lambda: True)
    monkeypatch.setattr(
        session_mod,
        "StitchingController",
        lambda parent: fake_controller(tmp_path, script, "hang"),
    )
    gui = MicroManagerGUI(mmcore=mmcore)
    qtbot.addWidget(gui)
    # closing with a job left running asks first: say "quit" unless a test
    # answers itself
    monkeypatch.setattr(gui, "_confirm", lambda *a, **k: True)
    yield gui
    if gui.stitching is not None:
        gui.stitching.shutdown()


def test_window_has_no_stitching_without_the_package(
    qtbot: QtBot, mmcore: CMMCorePlus, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pymmcore_gui._stitching as stitching_pkg
    from pymmcore_gui import MicroManagerGUI

    monkeypatch.setattr(stitching_pkg, "is_available", lambda: False)
    gui = MicroManagerGUI(mmcore=mmcore)
    qtbot.addWidget(gui)
    assert gui.stitching is None
    assert not hasattr(gui, "_stitching_status")


def test_close_asks_while_stitching(
    qtbot: QtBot,
    gui_with_stitching: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gui = gui_with_stitching
    session = gui.stitching
    assert session is not None
    assert gui._stitching_status.parent() is not None
    handle = session._submit(_run(tmp_path / "run.ome.zarr"))
    assert handle is not None
    qtbot.waitUntil(lambda: handle.state == JobState.RUNNING, timeout=10_000)

    asked: list[str] = []

    def keep_working(title: str, text: str, **_: Any) -> bool:
        asked.append(text)
        return False

    monkeypatch.setattr(gui, "_confirm", keep_working)
    assert not gui.close()
    assert "run is" in asked[0]
    assert handle.state == JobState.RUNNING

    monkeypatch.setattr(gui, "_confirm", lambda *a, **k: True)
    assert gui.close()
    assert handle.state == JobState.CANCELLED
    assert not session.controller.is_busy()


# ----------------------- "Stitch…" on reopened data (3a) -----------------------


def _saved_run(
    qtbot: QtBot, mmcore: CMMCorePlus, path: Path, grid: bool = True
) -> Path:
    """Acquire and save a small run; on the runner thread, as the GUI does."""
    w, h = mmcore.getImageWidth(), mmcore.getImageHeight()
    p = mmcore.getPixelSizeUm()
    seq = useq.MDASequence(
        grid_plan=useq.GridRowsColumns(
            rows=1, columns=2, overlap=(10, 10), fov_width=w * p, fov_height=h * p
        )
        if grid
        else None,
        channels=(useq.Channel(config="DAPI", exposure=10),),
    )
    with qtbot.waitSignal(mmcore.mda.events.sequenceFinished, timeout=30_000):
        mmcore.run_mda(seq, output=str(path))
    qtbot.waitUntil(lambda: mmcore.mda.status.phase.value == "idle", timeout=10_000)
    sink = mmcore.mda.get_sink()
    assert sink is not None
    return Path(sink.settings.output_path)  # type: ignore[attr-defined]


def test_reopened_grid_offers_stitch(
    qtbot: QtBot,
    gui_with_stitching: Any,
    mmcore: CMMCorePlus,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the window answers the request with the settings dialog; decline it
    monkeypatch.setattr(
        StitchingSettingsDialog, "exec", lambda _: QDialog.DialogCode.Rejected
    )
    gui = gui_with_stitching
    source = _saved_run(qtbot, mmcore, tmp_path / "run.ome.zarr")
    viewer = gui.acquire.viewers.open_acquisition(source)
    assert viewer._stitch_btn is not None
    assert not viewer._stitch_btn.isHidden()
    with qtbot.waitSignal(gui.acquire.viewers.stitchRequested) as blocker:
        viewer._stitch_btn.click()
    assert blocker.args is not None
    saved = blocker.args[0]
    assert saved.source == source
    assert saved.sequence.grid_plan is not None
    assert saved.summary["image_infos"]

    plain = _saved_run(qtbot, mmcore, tmp_path / "plain.ome.zarr", grid=False)
    assert gui.acquire.viewers.open_acquisition(plain)._stitch_btn is None


def test_stitch_saved_asks_then_queues(
    qtbot: QtBot,
    gui_with_stitching: Any,
    mmcore: CMMCorePlus,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pymmcore_gui._acquisition_loader import open_acquisition
    from pymmcore_gui._ndv_viewers import SavedAcquisition

    gui = gui_with_stitching
    mmcore.setProperty(mmcore.getCameraDevice(), "TransposeMirrorY", "1")
    source = _saved_run(qtbot, mmcore, tmp_path / "run.ome.tiff")
    loaded = open_acquisition(source)
    assert loaded.sequence is not None
    assert loaded.record is not None
    saved = SavedAcquisition(
        source, loaded.sequence, loaded.record.summary_meta, loaded.title
    )
    loaded.close()

    titles: list[str] = []

    def decline(dialog: StitchingSettingsDialog) -> int:
        titles.append(dialog.windowTitle())
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(StitchingSettingsDialog, "exec", decline)
    assert gui._stitch_saved(saved) is None
    assert titles == [f"Stitch {loaded.title}"]
    assert not gui.stitching.controller.jobs

    chosen = StitchSettingsV1(alignment=Alignment.STAGE)

    def accept(dialog: StitchingSettingsDialog) -> int:
        dialog.setValue(chosen)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(StitchingSettingsDialog, "exec", accept)
    handle = gui._stitch_saved(saved)
    assert handle is not None
    job = handle.job
    assert job.source == source
    assert job.settings == chosen
    assert settings.stitching == chosen
    p = mmcore.getPixelSizeUm()
    np.testing.assert_allclose(np.array(job.linear_part), np.diag([p, -p]))
    assert job.channel_names == ["DAPI"]


def test_submit_saved_needs_a_pixel_size(
    make_session: Callable[[str], StitchingSession], toasts: Toasts, tmp_path: Path
) -> None:
    session = make_session("ok")
    seq = useq.MDASequence(grid_plan=GRID)
    assert (
        session.submit_saved(tmp_path / "x.ome.zarr", seq, {}, StitchSettingsV1())
        is None
    )
    assert "no pixel size" in toasts.shown[-1][1]
    assert not session.controller.jobs


def test_stitch_button_matches_the_other_viewer_buttons(
    qtbot: QtBot,
    gui_with_stitching: Any,
    mmcore: CMMCorePlus,
    tmp_path: Path,
) -> None:
    source = _saved_run(qtbot, mmcore, tmp_path / "run.ome.zarr")
    viewer = gui_with_stitching.acquire.viewers.open_acquisition(source)
    assert viewer._stitch_btn is not None
    assert viewer._save_btn is not None
    assert viewer._stitch_btn.property("variant") == "subtle"
    assert viewer._stitch_btn.property("variant") == viewer._save_btn.property(
        "variant"
    )


# ----------------------------- Re-stitch (3b) -----------------------------


def _fake_mosaic(
    qtbot: QtBot, mmcore: CMMCorePlus, tmp_path: Path, **prov: Any
) -> Path:
    """A saved run carrying a mosaic's provenance, as if stitched from `raw`."""
    import zarr

    raw = tmp_path / "raw.ome.zarr"
    raw.mkdir()
    mosaic = _saved_run(qtbot, mmcore, tmp_path / "raw_stitched.ome.zarr", grid=False)
    record = {
        "schema_version": 1,
        "source": str(raw),
        "source_fingerprint": {"kind": "directory", "size": 1},
        "group": "",
        "tiles": [
            {"position_index": i, "stage_x_um": 100.0 * i, "stage_y_um": 0.0}
            for i in range(2)
        ],
        "settings": StitchSettingsV1(blending=Blending.MAX).model_dump(mode="json"),
        "pixel_size_um": 1.0,
        "linear_part": [[1.0, 0.0], [0.0, 1.0]],
        "z_step_um": None,
        "channel_names": ["DAPI"],
        "transforms": {"all": [np.eye(3).tolist(), np.eye(3).tolist()]},
        "registration": {"n_pairs": 1, "rejected_pairs": 0},
        "groups": [[0, 1]],
        **prov,
    }
    root = zarr.open_group(str(mosaic), mode="r+")
    root.attrs["pymmcore_gui"] = {"stitching": record}
    return mosaic


def test_reopened_mosaic_offers_restitch(
    qtbot: QtBot,
    gui_with_stitching: Any,
    mmcore: CMMCorePlus,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        StitchingSettingsDialog, "exec", lambda _: QDialog.DialogCode.Rejected
    )
    gui = gui_with_stitching
    mosaic = _fake_mosaic(qtbot, mmcore, tmp_path)
    viewer = gui.acquire.viewers.open_acquisition(mosaic)
    assert viewer._stitch_btn is not None
    assert "again" in viewer._stitch_btn.toolTip()
    assert viewer._stitch_menu_text == "Re-stitch…"
    with qtbot.waitSignal(gui.acquire.viewers.restitchRequested) as blocker:
        viewer._stitch_btn.click()
    assert blocker.args is not None
    assert blocker.args[0].path == mosaic


def _restitch_request(mosaic: Path) -> Any:
    from pymmcore_gui._acquisition_loader import open_acquisition
    from pymmcore_gui._ndv_viewers import StitchedMosaic

    loaded = open_acquisition(mosaic)
    try:
        assert loaded.stitching is not None
        return StitchedMosaic(mosaic, dict(loaded.stitching), loaded.title)
    finally:
        loaded.close()


def _accept_with(
    monkeypatch: pytest.MonkeyPatch, change: dict[str, Any] | None = None
) -> list[StitchingSettingsDialog]:
    shown: list[StitchingSettingsDialog] = []

    def accept(dialog: StitchingSettingsDialog) -> int:
        shown.append(dialog)
        if change:
            dialog.setValue(dialog.value().model_copy(update=change))
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(StitchingSettingsDialog, "exec", accept)
    return shown


def test_restitch_starts_from_the_mosaic_settings(
    qtbot: QtBot,
    gui_with_stitching: Any,
    mmcore: CMMCorePlus,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    gui = gui_with_stitching
    mosaic = _fake_mosaic(qtbot, mmcore, tmp_path)
    shown = _accept_with(monkeypatch, {"blending": Blending.AVERAGE})
    handle = gui._restitch(_restitch_request(mosaic))
    assert handle is not None
    assert shown[0].windowTitle() == "Re-stitch raw_stitched.ome.zarr"
    assert shown[0].note_label.isVisibleTo(shown[0])
    job = handle.job
    assert job.source == tmp_path / "raw.ome.zarr"
    assert job.settings.blending == Blending.AVERAGE
    # only blending changed: the earlier registration rides along
    group = job.groups[0]
    assert group.saved_registration is not None
    assert group.output.parent == mosaic.parent
    assert group.output.name == "raw_stitched-2.ome.zarr"
    # re-stitching doesn't change the defaults of new sequences
    assert settings.stitching == StitchSettingsV1()


def test_restitch_registers_again_after_an_alignment_change(
    qtbot: QtBot,
    gui_with_stitching: Any,
    mmcore: CMMCorePlus,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mosaic = _fake_mosaic(qtbot, mmcore, tmp_path)
    _accept_with(monkeypatch, {"reg_binning": 2})
    handle = gui_with_stitching._restitch(_restitch_request(mosaic))
    assert handle is not None
    assert handle.job.groups[0].saved_registration is None


def test_restitch_asks_for_a_moved_acquisition(
    qtbot: QtBot,
    gui_with_stitching: Any,
    mmcore: CMMCorePlus,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gui = gui_with_stitching
    mosaic = _fake_mosaic(qtbot, mmcore, tmp_path)
    request = _restitch_request(mosaic)
    moved = tmp_path / "moved" / "raw.ome.zarr"
    moved.parent.mkdir()
    (tmp_path / "raw.ome.zarr").rename(moved)
    _accept_with(monkeypatch)

    asked: list[Path] = []

    def give_up(missing: Path) -> None:
        asked.append(missing)

    monkeypatch.setattr(gui, "_locate_source", give_up)
    assert gui._restitch(request) is None  # the user gave up
    assert asked == [tmp_path / "raw.ome.zarr"]

    monkeypatch.setattr(gui, "_locate_source", lambda p: moved)
    handle = gui._restitch(request)
    assert handle is not None
    assert handle.job.source == moved


def test_mosaic_from_a_newer_version_is_refused(
    qtbot: QtBot,
    gui_with_stitching: Any,
    mmcore: CMMCorePlus,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gui = gui_with_stitching
    mosaic = _fake_mosaic(qtbot, mmcore, tmp_path, schema_version=2)
    shown = _accept_with(monkeypatch)
    assert gui._restitch(_restitch_request(mosaic)) is None
    assert not shown
    last = gui.nm.notifications()[-1]
    assert last.severity == "warning"
    assert "newer version" in last.message


# --------------------- "Stitch…" on a finished run's viewer ---------------------


def _grid_seq(mmcore: CMMCorePlus, grid: bool = True) -> useq.MDASequence:
    w, h = mmcore.getImageWidth(), mmcore.getImageHeight()
    p = mmcore.getPixelSizeUm()
    return useq.MDASequence(
        grid_plan=useq.GridRowsColumns(
            rows=1, columns=2, overlap=(10, 10), fov_width=w * p, fov_height=h * p
        )
        if grid
        else None
    )


def _run_in_gui(
    qtbot: QtBot, gui: Any, mmcore: CMMCorePlus, seq: useq.MDASequence, output: Any
) -> Any:
    viewers = gui.acquire.viewers
    with qtbot.waitSignal(mmcore.mda.events.sequenceFinished, timeout=30_000):
        mmcore.run_mda(seq, output=output)
    qtbot.waitUntil(lambda: mmcore.mda.status.phase.value == "idle", timeout=10_000)
    qtbot.wait(50)  # the queued GUI-thread relay of sequenceFinished
    assert viewers.active_viewer is not None
    return viewers.active_viewer


def test_finished_grid_run_offers_stitch(
    qtbot: QtBot,
    gui_with_stitching: Any,
    mmcore: CMMCorePlus,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        StitchingSettingsDialog, "exec", lambda _: QDialog.DialogCode.Rejected
    )
    gui = gui_with_stitching
    seq = _grid_seq(mmcore)
    viewer = _run_in_gui(qtbot, gui, mmcore, seq, str(tmp_path / "run.ome.zarr"))
    assert viewer._stitch_btn is not None
    assert not viewer._stitch_btn.isHidden()
    with qtbot.waitSignal(gui.acquire.viewers.stitchRequested) as blocker:
        viewer._stitch_btn.click()
    assert blocker.args is not None
    saved = blocker.args[0]
    assert saved.source == tmp_path / "run.ome.zarr"
    assert saved.summary["image_infos"]


@pytest.mark.parametrize("case", ["memory", "no-grid"])
def test_no_stitch_for_unsaved_or_gridless_runs(
    qtbot: QtBot,
    gui_with_stitching: Any,
    mmcore: CMMCorePlus,
    tmp_path: Path,
    case: str,
) -> None:
    seq = _grid_seq(mmcore, grid=case != "no-grid")
    output = "memory" if case == "memory" else str(tmp_path / "run.ome.zarr")
    viewer = _run_in_gui(qtbot, gui_with_stitching, mmcore, seq, output)
    assert viewer._stitch_btn is None


def test_no_stitch_for_a_cancelled_run(
    qtbot: QtBot, gui_with_stitching: Any, mmcore: CMMCorePlus, tmp_path: Path
) -> None:
    seq = _grid_seq(mmcore).model_copy(
        update={
            "time_plan": useq.TIntervalLoops(interval=timedelta(seconds=0.5), loops=20)
        }
    )

    def cancel_early(*_: object) -> None:
        mmcore.mda.cancel()

    mmcore.mda.events.frameReady.connect(cancel_early)
    try:
        viewer = _run_in_gui(
            qtbot, gui_with_stitching, mmcore, seq, str(tmp_path / "run.ome.zarr")
        )
    finally:
        mmcore.mda.events.frameReady.disconnect(cancel_early)
    assert str(mmcore.mda.status.finish_reason) == "canceled"
    assert viewer._stitch_btn is None
