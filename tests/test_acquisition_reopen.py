"""Tests for reopening acquisitions: drag-and-drop and "Re-use MDA…"."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock, patch

import numpy as np
import pytest
import useq
from ome_writers import (
    AcquisitionSettings,
    OmeTiffFormat,
    OmeZarrFormat,
    create_stream,
    useq_to_acquisition_settings,
)
from pymmcore_widgets.useq_widgets import PYMMCW_METADATA_KEY

import pymmcore_gui._modern_gui._acquire_viewers as acquire_viewers_module
from pymmcore_gui._modern_gui._acquire import AcquirePage
from pymmcore_gui._modern_gui._main_win import MainWindow
from pymmcore_gui._qt.QtCore import QThread
from pymmcore_gui._qt.QtWidgets import QFileDialog, QMenu, QMessageBox

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot

    from pymmcore_gui._array_viewer import MMArrayViewer


def _ch(*names: str) -> tuple[useq.Channel, ...]:
    """Real `Channel` objects -- `MDASequence.channels` is statically a
    `tuple[Channel, ...]`; pydantic coerces plain strings at runtime, but the
    stub doesn't reflect that, so tests spell it out explicitly."""
    return tuple(useq.Channel(config=n, exposure=10) for n in names)


def _fmt(name: str) -> OmeTiffFormat | OmeZarrFormat:
    """Same idea as `_ch`, for `AcquisitionSettings.format`."""
    return OmeTiffFormat() if name == "ome-tiff" else OmeZarrFormat()


class _FakeDropEvent:
    """Duck-types the subset of QDropEvent/QDragMoveEvent our handlers use.

    Constructing a real QDropEvent works but adds nothing here -- these
    tests exercise our own accept/route/reject logic, not Qt's own drag
    machinery, so a minimal stand-in keeps them focused and fast.
    """

    def __init__(self, paths: list[Path]) -> None:
        from pymmcore_gui._qt.QtCore import QUrl
        from pymmcore_gui._qt.QtGui import QDropEvent as _QDropEvent  # noqa: F401
        from pymmcore_gui._qt.QtWidgets import QApplication

        assert QApplication.instance() is not None
        from pymmcore_gui._qt.QtCore import QMimeData

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
        self._mime = mime
        self.accepted = False

    def mimeData(self) -> object:
        return self._mime

    def acceptProposedAction(self) -> None:
        self.accepted = True


def _write_acquisition(seq: useq.MDASequence, root: Path) -> Path:
    """Write `seq`'s full nominal acquisition, embedding it as real metadata."""
    dims = useq_to_acquisition_settings(seq, 8, 8, pixel_size_um=0.325)["dimensions"]
    settings = AcquisitionSettings(
        dimensions=tuple(dims),
        dtype="uint16",
        root_path=str(root),
        format=OmeTiffFormat(),
    )
    summary = {
        "format": "summary-dict",
        "version": "1.0",
        "mda_sequence": seq.model_dump(mode="json", exclude_unset=True),
    }
    n_frames = 1
    for dim in dims[:-2]:
        n_frames *= dim.count or 1
    with create_stream(settings) as stream:
        stream.set_global_metadata("pymmcore_plus", {"summary_metadata": summary})
        for i in range(n_frames):
            stream.append(np.full((8, 8), i, dtype="uint16"))
    return Path(settings.output_path)


def _open(page: AcquirePage, path: Path) -> MMArrayViewer:
    """`open_acquisition`, cast to the concrete MMArrayViewer subclass.

    `AcquireViewersManager.open_acquisition`'s declared return type is the
    generic `ndv.ArrayViewer` -- these tests need pymmcore-gui's own
    extensions (`mda_sequence`, `_show_context_menu`, ...).
    """
    return cast("MMArrayViewer", page.viewers.open_acquisition(path))


# --------------------------- drag-and-drop filtering ---------------------------


def test_dropped_paths_accepts_supported_and_rejects_the_rest(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    good = _write_acquisition(
        useq.MDASequence(channels=_ch("DAPI")), tmp_path / "a.ome.tiff"
    )
    bad_dir = tmp_path / "not_a_dataset"
    bad_dir.mkdir()
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("hello")

    event = _FakeDropEvent([good, bad_dir, bad_file])
    paths = MainWindow._dropped_acquisition_paths(event)  # type: ignore[arg-type]
    assert paths == [good]


def test_dropped_paths_empty_for_no_urls() -> None:
    event = _FakeDropEvent([])
    assert MainWindow._dropped_acquisition_paths(event) == []  # type: ignore[arg-type]


# --------------------------- AcquireViewersManager.open_acquisition ---------


def test_open_acquisition_creates_tab_with_filename_title(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    seq = useq.MDASequence(channels=_ch("DAPI", "FITC"))
    path = _write_acquisition(seq, tmp_path / "myfile.ome.tiff")

    viewer = _open(page, path)

    assert viewer.mda_sequence is not None
    assert viewer.mda_sequence.replace(uid=seq.uid) == seq
    assert viewer.source_title == path.name
    docks = [dw for dw in page.viewers._records if dw.windowTitle() == path.name]
    assert docks, [dw.windowTitle() for dw in page.viewers._records]


def test_open_acquisition_does_not_emit_mda_viewer_created(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    """A reopened acquisition must never enter the live-camera ROI machinery."""
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    path = _write_acquisition(
        useq.MDASequence(channels=_ch("DAPI")), tmp_path / "b.ome.tiff"
    )

    created = Mock()
    closed = Mock()
    page.viewers.mdaViewerCreated.connect(created)
    page.viewers.mdaViewerClosed.connect(closed)

    viewer = _open(page, path)
    created.assert_not_called()

    # find and close its dock
    dw = next(dw for dw, rec in page.viewers._records.items() if rec.viewer is viewer)
    dw.closeDockWidget()
    closed.assert_not_called()


def test_open_acquisition_multiple_datasets_get_separate_tabs(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    path_a = _write_acquisition(
        useq.MDASequence(channels=_ch("DAPI")), tmp_path / "a.ome.tiff"
    )
    path_b = _write_acquisition(
        useq.MDASequence(channels=_ch("FITC")), tmp_path / "b.ome.tiff"
    )

    viewer_a = _open(page, path_a)
    viewer_b = _open(page, path_b)

    assert viewer_a is not viewer_b
    assert len(page.viewers._records) == 2


def test_open_acquisition_no_sequence_metadata_has_no_reuse_callback(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    seq = useq.MDASequence(channels=_ch("DAPI"))
    dims = useq_to_acquisition_settings(seq, 8, 8, pixel_size_um=0.325)["dimensions"]
    settings = AcquisitionSettings(
        dimensions=tuple(dims),
        dtype="uint16",
        root_path=str(tmp_path / "bare.ome.tiff"),
        format=OmeTiffFormat(),
    )
    with create_stream(settings) as stream:
        stream.append(np.zeros((8, 8), dtype="uint16"))

    viewer = _open(page, Path(settings.output_path))
    assert viewer.mda_sequence is None
    assert viewer._show_context_menu(Mock()) is False


def test_open_acquisition_raises_for_unsupported_path(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    unsupported = tmp_path / "not_a_dataset"
    unsupported.mkdir()
    with pytest.raises(ValueError, match="Not a supported acquisition"):
        page.viewers.open_acquisition(unsupported)


def test_open_acquisition_close_releases_file_handle(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    path = _write_acquisition(
        useq.MDASequence(channels=_ch("DAPI")), tmp_path / "c.ome.tiff"
    )

    viewer = _open(page, path)
    dw = next(dw for dw, rec in page.viewers._records.items() if rec.viewer is viewer)
    tf = viewer.data_wrapper._tf
    assert tf.filehandle.closed is False
    dw.closeDockWidget()
    assert tf.filehandle.closed is True


# --------------------------- "Re-use MDA…" context menu ---------------------


def test_context_menu_absent_without_sequence(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    settings = AcquisitionSettings(
        dimensions=tuple(
            useq_to_acquisition_settings(
                useq.MDASequence(channels=_ch("DAPI")), 8, 8, pixel_size_um=0.325
            )["dimensions"]
        ),
        dtype="uint16",
        root_path=str(tmp_path / "bare2.ome.tiff"),
        format=OmeTiffFormat(),
    )
    with create_stream(settings) as stream:
        stream.append(np.zeros((8, 8), dtype="uint16"))
    viewer = _open(page, Path(settings.output_path))

    with patch.object(QMenu, "exec") as exec_:
        shown = viewer._show_context_menu(Mock())
    assert shown is False
    exec_.assert_not_called()


def test_context_menu_present_and_enabled_when_idle(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    path = _write_acquisition(
        useq.MDASequence(channels=_ch("DAPI")), tmp_path / "d.ome.tiff"
    )
    viewer = _open(page, path)

    with patch.object(QMenu, "exec") as exec_:
        shown = viewer._show_context_menu(Mock())
    assert shown is True
    exec_.assert_called_once()


def test_context_menu_disabled_while_mda_running(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    path = _write_acquisition(
        useq.MDASequence(channels=_ch("DAPI")), tmp_path / "e.ome.tiff"
    )
    viewer = _open(page, path)

    captured_menu: list[QMenu] = []
    real_menu_cls = QMenu

    def _spy_exec(self: QMenu, *a: object, **k: object) -> None:
        captured_menu.append(self)

    with (
        patch.object(real_menu_cls, "exec", _spy_exec),
        patch.object(mmcore.mda, "is_running", return_value=True),
    ):
        viewer._show_context_menu(Mock())

    assert captured_menu
    actions = captured_menu[0].actions()
    assert actions and actions[0].text() == "Re-use MDA…"
    assert actions[0].isEnabled() is False


def test_reuse_mda_emits_sequence_from_the_correct_viewer(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    seq_a = useq.MDASequence(channels=_ch("DAPI"))
    seq_b = useq.MDASequence(channels=_ch("FITC", "Cy5"))
    path_a = _write_acquisition(seq_a, tmp_path / "a.ome.tiff")
    path_b = _write_acquisition(seq_b, tmp_path / "b.ome.tiff")
    viewer_a = _open(page, path_a)
    page.viewers.open_acquisition(path_b)

    received: list[tuple[useq.MDASequence, str]] = []
    page.viewers.reuseMDARequested.connect(lambda s, t: received.append((s, t)))

    assert viewer_a._reuse_mda_callback is not None
    # AcquirePage.__init__ already connected this same signal to its own
    # confirmation handler (_on_reuse_mda_requested); patch its dialog so
    # that real, already-wired handler doesn't block on a modal no one can
    # click, while this test observes the signal's own payload.
    with patch.object(
        QMessageBox, "question", return_value=QMessageBox.StandardButton.No
    ):
        viewer_a._reuse_mda_callback()

    assert len(received) == 1
    got_seq, got_title = received[0]
    assert got_seq.replace(uid=seq_a.uid) == seq_a
    assert got_title == path_a.name


# --------------------------- confirmation / setValue -------------------------


def test_reuse_mda_cancelled_leaves_mda_unchanged(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    original = page.mda_widget.value()

    new_seq = useq.MDASequence(
        channels=_ch("FITC", "Cy5"),
        time_plan=useq.TIntervalLoops(interval=timedelta(seconds=1), loops=5),
    )
    with patch.object(
        QMessageBox, "question", return_value=QMessageBox.StandardButton.No
    ):
        page._on_reuse_mda_requested(new_seq, "some_file.ome.tiff")

    assert page.mda_widget.value().channels == original.channels


def test_reuse_mda_confirmed_calls_set_value_and_focuses_mda(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    new_seq = useq.MDASequence(
        channels=_ch("FITC", "Cy5"),
        time_plan=useq.TIntervalLoops(interval=timedelta(seconds=1), loops=5),
    )

    with patch.object(
        QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes
    ):
        page._on_reuse_mda_requested(new_seq, "some_file.ome.tiff")

    assert [c.config for c in page.mda_widget.value().channels] == ["FITC", "Cy5"]


def test_reuse_mda_preserves_save_destination_when_sequence_carries_none(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    """Reusing a sequence with no save-info metadata must not blank saving."""
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page.mda_widget.save_info.setValue(
        {
            "save_dir": str(tmp_path),
            "save_name": "keep_me",
            "format": "ome-tiff",
            "should_save": True,
        }
    )
    before = dict(page.mda_widget.save_info.value())

    new_seq = useq.MDASequence(channels=_ch("FITC"))  # no PYMMCW_METADATA_KEY metadata
    assert PYMMCW_METADATA_KEY not in new_seq.metadata
    with patch.object(
        QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes
    ):
        page._on_reuse_mda_requested(new_seq, "some_file.ome.tiff")

    after = dict(page.mda_widget.save_info.value())
    assert after == before


def test_reuse_mda_with_its_own_save_info_does_not_overwrite_on_next_run(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    """A sequence recovered *with* its own save-info applies that destination.

    This intentionally repoints the editor at the path the reused sequence
    was originally acquired to (consistent with restoring every other
    parameter) rather than silently keeping the previous destination. The
    dataset that was just opened is not at risk of being overwritten by the
    *next* run, though: `get_next_available_path` (upstream in
    `MDAWidget.prepare_mda`) auto-increments a save path that already exists
    on disk -- which this one, having just been read from, necessarily does.
    """
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    original_dir = str(tmp_path / "original_acquisition")
    new_seq = useq.MDASequence(
        channels=_ch("FITC"),
        metadata={
            PYMMCW_METADATA_KEY: {
                "save_dir": original_dir,
                "save_name": "old_acquisition",
                "format": "ome-tiff",
                "should_save": True,
            }
        },
    )
    with patch.object(
        QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes
    ):
        page._on_reuse_mda_requested(new_seq, "old_acquisition.ome.tiff")

    after = page.mda_widget.save_info.value()
    assert after["save_dir"] == original_dir
    assert after["save_name"] == "old_acquisition"
    assert after["should_save"] is True

    requested = Path(original_dir) / "old_acquisition"
    requested.parent.mkdir(parents=True, exist_ok=True)
    requested.touch()  # the dataset just opened already exists on disk
    next_path = page.mda_widget.get_next_available_path(requested)
    assert next_path != requested  # auto-incremented, not overwritten


def test_reuse_mda_disabled_while_mda_locked(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page._mda_locked = True

    new_seq = useq.MDASequence(channels=_ch("FITC"))
    with patch.object(QMessageBox, "question") as question:
        page._on_reuse_mda_requested(new_seq, "some_file.ome.tiff")
    question.assert_not_called()


# --------------------------- Save renames the tab ---------------------------


def test_reopened_viewer_gets_real_acquisition_record(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    """The Save button must re-export real metadata, not the current mic state."""
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    path = _write_acquisition(
        useq.MDASequence(channels=_ch("DAPI", "FITC")), tmp_path / "a.ome.tiff"
    )

    viewer = _open(page, path)

    assert viewer._acquisition_record is not None
    names = [d.name for d in viewer._acquisition_record.settings.dimensions]
    assert "c" in names


def test_save_renames_dock_and_source_title(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    """Saving a viewer's data renames its dock/tab to the new destination's filename."""
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    path = _write_acquisition(
        useq.MDASequence(channels=_ch("DAPI")), tmp_path / "orig.ome.tiff"
    )
    viewer = _open(page, path)
    dw = next(dw for dw, rec in page.viewers._records.items() if rec.viewer is viewer)
    assert dw.windowTitle() == "orig.ome.tiff"
    assert viewer.source_title == "orig.ome.tiff"

    dest = tmp_path / "renamed.ome.zarr"
    with patch.object(
        QFileDialog,
        "getSaveFileName",
        classmethod(lambda *a, **k: (str(dest), "OME-Zarr (*.ome.zarr)")),
    ):
        viewer._save_data()

    assert dw.windowTitle() == "renamed.ome.zarr"
    assert viewer.source_title == "renamed.ome.zarr"


def test_save_cancelled_leaves_dock_title_unchanged(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    path = _write_acquisition(
        useq.MDASequence(channels=_ch("DAPI")), tmp_path / "orig2.ome.tiff"
    )
    viewer = _open(page, path)
    dw = next(dw for dw, rec in page.viewers._records.items() if rec.viewer is viewer)

    with patch.object(
        QFileDialog, "getSaveFileName", classmethod(lambda *a, **k: ("", ""))
    ):
        viewer._save_data()

    assert dw.windowTitle() == "orig2.ome.tiff"
    assert viewer.source_title == "orig2.ome.tiff"


# --------------------------- background opening ---------------------------


def test_open_acquisition_async_creates_tab_off_the_gui_thread(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    seq = useq.MDASequence(channels=_ch("DAPI"))
    path = _write_acquisition(seq, tmp_path / "async.ome.tiff")

    gui_thread = QThread.currentThread()
    open_threads: list[QThread | None] = []
    real_open = acquire_viewers_module._open_acquisition

    def _record_thread(p: object) -> object:
        open_threads.append(QThread.currentThread())
        return real_open(p)

    with patch.object(acquire_viewers_module, "_open_acquisition", _record_thread):
        page.viewers.open_acquisition_async(path)
        qtbot.waitUntil(lambda: bool(page.viewers._records), timeout=5000)

    assert open_threads and open_threads[0] is not gui_thread
    titles = [dw.windowTitle() for dw in page.viewers._records]
    assert path.name in titles


def test_open_acquisition_async_reports_failure_instead_of_raising(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    unsupported = tmp_path / "not_a_dataset"
    unsupported.mkdir()

    with qtbot.waitSignal(page.viewers.acquisitionOpenFailed, timeout=5000) as blocker:
        page.viewers.open_acquisition_async(unsupported)
    assert "Not a supported acquisition" in blocker.args[0]
    assert not page.viewers._records


def test_drag_enter_resolves_paths_once_for_the_whole_drag(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    """Recognizing an acquisition costs real file I/O, so it happens once.

    `dragMoveEvent` fires on every mouse-move; re-probing there would open
    and re-parse each dropped dataset's OME metadata dozens of times a
    second.
    """
    win = MainWindow(mmcore=mmcore)
    qtbot.addWidget(win)
    path = _write_acquisition(
        useq.MDASequence(channels=_ch("DAPI")), tmp_path / "drag.ome.tiff"
    )
    event = _FakeDropEvent([path])

    probe = Mock(return_value=[path])
    with patch.object(MainWindow, "_dropped_acquisition_paths", staticmethod(probe)):
        win.dragEnterEvent(event)  # type: ignore[arg-type]
        assert event.accepted
        for _ in range(10):
            win.dragMoveEvent(event)  # type: ignore[arg-type]
        assert probe.call_count == 1

    assert win._drag_paths == [path]
