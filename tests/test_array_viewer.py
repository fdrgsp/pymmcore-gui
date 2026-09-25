"""Tests for the save-related helpers in `pymmcore_gui._array_viewer`.

`_save_multiposition`/`_save_as_tiff` (the old hand-rolled OME-TIFF writer)
are gone -- saving now goes through `pymmcore_gui._mda_export.export_acquisition`,
covered end-to-end in `tests/test_mda_export.py`. These tests cover what's
specific to the viewer: building an `AcquisitionRecord` from whatever a
viewer happens to be displaying (`_synthesize_record`, used whenever no live
record was attached at acquisition time), the save-path/format prompt
(`_prompt_save_path`), and the overwrite-confirmation flow
(`_export_with_overwrite_prompt`).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest
from ndv.models._viewer_model import InteractionMode
from ome_writers import (
    AcquisitionSettings,
    ScratchFormat,
    create_stream,
    dims_from_standard_axes,
)

from pymmcore_gui._array_viewer import (
    MMArrayViewer,
    _enable_1based_slider_labels,
    _patch_dim_row_1based,
    _prompt_save_path,
    _synthesize_record,
)
from pymmcore_gui._mda_export import AcquisitionRecord
from pymmcore_gui._qt.QtWidgets import QFileDialog, QMessageBox, QWidget

if TYPE_CHECKING:
    from pathlib import Path

    from ndv.views._qt._array_view import DimRow, _QDimsSliders
    from pytestqt.qtbot import QtBot


class _FakeViewer:
    """Duck-types just enough of MMArrayViewer for the save helpers."""

    def __init__(
        self,
        data: object,
        sizes: dict[object, int],
        dtype: str = "uint16",
        scales: dict[str, float] | None = None,
    ) -> None:
        self.data = data
        self.data_wrapper = SimpleNamespace(
            sizes=lambda: sizes,
            dtype=dtype,
            # Real DataWrapper.axis_scales()/coords derive physical scale and
            # channel/position labels straight from the data; this fake has
            # neither, so scale comes only from the (required) override and
            # every axis reports plain, unlabeled range coords.
            axis_scales=lambda: {},
            coords={name: range(size) for name, size in sizes.items()},
        )
        self.display_model = SimpleNamespace(scales=scales or {})
        self._widget = QWidget()

    def widget(self) -> QWidget:
        return self._widget


def test_clear_roi_removes_model_and_canvas_visual() -> None:
    visual = SimpleNamespace(remove=Mock())
    viewer = SimpleNamespace(roi=object(), _roi_view=visual)

    MMArrayViewer.clear_roi(viewer)  # type: ignore[arg-type]

    assert viewer.roi is None
    assert viewer._roi_view is None
    visual.remove.assert_called_once_with()


def test_existing_roi_editing_uses_pan_zoom_not_creation_mode() -> None:
    viewer = SimpleNamespace(
        _viewer_model=SimpleNamespace(interaction_mode=InteractionMode.CREATE_ROI),
        roi=object(),
        _roi_view=None,
        _create_roi_view=Mock(),
        _synchronize_roi=Mock(),
        set_roi_visual_selected=Mock(),
    )

    MMArrayViewer.set_existing_roi_editing_active(viewer, True)  # type: ignore[arg-type]

    assert viewer._viewer_model.interaction_mode is InteractionMode.PAN_ZOOM
    viewer._create_roi_view.assert_called_once_with()
    viewer._synchronize_roi.assert_called_once_with()
    viewer.set_roi_visual_selected.assert_called_once_with(True)


def test_synthesize_record_from_stream_view(qtbot: QtBot) -> None:
    """A live StreamView (the classic-GUI MDA fallback) keeps its real axis names."""
    settings = AcquisitionSettings(
        dimensions=tuple(
            dims_from_standard_axes({"t": 2, "c": ["DAPI"], "y": 8, "x": 8})
        ),
        dtype="uint16",
        format=ScratchFormat(),
    )
    with create_stream(settings) as stream:
        for _ in range(2):
            stream.append(np.zeros((8, 8), dtype="uint16"))
        view = stream.view()

        viewer = _FakeViewer(
            data=view,
            sizes={k: len(v) for k, v in view.coords.items()},
            dtype="uint16",
            scales={"x": 0.5, "y": 0.5},
        )
        record = _synthesize_record(viewer)

    assert record is not None
    assert record.view is view
    assert record.frame_meta == []
    names = [d.name for d in record.settings.dimensions]
    assert names == ["t", "c", "y", "x"]
    types = {d.name: d.type for d in record.settings.dimensions}
    assert types == {"t": "time", "c": "channel", "y": "space", "x": "space"}
    scales = {d.name: d.scale for d in record.settings.dimensions}
    assert scales["x"] == 0.5
    assert scales["y"] == 0.5


def test_synthesize_record_from_ring_buffer_like_wrapper(qtbot: QtBot) -> None:
    """A Preview-style wrapper with integer axis keys still resolves to y/x."""
    data = np.zeros((1, 8, 8), dtype="uint16")
    viewer = _FakeViewer(
        data=data,
        sizes={0: 1, 1: 8, 2: 8},
        dtype="uint16",
        scales={"x": 0.25, "y": 0.25},
    )
    record = _synthesize_record(viewer)

    assert record is not None
    assert record.view is data
    names = [d.name for d in record.settings.dimensions]
    assert names == ["0", "y", "x"]
    types = {d.name: d.type for d in record.settings.dimensions}
    assert types["0"] == "other"
    assert types["y"] == "space"
    assert types["x"] == "space"


def test_synthesize_record_returns_none_without_data_or_wrapper() -> None:
    viewer = SimpleNamespace(data=None, data_wrapper=None)
    # pyright checks SimpleNamespace structurally against _RecordSource (and
    # correctly finds display_model missing); mypy's SimpleNamespace stub is
    # permissive enough to already accept it, so only pyright needs silencing.
    assert _synthesize_record(viewer) is None  # pyright: ignore[reportArgumentType]


def test_synthesize_record_returns_none_for_single_axis(qtbot: QtBot) -> None:
    viewer = _FakeViewer(data=np.zeros(8), sizes={"x": 8})
    assert _synthesize_record(viewer) is None


@pytest.mark.parametrize(
    ("filter_str", "typed_name", "expected_suffix", "expected_fmt"),
    [
        ("OME-TIFF (*.ome.tiff *.ome.tif)", "acq", ".ome.tiff", "ome-tiff"),
        ("OME-TIFF (*.ome.tiff *.ome.tif)", "acq.ome.tif", ".ome.tif", "ome-tiff"),
        ("OME-Zarr (*.ome.zarr)", "acq", ".ome.zarr", "ome-zarr"),
        ("OME-Zarr (*.ome.zarr)", "acq.ome.zarr", ".ome.zarr", "ome-zarr"),
    ],
)
def test_prompt_save_path(
    monkeypatch: pytest.MonkeyPatch,
    qtbot: QtBot,
    tmp_path: Path,
    filter_str: str,
    typed_name: str,
    expected_suffix: str,
    expected_fmt: str,
) -> None:
    typed_path = str(tmp_path / typed_name)
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        classmethod(lambda *a, **k: (typed_path, filter_str)),
    )
    result = _prompt_save_path(QWidget())
    assert result is not None
    path, fmt = result
    assert fmt == expected_fmt
    assert str(path).endswith(expected_suffix)


def test_prompt_save_path_cancelled(
    monkeypatch: pytest.MonkeyPatch, qtbot: QtBot
) -> None:
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", classmethod(lambda *a, **k: ("", ""))
    )
    assert _prompt_save_path(QWidget()) is None


def _record_with_one_frame(tmp_path: Path) -> AcquisitionRecord:
    settings = AcquisitionSettings(
        dimensions=tuple(dims_from_standard_axes({"t": 1, "y": 8, "x": 8})),
        dtype="uint16",
        format=ScratchFormat(),
    )
    stream = create_stream(settings)
    stream.append(np.zeros((8, 8), dtype="uint16"))
    view = stream.view()
    stream.close()  # scratch backend keeps the arrays alive after close
    return AcquisitionRecord(settings=settings, summary_meta=None, view=view)


def test_export_with_overwrite_prompt_writes_new_file(
    qtbot: QtBot, tmp_path: Path
) -> None:
    record = _record_with_one_frame(tmp_path)
    path = tmp_path / "new.ome.zarr"
    # A stable widget instance, not a fresh one per call: `self.widget()` on a
    # real viewer always returns the same live Qt widget, and QProgressDialog
    # only keeps its parent alive via that same C++ ownership -- a throwaway
    # widget with no surviving Python reference gets GC'd out from under it.
    host = QWidget()
    fake = SimpleNamespace(widget=lambda: host, _on_saved=None)

    MMArrayViewer._export_with_overwrite_prompt(fake, record, path, "ome-zarr")  # type: ignore[arg-type]

    assert (path / "zarr.json").exists()


def test_export_with_overwrite_prompt_asks_before_clobbering(
    monkeypatch: pytest.MonkeyPatch, qtbot: QtBot, tmp_path: Path
) -> None:
    path = tmp_path / "existing.ome.zarr"
    record = _record_with_one_frame(tmp_path)
    host = QWidget()
    fake = SimpleNamespace(widget=lambda: host, _on_saved=None)
    MMArrayViewer._export_with_overwrite_prompt(fake, record, path, "ome-zarr")  # type: ignore[arg-type]
    first_write = json.loads((path / "zarr.json").read_text())

    # Declining the prompt must leave the existing data untouched.
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No
    )
    record2 = _record_with_one_frame(tmp_path)
    MMArrayViewer._export_with_overwrite_prompt(fake, record2, path, "ome-zarr")  # type: ignore[arg-type]
    assert json.loads((path / "zarr.json").read_text()) == first_write

    # Confirming it retries with overwrite=True and succeeds.
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    record3 = _record_with_one_frame(tmp_path)
    MMArrayViewer._export_with_overwrite_prompt(fake, record3, path, "ome-zarr")  # type: ignore[arg-type]
    assert (path / "zarr.json").exists()


# ---------------------------------------------------------------------------
# 1-based slider label tests
# ---------------------------------------------------------------------------


def _make_dims_sliders(n_frames: int = 5) -> tuple[_QDimsSliders, DimRow]:
    """Return a (dims_sliders, dim_row) pair with a freshly created DimRow."""
    from ndv.views._qt._array_view import DimRow, _QDimsSliders

    dims = _QDimsSliders()
    dims.create_sliders({"t": range(n_frames)})
    rows = dims.findChildren(DimRow)
    return dims, rows[0]


def test_patch_dim_row_1based_display(qtbot: QtBot) -> None:
    """Index label text and out-of label become 1-based after patching."""
    _dims, row = _make_dims_sliders(5)
    row.slider.setValue(0)

    _patch_dim_row_1based(row)

    assert row.index_label.text() == "1"
    assert row.out_of.text() == "/ 5"


def test_patch_dim_row_1based_slider_value_shows_1based(qtbot: QtBot) -> None:
    """Moving the slider updates the label to the 1-based position."""
    _dims, row = _make_dims_sliders(5)
    _patch_dim_row_1based(row)

    row.slider.setValue(3)  # 0-based index 3 → should display 4

    assert row.index_label.text() == "4"


def test_patch_dim_row_1based_still_notifies_index_changed(qtbot: QtBot) -> None:
    """Moving the slider must still propagate to `_QDimsSliders.currentIndexChanged`.

    That signal (wired in `_QDimsSliders.create_sliders` via
    `dim_row.slider.valueChanged.connect(self.currentIndexChanged)`) is what
    actually tells the viewer to redraw the displayed frame -- it rides on
    the same `QLabeledSlider._on_slider_value_changed` callback the 1-based
    patch touches, so a naive patch (disconnecting that callback instead of
    only adding to it) makes the label lie: it updates while the displayed
    frame silently stops changing.
    """
    dims, row = _make_dims_sliders(5)
    _patch_dim_row_1based(row)

    seen: list[object] = []
    dims.currentIndexChanged.connect(lambda: seen.append(dims.current_index()))
    row.slider.setValue(2)

    assert seen == [{"t": 2}]


def test_patch_dim_row_1based_label_edit_moves_to_correct_frame(qtbot: QtBot) -> None:
    """Typing a 1-based frame number in the label navigates to the right frame."""
    _dims, row = _make_dims_sliders(5)
    _patch_dim_row_1based(row)

    # Simulate the user typing "3" (1-based frame 3 = 0-based index 2).
    row.index_label.valueEdited.emit(3.0)

    assert row.slider.value() == 2


def test_patch_dim_row_1based_idempotent(qtbot: QtBot) -> None:
    """Calling _patch_dim_row_1based twice doesn't double-offset the display."""
    _dims, row = _make_dims_sliders(5)
    _patch_dim_row_1based(row)
    _patch_dim_row_1based(row)  # second call should be a no-op for signals

    row.slider.setValue(2)
    assert row.index_label.text() == "3"


def test_patch_dim_row_1based_range_grows(qtbot: QtBot) -> None:
    """Out-of label and editable range stay correct when the coord range extends."""
    from ndv.views._qt._array_view import DimRow, _QDimsSliders

    dims = _QDimsSliders()
    fake_widget = SimpleNamespace(dims_sliders=dims)
    _enable_1based_slider_labels(fake_widget)
    dims.create_sliders({"t": range(3)})

    # Extend range to 8 frames (as happens when more MDA frames arrive).
    dims.create_sliders({"t": range(8)})
    rows = dims.findChildren(DimRow)
    assert rows[0].out_of.text() == "/ 8"
    assert rows[0].index_label._max == 8


def test_enable_1based_slider_labels_wraps_create_sliders(qtbot: QtBot) -> None:
    """_enable_1based_slider_labels installs the 1-based patch on new rows too."""
    from ndv.views._qt._array_view import DimRow, _QDimsSliders

    dims = _QDimsSliders()
    fake_widget = SimpleNamespace(dims_sliders=dims)
    _enable_1based_slider_labels(fake_widget)

    dims.create_sliders({"t": range(4)})
    rows = dims.findChildren(DimRow)
    assert len(rows) == 1
    rows[0].slider.setValue(0)
    assert rows[0].index_label.text() == "1"
    assert rows[0].out_of.text() == "/ 4"
