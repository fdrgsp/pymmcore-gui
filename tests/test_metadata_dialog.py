"""Tests for the image metadata dialog (`widgets/_metadata_dialog.py`)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
import useq
from ndv.models import ChannelMode
from ome_writers import (
    AcquisitionSettings,
    OmeTiffFormat,
    create_stream,
    useq_to_acquisition_settings,
)
from pymmcore_plus.metadata import summary_metadata

from pymmcore_gui._image_metadata import DataOrigin
from pymmcore_gui._qt.QtCore import QEvent, Qt
from pymmcore_gui._qt.QtWidgets import QApplication, QPushButton, QTreeWidgetItem
from pymmcore_gui._settings import Settings
from pymmcore_gui.widgets._acquire import AcquirePage
from pymmcore_gui.widgets._metadata_dialog import (
    MetadataDialog,
    _AcquisitionPage,
    _FramePage,
    _Page,
    _RawPage,
    _SequencePage,
    _TimestampsPage,
    _tree_root,
)

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot

    from pymmcore_gui._array_viewer import MMArrayViewer

SEQUENCE = useq.MDASequence(
    channels=(
        useq.Channel(config="DAPI", exposure=10),
        useq.Channel(config="FITC", exposure=10),
    ),
    time_plan=useq.TIntervalLoops(interval=timedelta(seconds=1), loops=2),
    z_plan=useq.ZRangeAround(range=2, step=1),
)


def _write(root: Path, *, metadata: bool = True) -> Path:
    """Frame `i` (t, c, z order) has `delta_t == i` seconds and stage x `i`."""
    dims = useq_to_acquisition_settings(SEQUENCE, 8, 8, pixel_size_um=0.5)["dimensions"]
    settings = AcquisitionSettings(
        dimensions=tuple(dims),
        dtype="uint16",
        root_path=str(root),
        format=OmeTiffFormat(),
    )
    summary = {
        "format": "summary-dict",
        "version": "1.0",
        "datetime": "2026-11-20 14:03:12",
        "mda_sequence": SEQUENCE.model_dump(mode="json", exclude_unset=True),
    }
    with create_stream(settings) as stream:
        if metadata:
            stream.set_global_metadata("pymmcore_plus", {"summary_metadata": summary})
        for i in range(12):
            meta = {"delta_t": float(i), "exposure_time": 0.01, "position_x": float(i)}
            stream.append(
                np.full((8, 8), i, dtype="uint16"),
                frame_metadata=meta if metadata else None,
            )
    return Path(settings.output_path)


@pytest.fixture
def page(mmcore: CMMCorePlus, qtbot: QtBot) -> AcquirePage:
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    return page


def _show_dialog(viewer: MMArrayViewer) -> MetadataDialog:
    viewer.show_metadata()
    dialog = viewer._metadata_dialog
    assert dialog is not None
    return dialog


def _open(page: AcquirePage, path: Path) -> tuple[MMArrayViewer, MetadataDialog]:
    viewer = cast("MMArrayViewer", page.viewers.open_acquisition(path))
    return viewer, _show_dialog(viewer)


def _select(dialog: MetadataDialog, page_type: type[_Page]) -> Any:
    page = next(p for p in dialog._pages if isinstance(p, page_type))
    dialog._tabs.setCurrentWidget(page)
    return page


def _column(page: _TimestampsPage, header: str) -> int:
    model = page._proxy
    return next(
        c
        for c in range(model.columnCount())
        if model.headerData(c, Qt.Orientation.Horizontal) == header
    )


def _selected(page: _TimestampsPage) -> set[tuple[int, ...]]:
    assert page._snapshot is not None
    selection = page._table.selectionModel()
    assert selection is not None
    return {
        tuple(page._snapshot.frames[page._proxy.mapToSource(i).row()].index.values())
        for i in selection.selectedRows()
    }


def _children(item: QTreeWidgetItem) -> list[QTreeWidgetItem]:
    return [c for i in range(item.childCount()) if (c := item.child(i)) is not None]


def test_tabs_without_data_are_disabled(page: AcquirePage, tmp_path: Path) -> None:
    _, dialog = _open(page, _write(tmp_path / "bare.ome.tiff", metadata=False))

    enabled = {
        p.title: dialog._tabs.isTabEnabled(i) for i, p in enumerate(dialog._pages)
    }
    assert enabled == {
        "MDA Sequence": False,
        "Timestamps": False,
        "Frame": False,
        "Acquisition": False,
        "Raw": True,
    }
    assert all(
        dialog._tabs.tabToolTip(i)
        for i in range(dialog._tabs.count())
        if not dialog._tabs.isTabEnabled(i)
    )
    assert isinstance(dialog._tabs.currentWidget(), _RawPage)
    assert "OME-TIFF" in dialog._header.text()


def test_sequence_tab_shows_the_mda_widget_yaml(
    page: AcquirePage, tmp_path: Path
) -> None:
    viewer, dialog = _open(page, _write(tmp_path / "a.ome.tiff"))
    sequence_page = _select(dialog, _SequencePage)

    assert viewer.mda_sequence is not None
    expected = viewer.mda_sequence.yaml(exclude_unset=True, exclude_defaults=True)
    assert sequence_page._text.toPlainText() == expected
    assert sequence_page._reuse.isEnabled()
    # every button looks like the rest of the GUI's
    assert {b.property("variant") for b in dialog.findChildren(QPushButton)} == {
        "primary"
    }


def test_timestamps_table_sorts_numerically_and_hides_empty_columns(
    page: AcquirePage, tmp_path: Path
) -> None:
    _, dialog = _open(page, _write(tmp_path / "a.ome.tiff"))
    timestamps = _select(dialog, _TimestampsPage)
    table, proxy = timestamps._table, timestamps._proxy

    assert proxy.rowCount() == 12
    hidden = {
        str(proxy.headerData(c, Qt.Orientation.Horizontal))
        for c in range(proxy.columnCount())
        if table.isColumnHidden(c)
    }
    assert hidden == {"Y (µm)", "Z (µm)"}
    # t1 starts 6 s after t0 (frame i has delta_t i)
    assert timestamps._summary.text() == (
        "Started 2026-11-20 14:03:12 · Lasted 00:00:11.000 · Interval 6.000 s"
    )

    # as text, "10.000" sorts before "2.000"
    column = _column(timestamps, "Δt (s)")
    table.sortByColumn(column, Qt.SortOrder.AscendingOrder)
    assert [proxy.index(row, column).data() for row in range(3)] == [
        "0.000",
        "1.000",
        "2.000",
    ]


def test_follow_viewer_selects_the_displayed_frames(
    page: AcquirePage, tmp_path: Path, qtbot: QtBot
) -> None:
    viewer, dialog = _open(page, _write(tmp_path / "a.ome.tiff"))
    timestamps = _select(dialog, _TimestampsPage)
    model = viewer.display_model

    def shows(expected: set[tuple[int, ...]]) -> None:
        qtbot.waitUntil(lambda: _selected(timestamps) == expected, timeout=2000)

    # composite shows every channel, whatever current_index holds for "c"
    model.channel_mode = ChannelMode.COMPOSITE
    model.current_index.update({"t": 1, "c": 1, "z": 2})
    shows({(1, 0, 2), (1, 1, 2)})

    model.channel_mode = ChannelMode.GRAYSCALE
    shows({(1, 1, 2)})

    del model.current_index["z"]  # ndv shows plane 0 of an unset axis
    shows({(1, 1, 0)})

    # (ndv's Qt sliders do not support a slice in current_index)
    model.current_index["z"] = 1
    model.visible_axes = ("z", "x")  # an orthogonal view shows every z
    shows({(1, 1, 0), (1, 1, 1), (1, 1, 2)})


def test_double_click_moves_the_viewer(page: AcquirePage, tmp_path: Path) -> None:
    viewer, dialog = _open(page, _write(tmp_path / "a.ome.tiff"))
    timestamps = _select(dialog, _TimestampsPage)
    viewer.display_model.channel_mode = ChannelMode.GRAYSCALE
    assert timestamps._snapshot is not None
    row = next(
        row
        for row, frame in enumerate(timestamps._snapshot.frames)
        if frame.index == {"t": 0, "c": 1, "z": 1}
    )

    timestamps._table.doubleClicked.emit(
        timestamps._proxy.mapFromSource(timestamps._proxy.sourceModel().index(row, 0))
    )

    index = viewer.display_model.current_index
    assert (index["t"], index["c"], index["z"]) == (0, 1, 1)


def test_frame_tab_shows_one_item_per_displayed_channel(
    page: AcquirePage, tmp_path: Path, qtbot: QtBot
) -> None:
    viewer, dialog = _open(page, _write(tmp_path / "a.ome.tiff"))
    frame_page = _select(dialog, _FramePage)
    viewer.display_model.channel_mode = ChannelMode.COMPOSITE
    viewer.display_model.current_index.update({"t": 1, "z": 2})

    root = frame_page._tree.invisibleRootItem()
    assert root is not None
    qtbot.waitUntil(
        lambda: (
            [item.text(0) for item in _children(root)]
            == ["t 1 · c DAPI · z 2", "t 1 · c FITC · z 2"]
        ),
        timeout=2000,
    )
    fields = {item.text(0): item.text(1) for item in _children(_children(root)[0])}
    assert fields["Elapsed"] == "00:00:08.000"
    assert fields["Exposure (ms)"] == "10"


def test_appended_entries_show_up_while_open(
    page: AcquirePage, tmp_path: Path, qtbot: QtBot
) -> None:
    viewer = cast(
        "MMArrayViewer", page.viewers.open_acquisition(_write(tmp_path / "a.ome.tiff"))
    )
    record = viewer._acquisition_record
    assert record is not None
    later = record.frame_meta[6:]
    del record.frame_meta[6:]
    dialog = _show_dialog(viewer)
    timestamps = _select(dialog, _TimestampsPage)
    assert timestamps._proxy.rowCount() == 6

    record.frame_meta.extend(later)  # as a running acquisition appends
    qtbot.waitUntil(lambda: timestamps._proxy.rowCount() == 12, timeout=3000)


def test_raw_documents_load_lazily_and_copy_the_original_text(
    page: AcquirePage, tmp_path: Path
) -> None:
    viewer = cast(
        "MMArrayViewer", page.viewers.open_acquisition(_write(tmp_path / "a.ome.tiff"))
    )
    calls: list[str] = []

    def load() -> str:
        calls.append("a.json")
        return '{"a": [1, 2]}'

    viewer.data_origin = DataOrigin("OME-Zarr", "a.ome.zarr", lambda: {"a.json": load})
    dialog = _show_dialog(viewer)
    assert not calls

    raw = _select(dialog, _RawPage)
    assert calls == ["a.json"]
    assert raw._text.toPlainText() == '{\n  "a": [\n    1,\n    2\n  ]\n}'
    raw._copy()
    clipboard = QApplication.clipboard()
    assert clipboard is not None
    assert clipboard.text() == '{"a": [1, 2]}'


def test_closing_disconnects_from_the_viewer(
    page: AcquirePage, tmp_path: Path, qtbot: QtBot
) -> None:
    path = _write(tmp_path / "a.ome.tiff")
    viewer = cast("MMArrayViewer", page.viewers.open_acquisition(path))
    signal = viewer.display_model.current_index.value_changed
    n_connections = len(signal)

    dialog = _show_dialog(viewer)
    _select(dialog, _TimestampsPage)
    assert len(signal) == n_connections + 1
    qtbot.keyClick(dialog, Qt.Key.Key_Escape)  # type: ignore[no-untyped-call]
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    assert len(signal) == n_connections
    assert viewer._metadata_dialog is None
    assert Settings.instance().modern_window.metadata_dialog_tab == "Timestamps"

    # reopens on the last tab; closing the viewer closes it
    dialog = _show_dialog(viewer)
    assert isinstance(dialog._tabs.currentWidget(), _TimestampsPage)
    viewer.close()
    assert len(signal) == n_connections
    assert not dialog.isVisible()


def test_summary_lists_are_keyed_by_name(
    page: AcquirePage, tmp_path: Path, mmcore: CMMCorePlus
) -> None:
    """Devices, properties and presets read as `name | value`, not by index."""
    viewer, dialog = _open(page, _write(tmp_path / "a.ome.tiff"))
    record = viewer._acquisition_record
    assert record is not None
    record.summary_meta = summary_metadata(mmcore)
    dialog.refresh()
    acquisition = _select(dialog, _AcquisitionPage)

    def child(item: QTreeWidgetItem, key: str) -> QTreeWidgetItem:
        return next(c for c in _children(item) if c.text(0) == key)

    root = _tree_root(acquisition._tree)
    camera = child(child(root, "devices"), "Camera")
    exposure = child(child(camera, "properties"), "Exposure")
    assert exposure.text(1) == mmcore.getProperty("Camera", "Exposure")
    assert child(exposure, "data_type").text(1) == "float"
    binning = child(child(camera, "properties"), "Binning")
    assert child(binning, "allowed_values").text(1) == ", ".join(
        mmcore.getAllowedPropertyValues("Camera", "Binning")
    )

    dapi = child(child(child(root, "config_groups"), "Channel"), "presets")
    setting = child(child(child(dapi, "DAPI"), "settings"), "Dichroic-Label")
    assert setting.text(1) == "400DCLP"


def test_expand_all_resizes_the_key_column_once(
    page: AcquirePage, tmp_path: Path, mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Expand/Collapse All emit per item: resizing for each one is quadratic."""
    viewer, dialog = _open(page, _write(tmp_path / "a.ome.tiff"))
    record = viewer._acquisition_record
    assert record is not None
    record.summary_meta = summary_metadata(mmcore)
    dialog.refresh()
    tree = _select(dialog, _AcquisitionPage)._tree
    widths: list[int] = []
    tree.resizeColumnToContents = widths.append

    tree.expandAll()
    qtbot.waitUntil(lambda: widths == [0])
    tree.collapseAll()
    qtbot.waitUntil(lambda: widths == [0, 0])
