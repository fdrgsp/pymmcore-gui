"""Tests for `pymmcore_gui._mda_export` -- no Qt required.

Exercises `export_acquisition` against real `ome_writers` scratch streams
(the same live view an MDA viewer displays during a "memory" run), verifying
that replaying through a fresh stream reproduces both the pixel data and the
metadata a live, disk-backed acquisition would have written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
import tensorstore as ts
import tifffile
from ome_writers import (
    AcquisitionSettings,
    ScratchFormat,
    create_stream,
    dims_from_standard_axes,
)

from pymmcore_gui._mda_export import AcquisitionRecord, export_acquisition

if TYPE_CHECKING:
    from pymmcore_plus.metadata import SummaryMetaV1


def _summary_meta(**extra: object) -> SummaryMetaV1:
    return cast("SummaryMetaV1", {"format": "summary-dict", "version": "1.0", **extra})


def _zarr_attrs(root: str) -> dict[str, Any]:
    data = json.loads((Path(root) / "zarr.json").read_text())
    return cast("dict[str, Any]", data["attributes"])


def _read_zarr_array(root: str, sub: str = "0") -> np.ndarray:
    store = ts.open(
        {"driver": "zarr3", "kvstore": {"driver": "file", "path": f"{root}/{sub}"}}
    ).result()
    return np.asarray(store)


def test_export_ome_tiff_roundtrip(tmp_path: Path) -> None:
    settings = AcquisitionSettings(
        dimensions=tuple(
            dims_from_standard_axes({"t": 2, "c": ["DAPI", "FITC"], "y": 8, "x": 8})
        ),
        dtype="uint16",
        format=ScratchFormat(),
    )
    with create_stream(settings) as stream:
        frame_meta = []
        for t in range(2):
            for c in range(2):
                stream.append(
                    np.full((8, 8), t * 10 + c, dtype="uint16"),
                    frame_metadata={"delta_t": t * 0.5, "exposure_time": 0.02},
                )
                frame_meta.append({"delta_t": t * 0.5, "exposure_time": 0.02})
        view = stream.view()

        record = AcquisitionRecord(
            settings=settings,
            summary_meta=_summary_meta(),
            view=view,
            frame_meta=frame_meta,
        )

        calls: list[tuple[int, int]] = []

        def _progress(done: int, total: int) -> bool:
            calls.append((done, total))
            return True

        out = export_acquisition(
            record, tmp_path / "out.ome.tiff", "ome-tiff", progress=_progress
        )

    assert out is not None
    arr = tifffile.imread(out)
    assert arr.shape == (2, 2, 8, 8)
    assert arr.dtype == np.uint16
    for t in range(2):
        for c in range(2):
            assert np.all(arr[t, c] == t * 10 + c)

    assert calls == [(n, 4) for n in range(1, 5)]

    with tifffile.TiffFile(out) as tif:
        ome_xml = tif.ome_metadata or ""
        assert "DAPI" in ome_xml
        assert "FITC" in ome_xml
        assert "pymmcore_plus" in ome_xml
        assert "DeltaT" in ome_xml
        assert "ExposureTime" in ome_xml


def test_export_ome_zarr_roundtrip(tmp_path: Path) -> None:
    settings = AcquisitionSettings(
        dimensions=tuple(dims_from_standard_axes({"t": 3, "y": 8, "x": 8})),
        dtype="uint16",
        format=ScratchFormat(),
    )
    with create_stream(settings) as stream:
        for t in range(3):
            stream.append(np.full((8, 8), t, dtype="uint16"))
        view = stream.view()

        record = AcquisitionRecord(
            settings=settings,
            summary_meta=_summary_meta(),
            view=view,
        )
        out = export_acquisition(record, tmp_path / "out.ome.zarr", "ome-zarr")

    assert out is not None
    arr = _read_zarr_array(out)
    assert arr.shape == (3, 8, 8)
    for t in range(3):
        assert np.all(arr[t] == t)

    attrs = _zarr_attrs(out)
    assert "pymmcore_plus" in attrs
    assert attrs["pymmcore_plus"]["summary_metadata"]["format"] == "summary-dict"


def test_export_multiposition_ome_tiff_is_a_directory(tmp_path: Path) -> None:
    settings = AcquisitionSettings(
        dimensions=tuple(
            dims_from_standard_axes({"p": ["A1", "B2"], "c": ["DAPI"], "y": 8, "x": 8})
        ),
        dtype="uint16",
        format=ScratchFormat(),
    )
    with create_stream(settings) as stream:
        for p in range(2):
            stream.append(np.full((8, 8), p, dtype="uint16"))
        view = stream.view()

        record = AcquisitionRecord(
            settings=settings,
            summary_meta=None,
            view=view,
        )
        out = export_acquisition(record, tmp_path / "multi.ome.tiff", "ome-tiff")

    assert out is not None
    out_path = Path(out)
    assert out_path.is_dir()
    files = sorted(f.name for f in out_path.iterdir())
    assert files == ["multi_p000.ome.tiff", "multi_p001.ome.tiff"]
    assert tifffile.imread(out_path / "multi_p000.ome.tiff").shape == (8, 8)


def test_export_clamps_partial_unbounded_acquisition(tmp_path: Path) -> None:
    """A cancelled/unbounded run only exports what was actually written."""
    settings = AcquisitionSettings(
        dimensions=tuple(
            dims_from_standard_axes({"t": None, "c": ["DAPI"], "y": 8, "x": 8})
        ),
        dtype="uint16",
        format=ScratchFormat(),
    )
    with create_stream(settings) as stream:
        for t in range(3):  # only 3 timepoints ever acquired
            stream.append(np.full((8, 8), t, dtype="uint16"))
        view = stream.view()
        assert view.shape == (3, 1, 8, 8)

        record = AcquisitionRecord(settings=settings, summary_meta=None, view=view)
        out = export_acquisition(record, tmp_path / "partial.ome.zarr", "ome-zarr")

    assert out is not None
    arr = _read_zarr_array(out)
    assert arr.shape == (3, 1, 8, 8)


def test_export_nothing_acquired_raises(tmp_path: Path) -> None:
    settings = AcquisitionSettings(
        dimensions=tuple(dims_from_standard_axes({"t": None, "y": 8, "x": 8})),
        dtype="uint16",
        format=ScratchFormat(),
    )
    with create_stream(settings) as stream:
        view = stream.view()  # never appended to
        record = AcquisitionRecord(settings=settings, summary_meta=None, view=view)
        with pytest.raises(ValueError, match="Nothing to export"):
            export_acquisition(record, tmp_path / "empty.ome.tiff", "ome-tiff")


def test_export_cancellation_returns_none_and_stops_early(tmp_path: Path) -> None:
    settings = AcquisitionSettings(
        dimensions=tuple(dims_from_standard_axes({"t": 5, "y": 8, "x": 8})),
        dtype="uint16",
        format=ScratchFormat(),
    )
    with create_stream(settings) as stream:
        for t in range(5):
            stream.append(np.full((8, 8), t, dtype="uint16"))
        view = stream.view()
        record = AcquisitionRecord(settings=settings, summary_meta=None, view=view)

        calls: list[tuple[int, int]] = []

        def _progress(done: int, total: int) -> bool:
            calls.append((done, total))
            return done < 2

        out = export_acquisition(
            record, tmp_path / "cancelled.ome.zarr", "ome-zarr", progress=_progress
        )

    assert out is None
    assert calls == [(1, 5), (2, 5)]


def test_export_overwrite_flag(tmp_path: Path) -> None:
    settings = AcquisitionSettings(
        dimensions=tuple(dims_from_standard_axes({"t": 1, "y": 8, "x": 8})),
        dtype="uint16",
        format=ScratchFormat(),
    )
    path = tmp_path / "existing.ome.zarr"
    with create_stream(settings) as stream:
        stream.append(np.zeros((8, 8), dtype="uint16"))
        view = stream.view()
        record = AcquisitionRecord(settings=settings, summary_meta=None, view=view)

        export_acquisition(record, path, "ome-zarr")  # first write
        with pytest.raises(FileExistsError):
            export_acquisition(record, path, "ome-zarr", overwrite=False)
        # succeeds with overwrite=True
        out = export_acquisition(record, path, "ome-zarr", overwrite=True)

    assert out is not None
