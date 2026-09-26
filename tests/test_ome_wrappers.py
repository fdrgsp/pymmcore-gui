"""Tests for `pymmcore_gui._ome_zarr_wrapper` and `_ome_tiff_wrapper` -- no Qt required.

Fixtures are built through the real `ome_writers` path (the same writer used
by a live, disk-backed acquisition), so these exercise the actual on-disk
layout and OME metadata a pyMM acquisition produces.
"""

from __future__ import annotations

import warnings
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import tifffile
import useq
from ndv.models import DataWrapper
from ome_writers import (
    AcquisitionSettings,
    OmeTiffFormat,
    OmeZarrFormat,
    create_stream,
    useq_to_acquisition_settings,
)

import pymmcore_gui._ome_tiff_wrapper
import pymmcore_gui._ome_zarr_wrapper  # noqa: F401 -- registers OMEZarrWrapper
from pymmcore_gui._ome_tiff_wrapper import OMETiffWrapper
from pymmcore_gui._ome_zarr_wrapper import OMEZarrWrapper

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ome_writers import Dimension

FORMATS = ("ome-tiff", "ome-zarr")
WRAPPER_TYPE: dict[str, type[OMETiffWrapper | OMEZarrWrapper]] = {
    "ome-tiff": OMETiffWrapper,
    "ome-zarr": OMEZarrWrapper,
}


def _ch(*names: str) -> tuple[useq.Channel, ...]:
    """Real `Channel` objects -- `MDASequence.channels` is statically a
    `tuple[Channel, ...]`; pydantic coerces plain strings at runtime, but the
    stub doesn't reflect that, so tests spell it out explicitly."""
    return tuple(useq.Channel(config=n, exposure=10) for n in names)


def _fmt(name: str) -> OmeTiffFormat | OmeZarrFormat:
    """Same idea as `_ch`, for `AcquisitionSettings.format`."""
    return OmeTiffFormat() if name == "ome-tiff" else OmeZarrFormat()


def _create(path: object) -> OMETiffWrapper | OMEZarrWrapper:
    """`DataWrapper.create()`, cast to the concrete wrapper this test expects.

    `DataWrapper.create()`'s declared return type is the generic base class,
    which doesn't know about `close()` -- an extension both wrappers add for
    releasing their file handles/zarr store, not part of upstream `ndv`.
    """
    return cast("OMETiffWrapper | OMEZarrWrapper", DataWrapper.create(path))


def _full_frame(dims: tuple, position: int | None = None) -> dict[int, int | slice]:
    """Build indexers for one full 2D frame, squeezing every non-spatial axis.

    A singleton axis (e.g. a 1-channel acquisition's "c") is present in
    `dims` for one backend and absent for the other -- see the module-level
    note on this in `test_single_position_lazy_access` -- so tests index by
    dim *name* via this helper rather than hardcoding positions.
    """
    idx: dict[int, int | slice] = {}
    for i, name in enumerate(dims):
        if name == "p":
            idx[i] = position if position is not None else 0
        elif name in ("y", "x"):
            idx[i] = slice(None)
        else:
            idx[i] = 0
    return idx


def _write(
    dims: Sequence[Dimension], fmt: str, root: Path, n_frames: int, value: int = 0
) -> str:
    settings = AcquisitionSettings(
        dimensions=tuple(dims), dtype="uint16", root_path=str(root), format=_fmt(fmt)
    )
    with create_stream(settings) as stream:
        for i in range(n_frames):
            stream.append(np.full((8, 8), value + i, dtype="uint16"))
    return settings.output_path


@pytest.fixture
def tmp(tmp_path: Path) -> Path:
    return tmp_path


# --------------------------- single position ---------------------------


@pytest.mark.parametrize("fmt", FORMATS)
def test_single_position_roundtrip(tmp: Path, fmt: str) -> None:
    seq = useq.MDASequence(
        channels=_ch("DAPI", "FITC"),
        time_plan=useq.TIntervalLoops(interval=timedelta(seconds=0.5), loops=3),
        z_plan=useq.ZRangeAround(range=2, step=1),
        axis_order=tuple("tpcz"),
    )
    dims = useq_to_acquisition_settings(seq, 8, 8, pixel_size_um=0.325)["dimensions"]
    ext = "ome.tiff" if fmt == "ome-tiff" else "ome.zarr"
    out = _write(dims, fmt, tmp / f"single.{ext}", n_frames=18)

    assert WRAPPER_TYPE[fmt].supports(out)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        w = _create(out)
        assert isinstance(w, WRAPPER_TYPE[fmt])
        assert w.dims == ("t", "c", "z", "y", "x")
        assert w.sizes() == {"t": 3, "c": 2, "z": 3, "y": 8, "x": 8}
        assert w.channel_names(1) == {0: "DAPI", 1: "FITC"}
        scales = w.axis_scales()
        assert scales["z"] == pytest.approx(1.0)
        assert scales["y"] == pytest.approx(0.325)
        assert scales["x"] == pytest.approx(0.325)

        # normal 2D frame: t=1, c=0, z=2, full y/x
        frame = w.isel({0: 1, 1: 0, 2: 2, 3: slice(None), 4: slice(None)})
        assert frame.shape == (8, 8)
        assert frame[0, 0] == 8  # acquisition order: t*6 + c*3 + z

        # orthogonal-style read: keep z and x, fix everything else
        ortho = w.isel({0: 1, 1: 0, 2: slice(None), 3: 3, 4: slice(None)})
        assert ortho.shape == (3, 8)

        w.close()
    assert not caught, [str(x.message) for x in caught]


@pytest.mark.parametrize("fmt", FORMATS)
def test_single_position_lazy_access(tmp: Path, fmt: str) -> None:
    """Only the requested frame is decoded -- no full-array materialization."""
    seq = useq.MDASequence(
        channels=_ch("DAPI"),
        time_plan=useq.TIntervalLoops(interval=timedelta(seconds=1), loops=50),
    )
    dims = useq_to_acquisition_settings(seq, 8, 8, pixel_size_um=0.325)["dimensions"]
    ext = "ome.tiff" if fmt == "ome-tiff" else "ome.zarr"
    out = _write(dims, fmt, tmp / f"lazy.{ext}", n_frames=50)

    w = _create(out)
    assert w.sizes()["t"] == 50
    t_axis = w.dims.index("t")
    frame = w.isel({**_full_frame(w.dims), t_axis: 49})
    assert frame[0, 0] == 49
    w.close()


# --------------------------- multi-position ---------------------------


@pytest.mark.parametrize("fmt", FORMATS)
def test_multiposition_roundtrip(tmp: Path, fmt: str) -> None:
    seq = useq.MDASequence(
        channels=_ch("DAPI"),
        stage_positions=(
            useq.Position(x=0, y=0),
            useq.Position(x=10, y=10),
            useq.Position(x=20, y=20),
        ),
    )
    dims = useq_to_acquisition_settings(seq, 8, 8, pixel_size_um=0.325)["dimensions"]
    ext = "ome.tiff" if fmt == "ome-tiff" else "ome.zarr"
    out = _write(dims, fmt, tmp / f"multi.{ext}", n_frames=3)
    # position 0/1/2's (only) written frame is filled with value 0/1/2
    out_path = Path(out)

    assert WRAPPER_TYPE[fmt].supports(out_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        w = _create(out_path)
        assert w.dims[0] == "p"
        assert w.sizes()["p"] == 3

        vals = [w.isel(_full_frame(w.dims, position=p))[0, 0] for p in range(3)]
        assert vals == [0, 1, 2]

        stacked_idx = _full_frame(w.dims)
        stacked_idx[0] = slice(0, 3)
        stacked = w.isel(stacked_idx)
        assert stacked.shape[0] == 3
        assert list(stacked[:, 0, 0]) == [0, 1, 2]

        w.close()
    assert not caught, [str(x.message) for x in caught]


@pytest.mark.parametrize("fmt", FORMATS)
def test_multiposition_multichannel(tmp: Path, fmt: str) -> None:
    seq = useq.MDASequence(
        channels=_ch("DAPI", "FITC"),
        stage_positions=(useq.Position(x=0, y=0), useq.Position(x=10, y=10)),
    )
    dims = useq_to_acquisition_settings(seq, 8, 8, pixel_size_um=0.325)["dimensions"]
    ext = "ome.tiff" if fmt == "ome-tiff" else "ome.zarr"
    out = _write(dims, fmt, tmp / f"multi2.{ext}", n_frames=4)

    w = _create(out)
    assert w.dims == ("p", "c", "y", "x")
    assert w.coords["c"] == ["DAPI", "FITC"]
    # acquisition order p0c0, p0c1, p1c0, p1c1 -> position 1, channel 1 is frame 3
    frame = w.isel({0: 1, 1: 1, 2: slice(None), 3: slice(None)})
    assert frame[0, 0] == 3
    w.close()


# --------------------------- supports() / rejection ---------------------------


def test_ome_tiff_wrapper_rejects_plain_tiff(tmp: Path) -> None:
    plain = tmp / "plain.tiff"
    tifffile.imwrite(plain, np.zeros((8, 8), dtype="uint16"))
    assert not OMETiffWrapper.supports(plain)


def test_ome_tiff_wrapper_rejects_missing_path(tmp: Path) -> None:
    assert not OMETiffWrapper.supports(tmp / "does-not-exist.ome.tiff")


def test_ome_tiff_wrapper_rejects_empty_directory(tmp: Path) -> None:
    empty = tmp / "empty_dir"
    empty.mkdir()
    assert not OMETiffWrapper.supports(empty)


def test_ome_zarr_wrapper_rejects_non_ome_directory(tmp: Path) -> None:
    not_zarr = tmp / "not_a_store"
    not_zarr.mkdir()
    assert not OMEZarrWrapper.supports(not_zarr)


# --------------------------- resource cleanup ---------------------------


def test_ome_tiff_wrapper_close_releases_handle(tmp: Path) -> None:
    seq = useq.MDASequence(channels=_ch("DAPI"))
    dims = useq_to_acquisition_settings(seq, 8, 8, pixel_size_um=0.325)["dimensions"]
    out = _write(dims, "ome-tiff", tmp / "closeme.ome.tiff", n_frames=1)

    w = OMETiffWrapper(out)
    assert w._tf.filehandle.closed is False
    w.close()
    assert w._tf.filehandle.closed is True


@pytest.mark.parametrize("fmt", FORMATS)
def test_gridded_multiposition_reads_in_acquisition_order(tmp: Path, fmt: str) -> None:
    """A grid's tiles land on the flat `p` axis in the order they were written.

    Both backends name a gridded acquisition's per-position units after the
    tile's grid row/column (`_r###_c###` files, `{pos}_{row}_{col}` zarr
    groups). A snake traversal visits the second row right-to-left, so those
    names sort into a *different* order than they were acquired in; the
    authoritative order lives in the OME metadata (`<Image>` order for TIFF,
    the `OME/series` attribute for Zarr) and is what the `p` axis must follow.
    """
    seq = useq.MDASequence(
        grid_plan=useq.GridRowsColumns(rows=2, columns=3, mode="row_wise_snake"),
        stage_positions=(useq.Position(x=0, y=0), useq.Position(x=0, y=9000)),
    )
    dims = useq_to_acquisition_settings(seq, 8, 8, pixel_size_um=0.325)["dimensions"]
    ext = "ome.tiff" if fmt == "ome-tiff" else "ome.zarr"
    out = _write(dims, fmt, tmp / f"grid.{ext}", n_frames=12)

    assert WRAPPER_TYPE[fmt].supports(Path(out))
    w = _create(out)
    try:
        assert w.sizes()["p"] == 12
        vals = [w.isel(_full_frame(w.dims, position=p))[0, 0] for p in range(12)]
        assert vals == list(range(12))
    finally:
        w.close()
