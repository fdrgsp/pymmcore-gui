"""ndv `DataWrapper` for OME-TIFF files and pyMM multi-position directories.

`ome_writers`/pyMM writes one file per *stored* position for a multi-position
acquisition, named `<stem>[_<well>]_p###[_r###_c###].ome.tiff` -- the
`_r###_c###` tile suffix appears when a stage position holds more than one
grid tile. This wrapper opens either that directory or a single OME-TIFF file.

Reads are lazy at the single-plane level: only the page(s) needed to satisfy
a given `isel()` request are ever decoded, so opening a large acquisition
never materializes it into memory. Dimension labels, channel names, and
physical (space/time) coordinates all come straight from tifffile's own OME
metadata parsing (`TiffPageSeries.dims`/`.coords`) rather than a second,
hand-rolled XML reader.

Multi-position note: a pyMM multi-position acquisition writes one physical
file per position, but each file's OME-XML describes the *entire* dataset (a
single `<OME>` graph with one `<Image>` per position, each pointing at its
own file). Opening any *one* of the sibling files is therefore enough:
tifffile exposes every position as `tf.series[p]` and transparently opens
whichever companion file that position's pixels live in -- so this wrapper
only ever opens a single `TiffFile` handle, never one per position.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard

import numpy as np
import tifffile
from ndv.models import DataWrapper

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from tifffile import TiffPageSeries

_MULTIPOS_RE = re.compile(
    r"^(?P<stem>.+?)_p(?P<index>\d+)(?P<tile>_r\d+_c\d+)?\.ome\.tiff?$",
    re.IGNORECASE,
)


def _multiposition_files(directory: Path) -> list[Path]:
    """Return this directory's per-position OME-TIFF files.

    Filename order is *not* storage order once a grid is involved -- a snake
    traversal visits `(row 1, col 2)` before `(row 1, col 0)`, so sorting by
    `r###_c###` would silently relabel tiles. This ordering is only used to
    pick a file to open and to sanity-check the set on disk; the real order
    comes from the OME-XML (see `_ome_file_order`).
    """
    matches = [
        (int(m["index"]), m["tile"] or "", p)
        for p in directory.iterdir()
        if p.is_file() and (m := _MULTIPOS_RE.match(p.name))
    ]
    return [p for *_, p in sorted(matches)]


def _position_label(filename: str) -> str:
    """The `###[_r###_c###]` part of a position file's name, or the whole stem."""
    if m := _MULTIPOS_RE.match(filename):
        return f"{m['index']}{m['tile'] or ''}"
    return filename  # pragma: no cover


def _ome_file_order(xml_str: str | None) -> list[str]:
    """Per-position filenames in the OME graph's own `<Image>` order.

    A multi-file OME-TIFF set repeats the whole `<OME>` graph in every file,
    one `<Image>` per stored position **in acquisition order**, each naming
    its own file via `<TiffData><UUID FileName="...">`. That order is what
    `tifffile` exposes as `tf.series[i]`, so it -- not the directory listing
    -- is what the `p` axis's labels have to follow. Returns `[]` when the
    metadata is missing or doesn't name exactly one file per image.
    """
    if not xml_str:
        return []
    with contextlib.suppress(ET.ParseError):
        root = ET.fromstring(xml_str)
        names: list[str] = []
        for image in root.iter():
            if not image.tag.endswith("Image"):
                continue
            uuids = [
                name
                for el in image.iter()
                if el.tag.endswith("UUID") and (name := el.get("FileName"))
            ]
            if len(uuids) != 1:
                return []
            names.append(uuids[0])
        return names
    return []  # pragma: no cover


def _resolve_axis(indexer: int | slice, size: int) -> tuple[list[int], bool]:
    """Return (concrete index values, keep_axis) for one axis's `isel` request.

    `keep_axis` mirrors plain numpy/`DataWrapper` semantics: an int indexer
    selects and squeezes that axis; a slice keeps it (with however many
    values the slice resolves to).
    """
    if isinstance(indexer, slice):
        start, stop, step = indexer.indices(size)
        return list(range(start, stop, step)), True
    return [indexer], False


def _read_series(
    series: TiffPageSeries, indexers: Mapping[int, int | slice]
) -> np.ndarray:
    """Read the requested slice from `series`, decoding only the pages it needs.

    `indexers` are positional within this series' own dims (Y/X are always
    the last two, matching how OME-TIFF stores one raster plane per page --
    true regardless of which two dims ndv is displaying, e.g. an orthogonal
    Z/X view still reads one full Y*X page per Z).
    """
    leading_shape = series.shape[:-2]
    n_leading = len(leading_shape)
    y_idx = indexers.get(n_leading, slice(None))
    x_idx = indexers.get(n_leading + 1, slice(None))

    axes_info = [
        _resolve_axis(indexers.get(i, slice(None)), size)
        for i, size in enumerate(leading_shape)
    ]
    value_lists = [values for values, _ in axes_info]
    combos = list(itertools.product(*value_lists)) if value_lists else [()]

    planes = [
        _read_plane(series, combo, leading_shape, y_idx, x_idx) for combo in combos
    ]
    shape = tuple(len(values) for values in value_lists)
    arr = np.array(planes).reshape(*shape, *planes[0].shape)

    squeeze_axes = tuple(i for i, (_, keep) in enumerate(axes_info) if not keep)
    return arr.squeeze(axis=squeeze_axes) if squeeze_axes else arr


def _read_plane(
    series: TiffPageSeries,
    combo: tuple[int, ...],
    leading_shape: tuple[int, ...],
    y_idx: int | slice,
    x_idx: int | slice,
) -> np.ndarray:
    flat = int(np.ravel_multi_index(combo, leading_shape)) if combo else 0
    page = series.pages[flat]
    if page is None:
        raise ValueError(f"Missing page {flat} in {series}")
    # A plane belonging to another position's file (multi-position OME-TIFF,
    # where one merged OME-XML graph spans several sibling files) makes
    # tifffile transparently reopen-read-close that companion file on every
    # call, since it never keeps more than the file it was opened from
    # persistently open. That's expected here, not a sign of a stale handle,
    # so the warning it emits about it is suppressed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return np.asarray(page.asarray())[y_idx, x_idx]


def _coords_from_series(
    series: TiffPageSeries, dims: tuple[Hashable, ...]
) -> dict[Hashable, Sequence]:
    """Real (physical/labeled) coordinates from tifffile's OME parsing.

    `series.coords` already resolves channel names and physical space/time
    coordinates from the file's OME-XML; any dim it doesn't cover (e.g. an
    acquisition with no per-frame delta_t) falls back to `range(size)`,
    matching `DataWrapper.sizes()`'s own assumption for an unlabeled axis.
    """
    native_coords = series.coords or {}
    sizes = series.sizes
    coords: dict[Hashable, Sequence] = {}
    for dim, name in zip(series.dims, dims, strict=False):
        if (values := native_coords.get(dim)) is not None:
            coords[name] = list(values)
        else:
            coords[name] = range(sizes[dim])
    return coords


class OMETiffWrapper(DataWrapper):
    """`ndv.DataWrapper` for one OME-TIFF file or a pyMM multi-position directory."""

    PRIORITY = 45

    def __init__(self, data: Any) -> None:
        path = Path(data)
        self._is_multiposition = path.is_dir()

        if self._is_multiposition:
            position_files = _multiposition_files(path)
            if not position_files:
                raise ValueError(f"No per-position `*.ome.tiff` files found in {path}")
            # Any sibling file exposes every position via tf.series[p] -- see
            # the module docstring. Opening the first one is enough.
            self._tf = tifffile.TiffFile(position_files[0])
            if len(self._tf.series) < len(position_files):
                raise ValueError(
                    f"{path} has {len(position_files)} position file(s) but its "
                    f"OME metadata only describes {len(self._tf.series)}"
                )
            first_series = self._tf.series[0]
            inner_dims = tuple(d.lower() for d in first_series.dims)
            self._dims = ("p", *inner_dims)
            # Label each position from the file the OME graph assigns to it,
            # in that graph's order -- which is the same order tifffile
            # exposes as tf.series[i], and the order the data was acquired in.
            ordered = _ome_file_order(self._tf.ome_metadata)
            if len(ordered) == len(self._tf.series):
                labels = [_position_label(name) for name in ordered]
            else:
                labels = [_position_label(p.name) for p in position_files]
            coords: dict[Hashable, Sequence] = {"p": labels}
            coords.update(_coords_from_series(first_series, inner_dims))
        else:
            self._tf = tifffile.TiffFile(path)
            first_series = self._tf.series[0]
            self._dims = tuple(d.lower() for d in first_series.dims)
            coords = _coords_from_series(first_series, self._dims)

        self._dtype = np.dtype(first_series.dtype)
        self._coords = coords
        super().__init__(path)

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[Any]:
        if not isinstance(obj, (str, os.PathLike)):
            return False
        path = Path(obj)
        with contextlib.suppress(Exception):
            if path.is_dir():
                return bool(_multiposition_files(path))
            if path.suffix.lower() in (".tif", ".tiff") and ".ome" in path.name.lower():
                with tifffile.TiffFile(path) as tf:
                    return bool(tf.series) and bool(tf.ome_metadata)
        return False

    @property
    def dims(self) -> tuple[Hashable, ...]:
        return self._dims

    @property
    def coords(self) -> Mapping[Hashable, Sequence]:
        return self._coords

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def isel(self, index: Mapping[int, int | slice]) -> np.ndarray:
        if self._is_multiposition:
            return self._isel_multiposition(index)
        return _read_series(self._tf.series[0], index)

    def close(self) -> None:
        """Close this wrapper's single `TiffFile` handle (all positions, if any)."""
        with contextlib.suppress(Exception):
            self._tf.close()

    def global_metadata(self, namespace: str) -> dict[str, Any] | None:
        """Return this file's OME `MapAnnotation` for `namespace`, if any.

        A generic escape hatch for whatever a writer chose to stash as
        acquisition-level (not per-frame) metadata, decoding each `<M K="...">`
        entry's text as JSON. Callers that know a specific namespace's
        convention (e.g. pymmcore-plus's own "pymmcore_plus") interpret the
        returned dict themselves; this method has no opinion on its contents.
        """
        xml_str = self._tf.ome_metadata
        if not xml_str:
            return None
        root = ET.fromstring(xml_str)
        for annotation in root.iter():
            if (
                annotation.tag.endswith("MapAnnotation")
                and annotation.get("Namespace") == namespace
            ):
                result: dict[str, Any] = {}
                for entry in annotation.iter():
                    if entry.tag.endswith("M") and (key := entry.get("K")):
                        if entry.text:
                            with contextlib.suppress(json.JSONDecodeError):
                                result[key] = json.loads(entry.text)
                return result or None
        return None

    # ----------------------- internals -----------------------

    def _isel_multiposition(self, indexers: Mapping[int, int | slice]) -> np.ndarray:
        pos_idx = indexers.get(0, slice(None))
        inner = {i - 1: v for i, v in indexers.items() if i != 0}
        n_positions = len(self._tf.series)
        if isinstance(pos_idx, slice):
            start, stop, step = pos_idx.indices(n_positions)
            frames = [
                _read_series(self._tf.series[i], inner)
                for i in range(start, stop, step)
            ]
            return np.stack(frames, axis=0) if frames else np.empty((0,))
        return _read_series(self._tf.series[pos_idx], inner)
