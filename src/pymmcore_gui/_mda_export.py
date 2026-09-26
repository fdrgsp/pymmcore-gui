"""Export a live (in-memory or on-disk) MDA acquisition to OME-TIFF/OME-Zarr.

The "save" affordance on an MDA-backed viewer needs to write a *real*,
metadata-complete OME-TIFF or OME-Zarr regardless of whether the acquisition
itself streamed straight to disk or ran with output="memory" (the fallback
`MemoryMDAWidget.prepare_mda` uses whenever the Saving section is unchecked --
see `pymmcore_gui.widgets._mda_widget`). Rather than hand-rolling a second,
metadata-poor writer, this module replays the acquisition's live view through
a brand new `ome_writers` stream, so the export goes through the exact same
writer -- and therefore produces the exact same on-disk metadata -- as a live,
disk-backed acquisition would have.

Frames are streamed one at a time directly from the source view; the full
acquisition is never materialized as a single in-memory array (unlike
`np.asarray(large_stream_view)`, which the `ome_writers.StreamView` docstring
explicitly warns against).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from ome_writers import AcquisitionSettings, Channel, Dimension, Position, create_stream
from pymmcore_plus.mda._sink import _serialize_summary_meta

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from ndv.models import DataWrapper
    from pymmcore_plus.metadata import SummaryMetaV1

logger = logging.getLogger(__name__)

ExportFormat = Literal["ome-tiff", "ome-zarr"]
# (frames_done, frames_total) -> keep_going. Returning False cancels the export
# after the frame just written -- whatever was flushed to disk stays there,
# same as a user-cancelled live MDA leaves a partial file.
ProgressCallback = Callable[[int, int], bool]

_DimType = Literal["space", "time", "channel", "position", "other"]
_TYPE_BY_AXIS_NAME: dict[str, _DimType] = {
    "t": "time",
    "c": "channel",
    "z": "space",
    "p": "position",
}


@dataclass
class AcquisitionRecord:
    """Snapshot of everything needed to re-write a live acquisition to disk.

    Capture this once, at ``sequenceStarted`` (not at save time): the sink
    that produces ``settings``/``summary_meta`` is replaced wholesale the next
    time an MDA runs, so a viewer left open across two acquisitions needs its
    own copy rather than reaching back into what is by then a stale sink.

    Parameters
    ----------
    settings : AcquisitionSettings
        The sink's *resolved* settings (dimensions, dtype, positions, channel
        names, physical scales, etc.) -- i.e. exactly what was passed to
        `ome_writers.create_stream()` for the original acquisition. Available
        as `OmeWritersSink.settings` after `sequenceStarted`.
    summary_meta : SummaryMetaV1 | None
        The summary metadata emitted alongside `sequenceStarted`.
    view : Any
        The sink's live view (`MDARunner.get_view()` /
        `OmeWritersSink.get_view()`), indexable in acquisition order --
        i.e. in the same dimension order as `settings.dimensions`.
    frame_meta : list[dict[str, Any]]
        Per-frame metadata dicts, in acquisition order, already converted via
        `pymmcore_plus.mda.frame_meta_to_ome` (e.g. from `frameReady`). May be
        shorter than the number of frames actually written (a frame with no
        captured metadata is simply written with none).
    """

    settings: AcquisitionSettings
    summary_meta: SummaryMetaV1 | None
    view: Any
    frame_meta: list[dict[str, Any]] = field(default_factory=list)


def export_acquisition(
    record: AcquisitionRecord,
    path: str | Path,
    fmt: ExportFormat,
    *,
    overwrite: bool = False,
    progress: ProgressCallback | None = None,
    tiff_layout: str | None = None,
) -> str | None:
    """Replay `record` through a fresh `ome_writers` stream at `path`.

    Parameters
    ----------
    record : AcquisitionRecord
        The acquisition to export.
    path : str | Path
        Destination path. For OME-TIFF with multiple positions, this becomes
        a *directory* of per-position files (matching a live multi-position
        OME-TIFF acquisition); for OME-Zarr it is always a directory.
    fmt : "ome-tiff" | "ome-zarr"
        Output format.
    overwrite : bool
        Whether to overwrite an existing file/directory at `path`.
    tiff_layout : str | None
        OME-TIFF only: how the per-position files relate to each other, as
        `OmeTiffFormat.multi_file_metadata` (`"self-contained"`, `"master-tiff"`
        or `"redundant"`). `None` leaves the format's own default in place.
    progress : ProgressCallback | None
        Optional callback invoked after each frame is written, as
        `progress(frames_done, frames_total)`. Return False to cancel.

    Returns
    -------
    str | None
        The resolved output path (`AcquisitionSettings.output_path`), or
        `None` if `progress` requested cancellation.

    Raises
    ------
    ValueError
        If nothing has been acquired yet (no frames written to `record.view`).
    """
    dims = _clamp_dimensions(record.settings, record.view)
    if dims and dims[0].count is None:
        raise ValueError("Nothing to export: no frames have been acquired yet.")

    out_format: Any = fmt
    if tiff_layout is not None and fmt == "ome-tiff":
        out_format = {"name": "ome-tiff", "multi_file_metadata": tiff_layout}

    target = AcquisitionSettings.model_validate(
        {
            **record.settings.model_dump(
                exclude={"format", "root_path", "overwrite", "dimensions"}
            ),
            "dimensions": dims,
            "root_path": str(path),
            "format": out_format,
            "overwrite": overwrite,
        }
    )

    index_dims = target.dimensions[:-2]
    total = 1
    for dim in index_dims:
        total *= dim.count or 1
    ranges = [range(dim.count or 1) for dim in index_dims]
    n_frame_meta = len(record.frame_meta)

    cancelled = False
    with create_stream(target) as stream:
        if record.summary_meta is not None:
            summary = _serialize_summary_meta(record.summary_meta)
            payload = {"summary_metadata": summary}
            try:
                stream.set_global_metadata("pymmcore_plus", payload)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to attach summary metadata: %s", e, exc_info=True
                )

        for n, idx in enumerate(product(*ranges)):
            frame = np.asarray(record.view[idx])
            meta = record.frame_meta[n] if n < n_frame_meta else None
            stream.append(frame, frame_metadata=meta)
            if progress is not None and not progress(n + 1, total):
                cancelled = True
                break

    return None if cancelled else target.output_path


def _clamp_dimensions(settings: AcquisitionSettings, view: Any) -> list[Dimension]:
    """Shrink dimension counts to what was actually acquired.

    Handles two cases: an unbounded acquisition (`GeneratorMDASequence`, whose
    first dimension has `count=None`), and a cancelled/partial run that never
    reached its nominal size. `view.coords` (populated live, since the sink
    always requests `dynamic_shape=True`) gives the high-water extent per
    dimension.

    Only dimensions *without* explicit `coords` (i.e. not channel/position,
    which are validated to match `len(coords) == count`) are eligible --
    those are, in practice, exactly the ones that can be genuinely partial
    (time, z, or an inserted multi-camera axis); a channel or position axis
    either completes its full pass or the acquisition stops between passes.
    """
    coords_map: Mapping[str, Any] | None = getattr(view, "coords", None)
    if coords_map is None:
        return list(settings.dimensions)

    dims: list[Dimension] = []
    for dim in settings.dimensions:
        if dim.coords is None and dim.name in coords_map:
            n = len(coords_map[dim.name])
            if n and (dim.count is None or n < dim.count):
                dim = dim.model_copy(update={"count": n})
        dims.append(dim)
    return dims


def dimensions_from_wrapper(
    wrapper: DataWrapper, *, scale_overrides: Mapping[str, float] | None = None
) -> list[Dimension]:
    """Build `ome_writers.Dimension` specs from any ndv `DataWrapper`'s own metadata.

    A general-purpose bridge from ndv's `dims`/`coords`/`axis_scales()` to
    `ome_writers`' `Dimension`, usable for *any* wrapper -- a live acquisition's
    view, a reopened OME-TIFF/OME-Zarr's lazy wrapper, or (via `scale_overrides`)
    a viewer whose physical scale was only ever set as a display-model override
    rather than being recoverable from the data itself.

    Channel and position axes get real `Channel`/`Position` coords whenever the
    wrapper's own coords for that axis are more informative than a plain
    `range` (matching how `DataWrapper.channel_names()` decides the same
    thing); every other axis is left as an unlabeled, sequential dimension of
    the given count, with physical scale attached where available.
    """
    sizes = dict(wrapper.sizes())
    if len(sizes) < 2:
        return []
    names = list(sizes)
    n = len(names)
    wrapper_scales = wrapper.axis_scales()
    overrides = scale_overrides or {}

    dims: list[Dimension] = []
    for i, name in enumerate(names):
        is_frame_axis = i >= n - 2
        axis_name = ("y", "x")[i - (n - 2)] if is_frame_axis else str(name)
        dim_type: _DimType = (
            "space" if is_frame_axis else _TYPE_BY_AXIS_NAME.get(axis_name, "other")
        )
        # Only ever meaningful for space/time -- a channel/position axis's
        # "coords" are often numeric-looking *labels* (e.g. tifffile's
        # zero-padded position index strings), which axis_scales() would
        # otherwise happily (and wrongly) treat as an evenly-spaced physical
        # scale.
        scale = None
        if dim_type in ("space", "time"):
            scale = overrides.get(axis_name, wrapper_scales.get(axis_name))
        coords: list[str | float | Channel | Position] | None = None
        if not is_frame_axis:
            raw_coords = wrapper.coords.get(name)
            if raw_coords is not None and not isinstance(raw_coords, range):
                if dim_type == "channel":
                    coords = [Channel(name=str(v)) for v in raw_coords]
                elif dim_type == "position":
                    coords = [Position(name=str(v)) for v in raw_coords]
        dims.append(
            Dimension(
                name=axis_name,
                count=sizes[name],
                type=dim_type,
                scale=scale,
                unit="micrometer" if (dim_type == "space" and scale) else None,
                coords=coords,
            )
        )
    return dims


def record_from_wrapper(
    wrapper: DataWrapper,
    view: Any,
    *,
    summary_meta: SummaryMetaV1 | None = None,
    scale_overrides: Mapping[str, float] | None = None,
) -> AcquisitionRecord | None:
    """Build an `AcquisitionRecord` straight from any ndv `DataWrapper`'s own metadata.

    `view` must support the same acquisition-order tuple indexing
    `export_acquisition` relies on (see `AcquisitionRecord.view`) -- for a
    `DataWrapper`, that's `wrapper.isel(dict(enumerate(idx)))` wrapped in a
    tiny `__getitem__` adapter; callers that already have such a view (e.g.
    a live `StreamView`) can pass it directly instead.
    """
    dims = dimensions_from_wrapper(wrapper, scale_overrides=scale_overrides)
    if not dims:
        return None
    settings = AcquisitionSettings(
        dimensions=tuple(dims), dtype=str(np.dtype(wrapper.dtype))
    )
    return AcquisitionRecord(settings=settings, summary_meta=summary_meta, view=view)
