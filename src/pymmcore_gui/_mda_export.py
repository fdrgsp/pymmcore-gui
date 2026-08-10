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
from ome_writers import AcquisitionSettings, create_stream
from pymmcore_plus.mda._sink import _serialize_summary_meta

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from ome_writers import Dimension
    from pymmcore_plus.metadata import SummaryMetaV1

logger = logging.getLogger(__name__)

ExportFormat = Literal["ome-tiff", "ome-zarr"]
# (frames_done, frames_total) -> keep_going. Returning False cancels the export
# after the frame just written -- whatever was flushed to disk stays there,
# same as a user-cancelled live MDA leaves a partial file.
ProgressCallback = Callable[[int, int], bool]


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

    target = AcquisitionSettings.model_validate(
        {
            **record.settings.model_dump(
                exclude={"format", "root_path", "overwrite", "dimensions"}
            ),
            "dimensions": dims,
            "root_path": str(path),
            "format": fmt,
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
