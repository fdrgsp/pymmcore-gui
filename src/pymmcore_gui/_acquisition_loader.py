"""Reopen OME acquisitions with lazy data access and recovered metadata.

The wrapper's on-disk dimensions determine the displayed shape, including
partial acquisitions. Recovered sequence and summary metadata enable MDA
parameter reuse and faithful re-export; missing metadata does not prevent viewing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from useq import MDASequence

from pymmcore_gui._mda_export import (
    AcquisitionRecord,
    frame_ordinal,
    record_from_wrapper,
)
from pymmcore_gui._ome_tiff_wrapper import OMETiffWrapper
from pymmcore_gui._ome_zarr_wrapper import OMEZarrWrapper

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ndv.models import DataWrapper
    from ome_writers import Dimension
    from pymmcore_plus.metadata import SummaryMetaV1

logger = logging.getLogger(__name__)

_PYMMCORE_PLUS_NAMESPACE = "pymmcore_plus"


class _WrapperView:
    """Adapt acquisition-order tuple indices to DataWrapper.isel mappings.

    Pixel data is read only when indexed; unspecified spatial axes remain full slices.
    """

    def __init__(self, wrapper: DataWrapper) -> None:
        self._wrapper = wrapper

    def __getitem__(self, idx: tuple[int, ...]) -> Any:
        return self._wrapper.isel(dict(enumerate(idx)))


@dataclass
class LoadedAcquisition:
    """A reopened acquisition: lazy ndv-compatible data plus its recovered metadata.

    Parameters
    ----------
    wrapper : OMETiffWrapper | OMEZarrWrapper
        The lazy, ndv-compatible data. Pass this directly as the `data`
        argument to `ndv.ArrayViewer`/`MMArrayViewer` -- `DataWrapper.create`
        returns an already-a-`DataWrapper` argument unchanged, so no data is
        re-read or re-opened.
    sequence : useq.MDASequence | None
        The acquisition's original sequence, recovered from on-disk
        metadata, or `None` if none was found or it failed to validate.
    record : AcquisitionRecord | None
        Everything `MMArrayViewer`'s Save button needs to re-export this
        acquisition faithfully (real dimensions/scale/channel names, and the
        original summary and per-frame metadata when recoverable) -- attach this to the
        viewer as `_acquisition_record` so re-saving a reopened file (e.g. as
        a different format) doesn't fall back to stamping the *current*
        microscope state. `None` only when the wrapper's own dims are too
        sparse to describe (fewer than 2 axes), which shouldn't happen for
        any real acquisition.
    source_path : Path
        The file or directory this was opened from.
    title : str
        A display title for the viewer tab (`source_path.name`).
    """

    wrapper: OMETiffWrapper | OMEZarrWrapper
    sequence: MDASequence | None
    record: AcquisitionRecord | None
    source_path: Path
    title: str

    def close(self) -> None:
        """Release the wrapper's underlying file handle(s)/zarr store."""
        self.wrapper.close()


def supports_path(path: str | Path) -> bool:
    """Return whether `path` looks like a dataset this loader can open."""
    path = Path(path)
    return bool(OMETiffWrapper.supports(path) or OMEZarrWrapper.supports(path))


def open_acquisition(path: str | Path) -> LoadedAcquisition:
    """Open `path` (an OME-TIFF file, or a pyMM OME-TIFF/OME-Zarr directory).

    Raises
    ------
    ValueError
        If `path` isn't a format this loader recognizes, or fails to open.
    """
    path = Path(path)
    for cls, label in ((OMETiffWrapper, "OME-TIFF"), (OMEZarrWrapper, "OME-Zarr")):
        if cls.supports(path):
            try:
                wrapper: OMETiffWrapper | OMEZarrWrapper = cls(path)
            except Exception as e:
                raise ValueError(
                    f"Failed to open {label} acquisition {path}: {e}"
                ) from e
            break
    else:
        raise ValueError(f"Not a supported acquisition: {path}")

    sequence, summary_meta = _recover_summary_metadata(wrapper)
    record = record_from_wrapper(
        wrapper, _WrapperView(wrapper), summary_meta=summary_meta
    )
    if record is not None:
        record.frame_meta = _recover_frame_metadata(
            wrapper, record.settings.dimensions[:-2]
        )
    return LoadedAcquisition(
        wrapper=wrapper,
        sequence=sequence,
        record=record,
        source_path=path,
        title=path.name,
    )


def _recover_summary_metadata(
    wrapper: Any,
) -> tuple[MDASequence | None, SummaryMetaV1 | None]:
    """Recover the original sequence and summary metadata when available.

    Missing or malformed metadata leaves the acquisition viewable, with no sequence
    to reuse and export metadata derived from the data wrapper.
    """
    try:
        meta = wrapper.global_metadata(_PYMMCORE_PLUS_NAMESPACE)
        if not meta:
            return None, None
        summary_meta = meta.get("summary_metadata")
        if not summary_meta:
            return None, None
        sequence = None
        if raw_sequence := summary_meta.get("mda_sequence"):
            sequence = MDASequence.model_validate(raw_sequence)
            # SummaryMetaV1.mda_sequence is a real MDASequence, not the raw
            # JSON dict `global_metadata` decoded it from -- swap it back in
            # so this is usable directly as an AcquisitionRecord.summary_meta
            # (re-exporting re-serializes it the same way the original did).
            summary_meta = {**summary_meta, "mda_sequence": sequence}
        # Trusted, not schema-validated: exactly how a live sink's own
        # `summary_meta` is already handled elsewhere (e.g. `OmeWritersSink.
        # summary_meta`) -- this is whatever the writer put there, reshaped
        # only enough to be usable as one.
        return sequence, cast("SummaryMetaV1", summary_meta)
    except Exception as e:
        logger.warning("Could not recover metadata from %s: %s", wrapper, e)
        return None, None


def _recover_frame_metadata(
    wrapper: OMETiffWrapper | OMEZarrWrapper, dims: Sequence[Dimension]
) -> list[dict[str, Any] | None]:
    """Place the file's per-frame metadata at each frame's ordinal over `dims`.

    The list only reaches the last frame with metadata: a partial Zarr array's
    declared shape can be far larger than what was acquired.
    """
    try:
        entries = wrapper.frame_metadata()
    except Exception as e:
        logger.warning("Could not recover frame metadata from %s: %s", wrapper, e)
        return []

    shape = [dim.count or 0 for dim in dims]
    by_ordinal: dict[int, dict[str, Any]] = {}
    for index, meta in entries.items():
        if len(index) == len(shape) and all(
            isinstance(i, int) and 0 <= i < n for i, n in zip(index, shape, strict=True)
        ):
            by_ordinal[frame_ordinal(dims, index)] = meta
    if ignored := len(entries) - len(by_ordinal):
        logger.warning("Ignored %d malformed frame metadata entries", ignored)

    frame_meta: list[dict[str, Any] | None] = [None] * (max(by_ordinal, default=-1) + 1)
    for ordinal, meta in by_ordinal.items():
        frame_meta[ordinal] = meta
    return frame_meta
