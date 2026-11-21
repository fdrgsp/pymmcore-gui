"""Per-frame metadata of an acquisition, as the metadata dialog shows it.

Everything here is computed from a viewer's `AcquisitionRecord` -- the same
record its Save button exports -- so the dialog shows what gets saved.
"""

from __future__ import annotations

import csv
import itertools
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pymmcore_gui._mda_export import frame_index

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from ome_writers import AcquisitionSettings
    from useq import MDASequence

FORMAT_LABELS = {"ome-tiff": "OME-TIFF", "ome-zarr": "OME-Zarr", "scratch": "In memory"}
_POSITION_KEYS = ("position_x", "position_y", "position_z")
_KNOWN_KEYS = {"delta_t", "exposure_time", *_POSITION_KEYS}


@dataclass(frozen=True)
class DataOrigin:
    """Where a viewer's data lives: the metadata dialog's header and Raw tab."""

    format: str
    """A `FORMAT_LABELS` value."""
    path: str = ""
    raw_documents: Callable[[], Mapping[str, Callable[[], str]]] | None = None
    """Lists the raw metadata documents on disk, each loaded on demand."""


@dataclass(frozen=True, slots=True)
class FrameInfo:
    """One frame's metadata in seconds and µm (`None` when not recorded)."""

    index: dict[str, int]
    delta_t: float | None = None
    exposure: float | None = None
    position: tuple[float | None, float | None, float | None] = (None, None, None)
    extra: dict[str, Any] = field(default_factory=dict)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def frame_infos(
    settings: AcquisitionSettings,
    frame_meta: Sequence[Mapping[str, Any] | None],
    stop: int,
) -> list[FrameInfo]:
    """Normalize the entries of `frame_meta[:stop]` that have metadata.

    Entry *n* belongs to ordinal *n* over the record's original dimensions,
    whatever the live view's (asynchronously updated) extent says.
    """
    dims = settings.dimensions[:-2]
    names = [dim.name for dim in dims]
    return [
        FrameInfo(
            index=dict(zip(names, frame_index(dims, ordinal), strict=True)),
            delta_t=_number(meta.get("delta_t")),
            exposure=_number(meta.get("exposure_time")),
            position=(
                _number(meta.get("position_x")),
                _number(meta.get("position_y")),
                _number(meta.get("position_z")),
            ),
            extra={k: v for k, v in meta.items() if k not in _KNOWN_KEYS},
        )
        for ordinal, meta in enumerate(itertools.islice(frame_meta, stop))
        if meta is not None
    ]


def axis_labels(settings: AcquisitionSettings) -> dict[str, list[str]]:
    """Names of the frame axes' values, for axes with named coords (c, p)."""
    labels: dict[str, list[str]] = {}
    for dim in settings.dimensions[:-2]:
        if dim.coords:
            labels[dim.name] = [
                str(name if (name := getattr(coord, "name", coord)) is not None else i)
                for i, coord in enumerate(dim.coords)
            ]
    return labels


def format_elapsed(seconds: float | None) -> str:
    """`HH:MM:SS.mmm`, or an empty string for a missing value."""
    if seconds is None:
        return ""
    total_ms = round(abs(seconds) * 1000)
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    sign = "-" if seconds < 0 and total_ms else ""
    return f"{sign}{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def previous_intervals(frames: Sequence[FrameInfo]) -> list[float | None]:
    """Seconds since the previous timestamped frame at the same position.

    Frames are put in timestamp order within each position (`p`, if any), so
    the result does not depend on how the table is sorted. The first frame at
    a position, and a frame without a timestamp, get None.
    """
    result: list[float | None] = [None] * len(frames)
    by_position: defaultdict[int | None, list[tuple[float, int]]] = defaultdict(list)
    for row, frame in enumerate(frames):
        if frame.delta_t is not None:
            by_position[frame.index.get("p")].append((frame.delta_t, row))
    for stamps in by_position.values():
        stamps.sort()
        for (before, _), (now, row) in itertools.pairwise(stamps):
            result[row] = now - before
    return result


@dataclass(frozen=True, slots=True)
class TimepointInfo:
    """When one position's timepoint started, against the time plan."""

    position: int | None
    t: int
    start: float
    """Seconds: the earliest timestamp of this timepoint at this position."""
    interval: float | None
    """Seconds since the previous observed timepoint at this position."""
    requested: float | None
    """Seconds the time plan scheduled for that same gap."""

    @property
    def drift(self) -> float | None:
        """Relative deviation of `interval` from a positive `requested`."""
        if self.interval is None or not self.requested or self.requested <= 0:
            return None
        return (self.interval - self.requested) / self.requested


def _scheduled_starts(sequence: MDASequence | None, count: int) -> list[float]:
    """Seconds from the start at which the time plan schedules each timepoint."""
    deltas = getattr(getattr(sequence, "time_plan", None), "deltas", None)
    if deltas is None:
        return []
    return [delta.total_seconds() for delta in itertools.islice(deltas(), count)]


def timepoints(
    frames: Sequence[FrameInfo], sequence: MDASequence | None
) -> list[TimepointInfo]:
    """One row per (position, timepoint) with a timestamp, in that order.

    The requested gap comes from the plan's schedule between the two observed
    timepoints, so multiphase plans and skipped timepoints compare correctly.
    """
    starts: dict[tuple[int | None, int], float] = {}
    for frame in frames:
        if frame.delta_t is None or (t := frame.index.get("t")) is None:
            continue
        key = (frame.index.get("p"), t)
        starts[key] = min(starts.get(key, frame.delta_t), frame.delta_t)
    if not starts:
        return []

    schedule = _scheduled_starts(sequence, max(t for _, t in starts) + 1)
    rows: list[TimepointInfo] = []
    previous: dict[int | None, tuple[int, float]] = {}
    # p is None for every key or for none of them
    for p, t in sorted(starts, key=lambda key: (key[0] or 0, key[1])):
        start = starts[p, t]
        interval = requested = None
        if (before := previous.get(p)) is not None:
            prev_t, prev_start = before
            interval = start - prev_start
            if t < len(schedule):
                requested = schedule[t] - schedule[prev_t]
        rows.append(TimepointInfo(p, t, start, interval, requested))
        previous[p] = (t, start)
    return rows


def mean_interval(rows: Sequence[TimepointInfo]) -> float | None:
    """Mean seconds between consecutive observed timepoints of each position."""
    intervals = [row.interval for row in rows if row.interval is not None]
    return statistics.fmean(intervals) if intervals else None


class FrameLookup:
    """Finds the frames on screen without scanning every frame."""

    def __init__(self, frames: Sequence[FrameInfo]) -> None:
        self._axes = list(frames[0].index) if frames else []
        self._rows = {tuple(f.index.values()): row for row, f in enumerate(frames)}
        self._extents = [max(f.index[axis] for f in frames) + 1 for axis in self._axes]

    def rows(self, ranges: Mapping[str, range]) -> list[int]:
        """Rows whose index is in `ranges`, in record order.

        An axis that `ranges` leaves out matches every value.
        """
        choices = [
            ranges.get(axis, range(extent))
            for axis, extent in zip(self._axes, self._extents, strict=True)
        ]
        return [
            row
            for index in itertools.product(*choices)
            if (row := self._rows.get(index)) is not None
        ]


def frames_to_csv(
    frames: Sequence[FrameInfo],
    labels: Mapping[str, Sequence[str]],
    path: str | Path,
) -> None:
    """Write one row per frame, in record order, with units in the headers."""
    axes = list(frames[0].index) if frames else []
    named = [axis for axis in axes if axis in labels]
    intervals = previous_intervals(frames)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                *axes,
                *(f"{axis} name" for axis in named),
                "delta_t (s)",
                "since previous (s)",
                "exposure (ms)",
                "x (µm)",
                "y (µm)",
                "z (µm)",
            ]
        )
        for frame, interval in zip(frames, intervals, strict=True):
            exposure = None if frame.exposure is None else frame.exposure * 1000
            writer.writerow(
                [
                    *frame.index.values(),
                    *(value_label(labels[axis], frame.index[axis]) for axis in named),
                    *(
                        "" if value is None else value
                        for value in (
                            frame.delta_t,
                            interval,
                            exposure,
                            *frame.position,
                        )
                    ),
                ]
            )


def value_label(names: Sequence[str], i: int) -> str:
    """The name of value `i` of an axis, or the index when it has none."""
    return names[i] if 0 <= i < len(names) else str(i)
