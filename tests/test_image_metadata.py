"""Tests for `pymmcore_gui._image_metadata` -- no Qt required."""

from __future__ import annotations

import csv
import math
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import pytest
import useq
from ome_writers import (
    AcquisitionSettings,
    dims_from_standard_axes,
    useq_to_acquisition_settings,
)

from pymmcore_gui._image_metadata import (
    FrameInfo,
    FrameLookup,
    axis_labels,
    format_elapsed,
    frame_infos,
    frames_to_csv,
    mean_interval,
    previous_intervals,
    timepoints,
)

if TYPE_CHECKING:
    from pathlib import Path


def _settings(axes: dict[str, Any]) -> AcquisitionSettings:
    return AcquisitionSettings(
        dimensions=tuple(dims_from_standard_axes({**axes, "y": 8, "x": 8})),
        dtype="uint16",
    )


def test_frame_infos_index_by_original_dims_and_stop_at_captured_length() -> None:
    """Skipped slots and later appends don't shift indices; t may be unbounded."""
    settings = _settings({"t": None, "c": ["DAPI", "FITC"], "z": 3})
    frame_meta: list[dict[str, Any] | None] = [
        {"delta_t": 0.0, "exposure_time": 0.01, "position_x": 1.0},
        None,  # skipped
        {"delta_t": 0.2, "mda_event": "{}"},
        {"delta_t": math.nan},
        None,
        None,
        None,
        {"delta_t": 1.0},  # t1, c0, z1
    ]
    frames = frame_infos(settings, frame_meta, stop=8)
    frame_meta.append({"delta_t": 9.0})  # after the snapshot

    assert [f.index for f in frames] == [
        {"t": 0, "c": 0, "z": 0},
        {"t": 0, "c": 0, "z": 2},
        {"t": 0, "c": 1, "z": 0},
        {"t": 1, "c": 0, "z": 1},
    ]
    assert frames[0] == FrameInfo(
        index={"t": 0, "c": 0, "z": 0},
        delta_t=0.0,
        exposure=0.01,
        position=(1.0, None, None),
    )
    assert frames[1].extra == {"mda_event": "{}"}
    assert frames[2].delta_t is None  # not finite
    assert len(frame_infos(settings, frame_meta, stop=8)) == 4


def test_axis_labels_names_channels_and_positions() -> None:
    settings = _settings({"p": ["A1", "B2"], "c": ["DAPI"], "z": 2})
    assert axis_labels(settings) == {"p": ["A1", "B2"], "c": ["DAPI"]}


def test_position_labels_tell_wells_and_grid_tiles_apart() -> None:
    """Positions are named `fov0`... in every well, and alike in every grid tile."""
    plate = useq.WellPlatePlan(
        plate=useq.WellPlate.from_str("96-well"),
        a1_center_xy=(0, 0),
        selected_wells=((0, 0), (0, 1)),
        well_points_plan=useq.RandomPoints(num_points=2),
    )
    grid = useq.Position(
        name="cell",
        sequence=useq.MDASequence(grid_plan=useq.GridRowsColumns(rows=1, columns=2)),
    )
    for plan, expected in (
        (plate, ["A1 · fov0", "A1 · fov1", "A2 · fov0", "A2 · fov1"]),
        ((grid,), ["cell · 0,0", "cell · 0,1"]),
    ):
        seq = useq.MDASequence(stage_positions=plan)
        dims = useq_to_acquisition_settings(seq, 8, 8)["dimensions"]
        settings = AcquisitionSettings(dimensions=tuple(dims), dtype="uint16")
        assert axis_labels(settings)["p"] == expected


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (None, ""),
        (0, "00:00:00.000"),
        (1.2345, "00:00:01.234"),
        (3661.5, "01:01:01.500"),
        (-0.25, "-00:00:00.250"),
    ],
)
def test_format_elapsed(seconds: float | None, text: str) -> None:
    assert format_elapsed(seconds) == text


def _frame(delta_t: float | None, **index: int) -> FrameInfo:
    return FrameInfo(index=index, delta_t=delta_t)


def test_previous_intervals_stay_within_a_position_in_timestamp_order() -> None:
    frames = [
        _frame(0.0, p=0, t=0),
        _frame(0.1, p=1, t=0),
        _frame(5.0, p=0, t=1),
        _frame(None, p=1, t=1),
        _frame(4.0, p=1, t=2),  # earlier than its row order suggests
        _frame(2.0, p=1, t=3),
    ]
    assert previous_intervals(frames) == [
        None,
        None,
        5.0,
        None,
        pytest.approx(2.0),
        pytest.approx(1.9),
    ]


def test_timepoints_compare_each_gap_with_its_scheduled_gap() -> None:
    sequence = useq.MDASequence(
        time_plan=useq.MultiPhaseTimePlan(
            phases=[
                useq.TIntervalLoops(interval=timedelta(seconds=1), loops=2),
                useq.TIntervalLoops(interval=timedelta(seconds=10), loops=2),
            ]
        )
    )
    scheduled = [d.total_seconds() for d in sequence.time_plan.deltas()]  # type: ignore[union-attr]
    # timepoint 1 was not observed, so t2 is compared with t0
    frames = [
        _frame(0.0, t=0, c=0),
        _frame(0.05, t=0, c=1),
        _frame(scheduled[2] * 1.2, t=2, c=0),
    ]
    rows = timepoints(frames, sequence)
    assert [(r.t, r.start, r.requested) for r in rows] == [
        (0, 0.0, None),
        (2, scheduled[2] * 1.2, scheduled[2]),
    ]
    assert rows[1].drift == pytest.approx(0.2)


def test_timepoints_without_a_positive_requested_interval_have_no_drift() -> None:
    sequence = useq.MDASequence(
        time_plan=useq.TIntervalLoops(interval=timedelta(0), loops=3)
    )
    rows = timepoints([_frame(float(t), t=t) for t in range(3)], sequence)
    assert [r.requested for r in rows] == [None, 0.0, 0.0]
    assert all(r.drift is None for r in rows)


def test_mean_interval_is_taken_within_each_position() -> None:
    """Gaps are measured per position, never across positions."""
    frames = [
        _frame(start, p=p, t=t)
        for p, starts in enumerate(([0.0, 2.0, 4.0], [0.5, 3.5, 5.5]))
        for t, start in enumerate(starts)
    ]
    assert mean_interval(timepoints(frames, None)) == pytest.approx(2.25)


def test_timepoints_without_a_time_axis() -> None:
    assert timepoints([_frame(0.0, z=0)], None) == []
    assert mean_interval([]) is None


def test_frame_lookup_rows() -> None:
    frames = [
        _frame(float(n), t=t, c=c, z=z)
        for n, (t, c, z) in enumerate(
            (t, c, z) for t in range(2) for c in range(2) for z in range(3)
        )
        if n != 4  # no metadata for t0 c1 z1
    ]
    lookup = FrameLookup(frames)
    by_index = {tuple(f.index.values()): row for row, f in enumerate(frames)}

    assert lookup.rows({"t": range(1, 2), "c": range(2), "z": range(1, 2)}) == [
        by_index[1, 0, 1],
        by_index[1, 1, 1],
    ]
    assert lookup.rows({"t": range(1), "c": range(1, 2), "z": range(1, 2)}) == []
    # an axis left out matches every value
    assert len(lookup.rows({"t": range(1)})) == 5
    assert FrameLookup([]).rows({"t": range(1)}) == []


def test_frames_to_csv(tmp_path: Path) -> None:
    frames = [
        FrameInfo({"p": 0, "c": 1}, 0.0, 0.01, (1.5, None, None)),
        FrameInfo({"p": 0, "c": 0}, 0.25, None, (None, None, None)),
    ]
    path = tmp_path / "frames.csv"
    frames_to_csv(frames, {"c": ["DAPI", "FITC, 488"]}, path)

    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert rows == [
        [
            "p",
            "c",
            "c name",
            "delta_t (s)",
            "since previous (s)",
            "exposure (ms)",
            "x (µm)",
            "y (µm)",
            "z (µm)",
        ],
        ["0", "1", "FITC, 488", "0.0", "", "10.0", "1.5", "", ""],
        ["0", "0", "DAPI", "0.25", "0.25", "", "", "", ""],
    ]
