"""Tests for `pymmcore_gui._grid_axis` -- no Qt required.

`GridAxisLayout.build()` is exercised against real `ome_writers`
`AcquisitionSettings` (via `useq_to_acquisition_settings`, the same resolver
`OmeWritersSink` uses), and `GridAxisDataWrapper` against real scratch
`StreamView`s (via `create_stream`), mirroring `tests/test_mda_export.py`'s
style: no mocking of the writer or useq.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pytest
import useq
from ome_writers import AcquisitionSettings, create_stream, useq_to_acquisition_settings

from pymmcore_gui._grid_axis import (
    GridAxisDataWrapper,
    GridAxisLayout,
    GridAxisLayoutKind,
    GridOrder,
)


def _settings(
    sequence: useq.MDASequence, *, width: int = 4, height: int = 4
) -> AcquisitionSettings:
    d = useq_to_acquisition_settings(sequence, image_width=width, image_height=height)
    return AcquisitionSettings(
        dimensions=tuple(d["dimensions"]), plate=d.get("plate"), dtype="uint16"
    )


# --------------------------------------------------------------------------- #
# GridAxisLayout.build()
# --------------------------------------------------------------------------- #


def test_build_grid_only() -> None:
    seq = useq.MDASequence(grid_plan=useq.GridRowsColumns(rows=1, columns=3))
    layout = GridAxisLayout.build(seq, _settings(seq))
    assert layout.kind is GridAxisLayoutKind.GRID_ONLY
    assert layout.n_positions == 1
    assert layout.n_tiles == 3
    assert layout.exposed_axes == ("g",)
    assert [layout.flat_index(None, g) for g in range(3)] == [0, 1, 2]


@pytest.mark.parametrize(
    ("axis_order", "expected_order"),
    [
        (("t", "p", "g", "c"), GridOrder.POSITION_FIRST),
        (("t", "g", "p", "c"), GridOrder.GRID_FIRST),
    ],
)
def test_build_global_grid_both_axis_orders(
    axis_order: tuple[str, ...], expected_order: GridOrder
) -> None:
    seq = useq.MDASequence(
        axis_order=axis_order,
        stage_positions=(
            useq.AbsolutePosition(x=0, y=0),
            useq.AbsolutePosition(x=100, y=100),
        ),
        grid_plan=useq.GridRowsColumns(rows=1, columns=2),
    )
    layout = GridAxisLayout.build(seq, _settings(seq))
    assert layout.kind is GridAxisLayoutKind.REGULAR
    assert layout.order is expected_order
    assert (layout.n_positions, layout.n_tiles) == (2, 2)

    # Cross-check every flat_index() against the real flattened visit order.
    for flat, e in enumerate(seq):
        p, g = e.index[useq.Axis.POSITION], e.index[useq.Axis.GRID]
        assert layout.flat_index(p, g) == flat


def test_build_per_position_subsequence_grid_always_position_first() -> None:
    # Even with g before p in axis_order, a per-position subsequence grid
    # flattens position-first in ome_writers -- GridAxisLayout must match.
    seq = useq.MDASequence(
        axis_order=("t", "g", "p", "c"),
        stage_positions=(
            useq.AbsolutePosition(
                x=0,
                y=0,
                sequence=useq.MDASequence(
                    grid_plan=useq.GridRowsColumns(rows=1, columns=2)
                ),
            ),
            useq.AbsolutePosition(
                x=100,
                y=100,
                sequence=useq.MDASequence(
                    grid_plan=useq.GridRowsColumns(rows=1, columns=2)
                ),
            ),
        ),
    )
    layout = GridAxisLayout.build(seq, _settings(seq))
    assert layout.kind is GridAxisLayoutKind.REGULAR
    # Storage flattening is position-first regardless of axis_order (which,
    # for a per-position subsequence grid, only affects *acquisition* order,
    # e.g. interleaving p0g0,p1g0,p0g1,p1g1 -- not the flattened storage
    # slot), so check the arithmetic directly rather than acquisition order.
    assert layout.order is GridOrder.POSITION_FIRST
    assert [layout.flat_index(p, g) for p in range(2) for g in range(2)] == [0, 1, 2, 3]
    # Cross-check against settings.positions, the sink's own ground truth.
    settings = _settings(seq)
    assert len(settings.positions) == layout.n_flat == 4


def test_build_snake_traversal_follows_visit_order_not_row_major() -> None:
    seq = useq.MDASequence(grid_plan=useq.GridRowsColumns(rows=2, columns=2))
    layout = GridAxisLayout.build(seq, _settings(seq))
    assert layout.kind is GridAxisLayoutKind.GRID_ONLY
    events = list(seq)
    grid_cols = [e.index[useq.Axis.GRID] for e in events]
    # snake order visits g=0,1,2,3 in acquisition order regardless of
    # row/column -- row_wise_snake is useq's default mode.
    assert grid_cols == [0, 1, 2, 3]
    # but the underlying (row, col) is NOT simply g // cols, g % cols for the
    # second row (snake reverses direction) -- confirms g is visit order.
    row_col = [(gp.grid_row, gp.grid_col) for gp in seq.grid_plan]  # type: ignore[union-attr]
    assert row_col == [(0, 0), (0, 1), (1, 1), (1, 0)]


def test_build_repeated_timepoints_flat_index_is_idempotent() -> None:
    seq = useq.MDASequence(
        time_plan=useq.TIntervalLoops(interval=timedelta(0), loops=3),
        stage_positions=(
            useq.AbsolutePosition(x=0, y=0),
            useq.AbsolutePosition(x=100, y=100),
        ),
        grid_plan=useq.GridRowsColumns(rows=1, columns=2),
    )
    layout = GridAxisLayout.build(seq, _settings(seq))
    assert layout.kind is GridAxisLayoutKind.REGULAR
    # Same (p, g) revisited at every timepoint must always resolve identically.
    seen = {
        layout.flat_index(p, g) for p in range(2) for g in range(2) for _t in range(3)
    }
    assert seen == {0, 1, 2, 3}


def test_build_ragged_differently_sized_grids() -> None:
    seq = useq.MDASequence(
        stage_positions=(
            useq.AbsolutePosition(
                x=0,
                y=0,
                sequence=useq.MDASequence(
                    grid_plan=useq.GridRowsColumns(rows=1, columns=2)
                ),
            ),
            useq.AbsolutePosition(
                x=100,
                y=100,
                sequence=useq.MDASequence(
                    grid_plan=useq.GridRowsColumns(rows=1, columns=3)
                ),
            ),
        ),
    )
    layout = GridAxisLayout.build(seq, _settings(seq))
    assert layout.kind is GridAxisLayoutKind.RAGGED
    assert layout.tile_counts == (2, 3)
    assert layout.n_tiles == 3  # slider range = max
    assert layout.offsets == (0, 2)
    assert layout.flat_index(0, 0) == 0
    assert layout.flat_index(0, 1) == 1
    assert layout.flat_index(1, 0) == 2
    assert layout.flat_index(1, 2) == 4
    with pytest.raises(IndexError):
        layout.flat_index(0, 2)  # position 0 only has 2 tiles


def test_build_ragged_mixed_gridded_and_plain_positions() -> None:
    seq = useq.MDASequence(
        stage_positions=(
            useq.AbsolutePosition(
                x=0,
                y=0,
                sequence=useq.MDASequence(
                    grid_plan=useq.GridRowsColumns(rows=3, columns=1)
                ),
            ),
            useq.AbsolutePosition(x=0, y=100),
        ),
    )
    layout = GridAxisLayout.build(seq, _settings(seq))
    assert layout.kind is GridAxisLayoutKind.RAGGED
    assert layout.tile_counts == (3, 1)
    assert layout.offsets == (0, 3)
    assert layout.flat_index(1, 0) == 3
    with pytest.raises(IndexError):
        layout.flat_index(1, 1)  # the plain position has only one tile


def test_build_well_plate_falls_back_to_none() -> None:
    plate_plan = useq.WellPlatePlan(
        plate=96,  # pyright: ignore[reportArgumentType]
        a1_center_xy=(0, 0),
        selected_wells=((0, 1), (0, 1)),
        well_points_plan=useq.GridRowsColumns(rows=1, columns=2),
    )
    seq = useq.MDASequence(stage_positions=plate_plan)
    layout = GridAxisLayout.build(seq, _settings(seq))
    assert layout.kind is GridAxisLayoutKind.NONE
    assert layout.has_grid is False  # well-plate fov points never carry "g"


def test_build_ordinary_multi_position_no_grid_returns_none() -> None:
    seq = useq.MDASequence(
        stage_positions=(
            useq.AbsolutePosition(x=0, y=0),
            useq.AbsolutePosition(x=100, y=100),
        )
    )
    layout = GridAxisLayout.build(seq, _settings(seq))
    assert layout.kind is GridAxisLayoutKind.NONE
    assert layout.has_grid is False


def test_build_single_position_no_grid_returns_none() -> None:
    seq = useq.MDASequence(channels=["DAPI"])  # pyright: ignore
    layout = GridAxisLayout.build(seq, _settings(seq))
    assert layout.kind is GridAxisLayoutKind.NONE


def test_build_sink_fallback_settings_returns_none() -> None:
    # No position-typed dimension at all -- mimics
    # OmeWritersSink._unbounded_3d_settings' fallback.
    from ome_writers import dims_from_standard_axes

    seq = useq.MDASequence(grid_plan=useq.GridRowsColumns(rows=1, columns=2))
    settings = AcquisitionSettings(
        dimensions=tuple(dims_from_standard_axes({"t": 2, "y": 4, "x": 4})),
        dtype="uint16",
    )
    layout = GridAxisLayout.build(seq, settings)
    assert layout.kind is GridAxisLayoutKind.NONE


def test_build_non_mdasequence_returns_none() -> None:
    seq = useq.MDASequence(grid_plan=useq.GridRowsColumns(rows=1, columns=2))
    settings = _settings(seq)
    layout = GridAxisLayout.build(object(), settings)
    assert layout.kind is GridAxisLayoutKind.NONE


# --------------------------------------------------------------------------- #
# GridAxisDataWrapper
# --------------------------------------------------------------------------- #


def _stream_view_for(
    sequence: useq.MDASequence, *, width: int = 4, height: int = 4
) -> tuple[Any, Any, AcquisitionSettings]:
    settings = _settings(sequence, width=width, height=height)
    stream = create_stream(settings)
    for i, _e in enumerate(sequence):
        stream.append(np.full((height, width), i, dtype="uint16"))
    return stream, stream.view(), settings


def test_wrapper_regular_pixel_correctness_and_axis_order() -> None:
    seq = useq.MDASequence(
        stage_positions=(
            useq.AbsolutePosition(x=0, y=0),
            useq.AbsolutePosition(x=100, y=100),
        ),
        grid_plan=useq.GridRowsColumns(rows=1, columns=2),
        channels=["DAPI", "FITC"],  # pyright: ignore
    )
    settings = _settings(seq)
    layout = GridAxisLayout.build(seq, settings)
    assert layout.kind is GridAxisLayoutKind.REGULAR

    # Write a value unique per (flattened position, channel) -- decouples
    # verification from any assumption that acquisition order matches
    # flattened storage order (it does not, in general -- see the
    # per-position-subsequence test below).
    stream = create_stream(settings)
    for e in seq:
        p, g, c = (
            e.index[useq.Axis.POSITION],
            e.index[useq.Axis.GRID],
            e.index[useq.Axis.CHANNEL],
        )
        flat_pos = layout.flat_index(p, g)
        stream.append(np.full((4, 4), flat_pos * 10 + c, dtype="uint16"))
    view = stream.view()
    wrapper = GridAxisDataWrapper(view, layout)

    assert wrapper.dims[:2] == ("p", "g")
    assert dict(wrapper.sizes())["p"] == 2
    assert dict(wrapper.sizes())["g"] == 2

    p_axis = wrapper.dims.index("p")
    g_axis = wrapper.dims.index("g")
    c_axis = wrapper.dims.index("c")
    for p in range(2):
        for g in range(2):
            flat_pos = layout.flat_index(p, g)
            for c in range(2):
                result = wrapper.isel(
                    {p_axis: slice(p, p + 1), g_axis: slice(g, g + 1), c_axis: c}
                )
                assert (result == flat_pos * 10 + c).all()
                # Cross-check against indexing the raw flattened view directly
                # at the corresponding position.
                raw_expected = np.asarray(view[flat_pos, c])
                np.testing.assert_array_equal(np.squeeze(result), raw_expected)
    stream.close()


def test_wrapper_int_vs_slice_collapse_semantics() -> None:
    seq = useq.MDASequence(grid_plan=useq.GridRowsColumns(rows=1, columns=3))
    stream, view, settings = _stream_view_for(seq)
    layout = GridAxisLayout.build(seq, settings)
    wrapper = GridAxisDataWrapper(view, layout)
    g_axis = wrapper.dims.index("g")

    # bare int -> caller (e.g. ROI) wants the axis collapsed
    collapsed = wrapper.isel({g_axis: 1})
    assert collapsed.ndim == len(wrapper.dims) - 1

    # slice(v, v+1) -> ndv's own pipeline wants the axis retained, size 1
    retained = wrapper.isel({g_axis: slice(1, 2)})
    assert retained.ndim == len(wrapper.dims)
    assert retained.shape[g_axis] == 1
    np.testing.assert_array_equal(np.squeeze(retained), collapsed)
    stream.close()


def test_wrapper_rejects_multi_value_requests() -> None:
    seq = useq.MDASequence(grid_plan=useq.GridRowsColumns(rows=1, columns=3))
    stream, view, settings = _stream_view_for(seq)
    layout = GridAxisLayout.build(seq, settings)
    wrapper = GridAxisDataWrapper(view, layout)
    g_axis = wrapper.dims.index("g")
    with pytest.raises(ValueError, match="multi-value"):
        wrapper.isel({g_axis: slice(0, 2)})
    with pytest.raises(ValueError, match="multi-value"):
        wrapper.isel({g_axis: slice(None)})
    stream.close()


def test_wrapper_memory_shared_with_source_single_position() -> None:
    seq = useq.MDASequence(
        stage_positions=(
            useq.AbsolutePosition(x=0, y=0),
            useq.AbsolutePosition(x=100, y=100),
        )
    )
    # no grid here -- build a REGULAR-style layout manually isn't needed;
    # exercise the same integer-indexing path GRID_ONLY/REGULAR share via a
    # single-tile grid-only sequence instead, where memory sharing is
    # directly observable against the source per-position array.
    seq = useq.MDASequence(grid_plan=useq.GridRowsColumns(rows=1, columns=2))
    settings = _settings(seq)
    stream = create_stream(settings)
    frame0 = np.arange(16, dtype="uint16").reshape(4, 4)
    frame1 = np.full((4, 4), 7, dtype="uint16")
    stream.append(frame0)
    stream.append(frame1)
    view = stream.view()
    layout = GridAxisLayout.build(seq, settings)
    wrapper = GridAxisDataWrapper(view, layout)
    g_axis = wrapper.dims.index("g")

    result = wrapper.isel({g_axis: 0})
    # Compare against the source's own per-position array, not just another
    # wrapper output.
    source = np.asarray(
        view._get_from_position(0, (slice(None), slice(None)), (True, True))
    )
    assert np.shares_memory(result, source)
    stream.close()


def test_wrapper_composite_and_z_axes_pass_through() -> None:
    seq = useq.MDASequence(
        grid_plan=useq.GridRowsColumns(rows=1, columns=2),
        channels=["DAPI", "FITC"],  # pyright: ignore
        z_plan=useq.ZRangeAround(range=2, step=1),
    )
    stream, view, settings = _stream_view_for(seq)
    layout = GridAxisLayout.build(seq, settings)
    assert layout.kind is GridAxisLayoutKind.GRID_ONLY
    wrapper = GridAxisDataWrapper(view, layout)

    assert "z" in wrapper.dims and "c" in wrapper.dims
    assert wrapper.guess_z_axis() == wrapper.normalize_axis_key("z")
    # A small g axis (2 tiles) must never be mistaken for the channel axis.
    assert wrapper.guess_channel_axis() == wrapper.normalize_axis_key("c")
    stream.close()


def test_wrapper_guess_channel_and_z_never_pick_grid_axes() -> None:
    # No real channel or z axis at all -- guesses must return None, never
    # fall back to picking the (small) g axis.
    seq = useq.MDASequence(grid_plan=useq.GridRowsColumns(rows=1, columns=2))
    stream, view, settings = _stream_view_for(seq)
    layout = GridAxisLayout.build(seq, settings)
    wrapper = GridAxisDataWrapper(view, layout)
    assert wrapper.guess_channel_axis() is None
    assert wrapper.guess_z_axis() is None
    stream.close()


def test_wrapper_ragged_blank_fill_never_shows_another_position() -> None:
    seq = useq.MDASequence(
        stage_positions=(
            useq.AbsolutePosition(
                x=0,
                y=0,
                sequence=useq.MDASequence(
                    grid_plan=useq.GridRowsColumns(rows=1, columns=3)
                ),
            ),
            useq.AbsolutePosition(x=0, y=100),
        ),
    )
    stream, view, settings = _stream_view_for(seq)
    layout = GridAxisLayout.build(seq, settings)
    assert layout.kind is GridAxisLayoutKind.RAGGED
    wrapper = GridAxisDataWrapper(view, layout)
    p_axis, g_axis = wrapper.dims.index("p"), wrapper.dims.index("g")

    real_p1_g0 = wrapper.isel({p_axis: slice(1, 2), g_axis: slice(0, 1)})
    assert real_p1_g0.flat[0] == 3  # position 1's single tile is flat slot 3

    blank = wrapper.isel({p_axis: slice(1, 2), g_axis: slice(2, 3)})
    assert blank.shape == real_p1_g0.shape
    assert not blank.any()
    stream.close()


def test_wrapper_supports_never_autodetects() -> None:
    assert GridAxisDataWrapper.supports(object()) is False
