"""Expose a flattened `ome-writers` position axis as logical `p`/`g` viewer axes.

`ome-writers` intentionally flattens useq's `p` (stage position) and `g` (grid
tile) axes into a single writer position dimension -- storage never changes.
`GridAxisLayout` derives, from the planned `useq.MDASequence` and the sink's
*resolved* `AcquisitionSettings` alone (never by iterating acquired frames),
whether that flattened dimension can be truthfully presented as independent
`p`/`g` sliders. `GridAxisDataWrapper` is the read-only `ndv.DataWrapper` that
does the presenting: it translates a logical `(p, g)` selection back to the
single flattened integer position index the writer actually uses, with no
additional pixel copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import useq
from ndv.models import DataWrapper

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from ome_writers import AcquisitionSettings


class GridAxisLayoutKind(str, Enum):
    """What (if anything) a `GridAxisLayout` exposes in place of flattened `p`."""

    NONE = "none"
    """No grid axis: pass the raw, flattened `p` view through unmodified."""
    GRID_ONLY = "grid_only"
    """Only `g` is exposed; there is no independent stage position."""
    REGULAR = "regular"
    """Both `p` and `g` are exposed, as a rectangular `n_positions x n_tiles`."""


class GridOrder(str, Enum):
    """Flattening arithmetic, matching `ome_writers._build_stage_positions_plan`."""

    POSITION_FIRST = "position_first"
    """`flat = p * n_tiles + g` -- every per-position subsequence grid, or a
    global grid when `Axis.POSITION` precedes `Axis.GRID` in `axis_order`."""
    GRID_FIRST = "grid_first"
    """`flat = g * n_positions + p` -- a global grid only, when `Axis.GRID`
    precedes `Axis.POSITION` in `axis_order`."""


@dataclass(frozen=True, slots=True)
class GridAxisLayout:
    """A validated, planned mapping from logical `(p, g)` to a flattened slot.

    Never constructed directly outside this module -- use :meth:`build`, which
    always returns a usable layout, falling back to :meth:`none` for anything
    ragged, ambiguous, or otherwise unsupported.
    """

    kind: GridAxisLayoutKind
    n_positions: int
    n_tiles: int
    order: GridOrder
    n_flat: int
    raw_position_axis: int | None

    @property
    def exposed_axes(self) -> tuple[str, ...]:
        """Logical axis labels this layout adds, in slider-presentation order."""
        if self.kind is GridAxisLayoutKind.NONE:
            return ()
        if self.kind is GridAxisLayoutKind.GRID_ONLY:
            return ("g",)
        return ("p", "g") if self.order is GridOrder.POSITION_FIRST else ("g", "p")

    def flat_index(self, p: int | None, g: int | None) -> int:
        """Map a logical `(p, g)` pair to the writer's flattened position slot."""
        if self.kind is GridAxisLayoutKind.NONE:
            raise ValueError("flat_index() is not valid for a GridAxisLayoutKind.NONE")
        if self.kind is GridAxisLayoutKind.GRID_ONLY:
            g = 0 if g is None else g
            if not 0 <= g < self.n_tiles:
                raise IndexError(f"g={g} out of range for n_tiles={self.n_tiles}")
            return g
        p = 0 if p is None else p
        g = 0 if g is None else g
        if not 0 <= p < self.n_positions:
            raise IndexError(f"p={p} out of range for n_positions={self.n_positions}")
        if not 0 <= g < self.n_tiles:
            raise IndexError(f"g={g} out of range for n_tiles={self.n_tiles}")
        if self.order is GridOrder.POSITION_FIRST:
            return p * self.n_tiles + g
        return g * self.n_positions + p

    @classmethod
    def none(cls) -> GridAxisLayout:
        """The universal fallback: no grid axis, flattened `p` is used as-is."""
        return cls(
            kind=GridAxisLayoutKind.NONE,
            n_positions=0,
            n_tiles=0,
            order=GridOrder.POSITION_FIRST,
            n_flat=0,
            raw_position_axis=None,
        )

    @classmethod
    def build(cls, sequence: Any, settings: AcquisitionSettings) -> GridAxisLayout:
        """Derive and validate a layout without iterating MDA events.

        Never raises; falls back to :meth:`none` for anything unsupported,
        ragged, or where the derived layout disagrees with `settings` (the
        sink's ground-truth, resolved position list).
        """
        if not isinstance(sequence, useq.MDASequence):
            return cls.none()

        pos_dim_idx = settings.position_dimension_index
        if pos_dim_idx is None:
            # The sink fell back to generic (t, y, x)-style storage (e.g.
            # OmeWritersSink._unbounded_3d_settings) -- no position dim at all.
            return cls.none()

        stage_positions = sequence.stage_positions
        if isinstance(stage_positions, useq.WellPlatePlan):
            # Well-plate layouts are out of scope for this pass.
            return cls.none()

        has_global_grid = sequence.grid_plan is not None
        has_any_subseq_grid = any(
            p.sequence is not None and p.sequence.grid_plan is not None
            for p in stage_positions
        )
        if not has_global_grid and not has_any_subseq_grid:
            # Ordinary multi-position (or single-position) run: today's
            # flattened `p` is already correct, no adapter needed.
            return cls.none()

        try:
            kind, n_positions, n_tiles, order = cls._derive(
                sequence, stage_positions, has_global_grid
            )
        except ValueError:
            return cls.none()
        if n_tiles < 1:
            return cls.none()

        expected_n_flat = (
            n_tiles if kind is GridAxisLayoutKind.GRID_ONLY else (n_positions * n_tiles)
        )
        resolved_positions = settings.positions
        if len(resolved_positions) != expected_n_flat:
            # Integrity cross-check against the sink's actual resolved
            # positions -- catches any disagreement between our own
            # derivation and what ome-writers actually built (e.g. a grid
            # plan whose fov size resolves differently than expected).
            return cls.none()

        return cls(
            kind=kind,
            n_positions=n_positions,
            n_tiles=n_tiles,
            order=order,
            n_flat=expected_n_flat,
            raw_position_axis=pos_dim_idx,
        )

    @staticmethod
    def _derive(
        sequence: useq.MDASequence,
        stage_positions: tuple[useq.AbsolutePosition, ...],
        has_global_grid: bool,
    ) -> tuple[GridAxisLayoutKind, int, int, GridOrder]:
        """Return `(kind, n_positions, n_tiles, order)` or raise `ValueError`."""
        if not stage_positions:
            # Grid-plan only, no stage positions.
            assert sequence.grid_plan is not None
            n_tiles = len(list(sequence.grid_plan))
            return GridAxisLayoutKind.GRID_ONLY, 1, n_tiles, GridOrder.POSITION_FIRST

        n_positions = len(stage_positions)
        sub_grids = [
            p.sequence.grid_plan if p.sequence is not None else None
            for p in stage_positions
        ]
        if all(g is None for g in sub_grids):
            if not has_global_grid:
                raise ValueError("no grid plan found")  # pragma: no cover
            assert sequence.grid_plan is not None
            n_tiles = len(list(sequence.grid_plan))
            grid_first = (
                useq.Axis.GRID in sequence.axis_order
                and useq.Axis.POSITION in sequence.axis_order
                and sequence.axis_order.index(useq.Axis.GRID)
                < sequence.axis_order.index(useq.Axis.POSITION)
            )
            order = GridOrder.GRID_FIRST if grid_first else GridOrder.POSITION_FIRST
        elif all(g is not None for g in sub_grids):
            counts = [len(list(g)) for g in sub_grids]  # type: ignore[arg-type]
            if len(set(counts)) != 1:
                raise ValueError("ragged per-position grid sizes")
            n_tiles = counts[0]
            # Per-position subsequence grids always flatten position-first in
            # ome_writers, regardless of axis_order.
            order = GridOrder.POSITION_FIRST
        else:
            raise ValueError("mixed gridded/plain stage positions")

        return GridAxisLayoutKind.REGULAR, n_positions, n_tiles, order


class GridAxisDataWrapper(DataWrapper[Any]):
    """Read-only `ndv.DataWrapper` presenting a flattened writer position axis
    as logical `p`/`g` axes, with no additional pixel copy for a single
    `(p, g)` selection.
    """

    # Never auto-detected by DataWrapper.create(); always constructed
    # explicitly and passed straight into MMArrayViewer.
    PRIORITY: ClassVar[int] = 0

    def __init__(self, raw_view: Any, layout: GridAxisLayout) -> None:
        if layout.kind is GridAxisLayoutKind.NONE:
            raise ValueError("GridAxisDataWrapper requires a non-NONE layout")
        super().__init__(raw_view)
        self._layout = layout
        raw_dims = tuple(raw_view.dims)
        self._raw_dims = raw_dims
        pos_ax = layout.raw_position_axis
        assert pos_ax is not None
        self._raw_position_axis = pos_ax

        exposed = layout.exposed_axes
        self._dims: tuple[str, ...] = (
            raw_dims[:pos_ax] + exposed + raw_dims[pos_ax + 1 :]
        )
        self._p_axis = self._dims.index("p") if "p" in exposed else None
        self._g_axis = self._dims.index("g") if "g" in exposed else None
        # {logical wrapper-axis index: raw-view axis index} for every axis
        # that is *not* the position axis -- passed straight through.
        self._passthrough_raw_axis: dict[int, int] = {
            self._dims.index(name): i for i, name in enumerate(raw_dims) if i != pos_ax
        }

    @classmethod
    def supports(cls, obj: Any) -> bool:
        # Required by the ABC; never used for autodetection since instances
        # are always constructed explicitly (DataWrapper.create() returns an
        # already-built DataWrapper instance unchanged).
        return hasattr(obj, "dims") and hasattr(obj, "coords_changed")

    @property
    def layout(self) -> GridAxisLayout:
        return self._layout

    @property
    def dims(self) -> tuple[str, ...]:
        return self._dims

    @property
    def coords(self) -> Mapping[str, Sequence[Any]]:
        out = dict(self._data.coords)
        pos_name = self._raw_dims[self._raw_position_axis]
        out.pop(pos_name, None)
        # p/g extents are fixed at construction from the *planned* layout,
        # never the live/growing raw coords -- a partial acquisition must
        # never look like a smaller grid.
        if self._p_axis is not None:
            out["p"] = range(self._layout.n_positions)
        if self._g_axis is not None:
            out["g"] = range(self._layout.n_tiles)
        return out

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._data.dtype)

    def isel(self, index: Mapping[int, int | slice]) -> np.ndarray:
        p_val, p_collapse = self._resolve_singleton(
            index.get(self._p_axis) if self._p_axis is not None else None,
            self._layout.n_positions,
        )
        g_val, g_collapse = self._resolve_singleton(
            index.get(self._g_axis) if self._g_axis is not None else None,
            self._layout.n_tiles,
        )
        flat = self._layout.flat_index(p_val, g_val)

        raw_key: list[Any] = [slice(None)] * len(self._raw_dims)
        raw_key[self._raw_position_axis] = flat
        for my_axis, raw_axis in self._passthrough_raw_axis.items():
            if my_axis in index:
                raw_key[raw_axis] = index[my_axis]

        result = np.asarray(self._data[tuple(raw_key)])

        # Restore each retained (non-collapsed) p/g axis, smallest wrapper-
        # axis index first, so each expand_dims' `axis=` stays correct
        # against the growing array.
        for my_axis, collapse in sorted(
            (ax, collapse)
            for ax, collapse in ((self._p_axis, p_collapse), (self._g_axis, g_collapse))
            if ax is not None
        ):
            if not collapse:
                result = np.expand_dims(result, axis=my_axis)
        return result

    @staticmethod
    def _resolve_singleton(req: int | slice | None, n: int) -> tuple[int, bool]:
        """Resolve one p/g request to `(value, collapse)`.

        `None` (axis absent from the request) and a bare `int` (a direct
        caller, e.g. ROI extraction) both collapse the axis. A `slice(v,
        v+1)` (ndv's own resolve/request pipeline) retains it as a size-1
        axis. Any other slice is an explicit multi-value request, which this
        read-only, single-location adapter never supports.
        """
        if req is None:
            return 0, True
        if isinstance(req, int):
            return req, True
        if isinstance(req, slice) and req.stop is not None and req.start is not None:
            if req.stop - req.start == 1 and (req.step or 1) == 1:
                return req.start, False
        raise ValueError(
            "GridAxisDataWrapper does not support multi-value p/g requests "
            f"(got {req!r})"
        )

    def guess_channel_axis(self) -> Hashable | None:
        # Never fall back to the "smallest dimension" heuristic: a small g
        # (or p) axis could easily be smaller than a real channel axis, or
        # the only other axis at all.
        if "c" in self._dims:
            ax = self.normalize_axis_key("c")
            if self.sizes()[self._dims[ax]] <= self.MAX_CHANNELS:
                return ax
        return None

    def guess_z_axis(self) -> Hashable | None:
        # Never fall back to "last axis not in the last two dims": that
        # could pick g, p, or t when there is no genuine Z axis.
        if "z" in self._dims:
            return self.normalize_axis_key("z")
        return None
