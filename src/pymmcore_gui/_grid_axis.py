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
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import useq
from ndv.models import DataWrapper

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence
    from typing import TypeGuard

    from ome_writers import AcquisitionSettings


class GridAxisLayoutKind(str, Enum):
    """What (if anything) a `GridAxisLayout` exposes in place of flattened `p`."""

    NONE = "none"
    """No grid axis: pass the raw, flattened `p` view through unmodified."""
    GRID_ONLY = "grid_only"
    """Only `g` is exposed; there is no independent stage position."""
    REGULAR = "regular"
    """Both `p` and `g` are exposed, as a rectangular `n_positions x n_tiles`."""
    RAGGED = "ragged"
    """Both `p` and `g` are exposed, but positions have differing tile counts
    (including a plain, ungridded position, treated as one tile). `g`'s
    slider range is the largest per-position count; a position with fewer
    tiles reads as blank/zero once `g` exceeds its own count -- never another
    position's real tile."""


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
    # How many of the `n_flat` planned slots the source actually holds. Equal
    # to `n_flat` for a live run and for any complete dataset; smaller only
    # for a reopened, truncated one (a cancelled run). It can never be a
    # *gap*: `ome_writers`' `skip()` writes zero-filled placeholders rather
    # than omitting a slot, so a short source is always a prefix.
    n_stored: int = 0
    # True when the *sequence* has a grid somewhere (global or per-position)
    # that could not be exposed as independent sliders (ragged, well-plate,
    # or otherwise unsupported) -- meaningful only when `kind` is NONE. Lets
    # `_follow_index` tell "no grid at all" (a raw event's "p" already names
    # its correct flattened slot) apart from "grid exists but unsupported"
    # (arrival-order remapping is still required), without which a mixed
    # gridded/plain sequence's plain positions would resolve to the wrong
    # flattened slot once an earlier position's grid tiles have consumed
    # extra slots.
    has_grid: bool = False
    # RAGGED only: per-position tile count and cumulative flat offset (both
    # length n_positions). Precomputed once in `build()`; a table like this is
    # proportional to the number of positions, never to the number of frames.
    tile_counts: tuple[int, ...] = ()
    offsets: tuple[int, ...] = ()

    @property
    def exposed_axes(self) -> tuple[str, ...]:
        """Logical axis labels this layout adds, in slider-presentation order."""
        if self.kind is GridAxisLayoutKind.NONE:
            return ()
        if self.kind is GridAxisLayoutKind.GRID_ONLY:
            return ("g",)
        if self.kind is GridAxisLayoutKind.RAGGED:
            return ("p", "g")  # per-position flattening is always position-first
        return ("p", "g") if self.order is GridOrder.POSITION_FIRST else ("g", "p")

    def flat_index(self, p: int | None, g: int | None) -> int:
        """Map a logical `(p, g)` pair to the writer's flattened position slot.

        Raises `IndexError` for any planned slot that holds no frame:
        `p`/`g` outside the layout's overall range, a RAGGED `g` that is
        within the overall range but past this position's own tile count, or
        a slot beyond a truncated source's stored extent. Callers must treat
        all of these as "no data here" and never substitute another slot's
        tile.
        """
        if self.kind is GridAxisLayoutKind.NONE:
            raise ValueError("flat_index() is not valid for a GridAxisLayoutKind.NONE")
        if self.kind is GridAxisLayoutKind.GRID_ONLY:
            g = 0 if g is None else g
            if not 0 <= g < self.n_tiles:
                raise IndexError(f"g={g} out of range for n_tiles={self.n_tiles}")
            return self._stored(g)
        p = 0 if p is None else p
        g = 0 if g is None else g
        if not 0 <= p < self.n_positions:
            raise IndexError(f"p={p} out of range for n_positions={self.n_positions}")
        if self.kind is GridAxisLayoutKind.RAGGED:
            if not 0 <= g < self.tile_counts[p]:
                raise IndexError(
                    f"g={g} out of range for position {p} "
                    f"(has {self.tile_counts[p]} tiles)"
                )
            return self._stored(self.offsets[p] + g)
        if not 0 <= g < self.n_tiles:
            raise IndexError(f"g={g} out of range for n_tiles={self.n_tiles}")
        if self.order is GridOrder.POSITION_FIRST:
            return self._stored(p * self.n_tiles + g)
        return self._stored(g * self.n_positions + p)

    def _stored(self, flat: int) -> int:
        """`flat`, or `IndexError` if the source stops short of it."""
        if flat >= self.n_stored:
            raise IndexError(
                f"planned slot {flat} is beyond the {self.n_stored} slot(s) "
                "this source actually holds"
            )
        return flat

    @classmethod
    def none(cls, *, has_grid: bool = False) -> GridAxisLayout:
        """The universal fallback: no grid axis, flattened `p` is used as-is."""
        return cls(
            kind=GridAxisLayoutKind.NONE,
            n_positions=0,
            n_tiles=0,
            order=GridOrder.POSITION_FIRST,
            n_flat=0,
            raw_position_axis=None,
            has_grid=has_grid,
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

        spec = cls._derive(sequence, stage_positions, has_global_grid)
        if spec is None:
            # Ragged/unsupported: ome-writers still flattens these tiles into
            # its single position dimension, so events will carry real "g"
            # (and/or "p") values that a naive passthrough would misinterpret.
            return cls.none(has_grid=True)
        kind, n_positions, n_tiles, order, tile_counts = spec
        if n_tiles < 1 or (tile_counts is not None and any(c < 1 for c in tile_counts)):
            return cls.none(has_grid=True)

        offsets: tuple[int, ...] = ()
        if kind is GridAxisLayoutKind.RAGGED:
            assert tile_counts is not None
            expected_n_flat = sum(tile_counts)
            total = 0
            running: list[int] = []
            for count in tile_counts:
                running.append(total)
                total += count
            offsets = tuple(running)
        elif kind is GridAxisLayoutKind.GRID_ONLY:
            expected_n_flat = n_tiles
        else:
            expected_n_flat = n_positions * n_tiles

        # Integrity cross-check against the storage dimension actually built.
        # A source may hold *fewer* slots than planned (a reopened, cancelled
        # run truncates its tail, and `skip()` placeholders mean it can only
        # ever be a prefix) -- those trailing slots keep their identity and
        # read blank. More slots than planned means our derivation disagrees
        # with what the writer really built, so fall back. `count` is used
        # rather than `settings.positions`, which collapses to a single
        # synthetic Position whenever the dimension carries no coords (as a
        # reopened file's derived settings often do).
        n_stored = settings.dimensions[pos_dim_idx].count
        if n_stored is None or not 0 < n_stored <= expected_n_flat:
            return cls.none(has_grid=True)

        return cls(
            kind=kind,
            n_positions=n_positions,
            n_tiles=n_tiles,
            order=order,
            n_flat=expected_n_flat,
            raw_position_axis=pos_dim_idx,
            n_stored=n_stored,
            tile_counts=tile_counts or (),
            offsets=offsets,
        )

    @staticmethod
    def _derive(
        sequence: useq.MDASequence,
        stage_positions: tuple[useq.AbsolutePosition, ...],
        has_global_grid: bool,
    ) -> tuple[GridAxisLayoutKind, int, int, GridOrder, tuple[int, ...] | None] | None:
        """Return `(kind, n_positions, n_tiles, order, tile_counts)`.

        `tile_counts` is `None` except for `RAGGED` (where `n_tiles` is the
        largest per-position count, used only for the slider's overall
        range). Returns `None` when no derivable layout exists (defensive;
        the caller has already ruled out the no-grid-at-all case).
        """
        if not stage_positions:
            # Grid-plan only, no stage positions.
            assert sequence.grid_plan is not None
            n_tiles = len(list(sequence.grid_plan))
            return (
                GridAxisLayoutKind.GRID_ONLY,
                1,
                n_tiles,
                GridOrder.POSITION_FIRST,
                None,
            )

        n_positions = len(stage_positions)
        sub_grids = [
            p.sequence.grid_plan if p.sequence is not None else None
            for p in stage_positions
        ]
        if all(g is None for g in sub_grids):
            if not has_global_grid:
                return None  # pragma: no cover
            assert sequence.grid_plan is not None
            n_tiles = len(list(sequence.grid_plan))
            grid_first = (
                useq.Axis.GRID in sequence.axis_order
                and useq.Axis.POSITION in sequence.axis_order
                and sequence.axis_order.index(useq.Axis.GRID)
                < sequence.axis_order.index(useq.Axis.POSITION)
            )
            order = GridOrder.GRID_FIRST if grid_first else GridOrder.POSITION_FIRST
            return GridAxisLayoutKind.REGULAR, n_positions, n_tiles, order, None

        # At least one position carries its own grid subsequence (uniform,
        # ragged, or mixed with plain positions). Per-position subsequence
        # grids always flatten position-first in ome_writers, regardless of
        # axis_order. A position without its own grid inherits the global
        # one when there is one (ome_writers' subsequence-grid > global-grid
        # > no-grid priority); only with no grid at all does it count as a
        # single tile.
        global_plan = sequence.grid_plan  # == has_global_grid, but narrows
        n_global = len(list(global_plan)) if global_plan is not None else 1
        tile_counts = tuple(
            len(list(g)) if g is not None else n_global for g in sub_grids
        )
        if len(set(tile_counts)) == 1:
            # Uniform: the simple arithmetic case applies.
            n_tiles = tile_counts[0]
            return (
                GridAxisLayoutKind.REGULAR,
                n_positions,
                n_tiles,
                GridOrder.POSITION_FIRST,
                None,
            )
        return (
            GridAxisLayoutKind.RAGGED,
            n_positions,
            max(tile_counts),
            GridOrder.POSITION_FIRST,
            tile_counts,
        )


class GridAxisDataWrapper(DataWrapper[Any]):
    """Read-only `ndv.DataWrapper` presenting a flattened writer position axis.

    Exposes logical `p`/`g` axes instead, with no additional pixel copy for a
    single `(p, g)` selection.

    `raw_view` is either a live `ome_writers` `StreamView` (tuple-indexable)
    or another `DataWrapper` -- e.g. a reopened acquisition's lazy
    `OMETiffWrapper`/`OMEZarrWrapper`, which reads through `isel()` instead.
    Both are indexed with a plain **int** for the flattened position axis, so
    neither takes the `np.stack` path a length-1 slice would trigger.
    """

    # Never auto-detected by DataWrapper.create(); always constructed
    # explicitly and passed straight into MMArrayViewer.
    PRIORITY: ClassVar[int] = 0

    def __init__(self, raw_view: Any, layout: GridAxisLayout) -> None:
        if layout.kind is GridAxisLayoutKind.NONE:
            raise ValueError("GridAxisDataWrapper requires a non-NONE layout")
        super().__init__(raw_view)
        self._layout = layout
        self._source_wrapper = raw_view if isinstance(raw_view, DataWrapper) else None
        raw_dims = tuple(raw_view.dims)
        self._raw_dims = raw_dims
        pos_ax = layout.raw_position_axis
        assert pos_ax is not None
        self._raw_position_axis = pos_ax

        # Axis labels are normalized to `str` here (and in `coords`): ndv
        # only ever matches them by name, and the exposed "p"/"g" labels this
        # wrapper splices in are strings, so mixing in a raw non-str
        # `Hashable` key would make `dims` and `coords` disagree.
        exposed = layout.exposed_axes
        self._dims: tuple[str, ...] = (
            tuple(str(d) for d in raw_dims[:pos_ax])
            + exposed
            + tuple(str(d) for d in raw_dims[pos_ax + 1 :])
        )
        self._p_axis = self._dims.index("p") if "p" in exposed else None
        self._g_axis = self._dims.index("g") if "g" in exposed else None
        # {logical wrapper-axis index: raw-view axis index} for every axis
        # that is *not* the position axis -- passed straight through.
        self._passthrough_raw_axis: dict[int, int] = {
            self._dims.index(str(name)): i
            for i, name in enumerate(raw_dims)
            if i != pos_ax
        }

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[Any]:
        # Required by the ABC. Always False: this wrapper takes a mandatory
        # `layout` argument DataWrapper.create()'s `subclass(data)` can never
        # supply, so answering True here would only make every unwrapped
        # StreamView construction log a spurious "missing argument" warning
        # before falling through to ArrayLikeWrapper. Instances are always
        # constructed explicitly and passed straight into MMArrayViewer;
        # DataWrapper.create() returns an already-built DataWrapper instance
        # unchanged, so autodetection is never actually needed.
        return False

    @property
    def layout(self) -> GridAxisLayout:
        return self._layout

    @property
    def dims(self) -> tuple[str, ...]:
        return self._dims

    @property
    def coords(self) -> Mapping[Hashable, Sequence[Any]]:
        pos_name = self._raw_dims[self._raw_position_axis]
        out: dict[Hashable, Sequence[Any]] = {
            str(k): v for k, v in self._data.coords.items() if k != pos_name
        }
        # p/g extents are fixed at construction from the *planned* layout,
        # never the live/growing raw coords -- a partial acquisition must
        # never look like a smaller grid.
        if self._p_axis is not None:
            out["p"] = range(self._layout.n_positions)
        if self._g_axis is not None:
            out["g"] = range(self._layout.n_tiles)
        return out

    @property
    def dtype(self) -> np.dtype[Any]:
        return cast("np.dtype[Any]", np.dtype(self._data.dtype))

    def isel(self, index: Mapping[int, int | slice]) -> np.ndarray:
        p_val, p_collapse = self._resolve_singleton(
            index.get(self._p_axis) if self._p_axis is not None else None
        )
        g_val, g_collapse = self._resolve_singleton(
            index.get(self._g_axis) if self._g_axis is not None else None
        )

        raw_key: list[Any] = [slice(None)] * len(self._raw_dims)
        for my_axis, raw_axis in self._passthrough_raw_axis.items():
            if my_axis in index:
                raw_key[raw_axis] = index[my_axis]

        try:
            flat = self._layout.flat_index(p_val, g_val)
        except IndexError:
            # RAGGED only in practice: g is within the slider's overall
            # range but exceeds this specific position's own tile count.
            # There is no real frame here, so nothing is read at all -- the
            # shape is derived from the request instead. Never substitute
            # another position's real tile, and never pay for a decode just
            # to learn the shape of a frame that was never acquired.
            result = np.zeros(self._blank_shape(raw_key), dtype=self.dtype)
        else:
            raw_key[self._raw_position_axis] = flat
            result = self._read_raw(raw_key)

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

    def _blank_shape(self, raw_key: list[Any]) -> tuple[int, ...]:
        """Shape `_read_raw(raw_key)` would return, without reading anything.

        Raw extents are re-read per call rather than cached: a live
        `StreamView` grows along `t` as the acquisition proceeds, and a
        missing tile still has to match the shape its acquired siblings
        return *now*.
        """
        coords = self._data.coords
        shape: list[int] = []
        for axis, key in enumerate(raw_key):
            if axis == self._raw_position_axis or not isinstance(key, slice):
                continue  # an int index collapses its axis away
            shape.append(len(range(*key.indices(len(coords[self._raw_dims[axis]])))))
        return tuple(shape)

    def _read_raw(self, raw_key: list[Any]) -> np.ndarray:
        """Read one request from the wrapped source, in its own axis order."""
        if (src := self._source_wrapper) is not None:
            return np.asarray(src.isel(dict(enumerate(raw_key))))
        return np.asarray(self._data[tuple(raw_key)])

    @staticmethod
    def _resolve_singleton(req: int | slice | None) -> tuple[int, bool]:
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
