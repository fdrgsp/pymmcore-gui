"""Which tiles form which mosaic, and reading them from the saved acquisition.

GUI side (`plan_groups`, `build_job`): from the `MDASequence` alone, list each
mosaic's tiles with their index along the saved store's position axis. The
order mirrors how ome-writers flattens (stage position, grid point) into that
axis (`ome_writers._useq._build_positions`); a test keeps the two in step.

Worker side (`open_tiles`): open the store with the GUI's own OME wrappers and
hand out one lazy dask array per tile.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pymmcore_gui._stitching import _geometry
from pymmcore_gui._stitching._models import (
    Alignment,
    GroupV1,
    StitchJobV1,
    StitchSettingsV1,
    TileV1,
    tmp_path_for,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import dask.array
    import useq

    from pymmcore_gui._ome_tiff_wrapper import OMETiffWrapper
    from pymmcore_gui._ome_zarr_wrapper import OMEZarrWrapper

STITCHED_SUFFIX = "_stitched"
OUTPUT_EXTENSION = ".ome.zarr"
_KNOWN_EXTENSIONS = (".ome.zarr", ".ome.tiff", ".ome.tif", ".zarr", ".tiff", ".tif")


# ------------------------------ GUI side ------------------------------


@dataclass(frozen=True)
class PlannedGroup:
    name: str
    tiles: tuple[TileV1, ...]


def _is_stitchable_grid(grid: Any) -> bool:
    """A grid whose points have rows and columns, and more than one of them.

    Random points carry no row/column and are not meant to overlap.
    """
    if grid is None:
        return False
    points = list(grid)
    return len(points) > 1 and all(
        p.grid_row is not None and p.grid_col is not None for p in points
    )


def _z_origin(sequence: useq.MDASequence, pos_z: float | None) -> float:
    """Z of the first plane, as useq computes it for this position."""
    if sequence.z_plan is None:
        return float(pos_z or 0.0)
    offsets = list(sequence.z_plan)
    first = float(offsets[0]) if offsets else 0.0
    if sequence.z_plan.is_relative:
        return float(pos_z or 0.0) + first
    return first


def z_step(sequence: useq.MDASequence) -> float | None:
    """Signed distance between consecutive planes, or None without a stack."""
    if sequence.z_plan is None:
        return None
    offsets = [float(v) for v in sequence.z_plan]
    if len(offsets) < 2:
        return None
    return offsets[1] - offsets[0]


def plan_groups(sequence: useq.MDASequence) -> list[PlannedGroup]:
    """The mosaics a finished `sequence` produces (§3.6), in store order.

    - a global grid without stage positions: one mosaic;
    - a global relative grid x N stage positions: one mosaic per position;
    - per-position sub-sequence grids: one per position that has a grid.

    Well plates (and anything ome-writers would not save as a flat position
    axis) give no group: they are a later phase.
    """
    import useq

    if isinstance(sequence.stage_positions, useq.WellPlatePlan):
        return []

    global_grid = list(sequence.grid_plan) if sequence.grid_plan else None
    positions = list(sequence.stage_positions)

    if not positions:
        if not _is_stitchable_grid(sequence.grid_plan):
            return []
        assert global_grid is not None
        tiles = tuple(
            TileV1(
                position_index=i,
                name=f"{i:04d}",
                stage_x_um=float(gp.x or 0.0),
                stage_y_um=float(gp.y or 0.0),
                stage_z_um=_z_origin(sequence, gp.z),
            )
            for i, gp in enumerate(global_grid)
        )
        return [PlannedGroup(name="", tiles=tiles)]

    def _tile(index: int, pos: Any, gp: Any) -> TileV1:
        if gp.is_relative:
            x = (pos.x or 0.0) + (gp.x or 0.0)
            y = (pos.y or 0.0) + (gp.y or 0.0)
        else:
            x, y = gp.x or 0.0, gp.y or 0.0
        return TileV1(
            position_index=index,
            stage_x_um=float(x),
            stage_y_um=float(y),
            stage_z_um=_z_origin(sequence, pos.z),
        )

    def _name(p_idx: int, pos: useq.Position) -> str:
        return pos.name or f"p{p_idx}"

    order = sequence.axis_order
    grid_first = (
        global_grid is not None
        and "g" in order
        and "p" in order
        and order.index("g") < order.index("p")
    )
    tiles_by_pos: dict[int, list[TileV1]] = {}
    grids_by_pos: dict[int, Any] = {}
    index = 0
    if grid_first and global_grid:
        for gp in global_grid:
            for p_idx, pos in enumerate(positions):
                tiles_by_pos.setdefault(p_idx, []).append(_tile(index, pos, gp))
                grids_by_pos[p_idx] = sequence.grid_plan
                index += 1
    else:
        for p_idx, pos in enumerate(positions):
            grid: Any
            if pos.sequence is not None and pos.sequence.grid_plan is not None:
                grid = pos.sequence.grid_plan
            else:
                grid = sequence.grid_plan
            if grid is not None and (points := list(grid)):
                grids_by_pos[p_idx] = grid
                for gp in points:
                    tiles_by_pos.setdefault(p_idx, []).append(_tile(index, pos, gp))
                    index += 1
            else:
                index += 1  # a single field of view: not part of any mosaic

    groups: list[PlannedGroup] = []
    multiple = len(positions) > 1
    for p_idx, pos in enumerate(positions):
        if not _is_stitchable_grid(grids_by_pos.get(p_idx)):
            continue
        pos_tiles = tiles_by_pos[p_idx]
        group_name = _name(p_idx, pos) if multiple else (pos.name or "")
        groups.append(
            PlannedGroup(
                name=group_name,
                tiles=tuple(
                    t.model_copy(update={"name": f"{group_name or 'tile'}_{i}"})
                    for i, t in enumerate(pos_tiles)
                ),
            )
        )
    return groups


def acquisition_stem(source: Path) -> str:
    """`expt_001` for `expt_001.ome.zarr`, `expt_001.ome.tiff` or `expt_001/`."""
    name = source.name
    for ext in _KNOWN_EXTENSIONS:
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name).strip("_") or "pos"


def unique_output(
    directory: Path, stem: str, group_name: str = "", taken: set[Path] | None = None
) -> Path:
    """`<dir>/<stem>_stitched[_<group>].ome.zarr`, never an existing path (§3.5)."""
    base = f"{stem}{STITCHED_SUFFIX}"
    if group_name:
        base += f"_{_safe(group_name)}"
    taken = taken or set()
    n = 1
    while True:
        name = base if n == 1 else f"{base}-{n}"
        path = directory / f"{name}{OUTPUT_EXTENSION}"
        if path not in taken and not path.exists() and not tmp_path_for(path).exists():
            return path
        n += 1


def default_threads() -> int:
    """Half the CPU cores, at least one (§4.4)."""
    return max(1, (os.cpu_count() or 2) // 2)


def build_job(
    *,
    job_id: str,
    sequence: useq.MDASequence,
    source: Path,
    pixel_size_um: float,
    linear_part: np.ndarray,
    tile_shape_yx: tuple[int, int] | None = None,
    settings: StitchSettingsV1 | None = None,
    parent_pid: int | None = None,
    parent_create_time: float | None = None,
    memory_cap_bytes: int | None = None,
) -> StitchJobV1 | None:
    """The job for a finished, saved `sequence`, or None if it has no mosaic.

    When `tile_shape_yx` is known and no two tiles overlap, registration
    can't work, so alignment falls back to stage positions only (§3.1).
    """
    groups = plan_groups(sequence)
    if not groups:
        return None
    source = Path(os.path.abspath(source))
    settings = settings or StitchSettingsV1()
    linear = np.asarray(linear_part, dtype=float)

    if settings.alignment == Alignment.REGISTER and tile_shape_yx is not None:
        overlaps = [
            _geometry.overlap_fraction(
                linear,
                pixel_size_um,
                tile_shape_yx,
                [(t.stage_x_um, t.stage_y_um) for t in g.tiles],
            )
            for g in groups
        ]
        if max(overlaps) <= 0:
            settings = settings.model_copy(update={"alignment": Alignment.STAGE})

    stem = acquisition_stem(source)
    taken: set[Path] = set()
    job_groups = []
    for g in groups:
        out = unique_output(source.parent, stem, g.name, taken)
        taken.add(out)
        job_groups.append(GroupV1(name=g.name, tiles=list(g.tiles), output=out))

    step = z_step(sequence)
    return StitchJobV1(
        job_id=job_id,
        source=source,
        groups=job_groups,
        pixel_size_um=pixel_size_um,
        linear_part=(
            (float(linear[0, 0]), float(linear[0, 1])),
            (float(linear[1, 0]), float(linear[1, 1])),
        ),
        z_step_um=abs(step) if step else None,
        channel_names=[c.config for c in sequence.channels],
        settings=settings,
        threads=settings.worker_threads or default_threads(),
        memory_cap_bytes=memory_cap_bytes,
        parent_pid=parent_pid,
        parent_create_time=parent_create_time,
    )


# ----------------------------- worker side -----------------------------


SPATIAL_AND_NONSPATIAL = ("t", "c", "z", "y", "x")


class _LazyTile:
    """Array-like view of one position of a wrapper, for `dask.array.from_array`."""

    def __init__(
        self,
        wrapper: OMETiffWrapper | OMEZarrWrapper,
        p_axis: int | None,
        p_index: int,
        shape: tuple[int, ...],
        dtype: np.dtype,
        lock: threading.Lock | None,
    ) -> None:
        self._wrapper = wrapper
        self._p_axis = p_axis
        self._p_index = p_index
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)
        self._lock = lock

    def __getitem__(self, key: Any) -> np.ndarray:
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (self.ndim - len(key))
        index: dict[int, int | slice] = {}
        outer = 0
        for axis in range(self.ndim + (self._p_axis is not None)):
            if axis == self._p_axis:
                index[axis] = self._p_index
                continue
            index[axis] = key[outer]
            outer += 1
        if self._lock is None:
            return np.asarray(self._wrapper.isel(index))
        with self._lock:
            return np.asarray(self._wrapper.isel(index))


@dataclass
class OpenedTiles:
    """The saved acquisition, opened for reading tiles."""

    wrapper: OMETiffWrapper | OMEZarrWrapper
    dims: tuple[str, ...]
    """Per-tile dims, a subset of `t, c, z, y, x` in that order."""
    shape: tuple[int, ...]
    dtype: np.dtype
    channel_names: list[str]
    n_positions: int
    _p_axis: int | None
    _lock: threading.Lock | None

    @property
    def shape_yx(self) -> tuple[int, int]:
        return self.shape[-2], self.shape[-1]

    def sizes(self) -> dict[str, int]:
        return dict(zip(self.dims, self.shape, strict=True))

    def tile(self, position_index: int) -> dask.array.Array:
        """One tile as a lazy dask array, one chunk per full plane."""
        import dask.array as da

        if not 0 <= position_index < self.n_positions:
            raise IndexError(
                f"position {position_index} not in the acquisition "
                f"({self.n_positions} positions)"
            )
        src = _LazyTile(
            self.wrapper,
            self._p_axis,
            position_index,
            self.shape,
            self.dtype,
            self._lock,
        )
        chunks: Any = tuple(
            size if dim in ("y", "x") else 1
            for dim, size in zip(self.dims, self.shape, strict=True)
        )
        arr = da.from_array(  # type: ignore[no-untyped-call]
            src,
            chunks=chunks,
            meta=np.empty((0,) * len(self.shape), dtype=self.dtype),
            name=f"tile-{id(self.wrapper)}-{position_index}",
            asarray=False,
            lock=False,
        )
        return cast("dask.array.Array", arr)

    def close(self) -> None:
        self.wrapper.close()


def open_tiles(source: Path) -> OpenedTiles:
    """Open a saved acquisition for reading tiles. Raises ValueError if unusable."""
    from pymmcore_gui._ome_tiff_wrapper import OMETiffWrapper
    from pymmcore_gui._ome_zarr_wrapper import OMEZarrWrapper

    wrapper: OMETiffWrapper | OMEZarrWrapper
    lock: threading.Lock | None = None
    if OMEZarrWrapper.supports(source):
        wrapper = OMEZarrWrapper(source)
    elif OMETiffWrapper.supports(source):
        wrapper = OMETiffWrapper(source)
        # one tifffile handle, read from several threads
        lock = threading.Lock()
    else:
        raise ValueError(f"Not a saved acquisition this app can read: {source}")

    dims = [str(d).lower() for d in wrapper.dims]
    coords = wrapper.coords
    sizes = [len(coords[d]) for d in wrapper.dims]
    p_axis = dims.index("p") if "p" in dims else None
    n_positions = sizes[p_axis] if p_axis is not None else 1
    tile_dims = [d for d in dims if d != "p"]
    tile_shape = [s for d, s in zip(dims, sizes, strict=True) if d != "p"]
    if tile_dims[-2:] != ["y", "x"] or not set(tile_dims) <= set(
        SPATIAL_AND_NONSPATIAL
    ):
        wrapper.close()
        raise ValueError(f"Unsupported axes {tuple(dims)} in {source}")
    if tile_dims != [d for d in SPATIAL_AND_NONSPATIAL if d in tile_dims]:
        wrapper.close()
        raise ValueError(f"Unexpected axis order {tuple(dims)} in {source}")

    c_key = next((k for k in wrapper.dims if str(k).lower() == "c"), None)
    channels = [str(v) for v in coords[c_key]] if c_key is not None else ["0"]
    if len(set(channels)) != len(channels):
        channels = [f"{name} ({i})" for i, name in enumerate(channels)]
    return OpenedTiles(
        wrapper=wrapper,
        dims=tuple(tile_dims),
        shape=tuple(tile_shape),
        dtype=np.dtype(wrapper.dtype),
        channel_names=channels,
        n_positions=n_positions,
        _p_axis=p_axis,
        _lock=lock,
    )


def tile_centers(tiles: Sequence[TileV1]) -> list[tuple[float, float]]:
    return [(t.stage_x_um, t.stage_y_um) for t in tiles]
