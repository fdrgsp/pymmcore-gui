"""Registration and fusion adapter for multiview-stitcher."""

from __future__ import annotations

import contextlib
import logging
import math
import os
import threading
import time
import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pymmcore_gui._stitching import _geometry
from pymmcore_gui._stitching._inputs import OpenedTiles, locate_tiles, open_tiles
from pymmcore_gui._stitching._models import (
    METADATA_NAMESPACE,
    PROVENANCE_KEY,
    Alignment,
    Blending,
    GroupResultV1,
    GroupV1,
    Phase,
    ProgressEvent,
    ProvenanceV1,
    StitchError,
    StitchJobV1,
    StitchResultV1,
    is_tiff_output,
    same_source,
)
from pymmcore_gui._stitching._paths import remove_path, unique_path

if TYPE_CHECKING:
    import networkx as nx

logger = logging.getLogger(__name__)

Reporter = Callable[[ProgressEvent], None]

REG_KEY = "registered"
STAGE_KEY = "stage"

OUTPUT_CHUNK_2D = 2048
"""Output block side in 2D. Measured on a 10x10 grid of 2048² tiles read from
disk: 1024 -> 75 s of fusion, 2048 -> 27 s, 4096 -> 45 s (too few
blocks to keep the threads busy)."""
_PYRAMID_STEPS = 1000
BLOCK_MEMORY_FACTOR = 2
OUTPUT_CHUNK_3D = {"z": 8, "y": 512, "x": 512}


def _mvs_version() -> str:
    try:
        return version("multiview-stitcher")
    except PackageNotFoundError:  # pragma: no cover
        return "unknown"


def source_fingerprint(source: Path) -> dict[str, Any]:
    """Enough about the raw data to notice it moved or changed."""
    source = Path(source)
    st = source.stat()
    if source.is_file():
        return {
            "path": str(source),
            "kind": "file",
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
        }
    n_files = total = latest = 0
    for root, _dirs, files in os.walk(source):
        for name in files:
            with contextlib.suppress(OSError):
                fst = os.stat(os.path.join(root, name))
                n_files += 1
                total += fst.st_size
                latest = max(latest, fst.st_mtime_ns)
    return {
        "path": str(source),
        "kind": "directory",
        "n_files": n_files,
        "size": total,
        "mtime_ns": latest,
    }


class _Progress:
    """Turns engine steps into `ProgressEvent`s for one group."""

    def __init__(self, report: Reporter, group_index: int, n_groups: int) -> None:
        self._report = report
        self._group_index = group_index
        self._n_groups = n_groups
        self._lock = threading.Lock()
        self.phase = Phase.STARTING
        self.done = 0
        self.total = 0
        self._message = ""
        self._level_start = self._level_share = 0.0
        self._level_total = 1
        self._level_done = 0

    def set(
        self, phase: Phase, total: int = 0, message: str = "", done: int = 0
    ) -> None:
        with self._lock:
            self.phase, self.total, self.done = phase, total, done
            self._message = message
            self._emit()

    def set_pyramid_level(self, level: int, total: int) -> None:
        """Start pyramid level `level` (1-based) of `total` blocks.

        How many levels follow isn't known in advance, so level k fills the
        k-th half of what remains of the bar: progress never goes back.
        """
        with self._lock:
            self.phase = Phase.PYRAMID
            self._message = f"Writing pyramid level {level}"
            self._level_start = 1 - 0.5 ** (level - 1)
            self._level_share = 0.5**level
            self._level_total = max(total, 1)
            self._level_done = 0
            self._emit_pyramid()

    def advance(self, n: int = 1) -> None:
        with self._lock:
            if self.phase == Phase.PYRAMID:
                self._level_done += n
                self._emit_pyramid()
                return
            self.done += n
            self._emit()

    def _emit_pyramid(self) -> None:
        step = min(self._level_done / self._level_total, 1.0)
        self.total = _PYRAMID_STEPS
        self.done = round(
            (self._level_start + self._level_share * step) * _PYRAMID_STEPS
        )
        self._emit()

    def _emit(self) -> None:
        msg = self._message
        if not msg and self.total:
            unit = {Phase.REGISTERING: " pairs", Phase.FUSING: " blocks"}.get(
                self.phase, ""
            )
            msg = f"{self.phase.label} {self.done}/{self.total}{unit}"
        if msg and self._n_groups > 1:
            msg = f"Mosaic {self._group_index + 1}/{self._n_groups}: {msg}"
        self._report(
            ProgressEvent(
                phase=self.phase,
                group_index=self._group_index,
                n_groups=self._n_groups,
                done=self.done,
                total=self.total,
                message=msg,
            )
        )


@contextlib.contextmanager
def _track_block_batches(progress: _Progress) -> Iterator[None]:
    """Learn how many blocks fusion and each pyramid level will process.

    `fuse()` takes no progress callback; it iterates
    `misc_utils.ndindex_batches(nblocks, n_batch)` once for the fused level and
    once per pyramid level. Wrapping it gives the totals; `batch_func` counts
    finished blocks.
    """
    from multiview_stitcher import misc_utils

    original = misc_utils.ndindex_batches
    level = 0

    def wrapped(nblocks: Any, n_batch: Any) -> Any:
        nonlocal level
        total = int(np.prod(nblocks))
        if level == 0:
            progress.set(Phase.FUSING, total=total)
        else:
            progress.set_pyramid_level(level, total)
        level += 1
        return original(nblocks, n_batch)

    misc_utils.ndindex_batches = wrapped
    try:
        yield
    finally:
        misc_utils.ndindex_batches = original


def _sim(
    data: Any,
    dims: list[str],
    spacing: dict[str, float],
    affine: np.ndarray,
    channels: list[str],
    t_coords: list[int] | None,
) -> Any:
    from multiview_stitcher import param_utils
    from multiview_stitcher import spatial_image_utils as si_utils

    return si_utils.get_sim_from_array(
        data,
        dims=dims,
        scale=spacing,
        translation=dict.fromkeys(spacing, 0.0),
        affine=param_utils.affine_to_xaffine(affine),
        transform_key=STAGE_KEY,
        # Always named: without a "c" axis (OME-TIFF drops a single channel)
        # the sim gets one anyway, and the registration channel is looked
        # up by these names.
        c_coords=channels,
        t_coords=t_coords,
    )


def _msim(sim: Any) -> Any:
    from multiview_stitcher import msi_utils

    return msi_utils.get_msim_from_sim(sim, scale_factors=[])


class _GroupImages:
    """The group's tiles as multiview-stitcher images, placed by stage position."""

    def __init__(self, job: StitchJobV1, group: GroupV1, opened: OpenedTiles):
        self.job = job
        self.group = group
        self.opened = opened
        self.linear = np.array(job.linear_part, dtype=float)
        p = job.pixel_size_um
        sizes = opened.sizes()
        self.has_z = sizes.get("z", 1) > 1
        self.dims = list(opened.dims)
        self.channels = opened.channel_names
        named = job.channel_names
        if len(named) == len(self.channels) and len(set(named)) == len(named):
            if "c" not in self.dims or self.channels == [
                str(i) for i in range(len(named))
            ]:
                self.channels = list(named)
        self.t_coords = list(range(sizes["t"])) if "t" in sizes else None
        self.z_step = abs(job.z_step_um or 1.0)
        self.spacing = {"y": p, "x": p}
        if self.has_z:
            self.spacing = {"z": self.z_step, **self.spacing}
        self.arrays = []
        self.affines = []
        indices = locate_tiles(group.tiles, opened.stage_xy)
        if moved := sum(
            i != t.position_index for i, t in zip(indices, group.tiles, strict=True)
        ):
            logger.info("%d tiles found by stage position, not store order", moved)
        for tile, index in zip(group.tiles, indices, strict=True):
            arr = opened.tile(index)
            if "z" in self.dims and not self.has_z:
                arr = arr[(slice(None),) * self.dims.index("z") + (0,)]
            self.arrays.append(arr)
            self.affines.append(
                _geometry.tile_affine(
                    self.linear,
                    p,
                    opened.shape_yx,
                    (tile.stage_x_um, tile.stage_y_um),
                    tile.stage_z_um if self.has_z else None,
                )
            )
            if self.has_z and (job.z_step_um or 0) < 0:
                self.affines[-1][0, 0] = -1
        self.data_dims = [d for d in self.dims if d != "z" or self.has_z]
        self.msims = [
            _msim(
                _sim(a, self.data_dims, self.spacing, m, self.channels, self.t_coords)
            )
            for a, m in zip(self.arrays, self.affines, strict=True)
        ]

    def projection_msims(self) -> list[Any]:
        """2D (y, x) images from each tile's max projection, for registration."""
        z_axis = self.data_dims.index("z")
        dims2 = [d for d in self.data_dims if d != "z"]
        spacing2 = {"y": self.spacing["y"], "x": self.spacing["x"]}
        out = []
        for arr, m in zip(self.arrays, self.affines, strict=True):
            proj = arr.max(axis=z_axis)  # type: ignore[no-untyped-call]
            m2 = np.eye(3)
            m2[:2, :2] = m[1:3, 1:3]
            m2[:2, 2] = m[1:3, 3]
            out.append(
                _msim(_sim(proj, dims2, spacing2, m2, self.channels, self.t_coords))
            )
        return out

    def centers(self, ndim: int) -> np.ndarray:
        """Each tile's centre at its stage position, in (z,) y, x order."""
        rows = []
        for tile in self.group.tiles:
            yx = [tile.stage_y_um, tile.stage_x_um]
            if ndim == 3:
                nz = self.opened.sizes().get("z", 1)
                yx = [tile.stage_z_um + (nz - 1) / 2 * (self.job.z_step_um or 1), *yx]
            rows.append(yx)
        return np.array(rows)


def _pairwise_executor(
    threads: int, progress: _Progress, tried: list[int]
) -> Callable[[list[Any], list[tuple[int, int]], dict[str, Any]], list[Any]]:
    """Run pairwise registrations on our own threads, counting finished pairs.

    `tried` receives the number of pairs: upstream only reports the pairs
    that survive the quality filter.
    """
    from multiview_stitcher import registration

    def run(msims: list[Any], edges: list[tuple[int, int]], kwargs: dict) -> list:
        tried.append(len(edges))
        progress.set(Phase.REGISTERING, total=len(edges))

        def one(edge: tuple[int, int]) -> Any:
            ds = registration.register_pair_of_msims_over_time(
                msims[edge[0]], msims[edge[1]], **kwargs
            )
            result = ds.compute(scheduler="synchronous")
            progress.advance()
            return result

        with ThreadPoolExecutor(max_workers=threads) as pool:
            return list(pool.map(one, edges))

    return run


def _select_first_t(msims: list[Any]) -> list[Any]:
    from multiview_stitcher import msi_utils

    out = []
    for msim in msims:
        sim = msi_utils.get_sim_from_msim(msim)
        if "t" in sim.dims and sim.sizes["t"] > 1:
            msim = msi_utils.multiscale_sel_coords(
                msim, {"t": sim.coords["t"].values[0]}
            )
        out.append(msim)
    return out


def _corrections_by_t(params: list[Any]) -> dict[Any, list[np.ndarray]]:
    """Per timepoint (None = all), each tile's correction as a numpy matrix."""
    first = params[0]
    if "t" in first.dims and first.sizes["t"] > 1:
        return {
            t: [np.asarray(p.sel(t=t)) for p in params]
            for t in first.coords["t"].values
        }
    return {None: [np.asarray(p.isel(t=0) if "t" in p.dims else p) for p in params]}


def _register(
    images: _GroupImages, job: StitchJobV1, progress: _Progress
) -> tuple[dict[Any, list[np.ndarray]], dict[str, Any]]:
    """Corrections (applied after the stage placement) and what happened."""
    import networkx as nx
    from multiview_stitcher import mv_graph, registration

    settings = job.settings
    project = images.has_z and settings.z_max_projection
    reg_msims = images.projection_msims() if project else images.msims
    if not settings.register_every_timepoint:
        reg_msims = _select_first_t(reg_msims)

    binning = settings.reg_binning or _geometry.auto_binning(images.opened.shape_yx)
    reg_binning = {"y": binning, "x": binning}
    if images.has_z and not project:
        reg_binning["z"] = 1

    reg_channel = settings.reg_channel
    if reg_channel not in images.channels:
        reg_channel = images.channels[0]

    n_tiles = len(images.msims)
    ndim = 2 if project or not images.has_z else 3
    centers = images.centers(ndim)
    tried: list[int] = []
    try:
        res = cast(
            "dict[str, Any]",
            registration.register(
                reg_msims,
                transform_key=STAGE_KEY,
                reg_channel=reg_channel,
                registration_binning=reg_binning,
                pre_registration_pruning_method="keep_axis_aligned",
                groupwise_resolution_method=_geometry.resolution_method(n_tiles),
                post_registration_do_quality_filter=settings.reject_poor_matches,
                post_registration_quality_threshold=settings.quality_threshold,
                pairwise_executor=_pairwise_executor(job.threads, progress, tried),
                return_dict=True,
            ),
        )
    except mv_graph.NotEnoughOverlapError:
        # No usable edges remain; preserve every tile's stage placement.
        logger.warning("registration found no usable overlap")
        by_t: dict[Any, list[np.ndarray]] = {
            None: [np.eye(ndim + 1) for _ in range(n_tiles)]
        }
        components = [[i] for i in range(n_tiles)]
        accepted = 0
    else:
        graph: nx.Graph = res["pairwise_registration"]["graph"]
        graph.add_nodes_from(range(n_tiles))
        components = [sorted(c) for c in nx.connected_components(graph)]
        accepted = graph.number_of_edges()
        by_t = {
            t: _geometry.anchor_corrections(corr, centers, components)
            for t, corr in _corrections_by_t(res["params"]).items()
        }
    n_pairs = sum(tried)

    if project:
        by_t = {
            t: [_geometry.lift_2d_to_3d(c) for c in corr] for t, corr in by_t.items()
        }
    info = {
        "n_pairs": n_pairs,
        "rejected_pairs": max(n_pairs - accepted, 0),
        "isolated_tiles": sum(1 for c in components if len(c) == 1),
        "components": components,
        "binning": binning,
        "reg_channel": reg_channel,
        "max_projection": project,
    }
    return by_t, info


def _reusable(
    group: GroupV1, images: _GroupImages, fingerprint: dict[str, Any]
) -> dict[Any, list[np.ndarray]] | None:
    """The saved registration, if it still fits this source and these tiles."""
    saved = group.saved_registration
    if saved is None or not same_source(saved.source_fingerprint, fingerprint):
        return None
    by_t: dict[Any, list[np.ndarray]] = {}
    try:
        for key, stored in saved.transforms.items():
            t = None if key == "all" else int(key)
            by_t[t] = [np.asarray(m, dtype=float) for m in stored]
    except (TypeError, ValueError, OverflowError):
        return None
    n_dims = len(images.spacing) + 1
    expected_times = set(images.t_coords or [0])
    if set(by_t) != {None} and set(by_t) != expected_times:
        return None
    for matrices in by_t.values():
        if len(matrices) != len(group.tiles):
            return None
        if any(
            m.shape != (n_dims, n_dims)
            or not np.isfinite(m).all()
            or not np.allclose(m[-1], np.eye(n_dims)[-1])
            or np.linalg.det(m[:-1, :-1]) == 0
            for m in matrices
        ):
            return None
    logger.info("Reusing the saved registration of %d tiles", len(group.tiles))
    return by_t


def _apply_corrections(images: _GroupImages, by_t: dict[Any, list[np.ndarray]]) -> None:
    import xarray as xr
    from multiview_stitcher import msi_utils, param_utils

    for i, msim in enumerate(images.msims):
        if list(by_t) == [None]:
            xparams = param_utils.affine_to_xaffine(by_t[None][i])
        else:
            ts = list(by_t)
            xparams = xr.concat(
                [param_utils.affine_to_xaffine(by_t[t][i]) for t in ts], dim="t"
            ).assign_coords(t=ts)
        msi_utils.set_affine_transform(
            msim, xparams, transform_key=REG_KEY, base_transform_key=STAGE_KEY
        )


def _fusion_func(blending: Blending) -> Callable[..., Any]:
    from multiview_stitcher import fusion

    funcs: dict[Blending, Callable[..., Any]] = {
        Blending.SMOOTH: fusion.weighted_average_fusion,
        Blending.MAX: fusion.max_fusion,
        Blending.AVERAGE: fusion.simple_average_fusion,
    }
    return funcs[blending]


def _blocks_in_flight(job: StitchJobV1, images: _GroupImages) -> int:
    """Blocks fused at once: the thread budget, lowered to fit the memory cap.

    A block holds its float output plus the planes it reads from every tile
    it touches (one dask chunk per full plane). Blocks and the grid of tile
    centres are both aligned with the stage axes, so the tiles are counted
    there: per axis, a window of width `w` holds at most `floor(w / step) + 1`
    centres of a grid with that step.
    """
    n = job.threads
    if job.memory_cap_bytes is None:
        return n
    itemsize = images.opened.dtype.itemsize
    rows, cols = images.opened.shape_yx
    if images.has_z:
        side = (OUTPUT_CHUNK_3D["y"], OUTPUT_CHUNK_3D["x"])
        out_px = OUTPUT_CHUNK_3D["z"] * side[0] * side[1]
        planes = OUTPUT_CHUNK_3D["z"] + 2
    else:
        side = (OUTPUT_CHUNK_2D, OUTPUT_CHUNK_2D)
        out_px = OUTPUT_CHUNK_2D**2
        planes = 1
    centers = np.array([(t.stage_x_um, t.stage_y_um) for t in images.group.tiles])
    centers, counts = np.unique(centers.round(3), axis=0, return_counts=True)
    block_um = np.array(side[::-1]) * job.pixel_size_um
    tile_um = np.abs(images.linear) @ np.array([cols, rows])  # (x, y) footprint
    touched = int(counts.max())
    for axis in range(2):
        gaps = np.diff(np.unique(centers[:, axis]))
        if gaps.size:
            window = block_um[axis] + tile_um[axis]
            touched *= math.floor(window / gaps.min()) + 1
    touched = min(touched, len(images.arrays))
    estimate = out_px * 8 + touched * planes * rows * cols * (itemsize + 4)
    # multiview-stitcher's own intermediates (float copies, blending weights)
    # about double that: 2048² tiles in 2048² blocks measured ~510 MB a block
    per_block = BLOCK_MEMORY_FACTOR * estimate
    return max(1, min(n, job.memory_cap_bytes // max(per_block, 1)))


def _fuse(
    images: _GroupImages,
    job: StitchJobV1,
    transform_key: str,
    out_url: Path,
    progress: _Progress,
) -> None:
    from multiview_stitcher import fusion, msi_utils

    settings = job.settings
    workers = _blocks_in_flight(job, images)
    pool = ThreadPoolExecutor(max_workers=workers)

    def batch_func(fuse_block: Callable[[Any], Any], block_ids: list[Any]) -> None:
        for _ in pool.map(fuse_block, block_ids):
            progress.advance()

    sdims = ["z", "y", "x"] if images.has_z else ["y", "x"]
    output_spacing = {d: images.spacing[d] for d in sdims}
    chunks: int | dict[str, int] = (
        dict(OUTPUT_CHUNK_3D) if images.has_z else OUTPUT_CHUNK_2D
    )
    extra: dict[str, Any] = {}
    if settings.blend_width_um is not None:
        extra["blending_widths"] = dict.fromkeys(sdims, settings.blend_width_um)

    sims = [msi_utils.get_sim_from_msim(m) for m in images.msims]
    try:
        with _track_block_batches(progress):
            fusion.fuse(
                sims,
                transform_key=transform_key,
                fusion_func=_fusion_func(settings.blending),
                output_spacing=output_spacing,
                output_chunksize=chunks,
                output_zarr_url=str(out_url),
                zarr_options={"ome_zarr": True},
                batch_options={"batch_func": batch_func, "n_batch": workers * 4},
                **extra,
            )
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _provenance(
    job: StitchJobV1,
    group: GroupV1,
    transforms: dict[Any, list[np.ndarray]] | None,
    info: dict[str, Any],
    fingerprint: dict[str, Any],
    origin_xy_um: tuple[float, float] | None,
) -> dict[str, Any]:
    """How the mosaic was made, as JSON-ready data."""

    def _t_key(t: Any) -> str:
        return "all" if t is None else str(int(t))

    return ProvenanceV1(
        source=job.source,
        source_fingerprint=fingerprint,
        group=group.name,
        tiles=group.tiles,
        settings=job.settings,
        pixel_size_um=job.pixel_size_um,
        linear_part=job.linear_part,
        z_step_um=job.z_step_um,
        channel_names=job.channel_names,
        sequence=job.sequence,
        origin_xy_um=origin_xy_um,
        transforms=None
        if transforms is None
        else {_t_key(t): [m.tolist() for m in corr] for t, corr in transforms.items()},
        registration={k: v for k, v in info.items() if k != "components"},
        groups=info.get("components"),
        multiview_stitcher_version=_mvs_version(),
    ).model_dump(mode="json")


def _origin_xy(url: Path) -> tuple[float, float] | None:
    """Stage (x, y) of an OME-Zarr mosaic's first pixel, from its translation."""
    import zarr

    root: Any = zarr.open_group(str(url), mode="r")
    ms = root.attrs["multiscales"][0]
    axes = [str(a["name"]).lower() for a in ms["axes"]]
    for transform in ms["datasets"][0].get("coordinateTransformations", []):
        if transform.get("type") == "translation":
            origin = dict(zip(axes, transform["translation"], strict=True))
            return float(origin["x"]), float(origin["y"])
    return None


def _write_provenance(url: Path, record: dict[str, Any]) -> None:
    """Store `record` in an OME-Zarr mosaic's root attributes."""
    import zarr

    root = zarr.open_group(str(url), mode="r+")
    existing = root.attrs.get(METADATA_NAMESPACE)
    ns = dict(existing) if isinstance(existing, dict) else {}
    ns[PROVENANCE_KEY] = record
    root.attrs[METADATA_NAMESPACE] = ns


def _convert_to_tiff(
    fused: Path, tmp: Path, group: GroupV1, record: dict[str, Any], progress: _Progress
) -> None:
    """Copy the fused OME-Zarr into the OME-TIFF at `tmp`, then delete it."""
    from pymmcore_gui._stitching._tiff_output import zarr_to_ome_tiff

    progress.set(Phase.CONVERTING)
    remove_path(tmp)
    zarr_to_ome_tiff(
        fused,
        tmp,
        name=group.output.name,
        annotations={METADATA_NAMESPACE: {PROVENANCE_KEY: record}},
        progress=lambda done, total: progress.set(
            Phase.CONVERTING, total=total, done=done
        ),
    )
    remove_path(fused)


def _move_into_place(tmp: Path, final: Path) -> Path:
    """Rename the finished `.tmp` mosaic; never overwrite."""
    target = unique_path(final, check_temporary=False)
    os.replace(tmp, target)
    return target


def _stitch_group(
    job: StitchJobV1,
    group: GroupV1,
    opened: OpenedTiles,
    progress: _Progress,
    fingerprint: dict[str, Any],
) -> GroupResultV1:
    progress.set(Phase.READING)
    images = _GroupImages(job, group, opened)
    register = job.settings.alignment == Alignment.REGISTER and len(group.tiles) > 1

    transforms: dict[Any, list[np.ndarray]] | None = None
    info: dict[str, Any] = {}
    reused = False
    saved = _reusable(group, images, fingerprint) if register else None
    if saved is not None:
        progress.set(Phase.REGISTERING, message="Reusing the earlier registration")
        transforms, reused = saved, True
        if group.saved_registration is not None:
            info = dict(group.saved_registration.info)
        _apply_corrections(images, transforms)
    elif register:
        transforms, info = _register(images, job, progress)
        _apply_corrections(images, transforms)

    fused = group.fused_tmp
    for path in group.tmp_paths:
        remove_path(path)
    _fuse(images, job, REG_KEY if register else STAGE_KEY, fused, progress)

    record = _provenance(job, group, transforms, info, fingerprint, _origin_xy(fused))
    if is_tiff_output(group.output):
        _convert_to_tiff(fused, group.tmp_output, group, record, progress)
        progress.set(Phase.FINISHING)
    else:
        progress.set(Phase.FINISHING)
        _write_provenance(fused, record)
    final = _move_into_place(group.tmp_output, group.output)
    components = info.get("components") or [list(range(len(group.tiles)))]
    return GroupResultV1(
        name=group.name,
        output=final,
        n_tiles=len(group.tiles),
        registered=register,
        n_pairs=info.get("n_pairs", 0),
        rejected_pairs=info.get("rejected_pairs", 0),
        isolated_tiles=info.get("isolated_tiles", 0) if register else 0,
        n_components=len(components),
        reused_registration=reused,
    )


def run_stitch(job: StitchJobV1, report: Reporter) -> StitchResultV1:
    """Stitch every group of `job`, reporting progress through `report`.

    Raises `StitchError` for problems the user can act on; anything else is a
    bug, reported with its traceback.
    """
    import dask.config

    start = time.perf_counter()
    report(ProgressEvent(phase=Phase.STARTING, n_groups=len(job.groups)))
    try:
        opened = open_tiles(job.source)
    except (OSError, ValueError) as e:
        raise StitchError(f"Could not read the acquisition: {e}") from e

    try:
        needed = max(t.position_index for g in job.groups for t in g.tiles)
        if needed >= opened.n_positions:
            raise StitchError(
                f"The acquisition has {opened.n_positions} positions, but the "
                f"grid needs {needed + 1}. Was the run interrupted?"
            )
        if opened.sizes().get("z", 1) > 1 and not job.z_step_um:
            raise StitchError("The z-stack's step is unknown.")
        fingerprint = source_fingerprint(job.source)
        results = []
        # Parallelism comes from our own thread pools; nested dask threads
        # would oversubscribe the budget. Library warnings are shown
        # once each (in the worker, that is its log), never raised.
        with warnings.catch_warnings(), dask.config.set(scheduler="synchronous"):
            warnings.simplefilter("default")
            for i, group in enumerate(job.groups):
                progress = _Progress(report, i, len(job.groups))
                results.append(_stitch_group(job, group, opened, progress, fingerprint))
    finally:
        opened.close()
    return StitchResultV1(groups=results, elapsed_s=time.perf_counter() - start)
