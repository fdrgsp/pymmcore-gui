"""Synthetic grid acquisitions for the stitching tests, written by ome-writers.

A smooth random "world" image lives in stage coordinates (µm). Each tile is
sampled from it the way a camera with pixel->stage map `A` would see it,
centred on the tile's *true* stage position: its planned position plus a
known jitter. Registration has to remove that jitter.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import useq
from ome_writers import (
    AcquisitionSettings,
    OmeTiffFormat,
    OmeZarrFormat,
    create_stream,
    useq_to_acquisition_settings,
)
from scipy.ndimage import gaussian_filter, map_coordinates


@dataclass
class SynthAcquisition:
    path: Path
    sequence: useq.MDASequence
    planned_xy: np.ndarray
    """(n_positions, 2) planned stage centres, in store order."""
    jitter_xy: np.ndarray
    """(n_positions, 2) true minus planned centre."""
    world: np.ndarray
    world_origin_xy: tuple[float, float]
    world_pixel_um: float


def grid_sequence(
    rows: int,
    cols: int,
    shape: int,
    pixel_size: float,
    overlap: float = 15,
    channels: tuple[str, ...] = ("A",),
    z: int = 1,
    t: int = 1,
    **kwargs: Any,
) -> useq.MDASequence:
    fov = shape * pixel_size
    fields: dict[str, Any] = {
        "grid_plan": {
            "rows": rows,
            "columns": cols,
            "overlap": overlap,
            "fov_width": fov,
            "fov_height": fov,
        },
        "channels": [{"config": c, "exposure": 1} for c in channels],
        **kwargs,
    }
    if z > 1:
        fields["z_plan"] = {"range": z - 1, "step": 1}
    if t > 1:
        fields["time_plan"] = {"interval": 0, "loops": t}
    return useq.MDASequence.model_validate(fields)


def _world(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = gaussian_filter(rng.random((size, size)), 3)
    img += 0.5 * gaussian_filter(rng.random((size, size)), 1)
    img -= img.min()
    return cast("np.ndarray", img / img.max())


def write_grid(
    root: Path,
    sequence: useq.MDASequence,
    *,
    shape: int,
    pixel_size: float,
    linear: np.ndarray | None = None,
    jitter_um: float = 3.0,
    fmt: str = "ome-zarr",
    seed: int = 0,
    blank: frozenset[int] = frozenset(),
) -> SynthAcquisition:
    """Write `sequence` (a grid) with tiles cut from one world image.

    Tiles whose store index is in `blank` are flat, low-level noise: nothing
    registration can match.
    """
    linear = np.diag([pixel_size, pixel_size]) if linear is None else linear
    settings = useq_to_acquisition_settings(
        sequence, shape, shape, pixel_size_um=pixel_size
    )
    dims = settings["dimensions"]
    p_dim = next(d for d in dims if d.name == "p")
    positions = cast("list[Any]", p_dim.coords)
    planned = np.array([(pos.x_coord, pos.y_coord) for pos in positions])
    rng = np.random.default_rng(seed + 1)
    jitter = rng.uniform(-jitter_um, jitter_um, planned.shape)
    true = planned + jitter

    margin = shape * pixel_size * 1.5
    lo = true.min(axis=0) - margin
    hi = true.max(axis=0) + margin
    size = int(np.ceil((hi - lo).max() / pixel_size)) + 1
    world = _world(size, seed)

    # camera pixel (col, row) offsets from the image centre, as (2, H*W)
    rr, cc = np.mgrid[0:shape, 0:shape].astype(float)
    offs = np.stack([cc.ravel() - (shape - 1) / 2, rr.ravel() - (shape - 1) / 2])

    noise = np.random.default_rng(seed + 2)

    def tile(p: int, c: int, z: int, t: int) -> np.ndarray:
        if p in blank:
            return noise.integers(1000, 1010, (shape, shape)).astype("uint16")
        stage = true[p][:, None] + linear @ offs
        wx = (stage[0] - lo[0]) / pixel_size
        wy = (stage[1] - lo[1]) / pixel_size
        img = map_coordinates(world, [wy, wx], order=1).reshape(shape, shape)
        img = img * (1 - 0.15 * c) * (1 - 0.05 * z) + 0.01 * t
        return cast("np.ndarray", (img * 50_000 + 1000).astype("uint16"))

    acq = AcquisitionSettings(
        dimensions=tuple(dims),
        dtype="uint16",
        root_path=str(root),
        format=OmeTiffFormat() if fmt == "ome-tiff" else OmeZarrFormat(),
        plate=settings.get("plate"),
    )
    names = [d.name for d in dims[:-2]]
    with create_stream(acq) as stream:
        stream.set_global_metadata(
            "pymmcore_plus",
            {
                "summary_metadata": {
                    "format": "summary-dict",
                    "version": "1.0",
                    "mda_sequence": sequence.model_dump(mode="json"),
                }
            },
        )
        for idx in itertools.product(*(range(d.count or 1) for d in dims[:-2])):
            ix = dict(zip(names, idx, strict=True))
            stream.append(
                tile(ix.get("p", 0), ix.get("c", 0), ix.get("z", 0), ix.get("t", 0))
            )
    return SynthAcquisition(
        path=Path(acq.output_path),
        sequence=sequence,
        planned_xy=planned,
        jitter_xy=jitter,
        world=world,
        world_origin_xy=(float(lo[0]), float(lo[1])),
        world_pixel_um=pixel_size,
    )
