"""Write a fused OME-Zarr mosaic as a tiled, pyramidal OME-TIFF.

multiview-stitcher streams fusion only into Zarr, so an OME-TIFF mosaic is
fused into a temporary OME-Zarr first and copied here: one BigTIFF, each
plane tiled, the lower-resolution levels as SubIFDs of their full-resolution
plane (what OME-TIFF readers expect of a pyramid). The data are read in
bands of rows, so memory stays bounded by `band_bytes` however large the
mosaic.

How the mosaic was made is stored like ome-writers stores pymmcore-plus's
summary metadata: as a `MapAnnotation` in the GUI's namespace, which
`OMETiffWrapper.global_metadata` reads back.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any
from xml.sax.saxutils import escape, quoteattr

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

TILE = 512
"""Side of the square tiles each plane is stored in."""
BAND_BYTES = 256 * 1024**2
"""Upper bound on the data read at once while copying."""

OME_NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
_PIXEL_TYPES = {
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "float32": "float",
    "float64": "double",
}
_OME_ORDER = "tczyx"
"""Plane order of the pages: t slowest, then c, then z ("XYZCT" in OME)."""


def _levels(root: Any) -> tuple[list[Any], list[str], list[float]]:
    """The arrays of every level, the axis names and the level-0 scale."""
    ms = root.attrs["multiscales"][0]
    axes = [str(a["name"]).lower() for a in ms["axes"]]
    datasets = ms["datasets"]
    arrays = [root[d["path"]] for d in datasets]
    scale = [1.0] * len(axes)
    for transform in datasets[0].get("coordinateTransformations", []):
        if transform.get("type") == "scale":
            scale = [float(v) for v in transform["scale"]]
    return arrays, axes, scale


def _channel_names(root: Any, n: int) -> list[str]:
    channels = (root.attrs.get("omero") or {}).get("channels") or []
    names = [str(c.get("label", "")) for c in channels if isinstance(c, dict)]
    return names if len(names) == n else [f"Channel {i}" for i in range(n)]


def _ome_xml(
    name: str,
    sizes: dict[str, int],
    dtype: np.dtype,
    scale: dict[str, float],
    channels: list[str],
    annotations: dict[str, dict[str, Any]],
) -> str:
    """OME-XML for one image whose planes are stored in XYZCT order."""
    pixel_type = _PIXEL_TYPES.get(dtype.name)
    if pixel_type is None:
        raise ValueError(f"OME-TIFF can't store {dtype} pixels")
    n_planes = sizes["t"] * sizes["c"] * sizes["z"]
    physical = "".join(
        f' PhysicalSize{ax.upper()}="{scale[ax]!r}" PhysicalSize{ax.upper()}Unit="µm"'
        for ax in ("x", "y", "z")
        if ax in scale
    )
    channel_xml = "".join(
        f'<Channel ID="Channel:0:{i}" Name={quoteattr(c)} SamplesPerPixel="1"/>'
        for i, c in enumerate(channels)
    )
    refs = "".join(
        f'<AnnotationRef ID="Annotation:{i}"/>' for i in range(len(annotations))
    )
    maps = "".join(
        f'<MapAnnotation ID="Annotation:{i}" Namespace={quoteattr(ns)}><Value>'
        + "".join(
            f"<M K={quoteattr(k)}>{escape(json.dumps(v))}</M>"
            for k, v in entries.items()
        )
        + "</Value></MapAnnotation>"
        for i, (ns, entries) in enumerate(annotations.items())
    )
    return (
        (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f'<OME xmlns="{OME_NS}" Creator="pymmcore-gui">'
            f'<Image ID="Image:0" Name={quoteattr(name)}>'
            f'<Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="{pixel_type}"'
            f' SizeX="{sizes["x"]}" SizeY="{sizes["y"]}" SizeZ="{sizes["z"]}"'
            f' SizeC="{sizes["c"]}" SizeT="{sizes["t"]}"{physical}'
            ' BigEndian="false" Interleaved="false">'
            f"{channel_xml}"
            f'<TiffData IFD="0" PlaneCount="{n_planes}"/>'
            f"</Pixels>{refs}</Image>"
            + (f"<StructuredAnnotations>{maps}</StructuredAnnotations>" if maps else "")
            + "</OME>"
        )
        .encode("ascii", "xmlcharrefreplace")
        .decode("ascii")
    )  # TIFF tags are ASCII


def _plane_tiles(
    array: Any, axes: list[str], band_bytes: int, on_plane: Callable[[], None]
) -> Iterator[np.ndarray]:
    """Every plane's tiles, row by row, in OME plane order.

    Planes are read whole when they fit in `band_bytes`, else in bands of
    rows a multiple of the tile height (and of the array's chunk height when
    possible, so no chunk is read twice).
    """
    shape = dict(zip(axes, array.shape, strict=True))
    height, width = shape["y"], shape["x"]
    row_bytes = width * array.dtype.itemsize
    chunk_h = dict(zip(axes, array.chunks, strict=True))["y"]
    unit = (
        math.lcm(TILE, chunk_h)
        if math.lcm(TILE, chunk_h) * row_bytes <= band_bytes
        else TILE
    )
    band = max(unit, (band_bytes // max(row_bytes, 1)) // unit * unit)
    for t in range(shape.get("t", 1)):
        for c in range(shape.get("c", 1)):
            for z in range(shape.get("z", 1)):
                plane = {"t": t, "c": c, "z": z}
                for y0 in range(0, height, band):
                    key = tuple(
                        slice(y0, min(y0 + band, height))
                        if ax == "y"
                        else slice(None)
                        if ax == "x"
                        else plane[ax]
                        for ax in axes
                    )
                    rows = np.asarray(array[key])
                    for ty in range(0, rows.shape[0], TILE):
                        for tx in range(0, width, TILE):
                            yield rows[ty : ty + TILE, tx : tx + TILE]
                on_plane()


def zarr_to_ome_tiff(
    src: Path,
    dst: Path,
    *,
    name: str,
    annotations: dict[str, dict[str, Any]] | None = None,
    progress: Callable[[int, int], None] | None = None,
    band_bytes: int = BAND_BYTES,
) -> None:
    """Copy the OME-Zarr mosaic `src` into the pyramidal OME-TIFF `dst`.

    `annotations` maps an OME `MapAnnotation` namespace to its entries (each
    value stored as JSON). `progress(done, total)` counts planes written over
    every level.
    """
    import tifffile
    import zarr

    root = zarr.open_group(str(src), mode="r")
    arrays, axes, scale_list = _levels(root)
    sizes = dict.fromkeys(_OME_ORDER, 1)
    sizes.update(dict(zip(axes, arrays[0].shape, strict=True)))
    scale = {
        ax: s for ax, s in zip(axes, scale_list, strict=True) if ax in ("x", "y", "z")
    }
    if sizes["z"] == 1:
        scale.pop("z", None)
    dtype = np.dtype(arrays[0].dtype)
    xml = _ome_xml(
        name,
        sizes,
        dtype,
        scale,
        _channel_names(root, sizes["c"]),
        annotations or {},
    )
    # pages in OME plane order: the leading axes reordered to t, c, z
    lead = [ax for ax in _OME_ORDER if ax not in ("y", "x")]
    n_planes = sizes["t"] * sizes["c"] * sizes["z"]
    total = n_planes * len(arrays)
    done = 0

    def plane_written() -> None:
        nonlocal done
        done += 1
        if progress is not None:
            progress(done, total)

    with tifffile.TiffWriter(dst, bigtiff=True, ome=False) as tw:
        for level, array in enumerate(arrays):
            level_sizes = dict(zip(axes, array.shape, strict=True))
            shape = (
                *(level_sizes.get(ax, 1) for ax in lead),
                level_sizes["y"],
                level_sizes["x"],
            )
            tw.write(
                _plane_tiles(array, axes, band_bytes, plane_written),
                shape=shape,
                dtype=dtype,
                tile=(TILE, TILE),
                photometric="minisblack",
                metadata=None,
                description=xml if level == 0 else None,
                subifds=len(arrays) - 1 if level == 0 else None,
                subfiletype=1 if level else 0,
            )
