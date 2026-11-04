"""ndv `DataWrapper` for OME-Zarr / OME-NGFF stores.

Handles single-position images, multi-position layouts (bioformats2raw,
explicit ``series``, wells, and HCS plates), and pyramidal (multiscale)
datasets -- covering both what `ome_writers` produces for a pyMM acquisition
and OME-Zarr data written by other tools.

Requires the `yaozarrs` package (already a pymmcore-gui dependency via
`ome-writers[tensorstore]`).
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeGuard, cast

import numpy as np
from ndv.models import DataWrapper

from pymmcore_gui._utils import indexed_shape, isel_positions

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    from yaozarrs import ZarrGroup
    from yaozarrs._zarr import ZarrArray


class OMEZarrWrapper(DataWrapper):
    """`ndv.DataWrapper` for an OME-Zarr store, opened lazily via `yaozarrs`.

    Only the multiscale level referenced by ``datasets[0]`` is used; pyramid
    levels beyond the first are not exposed.
    """

    PRIORITY = 45

    def __init__(self, data: Any) -> None:
        import yaozarrs

        self._group: ZarrGroup = (
            yaozarrs.open_group(data) if isinstance(data, (str, os.PathLike)) else data
        )
        self._positions: list[str] = []
        self._dataset_path: str = ""
        self._arrays: dict[int | None, ZarrArray] = {}
        self._stores: dict[int, Any] = {}
        self._dims, self._coords = self._detect_structure()
        super().__init__(self._group)

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[Any]:
        with contextlib.suppress(Exception):
            from yaozarrs import ZarrGroup

            if isinstance(obj, ZarrGroup):
                return obj.ome_metadata() is not None

        if isinstance(obj, (str, os.PathLike)):
            with contextlib.suppress(Exception):
                import yaozarrs

                return yaozarrs.open_group(obj).ome_metadata() is not None
        return False

    @property
    def dims(self) -> tuple[Hashable, ...]:
        return self._dims

    @property
    def coords(self) -> Mapping[Hashable, Sequence]:
        return self._coords

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._array(0 if self._positions else None).dtype)

    def guess_channel_axis(self) -> int | None:
        """Use OME axis names rather than guessing from dimension sizes."""
        return next(
            (
                i
                for i, dim in enumerate(self._dims)
                if str(dim).lower() in self.COMMON_CHANNEL_NAMES
            ),
            None,
        )

    def isel(self, index: Mapping[int, int | slice]) -> np.ndarray:
        if self._positions:
            return isel_positions(
                index,
                len(self._positions),
                tuple(self.sizes().values()),
                self.dtype,
                self._read_position,
            )
        idx = tuple(index.get(i, slice(None)) for i in range(len(self._dims)))
        return self._read_array(self._array(None), idx)

    def close(self) -> None:
        """Release any resources the underlying zarr store may be holding open."""
        self._arrays.clear()
        self._stores.clear()
        store = getattr(self._group, "store", None)
        close = getattr(store, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()

    def global_metadata(self, namespace: str) -> Mapping[str, Any] | None:
        """Return this store's root-level attribute dict for `namespace`, if any.

        A generic escape hatch for whatever a writer chose to stash as
        acquisition-level (not per-frame) metadata under a top-level attrs
        key. Callers that know a specific namespace's convention (e.g.
        pymmcore-plus's own "pymmcore_plus") interpret the returned mapping
        themselves; this method has no opinion on its contents. Always reads
        the *root* group's attrs, even for a multi-position store, matching
        where `ome_writers` puts this kind of metadata.
        """
        return self._group.attrs.get(namespace)

    # ----------------------- Structure Detection -----------------------

    def _detect_structure(
        self,
    ) -> tuple[tuple[Hashable, ...], dict[Hashable, Sequence]]:
        """Detect NGFF structure and return (dims, coords).

        NGFF supports several layouts:
        - Single image: has `multiscales` directly
        - Bf2Raw: has `bioformats2raw.layout`, images in numbered subgroups
        - Series: has explicit `series` array listing image paths
        - Plate: has `plate` with wells containing images
        - Well: has `well` with images array
        """
        from yaozarrs import v04, v05

        meta: Any = self._group.ome_metadata()

        if hasattr(meta, "multiscales") and meta.multiscales:
            return self._init_single(meta)
        if isinstance(meta, (v04.Bf2Raw, v05.Bf2Raw)):
            return self._init_bioformats2raw()
        if isinstance(meta, (v04.Series, v05.Series)):
            self._positions = list(meta.series)
            return self._init_multiposition()
        if isinstance(meta, (v04.Plate, v05.Plate)):
            return self._init_plate(meta)
        if isinstance(meta, (v04.Well, v05.Well)):
            return self._init_well(meta)

        raise ValueError(f"Unknown NGFF structure: {self._group.store_path}")

    def _init_single(self, meta: Any) -> tuple[tuple[Hashable, ...], dict]:
        """Initialize single-position image."""
        ms = meta.multiscales[0]
        self._dataset_path = ms.datasets[0].path
        arr = cast("ZarrArray", self._group[self._dataset_path])
        assert arr.metadata.shape is not None
        shape = arr.metadata.shape
        dims = self._dims_from_axes(ms.axes, len(shape))
        coords = self._coords_from_axes(ms, shape, getattr(meta, "omero", None))
        return dims, coords

    def _init_bioformats2raw(self) -> tuple[tuple[Hashable, ...], dict]:
        """Initialize bioformats2raw layout."""
        if "OME" in self._group:
            ome_group = cast("ZarrGroup", self._group["OME"])
            attrs = ome_group.metadata.attributes
            # NGFF v0.5 nests everything under an "ome" key; v0.4 is flat.
            nested = attrs.get("ome")
            series = attrs.get("series") or (
                nested.get("series") if isinstance(nested, Mapping) else None
            )
            if series:
                # Authoritative acquisition order -- a grid's tiles are named
                # "{pos}_{row}_{col}" but visited in the grid's own traversal
                # order, so a snake row ends ..._1_2, _1_1, _1_0. Sorting the
                # group names instead would silently relabel those tiles.
                self._positions = list(series)
        if not self._positions:
            # Find numbered subgroups (0, 1, 2, ...)
            i = 0
            while str(i) in self._group:
                child = self._group[str(i)]
                if hasattr(child, "ome_metadata"):
                    child_meta: Any = cast("ZarrGroup", child).ome_metadata()
                    if hasattr(child_meta, "multiscales") and child_meta.multiscales:
                        self._positions.append(str(i))
                i += 1
        if not self._positions:
            raise ValueError("No positions found in bioformats2raw layout")
        return self._init_multiposition()

    def _init_well(self, meta: Any) -> tuple[tuple[Hashable, ...], dict]:
        """Initialize well."""
        self._positions = [fov.path for fov in meta.well.images]
        if not self._positions:
            raise ValueError("No FOV positions found in well")
        return self._init_multiposition()

    def _init_plate(self, meta: Any) -> tuple[tuple[Hashable, ...], dict]:
        """Initialize HCS plate."""
        for well_ref in meta.plate.wells:
            if well_ref.path not in self._group:
                continue
            well_group = cast("ZarrGroup", self._group[well_ref.path])
            well_meta: Any = well_group.ome_metadata()
            if hasattr(well_meta, "well"):
                self._positions.extend(
                    f"{well_ref.path}/{fov.path}" for fov in well_meta.well.images
                )
        if not self._positions:
            raise ValueError("No FOV positions found in plate")
        return self._init_multiposition()

    def _init_multiposition(self) -> tuple[tuple[Hashable, ...], dict]:
        """Finalize multi-position setup using the first position's metadata."""
        pos_group = cast("ZarrGroup", self._group[self._positions[0]])
        meta: Any = pos_group.ome_metadata()
        ms = meta.multiscales[0]
        self._dataset_path = ms.datasets[0].path
        arr = cast("ZarrArray", pos_group[self._dataset_path])
        assert arr.metadata.shape is not None
        shape = arr.metadata.shape
        inner_dims = self._dims_from_axes(ms.axes, len(shape))
        dims = ("p", *inner_dims)
        inner_coords = self._coords_from_axes(ms, shape, getattr(meta, "omero", None))
        coords: dict[Hashable, Sequence] = {"p": list(self._positions)}
        coords.update(inner_coords)
        return dims, coords

    @staticmethod
    def _dims_from_axes(axes: Any, ndim: int) -> tuple[Hashable, ...]:
        """Extract dimension names from axes or use integer indices."""
        if axes and len(axes) == ndim:
            return tuple(ax.name for ax in axes)
        return tuple(range(ndim))

    @staticmethod
    def _coords_from_axes(
        ms: Any, shape: tuple[int, ...], omero: Any
    ) -> dict[Hashable, Sequence]:
        """Build real (physical/channel-labeled) coordinates where available.

        Falls back to `range(size)` per axis when the metadata doesn't give
        anything more specific -- exactly what ndv's `DataWrapper.sizes()`
        already assumes for an unlabeled axis.
        """
        axes = ms.axes
        if not axes or len(axes) != len(shape):
            return {i: range(s) for i, s in enumerate(shape)}

        scale = translation = None
        for xform in ms.datasets[0].coordinateTransformations or ():
            if getattr(xform, "type", None) == "scale":
                scale = xform.scale
            elif getattr(xform, "type", None) == "translation":
                translation = xform.translation

        coords: dict[Hashable, Sequence] = {}
        channels = list(getattr(omero, "channels", None) or ())
        for i, (ax, size) in enumerate(zip(axes, shape, strict=False)):
            axis_type = getattr(ax, "type", None)
            if axis_type == "channel" and len(channels) == size:
                coords[ax.name] = [c.label or str(i) for i, c in enumerate(channels)]
            elif axis_type in ("space", "time") and scale is not None:
                start = translation[i] if translation else 0.0
                step = scale[i]
                coords[ax.name] = [start + step * n for n in range(size)]
            else:
                coords[ax.name] = range(size)
        return coords

    # ----------------------- Data Access -----------------------

    def _array(self, pos_idx: int | None) -> ZarrArray:
        """Return (and cache) one position's array node, or the single image's.

        Resolving it walks the zarr hierarchy, so cache it: `isel` is called for
        every frame and every slider move, and the layout is fixed at open time.
        """
        if (arr := self._arrays.get(pos_idx)) is None:
            group = (
                self._group
                if pos_idx is None
                else self._group[self._positions[pos_idx]]
            )
            arr = cast("ZarrArray", cast("ZarrGroup", group)[self._dataset_path])
            self._arrays[pos_idx] = arr
        return arr

    def _read_position(
        self, pos_idx: int, indexers: Mapping[int, int | slice]
    ) -> np.ndarray:
        if pos_idx < 0:
            pos_idx += len(self._positions)
        if not 0 <= pos_idx < len(self._positions):
            raise IndexError(f"Position index {pos_idx} out of range")
        inner_idx = tuple(
            indexers.get(i, slice(None)) for i in range(len(self._dims) - 1)
        )
        return self._read_array(self._array(pos_idx), inner_idx)

    def _read_array(self, arr: ZarrArray, idx: tuple) -> np.ndarray:
        shape = arr.metadata.shape
        assert shape is not None
        selected_shape = indexed_shape(shape, dict(enumerate(idx)))
        if 0 in selected_shape:
            return np.empty(selected_shape, dtype=np.dtype(arr.dtype))
        # TensorStore uses absolute indices, including negative slice bounds.
        normalized = tuple(
            slice(*index.indices(size))
            if isinstance(index, slice)
            else index + size
            if index < 0
            else index
            for index, size in zip(idx, shape, strict=True)
        )
        if (store := self._stores.get(id(arr))) is None:
            # to_tensorstore() re-opens the array and re-reads its metadata, so
            # keep the handle: every frame and slider move comes through here.
            store = self._stores[id(arr)] = arr.to_tensorstore()
        return np.asarray(store[normalized].read().result())
