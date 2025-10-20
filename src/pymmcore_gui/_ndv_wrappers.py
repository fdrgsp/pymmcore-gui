from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, TypeGuard

import numpy as np
from ndv import DataWrapper

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence
    from typing import Any, TypeAlias

    from pymmcore_plus.mda.handlers._5d_writer_base import _5DWriterBase

    Index: TypeAlias = int | slice


# --------------------------------------------------------------------------------
# this could be improved.  Just a quick Datawrapper for the pymmcore-plus 5D writer
# indexing and isel is particularly ugly at the moment.  TODO...


class _OME5DWrapper(DataWrapper["_5DWriterBase"]):
    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[_5DWriterBase]:
        if "pymmcore_plus.mda" in sys.modules:
            from pymmcore_plus.mda.handlers._5d_writer_base import _5DWriterBase

            return isinstance(obj, _5DWriterBase)
        return False

    @property
    def dims(self) -> tuple[Hashable, ...]:
        """Return the dimension labels for the data."""
        if not self.data.current_sequence:
            return ()
        return (*tuple(self.data.current_sequence.sizes), "y", "x")

    @property
    def coords(self) -> Mapping[Hashable, Sequence]:
        """Return the coordinates for the data."""
        if not self.data.current_sequence or not self.data.position_arrays:
            return {}
        coords: dict[Hashable, Sequence] = {
            dim: range(size) for dim, size in self.data.current_sequence.sizes.items()
        }
        ary = next(iter(self.data.position_arrays.values()))
        coords.update({"y": range(ary.shape[-2]), "x": range(ary.shape[-1])})
        return coords

    @property
    def dtype(self) -> np.dtype:
        """Return the dtype for the data."""
        try:
            return self.data.position_arrays["p0"].dtype
        except Exception:
            return super().dtype

    def isel(self, index: Mapping[int, int | slice]) -> np.ndarray:
        # oh lord look away.
        # this is a mess, partially caused by the ndv slice/model

        idx = [index.get(k, slice(None)) for k in range(len(self.dims))]
        try:
            pidx = self.dims.index("p")
        except ValueError:
            pidx = 0

        _pcoord: int | slice = index[pidx]
        pcoord: int = _pcoord.start if isinstance(_pcoord, slice) else _pcoord

        del idx[pidx]
        key = self.data.get_position_key(pcoord)
        data = self.data.position_arrays[key][tuple(idx)]
        # add back position dimension
        return np.expand_dims(data, axis=pidx)


class NGFFWriterWrapper(DataWrapper):
    """Wrapper for OMEWriterHandler objects from pymmcore-plus.

    This wrapper provides read access to data written by OMEWriterHandler,
    which uses the ome-writers library to write OME-Zarr or OME-TIFF files.
    """
