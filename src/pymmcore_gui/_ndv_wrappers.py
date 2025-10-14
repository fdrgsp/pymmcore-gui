from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any, TypeGuard

import numpy as np
import zarr
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

    PRIORITY = 45  # Higher priority than default, to catch before fallback

    def __init__(self, handler: Any) -> None:
        """Initialize wrapper with an OMEWriterHandler instance.

        Parameters
        ----------
        handler : OMEWriterHandler
            An OMEWriterHandler instance with an active stream.
        """
        from pathlib import Path

        # Store the handler
        self._handler = handler
        self._path = Path(handler.path)

        # Determine backend and set up reader
        self._backend: str | None = None
        self._readers: dict[int, Any] = {}  # Map position index to reader
        self._dims: tuple[Hashable, ...] = ()
        self._coords: Mapping[Hashable, Sequence] = {}
        self._num_positions: int = 1

        # Initialize the reader based on the file type
        self._group = zarr.open_group(str(self._path), mode="r")
        self._num_positions = len(list(self._group.keys()))

        self._extract_dims_from_zarr()

        # Call parent init with the first reader object
        super().__init__(self._group)

    def _extract_dims_from_zarr(self) -> None:
        """Extract dimension information from Zarr metadata."""
        # Try to read OME-NGFF metadata
        from pathlib import Path

        zarr_json = Path(self._path) / "zarr.json"
        if zarr_json.exists():
            with open(zarr_json) as f:
                metadata = json.load(f)
                attrs = metadata.get("attributes", {})

                # Look for OME-NGFF metadata
                if "ome" in attrs:
                    multiscales = attrs["ome"].get("multiscales", [])
                    if multiscales and "axes" in multiscales[0]:
                        axes = multiscales[0]["axes"]
                        dims = tuple(ax["name"] for ax in axes)
                        shape = self._data.shape
                        coords = {
                            dim: range(size)
                            for dim, size in zip(dims, shape, strict=False)
                        }
                        # Add position dimension if multiple positions
                        if self._num_positions > 1:
                            self._dims = ("p", *dims)
                            coords["p"] = range(self._num_positions)
                            self._coords = coords
                        else:
                            self._dims = dims
                            self._coords = coords
                        return

        # Fallback: use numeric dimensions
        shape = self._readers[0].shape
        dims = tuple(range(len(shape)))
        coords = {i: range(s) for i, s in enumerate(shape)}
        # Add position dimension if multiple positions
        if self._num_positions > 1:
            self._dims = ("p", *dims)
            coords["p"] = range(self._num_positions)
            self._coords = coords
        else:
            self._dims = dims
            self._coords = coords

    @property
    def dims(self) -> tuple[Hashable, ...]:
        """Return the dimension labels for the data."""
        return self._dims

    @property
    def coords(self) -> Mapping[Hashable, Sequence]:
        """Return the coordinates for the data."""
        return self._coords

    @property
    def dtype(self) -> np.dtype:
        """Return the dtype for the data."""
        return super().dtype

    def isel(self, index: Mapping[int, int | slice]) -> np.ndarray:
        """Return a slice of the data as a numpy array."""
        # Handle multi-position similar to _OME5DWrapper
        if self._num_positions > 1:
            idx = [index.get(k, slice(None)) for k in range(len(self.dims))]
            try:
                pidx = self.dims.index("p")
            except ValueError:
                pidx = 0

            _pcoord: int | slice = idx[pidx]
            pcoord: int = _pcoord.start if isinstance(_pcoord, slice) else _pcoord

            # Remove position dimension from index
            del idx[pidx]

            # Get data from the correct position
            if self._backend == "tiff":
                # For TIFF, data is already in memory
                data = np.asarray(self._readers[pcoord][tuple(idx)])
            else:
                # For Zarr/TensorStore
                data = self._asarray(self._readers[pcoord][tuple(idx)])

            # Add back position dimension
            return np.expand_dims(data, axis=pidx)
        else:
            # Single position, use the original logic
            idx = tuple(index.get(k, slice(None)) for k in range(len(self.dims)))

            if self._backend == "tiff":
                # For TIFF, data is already in memory
                return np.asarray(self._readers[0][idx])
            # For Zarr/TensorStore
            return self._asarray(self._readers[0][idx])

    def _asarray(self, data: Any) -> np.ndarray:
        """Convert data to a numpy array."""
        # For tensorstore, need to read the result
        if hasattr(data, "read") and callable(data.read):
            return np.asarray(data.read().result())
        return np.asarray(data)

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[Any]:
        """Check if this wrapper supports the given object.

        Returns True if obj is an OMEWriterHandler from pymmcore-plus.
        """
        # Check if it has the key attributes of OMEWriterHandler
        return (
            hasattr(obj, "path")
            and hasattr(obj, "stream")
            and hasattr(obj, "backend")
            and hasattr(obj.stream, "is_active")
            and hasattr(obj.stream, "append")
        )


# class OMEWriterWrapper(DataWrapper):
#     """Wrapper for OMEWriterHandler objects from pymmcore-plus.

#     This wrapper provides read access to data written by OMEWriterHandler,
#     which uses the ome-writers library to write OME-Zarr or OME-TIFF files.
#     """

#     PRIORITY = 45  # Higher priority than default, to catch before fallback

#     def __init__(self, handler: Any) -> None:
#         """Initialize wrapper with an OMEWriterHandler instance.

#         Parameters
#         ----------
#         handler : OMEWriterHandler
#             An OMEWriterHandler instance with an active stream.
#         """
#         from pathlib import Path

#         # Store the handler
#         self._handler = handler
#         self._path = Path(handler.path)

#         # Determine backend and set up reader
#         self._backend: str | None = None
#         self._readers: dict[int, Any] = {}  # Map position index to reader
#         self._dims: tuple[Hashable, ...] = ()
#         self._coords: Mapping[Hashable, Sequence] = {}
#         self._num_positions: int = 1

#         self._group: zarr.Group | None = None

#         # Initialize the reader based on the file type
#         self._init_reader()

#         # Call parent init with the first reader object
#         super().__init__(self._readers.get(0))

#     def _init_reader(self) -> None:
#         """Initialize the appropriate reader based on file extension."""
#         path_str = str(self._path)

#         # Determine backend from path
#         if path_str.endswith(".zarr"):
#             self._backend = "zarr"
#             self._init_zarr_python_reader()
#         elif path_str.endswith((".tiff", ".tif")):
#             self._backend = "tiff"
#             self._init_tiff_reader()
#         else:
#             # Try to infer from what exists on disk
#             if self._path.exists():
#                 if self._path.is_dir():
#                     self._backend = "zarr"
#                     self._init_zarr_python_reader()
#                 elif self._path.is_file():
#                     self._backend = "tiff"
#                     self._init_tiff_reader()
#             else:
#                 raise ValueError(
#                     f"Cannot determine backend for path: {self._path}. "
#                     "File does not exist yet."
#                 )

#     def _init_zarr_python_reader(self) -> None:
#         """Initialize zarr-python reader."""
#         import zarr

#         self._group = zarr.open_group(str(self._path), mode="r")
#         self._num_positions = len(list(self._group.keys()))
#         self._extract_dims_from_zarr()

#         # # Try to open position arrays (each position is a separate zarr hierarchy)
#         # pos = 0
#         # while True:
#         #     try:
#         #         pos_path = self._path / str(pos)
#         #         if not pos_path.exists():
#         #             break
#         #         # Open the group for this position and get the array at path "0"
#         #         grp = zarr.open_group(str(pos_path), mode="r")
#         #         self._readers[pos] = grp["0"]
#         #         pos += 1
#         #     except (KeyError, FileNotFoundError):
#         #         break

#         # self._num_positions = len(self._readers)
#         self._extract_dims_from_zarr()

#     def _extract_dims_from_zarr(self) -> None:
#         """Extract dimension information from Zarr metadata."""
#         # Try to read OME-NGFF metadata
#         from pathlib import Path

#         zarr_json = Path(self._path) / "zarr.json"
#         if zarr_json.exists():
#             with open(zarr_json) as f:
#                 metadata = json.load(f)
#                 attrs = metadata.get("attributes", {})

#                 # Look for OME-NGFF metadata
#                 if "ome" in attrs:
#                     multiscales = attrs["ome"].get("multiscales", [])
#                     if multiscales and "axes" in multiscales[0]:
#                         axes = multiscales[0]["axes"]
#                         dims = tuple(ax["name"] for ax in axes)
#                         shape = self._data.shape
#                         coords = {
#                             dim: range(size)
#                             for dim, size in zip(dims, shape, strict=False)
#                         }
#                         # Add position dimension if multiple positions
#                         if self._num_positions > 1:
#                             self._dims = ("p", *dims)
#                             coords["p"] = range(self._num_positions)
#                             self._coords = coords
#                         else:
#                             self._dims = dims
#                             self._coords = coords
#                         return

#         # Fallback: use numeric dimensions
#         shape = self._readers[0].shape
#         dims = tuple(range(len(shape)))
#         coords = {i: range(s) for i, s in enumerate(shape)}
#         # Add position dimension if multiple positions
#         if self._num_positions > 1:
#             self._dims = ("p", *dims)
#             coords["p"] = range(self._num_positions)
#             self._coords = coords
#         else:
#             self._dims = dims
#             self._coords = coords

#     def _init_tiff_reader(self) -> None:
#         """Initialize TIFF reader."""
#         try:
#             import tifffile
#         except ImportError as e:
#             raise ImportError("tifffile is required to read TIFF data") from e

#         # Check if path is a directory (ome-writers creates directories)
#         if self._path.is_dir():
#             # ome-writers creates files named "0", "1", etc. inside the directory
#             pos = 0
#             while True:
#                 tiff_path = self._path / str(pos)
#                 if tiff_path.exists():
#                     with tifffile.TiffFile(str(tiff_path)) as tif:
#                         self._readers[pos] = tif.asarray()
#                         if pos == 0:
#                             self._extract_dims_from_tiff(tif)
#                     pos += 1
#                 else:
#                     break
#         # For multi-position, we need to handle multiple files
#         # Check for traditional naming like file_p000.tiff, file_p001.tiff
#         elif "_p000" in self._path.name or "_p001" in self._path.name:
#             # This is already a position-specific file, find others
#             base = self._path.parent / self._path.stem.split("_p")[0]
#             pos = 0
#             while True:
#                 tiff_path = base.parent / f"{base.name}_p{pos:03d}{self._path.suffix}"
#                 if tiff_path.exists():
#                     with tifffile.TiffFile(str(tiff_path)) as tif:
#                         self._readers[pos] = tif.asarray()
#                         if pos == 0:
#                             self._extract_dims_from_tiff(tif)
#                     pos += 1
#                 else:
#                     break
#         else:
#             # Check if position-specific files exist
#             base = self._path.parent / self._path.stem
#             pos = 0
#             while True:
#                 tiff_path = base.parent / f"{base.name}_p{pos:03d}{self._path.suffix}"
#                 if tiff_path.exists():
#                     with tifffile.TiffFile(str(tiff_path)) as tif:
#                         self._readers[pos] = tif.asarray()
#                         if pos == 0:
#                             self._extract_dims_from_tiff(tif)
#                     pos += 1
#                 else:
#                     break

#             # If no position files found, use the original file
#             if not self._readers:
#                 with tifffile.TiffFile(str(self._path)) as tif:
#                     self._readers[0] = tif.asarray()
#                     self._extract_dims_from_tiff(tif)

#         self._num_positions = len(self._readers)

#     def _extract_dims_from_tiff(self, tif: Any) -> None:
#         """Extract dimension information from TIFF metadata."""
#         # Try to parse OME-XML metadata
#         try:
#             import ome_types

#             if hasattr(tif, "ome_metadata") and tif.ome_metadata:
#                 ome = ome_types.from_xml(tif.ome_metadata)
#                 if ome.images:
#                     pixels = ome.images[0].pixels
#                     # Build dimensions from OME metadata
#                     dim_order = str(pixels.dimension_order).lower()
#                     sizes = {
#                         "t": pixels.size_t or 1,
#                         "c": pixels.size_c or 1,
#                         "z": pixels.size_z or 1,
#                         "y": pixels.size_y,
#                         "x": pixels.size_x,
#                     }
#                     # Use the dimension order from OME
#                     dims = tuple(dim_order)
#                     coords: dict[Hashable, Sequence] = {
#                         dim: range(sizes[dim]) for dim in dims
#                     }
#                     # Add position dimension if multiple positions
#                     if self._num_positions > 1:
#                         self._dims = ("p", *dims)
#                         coords["p"] = range(self._num_positions)
#                         self._coords = coords
#                     else:
#                         self._dims = dims
#                         self._coords = coords
#                     return
#         except Exception:
#             pass

#         # Fallback: assume standard 5D TIFF (T, C, Z, Y, X)
#         shape = self._readers[0].shape
#         if len(shape) == 5:
#             dims = ("t", "c", "z", "y", "x")
#         elif len(shape) == 4:
#             dims = ("t", "z", "y", "x")
#         elif len(shape) == 3:
#             dims = ("z", "y", "x")
#         elif len(shape) == 2:
#             dims = ("y", "x")
#         else:
#             dims = tuple(range(len(shape)))

#         coords = {dim: range(size) for dim, size in zip(dims, shape, strict=False)}
#         # Add position dimension if multiple positions
#         if self._num_positions > 1:
#             self._dims = ("p", *dims)
#             coords["p"] = range(self._num_positions)
#             self._coords = coords
#         else:
#             self._dims = dims
#             self._coords = coords

#     @property
#     def dims(self) -> tuple[Hashable, ...]:
#         """Return the dimension labels for the data."""
#         return self._dims

#     @property
#     def coords(self) -> Mapping[Hashable, Sequence]:
#         """Return the coordinates for the data."""
#         return self._coords

#     @property
#     def dtype(self) -> np.dtype:
#         """Return the dtype for the data."""
#         return super().dtype

#     def isel(self, index: Mapping[int, int | slice]) -> np.ndarray:
#         """Return a slice of the data as a numpy array."""
#         # Handle multi-position similar to _OME5DWrapper
#         if self._num_positions > 1:
#             idx = [index.get(k, slice(None)) for k in range(len(self.dims))]
#             try:
#                 pidx = self.dims.index("p")
#             except ValueError:
#                 pidx = 0

#             _pcoord: int | slice = idx[pidx]
#             pcoord: int = _pcoord.start if isinstance(_pcoord, slice) else _pcoord

#             # Remove position dimension from index
#             del idx[pidx]

#             # Get data from the correct position
#             if self._backend == "tiff":
#                 # For TIFF, data is already in memory
#                 data = np.asarray(self._readers[pcoord][tuple(idx)])
#             else:
#                 # For Zarr/TensorStore
#                 data = self._asarray(self._readers[pcoord][tuple(idx)])

#             # Add back position dimension
#             return np.expand_dims(data, axis=pidx)
#         else:
#             # Single position, use the original logic
#             idx = tuple(index.get(k, slice(None)) for k in range(len(self.dims)))

#             if self._backend == "tiff":
#                 # For TIFF, data is already in memory
#                 return np.asarray(self._readers[0][idx])
#             # For Zarr/TensorStore
#             return self._asarray(self._readers[0][idx])

#     def _asarray(self, data: Any) -> np.ndarray:
#         """Convert data to a numpy array."""
#         # For tensorstore, need to read the result
#         if hasattr(data, "read") and callable(data.read):
#             return np.asarray(data.read().result())
#         return np.asarray(data)

#     @classmethod
#     def supports(cls, obj: Any) -> TypeGuard[Any]:
#         """Check if this wrapper supports the given object.

#         Returns True if obj is an OMEWriterHandler from pymmcore-plus.
#         """
#         # Check if it has the key attributes of OMEWriterHandler
#         return (
#             hasattr(obj, "path")
#             and hasattr(obj, "stream")
#             and hasattr(obj, "backend")
#             and hasattr(obj.stream, "is_active")
#             and hasattr(obj.stream, "append")
#         )
