from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard

import numpy as np
from ndv import DataWrapper

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence
    from typing import TypeAlias

    from pymmcore_plus.mda.handlers import OMEWriterHandler

    Index: TypeAlias = int | slice


class OMEWriterWrapperBase(DataWrapper["OMEWriterHandler"]):
    """Base class for OMEWriterHandler wrappers.

    Provides common functionality for wrapping OMEWriterHandler objects
    from pymmcore-plus that write data using the ome-writers library.

    This base class handles:
    - Path management
    - Dimension and coordinate tracking
    - Common property implementations

    Subclasses must implement:
    - _update_structure(): Update dims, coords, shape, dtype from disk
    - isel(): Data slicing implementation
    - _clear_cache(): Clear any cached data or file handles
    """

    # Class variables to satisfy ndv DataWrapper requirements
    _dims: tuple[Hashable, ...]
    _coords: Mapping[Hashable, Sequence]
    _dtype: np.dtype | None
    _shape: tuple[int, ...]

    def __init__(self, data: OMEWriterHandler) -> None:
        """Initialize the wrapper.

        Parameters
        ----------
        data : OMEWriterHandler
            The OMEWriterHandler instance that has written or is writing data.
        """
        # Initialize dimension and coordinate info BEFORE calling super().__init__
        self._dims: tuple[Hashable, ...] = ()
        self._coords: Mapping[Hashable, Sequence] = {}
        self._dtype: np.dtype | None = None
        self._shape: tuple[int, ...] = ()

        # Store path - do this before super init
        self._path = Path(data.path)

        # TODO: find anothe way ------------------------------------------------------
        # Call parent init - this will check if we have proper implementation
        # We pass a dummy object with shape attribute to satisfy ndv's checks
        class _DummyData:
            shape = ()
        super().__init__(_DummyData())  # type: ignore
        # ----------------------------------------------------------------------------

        # Replace with actual data
        self._data = data

        # Initial structure update
        self._update_structure()

    @abstractmethod
    def _update_structure(self) -> None:
        """Update the internal structure by reading from disk.

        Must update: _dims, _coords, _dtype, _shape
        """

    @abstractmethod
    def _clear_cache(self) -> None:
        """Clear any cached data or file handles."""

    @property
    def dims(self) -> tuple[Hashable, ...]:
        """Return the dimension labels for the data."""
        self._update_structure()  # Update in case new data was written
        return self._dims

    @property
    def coords(self) -> Mapping[Hashable, Sequence]:
        """Return the coordinates for the data."""
        self._update_structure()  # Update in case new data was written
        return self._coords

    @property
    def dtype(self) -> np.dtype:
        """Return the dtype of the data."""
        self._update_structure()  # Update in case new data was written
        if self._dtype is None:
            return np.dtype("uint8")  # Default fallback
        return self._dtype

    def refresh(self) -> None:
        """Manually refresh the data structure.

        This forces a re-read of the file structure from disk, which is useful
        when data is being written in real-time and you want to see the latest
        state.

        This will also clear any cached data and reset file handles.
        """
        self._clear_cache()
        self._update_structure()
        # Emit signal that dimensions may have changed
        self.dims_changed.emit()


class ZarrWriterWrapper(OMEWriterWrapperBase):
    """Wrapper for OMEWriterHandler objects that write Zarr files.

    This wrapper provides read access to data written by OMEWriterHandler
    using the acquire-zarr or tensorstore backends.

    For Zarr files, the data is organized as a group with multiple arrays,
    one per stage position. The wrapper combines these into a unified view
    with dimensions (position, channel, [z,] y, x).

    Examples
    --------
    >>> handler = OMEWriterHandler("/path/to/data.ome.zarr", backend="acquire-zarr")
    >>> mmc.mda.run(sequence, output=handler)
    >>> wrapper = ZarrWriterWrapper(handler)
    >>> print(wrapper.dims)  # ('p', 'c', 'y', 'x')
    >>> data = wrapper.isel({0: 0, 1: 0})  # First position, first channel
    """

    def __init__(self, data: OMEWriterHandler) -> None:
        """Initialize the Zarr wrapper."""
        self._zarr_group: Any = None
        super().__init__(data)

    def _update_structure(self) -> None:
        """Update structure for Zarr backend."""
        # Try to use sequence information if available
        if sequence := self._data.sequence:
            dims_list = []
            shape_list = []

            # Get dimension sizes by iterating through the sequence
            # This gives us accurate counts for all dimensions
            sizes = sequence.sizes

            # Build dimensions in order: t, p, g, c, z
            if sizes.get("t", 0) > 1:
                dims_list.append("t")
                shape_list.append(sizes["t"])

            if sizes.get("p", 0) > 1:
                dims_list.append("p")
                shape_list.append(sizes["p"])

            if sizes.get("g", 0) > 1:
                dims_list.append("g")
                shape_list.append(sizes["g"])

            if sizes.get("c", 0) > 1:
                dims_list.append("c")
                shape_list.append(sizes["c"])

            if sizes.get("z", 0) > 1:
                dims_list.append("z")
                shape_list.append(sizes["z"])

            # For Y and X, we need actual data
            if self._path.exists():
                import zarr

                self._zarr_group = zarr.open(str(self._path), mode="r")
                position_keys = sorted(
                    [k for k in self._zarr_group.keys() if k.isdigit()], key=int
                )

                if position_keys:
                    first_array = self._zarr_group[position_keys[0]]
                    self._dtype = np.dtype(first_array.dtype)
                    # Get Y, X from actual array
                    array_shape = first_array.shape
                    # Last two dimensions are always Y, X
                    dims_list.extend(["y", "x"])
                    shape_list.extend(array_shape[-2:])

                    self._dims = tuple(dims_list)
                    self._shape = tuple(shape_list)
                else:
                    # No data yet
                    self._dims = ()
                    self._shape = ()
            else:
                # No data on disk yet, but we can still provide partial info
                self._dims = tuple(dims_list) if dims_list else ()
                self._shape = tuple(shape_list) if shape_list else ()

            # Create coordinates
            self._coords = {
                dim: range(size)
                for dim, size in zip(self._dims, self._shape, strict=True)
            }
            return

        # Fallback to reading from disk if no sequence available
        if not self._path.exists():
            # No data written yet - set empty structure
            self._dims = ()
            self._coords = {}
            self._shape = ()
            return

        import zarr

        self._zarr_group = zarr.open(str(self._path), mode="r")

        # Get the list of position arrays
        position_keys = sorted(
            [k for k in self._zarr_group.keys() if k.isdigit()],
            key=int,
        )

        if not position_keys:
            self._dims = ()
            self._coords = {}
            self._shape = ()
            return

        # Get shape from first position
        first_array = self._zarr_group[position_keys[0]]
        self._dtype = np.dtype(first_array.dtype)

        # Shape is (n_positions, ...) + first_array.shape
        n_positions = len(position_keys)
        self._shape = (n_positions, *tuple(first_array.shape))

        # Get dimension names from OME metadata - fail if not available
        ome_attrs = dict(self._zarr_group.attrs)
        if "ome" not in ome_attrs or "multiscales" not in ome_attrs["ome"]:
            raise ValueError(
                f"No OME metadata found in {self._path}. "
                "Cannot determine dimension names without metadata."
            )

        multiscales = ome_attrs["ome"]["multiscales"][0]
        if "axes" not in multiscales:
            raise ValueError(
                f"No 'axes' information in OME metadata for {self._path}. "
                "Cannot determine dimension names without axis information."
            )

        axes = multiscales["axes"]
        # axes should match the first_array dimensions
        dim_names = tuple([ax["name"] for ax in axes])
        # Prepend 'p' for position
        self._dims = ("p", *dim_names)

        # Create coordinates
        self._coords = {
            dim: range(size) for dim, size in zip(self._dims, self._shape, strict=True)
        }

    def _clear_cache(self) -> None:
        """Clear zarr group handle."""
        self._zarr_group = None

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[OMEWriterHandler]:
        """Check if this wrapper supports the given object.

        Parameters
        ----------
        obj : Any
            The object to check.

        Returns
        -------
        bool
            True if obj is an OMEWriterHandler for Zarr format.
        """
        if not (
            hasattr(obj, "path")
            and hasattr(obj, "backend")
            and hasattr(obj, "stream")
            and hasattr(obj, "sequenceStarted")
        ):
            return False

        # Check if it's a Zarr file
        path = Path(obj.path)
        is_zarr = path.suffix == ".zarr" or path.name.endswith(".ome.zarr")
        return is_zarr

    def isel(self, index: Mapping[int, int | slice]) -> np.ndarray:
        """Select data by dimension index.

        Parameters
        ----------
        index : Mapping[int, int | slice]
            Mapping from dimension index to integer or slice.

        Returns
        -------
        np.ndarray
            The selected data as a numpy array.
        """
        self._update_structure()  # Update in case new data was written

        if not self._shape:
            return np.array([])

        # Convert index mapping to tuple
        idx = tuple(index.get(k, slice(None)) for k in range(len(self._dims)))

        import zarr

        if self._zarr_group is None:
            self._zarr_group = zarr.open(str(self._path), mode="r")

        # First index is position
        position_idx = idx[0] if idx else slice(None)
        rest_idx = idx[1:] if len(idx) > 1 else ()

        # Get position keys
        position_keys = sorted(
            [k for k in self._zarr_group.keys() if k.isdigit()],
            key=int,
        )

        if isinstance(position_idx, int):
            # Single position
            if position_idx < 0:
                position_idx = len(position_keys) + position_idx
            arr = self._zarr_group[position_keys[position_idx]]
            return np.asarray(arr[rest_idx])
        else:
            # Slice of positions
            selected_keys = position_keys[position_idx]
            if not selected_keys:
                return np.array([])

            # Load each position and stack
            arrays = [
                np.asarray(self._zarr_group[key][rest_idx]) for key in selected_keys
            ]
            return np.stack(arrays, axis=0)


class TiffWriterWrapper(OMEWriterWrapperBase):
    """Wrapper for OMEWriterHandler objects that write TIFF files.

    This wrapper provides read access to data written by OMEWriterHandler
    using the tiff backend.

    For TIFF files, all data is stored in a single multi-page file.
    The wrapper loads the entire file into memory for efficient access.

    Examples
    --------
    >>> handler = OMEWriterHandler("/path/to/data.ome.tiff", backend="tiff")
    >>> mmc.mda.run(sequence, output=handler)
    >>> wrapper = TiffWriterWrapper(handler)
    >>> print(wrapper.dims)  # ('c', 'y', 'x')
    >>> data = wrapper.isel({0: 0})  # First channel
    """

    def __init__(self, data: OMEWriterHandler) -> None:
        """Initialize the TIFF wrapper."""
        self._tiff_data: np.ndarray | None = None
        self._position_files: list[Path] = []  # List of position-specific files
        super().__init__(data)

    def _update_structure(self) -> None:
        """Update structure for TIFF backend."""
        import tifffile

        # First check for position-specific files (multi-position case)
        # Pattern: {base}_p000.ome.tiff, {base}_p001.ome.tiff, etc.
        base_path = self._path.parent / self._path.stem.replace(".ome", "")
        position_files = sorted(
            base_path.parent.glob(f"{base_path.name}_p*.ome.tiff")
        )

        if position_files:
            # Multi-position case: use position-specific files
            self._position_files = position_files
            tiff_path = position_files[0]
            is_multi_position = True
        elif self._path.exists():
            # Single file case
            tiff_path = self._path
            self._position_files = [self._path]
            is_multi_position = False
        else:
            # No data written yet
            self._dims = ()
            self._coords = {}
            self._shape = ()
            self._position_files = []
            return

        with tifffile.TiffFile(str(tiff_path)) as tif:
            if not tif.series:
                self._dims = ()
                self._coords = {}
                self._shape = ()
                return

            series = tif.series[0]
            base_shape = tuple(series.shape)
            self._dtype = np.dtype(series.dtype)

            # Get dimension order from OME-XML metadata
            if not tif.is_ome or not tif.ome_metadata:
                raise ValueError(
                    f"No OME metadata found in {tiff_path}. "
                    "Cannot determine dimension names without metadata."
                )

            # Parse OME-XML to extract dimension order
            # The OME-XML contains DimensionOrder attribute (e.g., "XYCZT")
            import xml.etree.ElementTree as ET

            root = ET.fromstring(tif.ome_metadata)
            # Find the Pixels element with namespace handling
            ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
            pixels = root.find(".//ome:Pixels", ns)

            if pixels is None:
                # Try without namespace
                pixels = root.find(".//Pixels")

            if pixels is None or "DimensionOrder" not in pixels.attrib:
                raise ValueError(
                    f"Cannot find DimensionOrder in OME metadata for {tiff_path}. "
                    "Cannot determine dimension names without this information."
                )

            dim_order = pixels.attrib["DimensionOrder"]
            # Convert OME dimension order (e.g., "XYCZT") to lowercase tuple
            # Filter out dimensions with size 1 (they may not be in the actual shape)
            size_map = {
                "X": int(pixels.attrib.get("SizeX", 1)),
                "Y": int(pixels.attrib.get("SizeY", 1)),
                "Z": int(pixels.attrib.get("SizeZ", 1)),
                "C": int(pixels.attrib.get("SizeC", 1)),
                "T": int(pixels.attrib.get("SizeT", 1)),
            }

            # Build dimension names from the order, excluding singleton dimensions
            base_dims = tuple(
                d.lower() for d in reversed(dim_order) if size_map.get(d, 1) > 1
            )

            # Add position dimension if we have multiple position files
            if is_multi_position:
                self._dims = ("p", *base_dims)
                self._shape = (len(self._position_files), *base_shape)
            else:
                self._dims = base_dims
                self._shape = base_shape

        # Create coordinates
        self._coords = {
            dim: range(size) for dim, size in zip(self._dims, self._shape, strict=True)
        }

    def _clear_cache(self) -> None:
        """Clear cached TIFF data."""
        self._tiff_data = None

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[OMEWriterHandler]:
        """Check if this wrapper supports the given object.

        Parameters
        ----------
        obj : Any
            The object to check.

        Returns
        -------
        bool
            True if obj is an OMEWriterHandler for TIFF format.
        """
        if not (
            hasattr(obj, "path")
            and hasattr(obj, "backend")
            and hasattr(obj, "stream")
            and hasattr(obj, "sequenceStarted")
        ):
            return False

        # Check if it's a TIFF file
        path = Path(obj.path)
        is_tiff_suffix = path.suffix in (".tif", ".tiff")
        is_ome_tiff = path.name.endswith((".ome.tif", ".ome.tiff"))
        return is_tiff_suffix or is_ome_tiff

    def isel(self, index: Mapping[int, int | slice]) -> np.ndarray:
        """Select data by dimension index.

        Parameters
        ----------
        index : Mapping[int, int | slice]
            Mapping from dimension index to integer or slice.

        Returns
        -------
        np.ndarray
            The selected data as a numpy array.
        """
        self._update_structure()  # Update in case new data was written

        if not self._shape:
            return np.array([])

        import tifffile

        # Check if we have multiple position files
        if len(self._position_files) > 1:
            # Multi-position case: first dimension is position
            idx = tuple(index.get(k, slice(None)) for k in range(len(self._dims)))
            position_idx = idx[0] if idx else slice(None)
            rest_idx = idx[1:] if len(idx) > 1 else ()

            if isinstance(position_idx, int):
                # Single position requested
                if position_idx < 0:
                    position_idx = len(self._position_files) + position_idx

                with tifffile.TiffFile(str(self._position_files[position_idx])) as tif:
                    data = tif.asarray()
                return np.asarray(data[rest_idx] if rest_idx else data)
            else:
                # Slice of positions requested
                position_indices = range(len(self._position_files))[position_idx]
                if not position_indices:
                    return np.array([])

                # Load each position and stack
                arrays = []
                for pos_idx in position_indices:
                    with tifffile.TiffFile(str(self._position_files[pos_idx])) as tif:
                        data = tif.asarray()
                        arrays.append(data[rest_idx] if rest_idx else data)
                return np.stack(arrays, axis=0)
        else:
            # Single file case
            idx = tuple(index.get(k, slice(None)) for k in range(len(self._dims)))

            # Load the entire TIFF into memory if not already loaded
            if self._tiff_data is None:
                with tifffile.TiffFile(str(self._position_files[0])) as tif:
                    self._tiff_data = tif.asarray()

            return np.asarray(self._tiff_data[idx])


# Legacy alias for backward compatibility
class NGFFWriterWrapper(DataWrapper["OMEWriterHandler"]):
    """Wrapper for OMEWriterHandler objects from pymmcore-plus.

    This is a factory wrapper that automatically creates the appropriate
    subclass (ZarrWriterWrapper or TiffWriterWrapper) based on the file format.

    .. deprecated::
        Use ZarrWriterWrapper or TiffWriterWrapper directly for better type safety
        and explicit format handling.

    Parameters
    ----------
    data : OMEWriterHandler
        The OMEWriterHandler instance that has written or is writing data.

    Examples
    --------
    >>> handler = OMEWriterHandler("/path/to/data.ome.zarr")
    >>> wrapper = NGFFWriterWrapper(handler)  # Auto-detects format
    >>> # Better: use specific wrapper
    >>> wrapper = ZarrWriterWrapper(handler)
    """

    def __new__(cls, data: OMEWriterHandler) -> ZarrWriterWrapper | TiffWriterWrapper:
        """Create appropriate wrapper based on file format."""
        path = Path(data.path)

        # Check if it's a Zarr file
        is_zarr = path.suffix == ".zarr" or path.name.endswith(".ome.zarr")
        if is_zarr:
            return ZarrWriterWrapper(data)  # type: ignore[return-value]

        # Check if it's a TIFF file
        is_tiff_suffix = path.suffix in (".tif", ".tiff")
        is_ome_tiff = path.name.endswith((".ome.tif", ".ome.tiff"))
        if is_tiff_suffix or is_ome_tiff:
            return TiffWriterWrapper(data)  # type: ignore[return-value]

        raise ValueError(
            f"Cannot determine file format from path: {path}. "
            "Expected .ome.zarr or .ome.tiff extension."
        )
