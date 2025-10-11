"""Example showing how to use OMEWriterHandler with ndv.

This example demonstrates how to:
1. Create an OMEWriterHandler to write data
2. Wrap it with ndv's OMEWriterWrapper for visualization
3. View the data in ndv

Note: This requires pymmcore-plus with ome-writers support installed.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

import ndv
import numpy as np
from ndv import DataWrapper
from ome_writers import Dimension

from pymmcore_plus.mda.handlers import OMEWriterHandler

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence
    from typing import Any, Union

    from typing_extensions import TypeAlias

    Index: TypeAlias = Union[int, slice]




class OMEWriterWrapper(DataWrapper):
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
        self._reader: Any = None  # Will be set by _init_reader
        self._dims: tuple[Hashable, ...] = ()
        self._coords: Mapping[Hashable, Sequence] = {}

        # Initialize the reader based on the file type
        self._init_reader()

        # Call parent init with the reader object
        super().__init__(self._reader)

    def _init_reader(self) -> None:
        """Initialize the appropriate reader based on file extension."""
        path_str = str(self._path)

        # Determine backend from path
        if path_str.endswith('.zarr') or path_str.endswith('.ome.zarr'):
            self._backend = 'zarr'
            self._init_zarr_reader()
        elif path_str.endswith('.tiff') or path_str.endswith('.ome.tiff'):
            self._backend = 'tiff'
            self._init_tiff_reader()
        else:
            # Try to infer from what exists on disk
            if self._path.exists():
                if self._path.is_dir():
                    self._backend = 'zarr'
                    self._init_zarr_reader()
                elif self._path.is_file():
                    self._backend = 'tiff'
                    self._init_tiff_reader()
            else:
                raise ValueError(
                    f"Cannot determine backend for path: {self._path}. "
                    "File does not exist yet."
                )

    def _init_zarr_reader(self) -> None:
        """Initialize Zarr reader."""
        # Try tensorstore first, then zarr
        if (ts := sys.modules.get("tensorstore")):
            # Use tensorstore to read
            try:
                spec = {
                    "driver": "zarr3",
                    "kvstore": {"driver": "file", "path": str(self._path / "0")},
                }
                self._reader = ts.open(spec).result()
                self._extract_dims_from_tensorstore()
            except Exception:
                # Fall back to zarr
                self._init_zarr_python_reader()
        else:
            self._init_zarr_python_reader()

    def _init_zarr_python_reader(self) -> None:
        """Initialize zarr-python reader."""
        try:
            import zarr
        except ImportError as e:
            raise ImportError(
                "Either tensorstore or zarr is required to read Zarr data"
            ) from e

        # Open the zarr array (position 0 for now)
        self._reader = zarr.open_array(str(self._path), mode="r", path="0")
        self._extract_dims_from_zarr()

    def _init_tiff_reader(self) -> None:
        """Initialize TIFF reader."""
        try:
            import tifffile
        except ImportError as e:
            raise ImportError("tifffile is required to read TIFF data") from e

        # For multi-position, we need to handle multiple files
        # For now, open the first one (position 0)
        if '_p000' in self._path.name or '_p001' in self._path.name:
            # This is already a position-specific file
            tiff_path = self._path
        else:
            # Check if position-specific files exist
            base = self._path.parent / self._path.stem
            pos0 = base.parent / f"{base.name}_p000{self._path.suffix}"
            if pos0.exists():
                tiff_path = pos0
            else:
                tiff_path = self._path

        with tifffile.TiffFile(str(tiff_path)) as tif:
            self._reader = tif.asarray()
            self._extract_dims_from_tiff(tif)

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
                        self._dims = tuple(ax["name"] for ax in axes)
                        shape = self._reader.shape
                        self._coords = {
                            dim: range(size) for dim, size in zip(self._dims, shape)
                        }
                        return

        # Fallback: use numeric dimensions
        shape = self._reader.shape
        self._dims = tuple(range(len(shape)))
        self._coords = {i: range(s) for i, s in enumerate(shape)}

    def _extract_dims_from_tensorstore(self) -> None:
        """Extract dimension information from TensorStore."""
        # TensorStore has labels in the domain
        labels = self._reader.domain.labels
        if labels and any(labels):
            self._dims = tuple(
                str(label) if label else i for i, label in enumerate(labels)
            )
        else:
            self._dims = tuple(range(len(self._reader.shape)))

        shape = self._reader.domain.shape
        self._coords = {dim: range(size) for dim, size in zip(self._dims, shape)}

    def _extract_dims_from_tiff(self, tif: Any) -> None:
        """Extract dimension information from TIFF metadata."""
        # Try to parse OME-XML metadata
        try:
            import ome_types

            if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                ome = ome_types.from_xml(tif.ome_metadata)
                if ome.images:
                    pixels = ome.images[0].pixels
                    # Build dimensions from OME metadata
                    dim_order = str(pixels.dimension_order).lower()
                    sizes = {
                        't': pixels.size_t or 1,
                        'c': pixels.size_c or 1,
                        'z': pixels.size_z or 1,
                        'y': pixels.size_y,
                        'x': pixels.size_x,
                    }
                    # Use the dimension order from OME
                    self._dims = tuple(dim_order)
                    self._coords = {dim: range(sizes[dim]) for dim in self._dims}
                    return
        except Exception:
            pass

        # Fallback: assume standard 5D TIFF (T, C, Z, Y, X)
        shape = self._reader.shape
        if len(shape) == 5:
            self._dims = ('t', 'c', 'z', 'y', 'x')
        elif len(shape) == 4:
            self._dims = ('t', 'z', 'y', 'x')
        elif len(shape) == 3:
            self._dims = ('z', 'y', 'x')
        elif len(shape) == 2:
            self._dims = ('y', 'x')
        else:
            self._dims = tuple(range(len(shape)))

        self._coords = {dim: range(size) for dim, size in zip(self._dims, shape)}

    @property
    def dims(self) -> tuple[Hashable, ...]:
        """Return the dimension labels for the data."""
        return self._dims

    @property
    def coords(self) -> Mapping[Hashable, Sequence]:
        """Return the coordinates for the data."""
        return self._coords

    def isel(self, index: Mapping[int, int | slice]) -> np.ndarray:
        """Return a slice of the data as a numpy array."""
        idx = tuple(index.get(k, slice(None)) for k in range(len(self.dims)))

        if self._backend == 'tiff':
            # For TIFF, data is already in memory
            return np.asarray(self._reader[idx])
        # For Zarr/TensorStore
        return self._asarray(self._reader[idx])

    def _asarray(self, data: Any) -> np.ndarray:
        """Convert data to a numpy array."""
        # For tensorstore, need to read the result
        if hasattr(data, 'read') and callable(data.read):
            return np.asarray(data.read().result())
        return np.asarray(data)

    @classmethod
    def supports(cls, obj: Any) -> bool:
        """Check if this wrapper supports the given object.

        Returns True if obj is an OMEWriterHandler from pymmcore-plus.
        """
        # Check if it has the key attributes of OMEWriterHandler
        return (
            hasattr(obj, 'path')
            and hasattr(obj, 'stream')
            and hasattr(obj, 'backend')
            and hasattr(obj.stream, 'is_active')
            and hasattr(obj.stream, 'append')
        )





def example():
    """Example of using OMEWriterHandler with ndv."""
    # Create a temporary OME-Zarr file
    import tempfile
    from pathlib import Path

    # Create temporary directory
    tmp_dir = Path(tempfile.mkdtemp(suffix=".ome.zarr"))
    print(f"Writing data to: {tmp_dir}")

    # Create an OMEWriterHandler
    handler = OMEWriterHandler(
        path=str(tmp_dir),
        backend="acquire-zarr",  # or "acquire-zarr" or "tiff"
        overwrite=True,
    )

    # Simulate a microscopy acquisition with dimensions (t, c, z, y, x)
    # Let's create a simple 4D dataset: 3 timepoints, 2 channels, 5 z-slices
    nt, nc, nz, ny, nx = 3, 2, 5, 128, 128

    # Create some example data
    print("Generating example data...")
    data = np.random.randint(0, 65535, size=(nt, nc, nz, ny, nx), dtype=np.uint16)

    # Create dimension specifications
    dims = [
        Dimension(label="t", size=nt, unit=(1.0, "s")),
        Dimension(label="c", size=nc),
        Dimension(label="z", size=nz, unit=(1.0, "um")),
        Dimension(label="y", size=ny, unit=(0.5, "um"), chunk_size=64),
        Dimension(label="x", size=nx, unit=(0.5, "um"), chunk_size=64),
    ]

    # Create the stream
    from ome_writers import create_stream

    handler.stream = create_stream(
        str(tmp_dir),
        dtype=data.dtype,
        dimensions=dims,
        backend="tensorstore",
        overwrite=True,
    )

    # Write data frame by frame (as would happen during acquisition)
    print("Writing data to OME-Zarr...")
    for t in range(nt):
        for c in range(nc):
            for z in range(nz):
                frame = data[t, c, z]
                handler.stream.append(frame)

    # Flush the stream to ensure all data is written
    handler.stream.flush()
    print("Data written successfully!")

    # Now visualize the data using ndv
    print("\nVisualizing with ndv...")

    # The OMEWriterWrapper will automatically detect the handler
    # and provide read access to the data
    ndv.imshow(handler)

    print("Viewer created! Close the window to continue...")

    # Alternatively, you can also open the Zarr file directly
    # (once the stream is flushed)
    print("\nAlternative: Opening Zarr file directly...")
    import zarr

    zarr_data = zarr.open_array(str(tmp_dir), mode="r", path="0")
    ndv.imshow(zarr_data)

example()