"""Example showing how to use OMEWriterHandler with ndv.

This example demonstrates how to:
1. Create an OMEWriterHandler to write data
2. Wrap it with ndv's OMEWriterWrapper for visualization
3. View the data in ndv

Note: This requires pymmcore-plus with ome-writers support installed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ome_writers import Dimension, create_stream
from pymmcore_plus.mda.handlers import OMEWriterHandler

from pymmcore_gui._ndv_wrappers import NGFFWriterWrapper

if TYPE_CHECKING:
    from typing import TypeAlias

    Index: TypeAlias = int | slice


backend = "acquire-zarr"
# backend="acquire-zarr"
# backend = "tensorstore"
# backend="tiff"

# Create temporary directory
# tmp_dir = Path(
#     tempfile.mkdtemp(suffix=".ome.zarr" if backend != "tiff" else ".ome.tiff")
# )
tmp_dir = Path("/Users/fdrgsp/Desktop/t")
print(f"Writing data to: {tmp_dir}")


# Simulate a microscopy acquisition with dimensions (t, c, z, y, x)
# Let's create a simple 4D dataset: 3 timepoints, 2 channels, 5 z-slices
# and 2 positions
nt, nc, nz, ny, nx = 3, 2, 5, 128, 128
num_positions = 2

# Create dimension specifications (same for all positions)
dims = [
    Dimension(label="t", size=nt, unit=(1.0, "s")),
    Dimension(label="c", size=nc),
    Dimension(label="z", size=nz, unit=(1.0, "um")),
    Dimension(label="y", size=ny, unit=(0.5, "um"), chunk_size=64),
    Dimension(label="x", size=nx, unit=(0.5, "um"), chunk_size=64),
]

# Create some example data
print(f"Generating example data with {num_positions} positions...")

# Create the handler
handler = OMEWriterHandler(
    path=str(tmp_dir),
    backend=backend,
    overwrite=True,
)

# Write data for each position
print("Writing data to OME-Zarr...")
for pos in range(num_positions):
    # Create a stream for each position
    stream = create_stream(
        str(tmp_dir / str(pos)),
        dtype=np.uint16,
        dimensions=dims,
        backend=backend,
        overwrite=True,
    )

    # Generate unique data for this position
    data = np.random.randint(0, 65535, size=(nt, nc, nz, ny, nx), dtype=np.uint16)
    # Add a constant offset to make positions visually distinct
    data = data + pos * 10000

    # Write data frame by frame (as would happen during acquisition)
    for t in range(nt):
        for c in range(nc):
            for z in range(nz):
                frame = data[t, c, z]
                stream.append(frame)

    # Flush the stream to ensure all data is written
    stream.flush()
    print(f"  Position {pos} written successfully!")

from rich import print
print(list(tmp_dir.iterdir()))

print("\nAll positions written successfully!")

# Now visualize the data using ndv
print("\nVisualizing with ndv...")

# The OMEWriterWrapper will automatically detect the handler
# and provide read access to the data
wrapper = NGFFWriterWrapper(handler)

# print(wrapper.data)
# print(wrapper.data.dtype)
# print(wrapper.data.shape)
# ndv.imshow(wrapper)
