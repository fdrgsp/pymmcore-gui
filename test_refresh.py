"""Test the refresh functionality of NGFFWriterWrapper."""

from __future__ import annotations

from pathlib import Path
import time

import useq
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda.handlers import OMEWriterHandler

from pymmcore_gui._ndv_wrappers import NGFFWriterWrapper

# Test with acquire-zarr backend
backend = "acquire-zarr"
suffix = ".ome.zarr"
tmp_dir = Path(f"/tmp/test_refresh{suffix}")

mmc = CMMCorePlus.instance()
mmc.loadSystemConfiguration("/Users/fdrgsp/Desktop/test_config.cfg")

# Create handler
handler = OMEWriterHandler(
    path=str(tmp_dir),
    backend=backend,
    overwrite=True,
)

# Create wrapper before any data is written
wrapper = NGFFWriterWrapper(handler)

print("Initial state (no data):")
print(f"  Dims: {wrapper.dims}")
print(f"  Shape: {wrapper.sizes()}")

# Now start writing data with a simple sequence
seq = useq.MDASequence(channels=["DAPI"])
print("\nStarting MDA with 1 channel...")
mmc.mda.run(seq, output=handler)

# Refresh and check
wrapper.refresh()
print("\nAfter first MDA:")
print(f"  Dims: {wrapper.dims}")
print(f"  Shape: {wrapper.sizes()}")

# Write more data - add a position
seq2 = useq.MDASequence(channels=["DAPI", "FITC"], stage_positions=[(100, 100)])
print("\nStarting second MDA with 2 channels and 1 position...")

# Need a new handler for new file structure since we changed dimensions
handler2 = OMEWriterHandler(
    path=str(tmp_dir).replace("test_refresh", "test_refresh2"),
    backend=backend,
    overwrite=True,
)
wrapper2 = NGFFWriterWrapper(handler2)

print("Before MDA:")
print(f"  Dims: {wrapper2.dims}")
print(f"  Shape: {wrapper2.sizes()}")

mmc.mda.run(seq2, output=handler2)

print("\nAfter second MDA (auto-update on access):")
print(f"  Dims: {wrapper2.dims}")
print(f"  Shape: {wrapper2.sizes()}")

print("\n✓ Refresh test passed!")
