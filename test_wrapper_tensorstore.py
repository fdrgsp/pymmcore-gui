"""Test script for NGFFWriterWrapper with tensorstore backend."""

from __future__ import annotations

from pathlib import Path

import useq
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda.handlers import OMEWriterHandler

from pymmcore_gui._ndv_wrappers import NGFFWriterWrapper

# Test with tensorstore backend
backend = "tensorstore"
suffix = ".ome.zarr"
tmp_dir = Path(f"/tmp/test_data_tensorstore{suffix}")

seq = useq.MDASequence(channels=["DAPI", "FITC"], stage_positions=[(0, 0), (100, 100)])

mmc = CMMCorePlus.instance()
mmc.loadSystemConfiguration("/Users/fdrgsp/Desktop/test_config.cfg")

handler = OMEWriterHandler(
    path=str(tmp_dir),
    backend=backend,
    overwrite=True,
)

print(f"Running MDA with backend: {backend}")
mmc.mda.run(seq, output=handler)
print("MDA finished")

print("\n" + "=" * 60)
print("Testing NGFFWriterWrapper with tensorstore")
print("=" * 60)

wrapper = NGFFWriterWrapper(handler)

print("\nWrapper created successfully!")
print(f"Data type: {type(wrapper.data)}")
print(f"Dims: {wrapper.dims}")
print(f"Shape: {wrapper.sizes()}")
print(f"Coords: {dict(wrapper.coords)}")
print(f"Dtype: {wrapper.dtype}")

# Test data access
print("\n" + "=" * 60)
print("Testing data access")
print("=" * 60)

# Get first position, first channel
data_slice = wrapper.isel({0: 0, 1: 0})
print(f"\nFirst position, first channel shape: {data_slice.shape}")
print(f"Data type: {data_slice.dtype}")
print(f"Min: {data_slice.min()}, Max: {data_slice.max()}")

# Get all data for position 0
data_slice = wrapper.isel({0: 0})
print(f"\nPosition 0 shape: {data_slice.shape}")

print("\n✓ All tensorstore tests passed!")
