"""Complete test script for NGFFWriterWrapper with visualization."""

from __future__ import annotations

from pathlib import Path

import useq
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda.handlers import OMEWriterHandler

from pymmcore_gui._ndv_wrappers import NGFFWriterWrapper

# Test with acquire-zarr backend
backend = "acquire-zarr"
# backend = "tiff"

suffix = ".ome.zarr" if backend != "tiff" else ".ome.tiff"
tmp_dir = Path(f"/Users/fdrgsp/Desktop/t/data{suffix}")

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
print("Testing NGFFWriterWrapper")
print("=" * 60)

wrapper = NGFFWriterWrapper(handler)

print("\nWrapper created successfully!")
print(f"Data type: {type(wrapper.data)}")
print(f"Dims: {wrapper.dims}")
print(f"Shape: {wrapper.sizes()}")
print(f"Coords: {dict(wrapper.coords)}")
print(f"Dtype: {wrapper.dtype}")
print(f"Summary: {wrapper.summary_info()}")

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

# Test the wrapper is recognized by ndv
print("\n" + "=" * 60)
print("Testing ndv compatibility")
print("=" * 60)

try:
    import ndv

    # Test that DataWrapper.create can handle this
    from ndv import DataWrapper

    # Should recognize it's already a DataWrapper
    wrapped = DataWrapper.create(wrapper)
    print("✓ Wrapper is recognized by ndv.DataWrapper.create()")
    print(f"  Wrapped type: {type(wrapped)}")

    # # Uncomment to actually show in ndv viewer
    # print("\nShowing in ndv viewer...")
    # ndv.imshow(wrapper)

except ImportError:
    print("ndv not available - skipping visualization test")

print("\n✓ All tests passed!")
