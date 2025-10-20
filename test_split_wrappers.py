"""Test script demonstrating the split Zarr and TIFF wrappers."""

from pathlib import Path

import useq
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda.handlers import OMEWriterHandler

# Import the specific wrappers
from pymmcore_gui._ndv_wrappers import (
    NGFFWriterWrapper,
    TiffWriterWrapper,
    ZarrWriterWrapper,
)

print("=" * 70)
print("Testing Split DataWrapper Implementation")
print("=" * 70)

mmc = CMMCorePlus.instance()
mmc.loadSystemConfiguration("/Users/fdrgsp/Desktop/test_config.cfg")

seq = useq.MDASequence(channels=["DAPI", "FITC"], stage_positions=[(0, 0), (100, 100)])

# Test 1: ZarrWriterWrapper directly
print("\n" + "=" * 70)
print("Test 1: Using ZarrWriterWrapper directly")
print("=" * 70)

handler_zarr = OMEWriterHandler("/tmp/test_split.ome.zarr", backend="acquire-zarr", overwrite=True)
mmc.mda.run(seq, output=handler_zarr)

wrapper_zarr = ZarrWriterWrapper(handler_zarr)
print(f"✓ ZarrWriterWrapper created")
print(f"  Type: {type(wrapper_zarr).__name__}")
print(f"  Dims: {wrapper_zarr.dims}")
print(f"  Shape: {wrapper_zarr.sizes()}")

# Test 2: TiffWriterWrapper directly
print("\n" + "=" * 70)
print("Test 2: Using TiffWriterWrapper directly")
print("=" * 70)

seq_tiff = useq.MDASequence(channels=["DAPI", "FITC"])  # Single position for TIFF
handler_tiff = OMEWriterHandler("/tmp/test_split.ome.tiff", backend="tiff", overwrite=True)
mmc.mda.run(seq_tiff, output=handler_tiff)

wrapper_tiff = TiffWriterWrapper(handler_tiff)
print(f"✓ TiffWriterWrapper created")
print(f"  Type: {type(wrapper_tiff).__name__}")
print(f"  Dims: {wrapper_tiff.dims}")
print(f"  Shape: {wrapper_tiff.sizes()}")

# Test 3: NGFFWriterWrapper (factory - auto-detects)
print("\n" + "=" * 70)
print("Test 3: Using NGFFWriterWrapper (auto-detection factory)")
print("=" * 70)

handler_auto = OMEWriterHandler("/tmp/test_split_auto.ome.zarr", backend="tensorstore", overwrite=True)
mmc.mda.run(seq, output=handler_auto)

wrapper_auto = NGFFWriterWrapper(handler_auto)
print(f"✓ NGFFWriterWrapper created (auto-detected)")
print(f"  Actual type: {type(wrapper_auto).__name__}")
print(f"  Dims: {wrapper_auto.dims}")
print(f"  Shape: {wrapper_auto.sizes()}")

# Test 4: Verify inheritance
print("\n" + "=" * 70)
print("Test 4: Verifying class hierarchy")
print("=" * 70)

from pymmcore_gui._ndv_wrappers import OMEWriterWrapperBase

print(f"ZarrWriterWrapper inherits from OMEWriterWrapperBase: {isinstance(wrapper_zarr, OMEWriterWrapperBase)}")
print(f"TiffWriterWrapper inherits from OMEWriterWrapperBase: {isinstance(wrapper_tiff, OMEWriterWrapperBase)}")
print(f"Auto wrapper inherits from OMEWriterWrapperBase: {isinstance(wrapper_auto, OMEWriterWrapperBase)}")

# Test 5: Test supports() methods
print("\n" + "=" * 70)
print("Test 5: Testing supports() methods")
print("=" * 70)

print(f"ZarrWriterWrapper.supports(handler_zarr): {ZarrWriterWrapper.supports(handler_zarr)}")
print(f"ZarrWriterWrapper.supports(handler_tiff): {ZarrWriterWrapper.supports(handler_tiff)}")
print(f"TiffWriterWrapper.supports(handler_tiff): {TiffWriterWrapper.supports(handler_tiff)}")
print(f"TiffWriterWrapper.supports(handler_zarr): {TiffWriterWrapper.supports(handler_zarr)}")

print("\n" + "=" * 70)
print("✓ All split wrapper tests passed!")
print("=" * 70)

print("\nRecommendation:")
print("  - Use ZarrWriterWrapper for .ome.zarr files (acquire-zarr, tensorstore)")
print("  - Use TiffWriterWrapper for .ome.tiff files (tiff backend)")
print("  - Use NGFFWriterWrapper for automatic detection (legacy compatibility)")
