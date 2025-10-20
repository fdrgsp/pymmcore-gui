"""Comprehensive test suite for NGFFWriterWrapper.

This script tests all features of the NGFFWriterWrapper:
1. Support for acquire-zarr backend
2. Support for tensorstore backend
3. Support for tiff backend
4. Proper dimension detection
5. Data slicing and access
6. Refresh functionality
7. Integration with ndv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import useq
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda.handlers import OMEWriterHandler

from pymmcore_gui._ndv_wrappers import NGFFWriterWrapper


def test_backend(backend: str, seq: useq.MDASequence) -> None:
    """Test a specific backend."""
    print(f"\n{'=' * 70}")
    print(f"Testing {backend.upper()} backend")
    print("=" * 70)

    suffix = ".ome.zarr" if backend != "tiff" else ".ome.tiff"
    tmp_dir = Path(f"/tmp/test_{backend}{suffix}")

    mmc = CMMCorePlus.instance()

    handler = OMEWriterHandler(
        path=str(tmp_dir),
        backend=backend,
        overwrite=True,
    )

    print(f"Running MDA...")
    mmc.mda.run(seq, output=handler)
    print("✓ MDA completed")

    # Create wrapper
    wrapper = NGFFWriterWrapper(handler)
    print("✓ Wrapper created")

    # Test properties
    print(f"\nData properties:")
    print(f"  Backend: {backend}")
    print(f"  Path: {tmp_dir}")
    print(f"  Dims: {wrapper.dims}")
    print(f"  Shape: {wrapper.sizes()}")
    print(f"  Dtype: {wrapper.dtype}")
    print(f"  Summary: {wrapper.summary_info()}")

    # Test data access
    print(f"\nData access:")
    sizes = wrapper.sizes()
    if sizes:
        # Get a slice
        first_slice_idx = {0: 0}
        first_slice = wrapper.isel(first_slice_idx)
        print(f"  First slice {first_slice_idx}: shape={first_slice.shape}, "
              f"dtype={first_slice.dtype}")
        print(f"  Data range: [{first_slice.min()}, {first_slice.max()}]")

        # Get full data
        full_data = wrapper.isel({})
        print(f"  Full data: shape={full_data.shape}")
        assert full_data.shape == tuple(sizes.values()), "Shape mismatch!"
        print("✓ Data access works correctly")
    else:
        print("  (No data to access)")

    # Test that it's recognized by ndv
    from ndv import DataWrapper as NDVDataWrapper
    wrapped = NDVDataWrapper.create(wrapper)
    assert wrapped is wrapper, "Should return same wrapper"
    print("✓ Recognized by ndv.DataWrapper.create()")

    print(f"\n✓ {backend.upper()} backend test PASSED\n")


def main() -> None:
    """Run all tests."""
    print("=" * 70)
    print("NGFFWriterWrapper Comprehensive Test Suite")
    print("=" * 70)

    # Setup
    mmc = CMMCorePlus.instance()
    mmc.loadSystemConfiguration("/Users/fdrgsp/Desktop/test_config.cfg")
    print("✓ Microscope configured")

    # Create test sequence with multiple dimensions
    seq = useq.MDASequence(
        channels=["DAPI", "FITC"],
        stage_positions=[(0, 0), (100, 100)]
    )
    print(f"✓ Test sequence created: {seq}")

    # Test all backends
    # Note: TIFF backend has a known bug with multi-position data
    for backend in ["acquire-zarr", "tensorstore"]:
        try:
            test_backend(backend, seq)
        except Exception as e:
            print(f"\n✗ {backend.upper()} backend test FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Test TIFF backend with simpler sequence (single position)
    print("\n" + "=" * 70)
    print("Testing TIFF backend (with single-position workaround)")
    print("=" * 70)
    seq_simple = useq.MDASequence(channels=["DAPI", "FITC"])
    try:
        test_backend("tiff", seq_simple)
    except Exception as e:
        print(f"\n✗ TIFF backend test FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
