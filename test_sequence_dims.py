"""Test that wrappers use sequence information for dimension detection."""

import tempfile
from pathlib import Path

import useq
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda.handlers import OMEWriterHandler

from pymmcore_gui._ndv_wrappers import ZarrWriterWrapper

# Configure microscope
mmc = CMMCorePlus.instance()
mmc.loadSystemConfiguration("tests/test_config.cfg")

print("=" * 70)
print("Testing Sequence-Based Dimension Detection")
print("=" * 70)

# Create multi-dimensional sequence
seq = useq.MDASequence(
    channels=["DAPI", "FITC"],  # 2 channels (available in test config)
    stage_positions=[(0, 0), (100, 100)],  # 2 positions
)

with tempfile.TemporaryDirectory() as tmpdir:
    output_path = Path(tmpdir) / "test.ome.zarr"

    handler = OMEWriterHandler(
        path=str(output_path),
        backend="tensorstore",
        overwrite=True,
    )

    # Create wrapper BEFORE running MDA
    print("\n1. Creating wrapper BEFORE MDA starts...")
    wrapper = ZarrWriterWrapper(handler)
    print(f"   dims: {wrapper.dims}")
    print(f"   shape: {dict(zip(wrapper.dims, wrapper.sizes().values(), strict=True)) if wrapper.dims else 'empty'}")

    # Sequence is set when MDA starts
    print("\n2. Starting MDA...")

    # Use a callback to check dimensions during acquisition
    frame_count = [0]

    def on_frame(image, event, metadata):
        frame_count[0] += 1
        if frame_count[0] == 1:  # First frame
            print(f"\n3. After first frame acquired:")
            print(f"   Sequence available: {handler.sequence is not None}")
            if handler.sequence:
                print(f"   Sequence sizes: {handler.sequence.sizes}")
            wrapper.refresh()  # Force update
            print(f"   Wrapper dims: {wrapper.dims}")
            print(f"   Wrapper shape: {dict(zip(wrapper.dims, wrapper.sizes().values(), strict=True)) if wrapper.dims else 'empty'}")

    mmc.mda.events.frameReady.connect(on_frame)

    mmc.mda.run(seq, output=handler)

    mmc.mda.events.frameReady.disconnect(on_frame)

    print(f"\n4. After MDA finished:")
    wrapper.refresh()
    print(f"   dims: {wrapper.dims}")
    print(f"   shape: {dict(zip(wrapper.dims, wrapper.sizes().values(), strict=True))}")
    print(f"   dtype: {wrapper.dtype}")

    # Verify dimensions match sequence
    expected_sizes = seq.sizes
    actual_sizes = wrapper.sizes()

    print(f"\n5. Validation:")
    print(f"   Expected from sequence: {expected_sizes}")
    print(f"   Actual from wrapper: {actual_sizes}")

    # Check that we have the expected dimensions
    assert "p" in wrapper.dims, "Missing position dimension"
    assert "c" in wrapper.dims, "Missing channel dimension"
    assert "y" in wrapper.dims, "Missing y dimension"
    assert "x" in wrapper.dims, "Missing x dimension"

    assert actual_sizes["p"] == expected_sizes["p"], f"Position count mismatch"
    assert actual_sizes["c"] == expected_sizes["c"], f"Channel count mismatch"

    print("\n✓ All dimensions correctly detected from sequence!")

print("\n" + "=" * 70)
print("✓ Sequence-Based Detection Test PASSED")
print("=" * 70)
