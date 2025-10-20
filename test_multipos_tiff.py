"""Test multi-position TIFF support."""

import tempfile
from pathlib import Path

import useq
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda.handlers import OMEWriterHandler

from pymmcore_gui._ndv_wrappers import TiffWriterWrapper

# Configure microscope
mmc = CMMCorePlus.instance()
mmc.loadSystemConfiguration("tests/test_config.cfg")

print("=" * 70)
print("Testing Multi-Position TIFF Support")
print("=" * 70)

# Create multi-position sequence
seq = useq.MDASequence(
    channels=["DAPI", "FITC"],
    stage_positions=[(0, 0), (100, 100), (200, 200)],  # 3 positions
)

# Run MDA with TIFF backend
with tempfile.TemporaryDirectory() as tmpdir:
    output_path = Path(tmpdir) / "multipos.ome.tiff"

    handler = OMEWriterHandler(
        path=str(output_path),
        backend="tiff",
        overwrite=True,
    )

    print(f"\nRunning MDA with {len(seq.stage_positions)} positions...")
    mmc.mda.run(seq, output=handler)
    print("✓ MDA completed")

    # Check what files were created
    print(f"\nFiles in {tmpdir}:")
    for f in sorted(Path(tmpdir).glob("*.tiff")):
        print(f"  - {f.name} ({f.stat().st_size} bytes)")

    # Create wrapper
    wrapper = TiffWriterWrapper(handler)
    print(f"\n✓ Wrapper created")

    # Check dimensions
    print(f"\nDimensions:")
    print(f"  dims: {wrapper.dims}")
    print(f"  shape: {dict(zip(wrapper.dims, wrapper.sizes().values()))}")
    print(f"  dtype: {wrapper.dtype}")

    # Verify we have position dimension
    assert "p" in wrapper.dims, "Missing position dimension!"
    assert wrapper.sizes()["p"] == 3, f"Expected 3 positions, got {wrapper.sizes()['p']}"
    assert wrapper.sizes()["c"] == 2, f"Expected 2 channels, got {wrapper.sizes()['c']}"

    print(f"\n✓ Position dimension correctly detected!")

    # Test data access for each position
    print(f"\nTesting data access:")
    for p in range(3):
        data = wrapper.isel({0: p, 1: 0})  # Position p, channel 0
        print(f"  Position {p}: shape={data.shape}, dtype={data.dtype}")
        assert data.shape == (512, 512), f"Unexpected shape: {data.shape}"

    print(f"\n✓ Data access works for all positions!")

    # Test slice of positions
    print(f"\nTesting position slicing:")
    data = wrapper.isel({0: slice(0, 2), 1: 0})  # First 2 positions, channel 0
    print(f"  Slice [0:2]: shape={data.shape}")
    assert data.shape == (2, 512, 512), f"Unexpected shape: {data.shape}"

    print(f"\n✓ Position slicing works!")

print("\n" + "=" * 70)
print("✓ Multi-Position TIFF Test PASSED")
print("=" * 70)
