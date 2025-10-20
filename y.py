"""Example showing how to use OMEWriterHandler with ndv.

This example demonstrates how to:
1. Create an OMEWriterHandler to write data
2. Wrap it with ndv's OMEWriterWrapper for visualization
3. View the data in ndv

Note: This requires pymmcore-plus with ome-writers support installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import ndv
import useq
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda.handlers import OMEWriterHandler

from pymmcore_gui._ndv_wrappers import TiffWriterWrapper, ZarrWriterWrapper

if TYPE_CHECKING:
    from typing import TypeAlias

    Index: TypeAlias = int | slice


# backend = "acquire-zarr"
backend = "tensorstore"
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


mmc.mda.run(seq, output=handler)


wrapper = (
    TiffWriterWrapper(handler) if backend == "tiff" else ZarrWriterWrapper(handler)
)

print(wrapper.data)
print(wrapper.dtype)
print(wrapper.sizes())
ndv.imshow(wrapper)
