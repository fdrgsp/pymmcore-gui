from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import tifffile

from pymmcore_gui._array_viewer import _save_as_tiff, _save_multiposition

if TYPE_CHECKING:
    from pathlib import Path


def test_save_as_ome_tiff_with_scales(tmp_path: Path) -> None:
    path = tmp_path / "image.ome.tif"
    data = np.zeros((2, 3, 16, 16), dtype=np.uint16)

    _save_as_tiff(
        data,
        str(path),
        pixel_size_um=0.65,
        z_step_um=0.5,
        axes="CZYX",
    )

    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].axes == "CZYX"
        assert tif.series[0].shape == data.shape
        assert "0.65" in (tif.ome_metadata or "")
        assert "0.5" in (tif.ome_metadata or "")


def test_save_multiposition_as_separate_ome_tiffs(tmp_path: Path) -> None:
    data = np.zeros((2, 2, 16, 16), dtype=np.uint16)

    _save_multiposition(
        data,
        {"p": 2, "c": 2},
        str(tmp_path / "mda.ome.tif"),
        pixel_size_um=None,
        z_step_um=None,
        axes="CYX",
    )

    files = sorted(tmp_path.glob("*.ome.tif"))
    assert [path.name for path in files] == [
        "mda_p000.ome.tif",
        "mda_p001.ome.tif",
    ]
    assert all(tifffile.imread(path).shape == (2, 16, 16) for path in files)
