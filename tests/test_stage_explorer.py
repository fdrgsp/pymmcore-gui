from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import useq

from pymmcore_gui.widgets._stage_explorer import ThemedStageExplorer

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pytest import MonkeyPatch
    from pytestqt.qtbot import QtBot


def test_acquisition_display_uses_cached_camera_geometry(
    mmcore: CMMCorePlus, qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    explorer = ThemedStageExplorer(mmcore=mmcore)
    qtbot.addWidget(explorer)
    explorer.show()
    qtbot.waitExposed(explorer)
    fov = explorer._fov_w_h()

    core = MagicMock(wraps=mmcore)
    for name in (
        "getImageWidth",
        "getImageHeight",
        "getPixelSizeUm",
        "getImageBitDepth",
    ):
        getattr(core, name).side_effect = AssertionError("camera is busy acquiring")

    with monkeypatch.context() as patch:
        patch.setattr(explorer, "_mmc", core)
        image = np.zeros((32, 32), dtype=np.uint16)
        explorer._on_config_set("Channel", "DAPI")
        explorer._on_frame_ready(image, useq.MDAEvent(x_pos=100, y_pos=200))
        explorer._update_stage_pos_marker(100, 200)
        assert explorer._fov_w_h() == fov
        assert len(explorer._tiles) == 1
