from __future__ import annotations

import datetime
import gc
from queue import PriorityQueue
from typing import TYPE_CHECKING, cast

import pytest
import useq
from PyQt6.QtWidgets import QApplication, QWidget
from useq import MDASequence

from pymmcore_gui._ndv_viewers import NDVViewersManager
from pymmcore_gui.writers import (
    OMETiffWriterMM,
    OMEZarrWriterMM,
    TensorStoreWriterMM,
    TiffSequenceWriterMM,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot


writers = [
    (OMETiffWriterMM, "t.ome.tiff"),
    (OMEZarrWriterMM, "z.ome.zarr"),
    (TensorStoreWriterMM, "ts.tensorstore.zarr"),
    (TiffSequenceWriterMM, "t_seq"),
]


@pytest.mark.parametrize("writers", writers)
def test_viewers_manager(
    writers: tuple, mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    """Ensure that the viewers manager creates and cleans up viewers during MDA."""
    dummy = QWidget()
    manager = NDVViewersManager(dummy, mmcore)

    writer, name = writers

    if writer == TensorStoreWriterMM:
        wr = writer(driver="zarr", path=tmp_path / name, delete_existing=True)
    else:
        wr = writer(tmp_path / name)

    assert len(manager) == 0
    with qtbot.waitSignal(mmcore.mda.events.sequenceFinished, timeout=1000):
        mmcore.mda.run(
            MDASequence(
                stage_positions=[(0, 0)],
                time_plan=useq.TIntervalLoops(
                    interval=datetime.timedelta(seconds=0.1), loops=2
                ),
                channels=["DAPI", "FITC"],  # pyright: ignore
                z_plan=useq.ZRangeAround(range=4, step=1),
                metadata={"mm_handler": wr},
            ),
        )
        assert len(manager) == 1

    with qtbot.waitSignal(dummy.destroyed, timeout=1000):
        dummy.deleteLater()
    QApplication.processEvents()
    gc.collect()
    # only checking for strong references when WE have created the datahandler.
    # otherwise... the NDV datawrapper itself may be holding a strong ref?
    # need to look into this...
    if len(manager):
        for viewer in manager.viewers():
            if "vispy" in type(viewer._canvas).__name__.lower():
                # don't even bother... vispy is a mess of hard references
                del viewer._canvas
                continue
            referrers = gc.get_referrers(viewer)[1:]
            pytest.fail(f"Viewer {viewer} not deleted. Still referenced by {referrers}")
