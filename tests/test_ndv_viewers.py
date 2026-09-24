from __future__ import annotations

import datetime
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import ndv
import useq
from useq import MDASequence

import pymmcore_gui._ndv_viewers as viewers_module
from pymmcore_gui._grid_axis import GridAxisDataWrapper, GridAxisLayoutKind
from pymmcore_gui._ndv_viewers import NDVViewersManager, _StreamSignalBridge
from pymmcore_gui._qt.QtWidgets import QApplication, QWidget

if TYPE_CHECKING:
    import pytest
    from ndv.models import ArrayDisplayModel
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot


class _Emitter:
    def __init__(self) -> None:
        self.calls = 0

    def emit(self) -> None:
        self.calls += 1


def test_stream_dimension_bridge_marshals_to_gui_thread(
    qtbot: QtBot, qapp: QApplication
) -> None:
    """Live coordinate growth must never mutate ndv widgets from its writer thread."""
    parent = QWidget()
    qtbot.addWidget(parent)
    callback_threads: list[int] = []
    bridge = _StreamSignalBridge(
        lambda: callback_threads.append(threading.get_ident()), parent
    )

    worker = threading.Thread(target=bridge.dimsChanged.emit)
    worker.start()
    worker.join()

    assert callback_threads == []
    qtbot.waitUntil(lambda: bool(callback_threads))
    assert callback_threads == [threading.get_ident()]


class _FakeViewer(ndv.ArrayViewer):
    def __init__(self, data: object = None, /, **kwargs: object) -> None:
        self._fake_data = data
        self.kwargs = kwargs
        self._fake_display_model: SimpleNamespace | ArrayDisplayModel = SimpleNamespace(
            current_index={}
        )
        self._fake_data_wrapper = SimpleNamespace(
            dims_changed=_Emitter(), data_changed=_Emitter(), sizes=lambda: {}
        )
        self._widget = QWidget()
        self._acquisition_record: object | None = None

    @property
    def data(self) -> object:
        return self._fake_data

    @data.setter
    def data(self, data: object) -> None:
        self._fake_data = data

    @property
    def display_model(self) -> ArrayDisplayModel:
        return cast("ArrayDisplayModel", self._fake_display_model)

    @display_model.setter
    def display_model(self, model: ArrayDisplayModel) -> None:
        self._fake_display_model = model

    @property
    def data_wrapper(self) -> SimpleNamespace:
        return self._fake_data_wrapper

    def widget(self) -> QWidget:
        return self._widget


def test_viewers_manager(
    mmcore: CMMCorePlus, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use the ome-writers sink view and release it with the parent."""
    monkeypatch.setattr(viewers_module, "MMArrayViewer", _FakeViewer)
    dummy = QWidget()
    manager = NDVViewersManager(dummy, mmcore)

    assert len(manager) == 0
    mmcore.mda.run(
        MDASequence(
            time_plan=useq.TIntervalLoops(
                interval=datetime.timedelta(seconds=0.01), loops=2
            ),
            channels=["DAPI"],  # pyright: ignore
        ),
        output="memory",
    )
    qtbot.wait(20)

    assert len(manager) == 1
    viewer = next(manager.viewers())
    assert isinstance(viewer, _FakeViewer)
    assert viewer.data is not None
    assert viewer.display_model.current_index["t"] == 1
    # No grid in this sequence: the plain flattened path is used, and a real
    # AcquisitionRecord is still attached (closing the asymmetry with
    # AcquireViewersManager, which always attached one).
    assert manager._layout.kind is GridAxisLayoutKind.NONE
    assert viewer._acquisition_record is not None

    with qtbot.waitSignal(dummy.destroyed, timeout=1000):
        dummy.deleteLater()
    QApplication.processEvents()
    assert manager._active_mda_viewer is None


def test_viewers_manager_grid_run(
    mmcore: CMMCorePlus, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A position+grid sequence is wrapped and followed by real (p, g)."""
    monkeypatch.setattr(viewers_module, "MMArrayViewer", _FakeViewer)
    dummy = QWidget()
    manager = NDVViewersManager(dummy, mmcore)

    mmcore.mda.run(
        MDASequence(
            stage_positions=(
                useq.AbsolutePosition(x=0, y=0),
                useq.AbsolutePosition(x=100, y=100),
            ),
            grid_plan=useq.GridRowsColumns(rows=1, columns=2),
        ),
        output="memory",
    )
    qtbot.wait(20)

    assert manager._layout.kind is GridAxisLayoutKind.REGULAR
    viewer = next(manager.viewers())
    assert isinstance(viewer, _FakeViewer)
    assert isinstance(viewer.data, GridAxisDataWrapper)
    # Last planned event is (p=1, g=1); following must land there regardless
    # of arrival order.
    assert viewer.display_model.current_index["p"] == 1
    assert viewer.display_model.current_index["g"] == 1
    assert viewer._acquisition_record is not None


def test_viewers_manager_ragged_grid_run(
    mmcore: CMMCorePlus, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mixed gridded/plain positions expose a padded, blank-filled g slider."""
    monkeypatch.setattr(viewers_module, "MMArrayViewer", _FakeViewer)
    dummy = QWidget()
    manager = NDVViewersManager(dummy, mmcore)

    mmcore.mda.run(
        MDASequence(
            axis_order=("p", "g", "c"),
            stage_positions=(
                useq.AbsolutePosition(
                    x=0.0,
                    y=0.0,
                    sequence=MDASequence(
                        grid_plan=useq.GridRowsColumns(rows=3, columns=1)
                    ),
                ),
                useq.AbsolutePosition(x=0.0, y=100.0),
            ),
            channels=["Cy5"],  # pyright: ignore
        ),
        output="memory",
    )
    qtbot.wait(20)

    assert manager._layout.kind is GridAxisLayoutKind.RAGGED
    assert manager._layout.tile_counts == (3, 1)
    viewer = next(manager.viewers())
    assert isinstance(viewer.data, GridAxisDataWrapper)
    # Last planned event is the plain position's single tile: (p=1, g=0),
    # the correct flattened slot 3 -- not slot 1 (which would silently read
    # position 0's second tile).
    assert viewer.display_model.current_index["p"] == 1
    assert viewer.display_model.current_index["g"] == 0
    assert viewer.data.layout.flat_index(1, 0) == 3

    # A g value beyond position 1's own tile count reads as blank, never a
    # different position's real data.
    blank = viewer.data.isel({0: 1, 1: 2})
    real = viewer.data.isel({0: 0, 1: 1})
    assert blank.shape == real.shape
    assert not blank.any()
