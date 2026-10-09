from __future__ import annotations

import datetime
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import ndv
import pytest
import useq
from useq import MDASequence

import pymmcore_gui._ndv_viewers as viewers_module
from pymmcore_gui._ndv_viewers import AcquireViewersManager, _StreamSignalBridge
from pymmcore_gui._qt.QtAds import CDockManager
from pymmcore_gui._qt.QtWidgets import QApplication, QWidget

if TYPE_CHECKING:
    from ndv.models import ArrayDisplayModel
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot

    from pymmcore_gui._array_viewer import MMArrayViewer


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
            dims_changed=_Emitter(), data_changed=_Emitter()
        )
        self._widget = QWidget()

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
    dock_manager = CDockManager(dummy)
    manager = AcquireViewersManager(dock_manager, mmcore, parent=dummy)

    assert not manager._records
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

    qtbot.waitUntil(lambda: len(manager._records) == 1)
    viewer = manager.active_viewer
    assert isinstance(viewer, _FakeViewer)
    assert viewer.data is not None
    assert viewer.display_model.current_index["t"] == 1
    assert viewer.data_wrapper.data_changed.calls > 0

    with qtbot.waitSignal(dummy.destroyed, timeout=1000):
        dummy.deleteLater()
    QApplication.processEvents()
    assert manager.active_viewer is None
    assert not manager._records
    assert not manager._connected


@pytest.mark.parametrize("z_plan", [None, useq.ZRangeAround(range=2, step=1)])
def test_live_viewer_shows_z_buttons_only_for_z_stacks(
    mmcore: CMMCorePlus, qtbot: QtBot, z_plan: useq.ZRangeAround | None
) -> None:
    dummy = QWidget()
    qtbot.addWidget(dummy)
    manager = AcquireViewersManager(CDockManager(dummy), mmcore, parent=dummy)
    created: list[MMArrayViewer] = []
    manager.mdaViewerCreated.connect(created.append)

    mmcore.mda.run(
        MDASequence(channels=["DAPI"], z_plan=z_plan),  # pyright: ignore
        output="memory",
    )
    qtbot.waitUntil(lambda: bool(created))

    viewer = created[0]
    has_z = z_plan is not None
    assert viewer._roll_axes_btn is not None
    assert viewer._roll_axes_btn.isHidden() is not has_z
    assert viewer.widget().ndims_btn.isHidden() is not has_z
