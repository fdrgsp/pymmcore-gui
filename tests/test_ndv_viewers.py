from __future__ import annotations

import datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING

import ndv
import useq
from useq import MDASequence

import pymmcore_gui._ndv_viewers as viewers_module
from pymmcore_gui._ndv_viewers import NDVViewersManager
from pymmcore_gui._qt.QtWidgets import QApplication, QWidget

if TYPE_CHECKING:
    import pytest
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot


class _Emitter:
    def __init__(self) -> None:
        self.calls = 0

    def emit(self) -> None:
        self.calls += 1


class _FakeViewer(ndv.ArrayViewer):
    def __init__(self, data: object = None, /, **kwargs: object) -> None:
        self._fake_data = data
        self.kwargs = kwargs
        self._fake_display_model = SimpleNamespace(current_index={})
        self._fake_data_wrapper = SimpleNamespace(
            dims_changed=_Emitter(), data_changed=_Emitter()
        )
        self._widget = QWidget()

    @property
    def data(self) -> object:
        return self._fake_data

    @property
    def display_model(self) -> SimpleNamespace:
        return self._fake_display_model

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
    assert viewer.data_wrapper.data_changed.calls > 0

    with qtbot.waitSignal(dummy.destroyed, timeout=1000):
        dummy.deleteLater()
    QApplication.processEvents()
    assert manager._active_mda_viewer is None
