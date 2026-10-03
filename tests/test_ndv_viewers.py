from __future__ import annotations

import datetime
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import ndv
import useq
from useq import MDASequence

import pymmcore_gui._ndv_viewers as viewers_module
from pymmcore_gui._ndv_viewers import AcquireViewersManager, _StreamSignalBridge
from pymmcore_gui._qt.QtAds import CDockManager
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


def test_iterator_run_follows_frame_count(
    mmcore: CMMCorePlus, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An iterator-driven run is stored along one ``t`` axis; follow it by count.

    The events' own index keys (here ``c`` and ``p``) do not exist in that
    view, so following them would leave the slider on the first frame.
    """
    monkeypatch.setattr(viewers_module, "MMArrayViewer", _FakeViewer)
    dummy = QWidget()
    qtbot.addWidget(dummy)
    manager = AcquireViewersManager(CDockManager(dummy), mmcore, parent=dummy)

    events = [
        useq.MDAEvent(channel="DAPI", index={"c": 0}),  # pyright: ignore
        useq.MDAEvent(channel="FITC", index={"c": 1}),  # pyright: ignore
        useq.MDAEvent(index={"p": 3}),  # pyright: ignore
    ]
    mmcore.mda.run(iter(events), output="memory")

    qtbot.waitUntil(lambda: len(manager._records) == 1)
    viewer = manager.active_viewer
    assert isinstance(viewer, _FakeViewer)
    qtbot.waitUntil(lambda: viewer.display_model.current_index.get("t") == 2)
    assert set(viewer.display_model.current_index) == {"t"}
    # The runner's empty placeholder sequence must not be offered for re-use.
    assert getattr(viewer, "mda_sequence", None) is None


def test_viewers_manager_skips_runs_it_does_not_accept(
    mmcore: CMMCorePlus, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run owned by another page's viewer workspace opens no viewer here."""
    monkeypatch.setattr(viewers_module, "MMArrayViewer", _FakeViewer)
    dummy = QWidget()
    qtbot.addWidget(dummy)
    accepted = AcquireViewersManager(
        CDockManager(dummy), mmcore, parent=dummy, title_prefix=lambda: "Smart"
    )
    other = QWidget()
    qtbot.addWidget(other)
    refused = AcquireViewersManager(
        CDockManager(other), mmcore, parent=other, accepts_run=lambda: False
    )

    mmcore.mda.run(iter([useq.MDAEvent(), useq.MDAEvent()]), output="memory")

    qtbot.waitUntil(lambda: len(accepted._records) == 1)
    QApplication.processEvents()
    assert not refused._records
    assert refused.active_viewer is None
    (record,) = accepted._records.values()
    assert str(getattr(record.viewer, "source_title", "")).startswith("Smart ")
