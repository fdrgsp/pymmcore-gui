from __future__ import annotations

import datetime
import gc
import threading
import weakref
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import ndv
import numpy as np
import pytest
import useq
from useq import MDASequence

import pymmcore_gui._ndv_viewers as viewers_module
from pymmcore_gui._ndv_viewers import AcquireViewersManager, _StreamSignalBridge
from pymmcore_gui._qt.QtAds import CDockManager, DockWidgetArea
from pymmcore_gui._qt.QtCore import QPoint, Qt
from pymmcore_gui._qt.QtWidgets import QApplication, QMenu, QWidget
from pymmcore_gui.widgets.image_preview._snap_album import SnappedImage

if TYPE_CHECKING:
    from ndv.models import ArrayDisplayModel
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot

    from pymmcore_gui._array_viewer import MMArrayViewer
    from pymmcore_gui._qt.QtAds import CDockWidget
    from pymmcore_gui._qt.QtGui import QAction


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


def test_frame_relay_releases_pixels_before_gui_dispatch(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Only metadata should stay queued when the GUI cannot keep up."""
    parent = QWidget()
    qtbot.addWidget(parent)
    manager = AcquireViewersManager(CDockManager(parent), mmcore, parent=parent)
    frames: list[weakref.ReferenceType[np.ndarray]] = []

    def emit_frame() -> None:
        frame = np.zeros((8, 8), dtype=np.uint16)
        frames.append(weakref.ref(frame))
        mmcore.mda.events.frameReady.emit(frame, useq.MDAEvent(), {})

    worker = threading.Thread(target=emit_frame)
    worker.start()
    worker.join()

    assert frames[0]() is None
    qtbot.wait(0)
    manager._disconnect()


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

    def set_acquisition_running(self, running: bool) -> None:
        self.acquisition_running = running


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
    assert not viewer.acquisition_running

    with qtbot.waitSignal(dummy.destroyed, timeout=1000):
        dummy.deleteLater()
    QApplication.processEvents()
    assert manager.active_viewer is None
    assert not manager._records
    assert not manager._connected


@pytest.mark.parametrize("cancel", [False, True])
def test_acquiring_viewer_disables_save_and_roi(
    mmcore: CMMCorePlus, qtbot: QtBot, cancel: bool
) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    manager = AcquireViewersManager(CDockManager(parent), mmcore, parent=parent)
    created: list[MMArrayViewer] = []

    def on_created(viewer: MMArrayViewer) -> None:
        created.append(viewer)
        assert viewer._save_btn is not None
        assert not viewer._save_btn.isEnabled()
        assert not viewer.widget().add_roi_btn.isEnabled()
        if cancel:
            mmcore.mda.cancel()

    manager.mdaViewerCreated.connect(on_created)
    sequence = MDASequence(
        time_plan=useq.TIntervalLoops(interval=datetime.timedelta(0), loops=2)
    )
    mmcore.mda.run(sequence, output="memory")
    qtbot.waitUntil(lambda: bool(created))

    viewer = created[0]
    assert viewer._save_btn is not None
    qtbot.waitUntil(viewer._save_btn.isEnabled)
    assert viewer.widget().add_roi_btn.isEnabled()


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


def _tab_action(
    manager: AcquireViewersManager, dock: CDockWidget, text: str
) -> QAction:
    menu = manager._tab_menu(dock)
    return next(a for a in menu.actions() if a.text() == text)


def _viewer_tabs(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> tuple[AcquireViewersManager, list[CDockWidget]]:
    parent = QWidget()
    qtbot.addWidget(parent)
    manager = AcquireViewersManager(CDockManager(parent), mmcore, parent=parent)
    for _ in range(3):
        manager.open_snap(SnappedImage(np.zeros((8, 12), dtype=np.uint16)))
    return manager, list(manager._records)


def test_viewer_tab_menu_names_its_close_actions(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    manager, docks = _viewer_tabs(mmcore, qtbot)
    menu = manager._tab_menu(docks[0])
    assert [a.text() for a in menu.actions()] == [
        "Close",
        "Close All But This",
        "Close All",
    ]
    assert all(a.isEnabled() for a in menu.actions())


def test_viewer_tab_menu_survives_garbage_collection(
    mmcore: CMMCorePlus, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real menu, after a collection: this used to segfault on right-click.

    Automatic collection is off under pytest, so it has to be asked for here, and
    before anything in this test holds on to the tab.
    """
    manager, [dock, *_] = _viewer_tabs(mmcore, qtbot)
    # This exercises tab/menu lifetimes, not the image canvas's OpenGL context.
    for record in manager._records.values():
        record.viewer.widget().hide()
    parent = manager._parent_widget
    assert parent is not None
    parent.show()
    qtbot.waitExposed(parent)
    gc.collect()

    opened: list[QMenu] = []

    class PopupMenu(QMenu):
        # The manager uses only the instance overload, not QMenu's static helper.
        def exec(  # pyright: ignore[reportIncompatibleMethodOverride]
            self, p: QPoint | None = None, action: QAction | None = None
        ) -> QAction | None:
            opened.append(self)
            self.popup(p if p is not None else QPoint())
            return None

    # PySide resolves exec on the instance, so patch the factory with a subclass.
    # The real popup stays visible without a timer racing a nested event loop.
    monkeypatch.setattr(viewers_module, "QMenu", PopupMenu)
    tab = dock.tabWidget()
    assert tab is not None
    assert tab.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu
    tab.customContextMenuRequested.emit(QPoint(1, 1))
    try:
        qtbot.waitUntil(lambda: bool(opened) and opened[0].isVisible())
        assert [action.text() for action in opened[0].actions()] == [
            "Close",
            "Close All But This",
            "Close All",
        ]
    finally:
        for menu in opened:
            menu.close()


@pytest.mark.parametrize(
    ("text", "remaining"),
    [("Close", [1, 2]), ("Close All But This", [0]), ("Close All", [])],
)
def test_viewer_tab_menu_closes_the_right_tabs(
    mmcore: CMMCorePlus, qtbot: QtBot, text: str, remaining: list[int]
) -> None:
    manager, docks = _viewer_tabs(mmcore, qtbot)
    _tab_action(manager, docks[0], text).trigger()
    qtbot.wait(0)
    assert list(manager._records) == [docks[i] for i in remaining]


def test_close_all_but_this_is_disabled_on_a_lone_tab(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    manager, docks = _viewer_tabs(mmcore, qtbot)
    _tab_action(manager, docks[0], "Close All But This").trigger()
    qtbot.wait(0)
    assert not _tab_action(manager, docks[0], "Close All But This").isEnabled()
    assert _tab_action(manager, docks[0], "Close All").isEnabled()


def test_close_all_stays_within_the_tab_dock_area(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """A viewer split off to the side is not closed from another area's tab."""
    manager, [first, second, split] = _viewer_tabs(mmcore, qtbot)
    manager._dock_manager.addDockWidget(DockWidgetArea.RightDockWidgetArea, split)
    assert first.dockAreaWidget() is not split.dockAreaWidget()

    _tab_action(manager, first, "Close All").trigger()
    qtbot.wait(0)
    assert list(manager._records) == [split]
    assert second not in manager._records
