from __future__ import annotations

import sys
from datetime import timedelta
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import useq

from pymmcore_gui import MicroManagerGUI
from pymmcore_gui._app import MMQApplication
from pymmcore_gui._notification_manager import NotificationManager
from pymmcore_gui._qt.QtGui import QAction
from pymmcore_gui._qt.QtWidgets import QApplication, QMenu
from pymmcore_gui.widgets._panels import PanelKey
from pymmcore_gui.widgets._stage_explorer import ThemedStageExplorer

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot

    from pymmcore_gui._settings import Settings


@pytest.fixture
def gui(
    qtbot: QtBot, qapp: QApplication, mmcore: CMMCorePlus
) -> Iterator[MicroManagerGUI]:
    gui = MicroManagerGUI(mmcore=mmcore)
    qtbot.addWidget(gui)
    yield gui


def test_main_window_close_stops_stage_explorer(gui: MicroManagerGUI) -> None:
    gui.acquire.panel_button(PanelKey.STAGE_EXPLORER).click()
    explorer = gui.acquire.panel_widget(PanelKey.STAGE_EXPLORER)
    assert isinstance(explorer, ThemedStageExplorer)
    assert explorer._stage_poller.isRunning()

    gui.close()

    assert not explorer._stage_poller.isRunning()


def _file_menu_actions(gui: MicroManagerGUI) -> dict[str, QAction]:
    menu_bar = gui.menuBar()
    assert menu_bar is not None
    (file_action,) = [a for a in menu_bar.actions() if a.text() == "&File"]
    file_menu = file_action.menu()
    assert isinstance(file_menu, QMenu)
    return {a.text(): a for a in file_menu.actions() if not a.isSeparator()}


def test_file_menu_has_about(gui: MicroManagerGUI) -> None:
    actions = _file_menu_actions(gui)
    assert list(actions) == ["About pymmcore-gui…"]
    assert actions["About pymmcore-gui…"].menuRole() == QAction.MenuRole.AboutRole


def test_file_menu_about_shows_one_dialog(gui: MicroManagerGUI, qtbot: QtBot) -> None:
    from pymmcore_gui.widgets._about_widget import AboutWidget

    about = _file_menu_actions(gui)["About pymmcore-gui…"]
    about.trigger()
    dialog = gui._about
    assert isinstance(dialog, AboutWidget)
    qtbot.waitUntil(dialog.isVisible)

    dialog.close()
    about.trigger()
    assert gui._about is dialog
    qtbot.waitUntil(dialog.isVisible)


@pytest.mark.filterwarnings("ignore:No device with label")
def test_shutter_bar_refreshes_loaded_devices(
    gui: MicroManagerGUI, qtbot: QtBot
) -> None:
    bar = gui.acquire._shutters
    assert (layout := bar.layout()) is not None
    assert layout.count() == 3

    with qtbot.waitSignal(gui._mmc.events.systemConfigurationLoaded):
        gui._mmc.loadSystemConfiguration()
    assert layout.count() == 2


def test_save_state_uses_modern_settings(
    gui: MicroManagerGUI, settings: Settings
) -> None:
    classic = settings.window.model_dump()
    gui.acquire.panel_button(PanelKey.STAGES).click()
    gui._save_state()

    assert settings.modern_window.geometry
    assert PanelKey.STAGES in settings.modern_window.acquire_panels
    assert settings.window.model_dump() == classic


def test_ndv_viewers_in_main_window(gui: MicroManagerGUI, qtbot: QtBot) -> None:
    manager = gui.acquire.viewers
    assert not manager._records
    with qtbot.waitSignal(manager.mdaViewerCreated):
        gui._mmc.mda.run(
            useq.MDASequence(
                time_plan=useq.TIntervalLoops(interval=timedelta(0), loops=2),
                channels=(
                    useq.Channel(config="DAPI", exposure=None),
                    useq.Channel(config="FITC", exposure=None),
                ),
            ),
            output="memory",
        )
    assert len(manager._records) == 1
    assert manager.active_viewer is not None
    assert manager.active_viewer.data is not None


def test_main_window_notifications(gui: MicroManagerGUI) -> None:
    assert isinstance(gui.nm, NotificationManager)
    with patch.object(gui.nm, "show_error_message") as mock_show_error:
        app = QApplication.instance()
        assert isinstance(app, MMQApplication)
        app.exceptionRaised.emit(ValueError("Boom!"))
        mock_show_error.assert_called_once()
        assert mock_show_error.call_args.args[:2] == ("Boom!", "See traceback")
        assert callable(mock_show_error.call_args.kwargs["on_action"])


def test_snap_updates_preview_after_camera_format_changes(
    gui: MicroManagerGUI, qtbot: QtBot
) -> None:
    manager = gui.acquire.viewers
    assert manager.preview is None
    with qtbot.waitSignal(manager.previewCreated):
        manager.ensure_preview()
    gui._mmc.snapImage()
    preview = manager.preview
    assert preview is not None
    original = preview.dtype_shape
    assert original is not None

    gui._mmc.setProperty(gui._mmc.getCameraDevice(), "PixelType", "32bitRGB")
    with qtbot.waitSignal(gui._mmc.events.imageSnapped):
        gui._mmc.snapImage()
    assert preview.dtype_shape != original
    assert manager.preview is preview

    gui._mmc.setExposure(42)
    with qtbot.waitSignal(gui._mmc.events.imageSnapped):
        gui._mmc.snapImage()
    assert manager.preview is preview


@pytest.mark.skipif(
    sys.platform == "darwin", reason="need to debug hanging test on macOS CI"
)
def test_stream_updates_preview(gui: MicroManagerGUI, qtbot: QtBot) -> None:
    manager = gui.acquire.viewers
    with qtbot.waitSignal(manager.previewCreated):
        gui.acquire._live_btn.click()
    try:
        qtbot.waitUntil(gui._mmc.isSequenceRunning)
        assert manager.preview is not None
        gui._mmc.setExposure(11)
        qtbot.waitUntil(
            lambda: (
                manager.preview is not None and manager.preview.viewer.data is not None
            )
        )
    finally:
        gui._mmc.stopSequenceAcquisition()
    assert not gui._mmc.isSequenceRunning()
