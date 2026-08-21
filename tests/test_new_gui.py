from __future__ import annotations

import math
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock, patch

import numpy as np
import pytest
import useq
from pymmcore_plus import PropertyType
from pymmcore_widgets import MDAWidget as UpstreamMDAWidget
from pymmcore_widgets import StageWidget, XYZStageWidget
from pymmcore_widgets.mda._core_channels import PROPERTY_SEPARATOR
from pymmcore_widgets.useq_widgets._positions import MDAButton, _MDAPopup

import pymmcore_gui._modern_gui._acquire as acquire_module
import pymmcore_gui._modern_gui._acquire_toolbar as acquire_toolbar_module
import pymmcore_gui._modern_gui._acquire_viewers as acquire_viewers_module
from pymmcore_gui._app import create_mmgui
from pymmcore_gui._array_viewer import _icon_avg_rgb
from pymmcore_gui._layouts import DEFAULT_LAYOUT_NAME
from pymmcore_gui._modern_gui._acquire import (
    _MDA_DOCK_WIDTH,
    _RIGHT_DOCK_MAX_WIDTH,
    _RIGHT_DOCK_MIN_USABLE_WIDTH,
    AcquirePage,
)
from pymmcore_gui._modern_gui._acquire_presets import AcquisitionPresetSelector
from pymmcore_gui._modern_gui._acquire_stages import StagesPanel
from pymmcore_gui._modern_gui._configurations import ConfigurationsPage
from pymmcore_gui._modern_gui._hardware import HardwareSetupPage
from pymmcore_gui._modern_gui._installation import (
    InstallationPage,
    InstallReleaseDialog,
)
from pymmcore_gui._modern_gui._main_win import MainWindow
from pymmcore_gui._modern_gui._panels import PANELS, PanelKey, StageKind
from pymmcore_gui._modern_gui._startup import StartupDialog
from pymmcore_gui._modern_gui._theme import (
    UI_FONT_SIZE_PT,
    UI_FONT_WEIGHT,
    qcolor,
    set_theme,
    set_zoom,
    theme,
    ui_font,
)
from pymmcore_gui._modern_gui._theme._dark import DARK_THEME
from pymmcore_gui._modern_gui._theme._light import LIGHT_THEME
from pymmcore_gui._qt.QtAds import (
    CDockManager,
    CDockWidget,
    DockWidgetArea,
    SideBarLocation,
)
from pymmcore_gui._qt.QtCore import QPoint, QRect, QSize, Qt
from pymmcore_gui._qt.QtGui import QAction, QCursor, QImage, QPainter, QPalette
from pymmcore_gui._qt.QtWidgets import (
    QWIDGETSIZE_MAX,
    QAbstractButton,
    QAbstractSlider,
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStyle,
    QStyleOption,
    QToolBar,
    QToolButton,
    QWidget,
)
from pymmcore_gui.widgets._active_channel_table import CURRENT_CHANNEL_COLUMN
from pymmcore_gui.widgets._mda_widget import MemoryMDAWidget
from pymmcore_gui.widgets._stage_explorer import ThemedStageExplorer

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from pymmcore_plus import CMMCorePlus
    from pymmcore_widgets.mda._collapsible_mda import CollapsibleCoreMDATabs
    from pymmcore_widgets.mda._core_grid import CoreConnectedGridPlanWidget
    from pymmcore_widgets.useq_widgets._data_table import DataTable
    from pytestqt.qtbot import QtBot
    from qtpy.QtCore import QModelIndex

    from pymmcore_gui._app import WindowProtocol
    from pymmcore_gui._modern_gui._theme import Color
    from pymmcore_gui._qt.QtAds import CDockAreaWidget
    from pymmcore_gui._qt.QtGui import QIcon
    from pymmcore_gui._settings import Settings


def _row_index(table: DataTable, row: int, col: int = 0) -> QModelIndex:
    """Return ``table.model().index(row, col)``, asserting the model exists."""
    model = table.model()
    assert model is not None
    return model.index(row, col)


def test_accepting_startup_config_selects_acquire(
    mmcore: CMMCorePlus,
    settings: Settings,
    qtbot: QtBot,
) -> None:
    config = Path(__file__).with_name("test_config.cfg")
    settings.recent_configs = [config]

    def accept_after_processing_events(_dialog: StartupDialog) -> int:
        # The real modal dialog runs a nested event loop before returning.
        QApplication.processEvents()
        return QDialog.DialogCode.Accepted

    with patch.object(StartupDialog, "exec", accept_after_processing_events):
        window = create_mmgui(
            mm_config=None,
            mmcore=mmcore,
            install_sys_excepthook=False,
            install_sentry=False,
            exec_app=False,
            window_cls=cast("type[WindowProtocol]", MainWindow),
        )
    assert isinstance(window, MainWindow)
    qtbot.addWidget(window)

    assert window._stack.currentWidget() is window._acquire
    assert window._mode_tabs._tabs[window._stack.indexOf(window._acquire)].active

    # A config loaded later from inside the running UI must not force a tab switch.
    window._mode_tabs._select(window._stack.indexOf(window._configurations))
    mmcore.loadSystemConfiguration(str(config))
    assert window._stack.currentWidget() is window._configurations


def test_explicit_startup_config_selects_acquire(
    mmcore: CMMCorePlus,
    qtbot: QtBot,
) -> None:
    config = Path(__file__).with_name("test_config.cfg")
    window = create_mmgui(
        mm_config=config,
        mmcore=mmcore,
        install_sys_excepthook=False,
        install_sentry=False,
        exec_app=False,
        window_cls=cast("type[WindowProtocol]", MainWindow),
    )
    assert isinstance(window, MainWindow)
    qtbot.addWidget(window)

    # `-c` reaches create_mmgui as an explicit mm_config before app.exec().
    assert window._stack.currentWidget() is window._acquire
    QApplication.processEvents()
    assert window._stack.currentWidget() is window._acquire


def test_pixel_calibration_locks_other_main_window_modes(
    mmcore: CMMCorePlus,
    qtbot: QtBot,
) -> None:
    window = MainWindow(mmcore=mmcore)
    qtbot.addWidget(window)
    config_index = window._stack.indexOf(window._configurations)
    install_index = window._stack.indexOf(window._installation)
    hardware_index = window._stack.indexOf(window._hardware)
    acquire_index = window._stack.indexOf(window._acquire)
    window._mode_tabs._select(config_index)

    window._configurations.calibrationRunningChanged.emit(True)

    assert window._stack.currentWidget() is window._configurations
    assert not window._mode_tabs._tabs[install_index].isEnabled()
    assert not window._mode_tabs._tabs[hardware_index].isEnabled()
    assert not window._mode_tabs._tabs[acquire_index].isEnabled()
    window._mode_tabs._select(acquire_index)
    assert window._stack.currentWidget() is window._configurations

    window._configurations.calibrationRunningChanged.emit(False)
    assert window._mode_tabs._tabs[install_index].isEnabled()
    assert window._mode_tabs._tabs[hardware_index].isEnabled()
    assert window._mode_tabs._tabs[acquire_index].isEnabled()


@contextmanager
def _fake_installs(paths: Sequence[Path]) -> Iterator[None]:
    """Show *paths* to ``InstallWidget`` without touching the network.

    Its constructor otherwise fetches the release listing from micro-manager.org
    and ctypes-loads every installed ``mmgr_dal`` library. Discovery stays
    honest about the disk -- only directories that still exist are reported --
    so uninstalling really has to remove one for it to leave the table.
    """
    module = "pymmcore_widgets._install_widget"

    def find(return_first: bool = True) -> str | list[str] | None:
        live = [str(p) for p in paths if p.is_dir()]
        if not return_first:
            return live
        return live[0] if live else None

    with (
        patch(f"{module}.available_versions", return_value={}),
        patch(f"{module}.find_micromanager", find),
        patch(f"{module}._try_get_device_interface_version", return_value="75"),
    ):
        yield


def _fake_install_dir(parent: Path, name: str) -> Path:
    """Create a directory that looks like a Micro-Manager install."""
    path = parent / name
    path.mkdir()
    (path / "libmmgr_dal_DemoCamera").write_bytes(b"")
    return path


def test_installation_page_builds_its_widget_on_demand(
    qtbot: QtBot, tmp_path: Path
) -> None:
    page = InstallationPage()
    qtbot.addWidget(page)
    # constructing the page must not construct (or reach out on behalf of) the
    # wrapped widget
    assert page.install_widget is None

    older = _fake_install_dir(tmp_path, "Micro-Manager-75.20240101")
    newer = _fake_install_dir(tmp_path, "Micro-Manager-75.20250202")
    with _fake_installs([older, newer]):
        widget = page.ensure_loaded()
        assert page.ensure_loaded() is widget  # built once

    assert widget is not None
    assert page.install_widget is widget
    assert widget.toolbar.isHidden()  # the page toolbar stands in for it
    assert widget.install_row.isHidden()  # the Install button + dialog replace it
    assert widget.table.rowCount() == 2

    # Install sits directly before Uninstall in the toolbar
    assert list(page._buttons) == [
        "Refresh",
        "Reveal",
        "Set Active",
        "Install",
        "Uninstall",
    ]

    # buttons follow the actions they replace: nothing selected, nothing to do
    assert page._buttons["Refresh"].isEnabled()
    assert page._buttons["Install"].isEnabled()  # never depends on a selection
    assert not page._buttons["Uninstall"].isEnabled()
    assert not page._buttons["Set Active"].isEnabled()

    # rows are sorted newest first, and `older` is the active one
    widget.table.selectRow(0)
    assert page._buttons["Uninstall"].isEnabled()
    assert page._buttons["Set Active"].isEnabled()
    widget.table.selectRow(1)
    assert not page._buttons["Set Active"].isEnabled()  # already active


def test_installation_page_reports_a_changed_active_install(
    qtbot: QtBot, tmp_path: Path
) -> None:
    older = _fake_install_dir(tmp_path, "Micro-Manager-75.20240101")
    newer = _fake_install_dir(tmp_path, "Micro-Manager-75.20250202")
    with patch(
        "pymmcore_gui._modern_gui._installation.find_micromanager",
        return_value=str(older),
    ):
        page = InstallationPage()
    qtbot.addWidget(page)
    with _fake_installs([older, newer]):
        widget = page.ensure_loaded()
    assert widget is not None
    assert page.active_install == str(older)

    # every path that can change the active install (these buttons, the table's
    # own Delete-key uninstall, a finished download) ends in a table refresh
    with (
        patch(
            "pymmcore_gui._modern_gui._installation.find_micromanager",
            return_value=str(newer),
        ),
        _fake_installs([newer, older]),
        qtbot.waitSignal(page.activeInstallChanged) as blocker,
    ):
        widget.table.refresh()

    assert blocker.args == [str(newer)]
    assert page.active_install == str(newer)


def test_uninstalling_an_install_removes_it_from_the_table(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """The row must not come back: pymmcore-plus keeps listing deleted installs.

    ``find_micromanager(return_first=False)`` returns a module-level cache that
    nothing evicts from, so a table that simply re-lists itself after deleting
    shows the row it just removed.
    """
    page = InstallationPage()
    qtbot.addWidget(page)
    older = _fake_install_dir(tmp_path, "Micro-Manager-75.20240101")
    newer = _fake_install_dir(tmp_path, "Micro-Manager-75.20250202")

    with _fake_installs([older, newer]):
        widget = page.ensure_loaded()
        assert widget is not None
        widget.table.selectRow(1)  # rows are newest first, so this is `older`
        with patch.object(
            QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes
        ):
            page._buttons["Uninstall"].click()

        assert not older.exists()
        assert newer.is_dir()
        assert widget.table.rowCount() == 1
        remaining = widget.table.item(0, widget.table.VER_COL)
        assert remaining is not None
        assert remaining.text() == newer.name


def test_install_button_asks_which_release_then_installs(
    qtbot: QtBot, tmp_path: Path
) -> None:
    page = InstallationPage()
    qtbot.addWidget(page)
    install = _fake_install_dir(tmp_path, "Micro-Manager-75.20240101")

    with _fake_installs([install]):
        widget = page.ensure_loaded()
        assert widget is not None

        dialogs: list[InstallReleaseDialog] = []

        def choose_second_release(dialog: InstallReleaseDialog) -> int:
            dialogs.append(dialog)
            dialog._releases.setCurrentIndex(1)
            return QDialog.DialogCode.Accepted

        with (
            patch.object(InstallReleaseDialog, "exec", choose_second_release),
            patch.object(widget, "install") as start,
        ):
            page._buttons["Install"].click()

    assert len(dialogs) == 1
    # the dialog offers exactly what the widget resolved as installable
    combo = widget.version_combo
    assert [
        dialogs[0]._releases.itemText(i) for i in range(dialogs[0]._releases.count())
    ] == [combo.itemText(i) for i in range(combo.count())]
    # ...and the choice goes through the widget's public install API
    start.assert_called_once_with("latest")


def test_cancelling_the_release_dialog_installs_nothing(
    qtbot: QtBot, tmp_path: Path
) -> None:
    page = InstallationPage()
    qtbot.addWidget(page)
    install = _fake_install_dir(tmp_path, "Micro-Manager-75.20240101")

    with _fake_installs([install]):
        widget = page.ensure_loaded()
        assert widget is not None
        before = widget.version_combo.currentText()
        with (
            patch.object(
                InstallReleaseDialog, "exec", return_value=QDialog.DialogCode.Rejected
            ),
            patch.object(widget, "install") as start,
        ):
            page._buttons["Install"].click()

    start.assert_not_called()
    assert widget.version_combo.currentText() == before


def test_an_uninstall_that_fails_says_so(qtbot: QtBot, tmp_path: Path) -> None:
    page = InstallationPage()
    qtbot.addWidget(page)
    install = _fake_install_dir(tmp_path, "Micro-Manager-75.20240101")

    with _fake_installs([install]):
        widget = page.ensure_loaded()
        assert widget is not None
        widget.table.selectRow(0)
        with (
            patch(
                "shutil.rmtree", side_effect=PermissionError("Operation not permitted")
            ),
            patch.object(
                QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes
            ),
            patch.object(QMessageBox, "warning") as warning,
        ):
            page._buttons["Uninstall"].click()

    # silently swallowing the failure (what ignore_errors=True does upstream)
    # would leave the row in place with no explanation
    assert install.is_dir()
    warning.assert_called_once()
    assert "Operation not permitted" in warning.call_args[0][2]


def test_uninstall_emits_about_to_uninstall_before_deleting(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """The signal (if connected) must fire before shutil.rmtree, with the paths chosen.

    Regression test: nothing released a still-loaded device adapter DLL before
    an uninstall attempted to delete it, so a Windows session uninstalling
    whichever install was actively driving the connected hardware failed with
    a file-lock error. See MainWindow._prepare_uninstall.
    """
    page = InstallationPage()
    qtbot.addWidget(page)
    install = _fake_install_dir(tmp_path, "Micro-Manager-75.20240101")

    order: list[str] = []
    seen: list[set[str]] = []

    def prepare(paths: set[str]) -> None:
        order.append("prepared")
        seen.append(paths)

    page.aboutToUninstall.connect(prepare)

    with _fake_installs([install]):
        widget = page.ensure_loaded()
        assert widget is not None
        widget.table.selectRow(0)
        with (
            patch("shutil.rmtree", side_effect=lambda p: order.append("deleted")),
            patch.object(
                QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes
            ),
        ):
            page._buttons["Uninstall"].click()

    assert order == ["prepared", "deleted"]
    assert seen == [{str(install)}]


def test_window_opens_on_hardware_setup(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    window = MainWindow(mmcore=mmcore)
    qtbot.addWidget(window)
    assert window._stack.currentWidget() is window._hardware
    # Installation leads the tab order but is not built until it's opened.
    assert window._stack.indexOf(window._installation) == 0
    assert window._installation.install_widget is None


def test_window_without_micromanager_opens_on_installation(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    with patch(
        "pymmcore_gui._modern_gui._main_win.find_micromanager", return_value=None
    ):
        window = MainWindow(mmcore=mmcore)
    qtbot.addWidget(window)
    assert window._stack.currentWidget() is window._installation
    assert window._mode_tabs._tabs[0].active


def test_switching_active_install_repoints_the_running_core(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    window = MainWindow(mmcore=mmcore)
    qtbot.addWidget(window)
    assert [d for d in mmcore.getLoadedDevices() if d != "Core"]

    new_install = tmp_path / "Micro-Manager-75.20250202"
    new_install.mkdir()
    with patch.object(window, "_confirm_install_switch", return_value=True) as confirm:
        window._installation.activeInstallChanged.emit(str(new_install))

    confirm.assert_called_once_with(str(new_install))
    assert list(mmcore.getDeviceAdapterSearchPaths()) == [str(new_install)]
    # everything that belonged to the replaced installation is gone
    assert [d for d in mmcore.getLoadedDevices() if d != "Core"] == []
    assert not mmcore.getAvailableConfigGroups()
    assert not window._hardware.model.devices


def test_declining_an_install_switch_keeps_the_session(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    window = MainWindow(mmcore=mmcore)
    qtbot.addWidget(window)
    loaded = list(mmcore.getLoadedDevices())
    paths = list(mmcore.getDeviceAdapterSearchPaths())

    new_install = tmp_path / "Micro-Manager-75.20250202"
    new_install.mkdir()
    with patch.object(window, "_confirm_install_switch", return_value=False):
        window._installation.activeInstallChanged.emit(str(new_install))

    assert list(mmcore.getLoadedDevices()) == loaded
    assert list(mmcore.getDeviceAdapterSearchPaths()) == paths


def test_uninstalling_the_active_install_unloads_devices_first(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    """Uninstalling the install actively driving the core releases it first.

    Regression test: Windows keeps a loaded ``mmgr_dal_*.dll`` locked against
    deletion until the process that loaded it lets go, so uninstalling the
    active install failed with ``PermissionError: [WinError 5] Access is
    denied`` even after the user confirmed the delete.
    """
    window = MainWindow(mmcore=mmcore)
    qtbot.addWidget(window)
    assert [d for d in mmcore.getLoadedDevices() if d != "Core"]

    active = _fake_install_dir(tmp_path, "Micro-Manager-active")
    mmcore.setDeviceAdapterSearchPaths([str(active)])

    window._installation.aboutToUninstall.emit({str(active)})

    assert [d for d in mmcore.getLoadedDevices() if d != "Core"] == []


def test_uninstalling_an_unrelated_install_leaves_devices_loaded(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    """Deleting an install that isn't in use must not disturb a live session."""
    window = MainWindow(mmcore=mmcore)
    qtbot.addWidget(window)
    loaded_before = list(mmcore.getLoadedDevices())
    assert [d for d in loaded_before if d != "Core"]

    active = _fake_install_dir(tmp_path, "Micro-Manager-active")
    unrelated = tmp_path / "Micro-Manager-old"
    unrelated.mkdir()
    mmcore.setDeviceAdapterSearchPaths([str(active)])

    window._installation.aboutToUninstall.emit({str(unrelated)})

    assert list(mmcore.getLoadedDevices()) == loaded_before


def test_install_switch_prompt_uses_button_variants(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    window = MainWindow(mmcore=mmcore)
    qtbot.addWidget(window)
    messages: list[QMessageBox] = []

    def inspect_without_showing(message: QMessageBox) -> int:
        messages.append(message)
        return 0

    with patch.object(QMessageBox, "exec", inspect_without_showing):
        # no button was clicked -> the session is kept, the switch deferred
        assert not window._confirm_install_switch(str(tmp_path / "Micro-Manager"))

    assert len(messages) == 1
    buttons = {button.text(): button for button in messages[0].buttons()}
    assert buttons["Switch now"].property("variant") == "danger"
    assert buttons["Keep this session"].property("variant") == "subtle"


def test_unsaved_configuration_prompt_uses_button_variants(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    window = MainWindow(mmcore=mmcore)
    qtbot.addWidget(window)
    messages: list[QMessageBox] = []

    def inspect_without_showing(message: QMessageBox) -> int:
        messages.append(message)
        return 0

    with (
        patch.object(window._configurations, "dirty_parts", return_value=["Pixel"]),
        patch.object(QMessageBox, "exec", inspect_without_showing),
    ):
        assert window._prompt_unsaved_configuration_changes() == "cancel"

    assert len(messages) == 1
    buttons = {button.text(): button for button in messages[0].buttons()}
    assert buttons["Save to core"].property("variant") == "subtle"
    assert buttons["Save to file…"].property("variant") == "primary"
    assert buttons["Discard changes and continue"].property("variant") == "danger"
    assert buttons["Cancel"].property("variant") == "subtle"


def test_discarding_configuration_changes_before_switching_cleans_page(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    window = MainWindow(mmcore=mmcore)
    qtbot.addWidget(window)
    configurations = window._configurations
    configurations_index = window._stack.indexOf(configurations)
    acquire_index = window._stack.indexOf(window._acquire)
    window._mode_tabs._select(configurations_index)

    saved_groups = [group.name for group in configurations._group_editor.data()]
    configurations._group_editor._add_group()
    selected = configurations._pixel_config._selected_preset()
    assert selected is not None
    row, preset = selected
    resolution_id = str(preset.name)
    saved_pixel_size = float(mmcore.getPixelSizeUmByID(resolution_id))
    pixel_size = configurations._pixel_config._px_table.table().cellWidget(row, 1)
    assert isinstance(pixel_size, QDoubleSpinBox)
    pixel_size.setValue(saved_pixel_size + 0.25)
    assert configurations._group_dirty
    assert configurations._pixel_dirty

    with patch.object(
        window, "_prompt_unsaved_configuration_changes", return_value="discard"
    ):
        window._mode_tabs._select(acquire_index)

    assert window._stack.currentWidget() is window._acquire
    assert not configurations.is_dirty()
    assert [group.name for group in configurations._group_editor.data()] == saved_groups
    reloaded_preset = next(
        preset
        for preset in configurations._pixel_config._resID_map.values()
        if str(preset.name) == resolution_id
    )
    assert reloaded_preset.pixel_size_um == saved_pixel_size

    window._mode_tabs._select(configurations_index)
    assert window._stack.currentWidget() is configurations
    assert not configurations.is_dirty()


def test_new_gui_uses_one_application_font(
    mmcore: CMMCorePlus,
    qtbot: QtBot,
) -> None:
    set_theme(DARK_THEME)
    roots = (
        HardwareSetupPage(mmcore),
        ConfigurationsPage(mmcore),
        MemoryMDAWidget(mmcore),
    )
    for root in roots:
        qtbot.addWidget(root)

    app_font = QApplication.font()
    expected_size = UI_FONT_SIZE_PT * theme().zoom_factor
    assert math.isclose(app_font.pointSizeF(), expected_size)
    assert app_font.weight() == UI_FONT_WEIGHT

    painted_font = ui_font()
    assert painted_font.family() == app_font.family()
    assert math.isclose(painted_font.pointSizeF(), expected_size)
    assert painted_font.weight() == UI_FONT_WEIGHT

    mismatches: list[str] = []
    for root in roots:
        for widget in (root, *root.findChildren(QWidget)):
            font = widget.font()
            if (
                font.family() != app_font.family()
                or not math.isclose(font.pointSizeF(), expected_size)
                or font.weight() != UI_FONT_WEIGHT
            ):
                mismatches.append(
                    f"{type(widget).__name__}: "
                    f"{font.family()} {font.pointSizeF()}pt weight={font.weight()}"
                )

    assert not mismatches, "\n".join(mismatches)


def test_acquire_page_dock_layout(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    set_theme(DARK_THEME)
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    tabs = page._mda._collapsible_tabs()

    assert page._viewers.preview is None
    assert page._central_dock_area.dockWidgetsCount() == 1  # outer viewer shell
    assert page._viewer_dock_manager.dockAreaCount() == 0
    # AcquirePage doesn't use TabPage's left sidebar or its (now-removed)
    # right/bottom regions -- the dock manager supplies all docking itself.
    assert page.left.isHidden()
    assert not hasattr(page, "right")
    assert not hasattr(page, "bottom")

    dm = page._dock_manager
    assert dm.centralWidget() is page._central
    assert page._central.widget() is page._viewer_dock_manager
    assert page._viewer_dock_manager.centralWidget() is None
    DF = CDockWidget.DockWidgetFeature
    assert page._central.features().value & DF.NoTab.value
    assert not page._central.features().value & DF.DockWidgetClosable.value
    assert page._central_dock_area.allowedAreas() == DockWidgetArea.OuterDockAreas
    assert page._mda_dock.dockManager() is dm
    assert page._viewers._dock_manager is page._viewer_dock_manager
    assert page._viewers._dock_manager is not dm

    assert page._mda_dock.widget() is page._mda
    assert not page._mda_dock.isClosed()
    assert page.panel_button(PanelKey.MDA).isChecked()
    assert page._mda.prepare_mda() == "memory"

    # Groups and Presets is the one other default-open panel.
    assert page.panel_widget(PanelKey.PRESETS) is not None
    assert page.panel_dock(PanelKey.PRESETS) is not None
    assert page.panel_button(PanelKey.PRESETS).isChecked()

    # Every *other* panel is lazy: no widget or dock before first open.
    assert page.panel_widget(PanelKey.PROPERTIES) is None
    assert page.panel_dock(PanelKey.PROPERTIES) is None
    assert not page.panel_button(PanelKey.PROPERTIES).isChecked()
    assert page.panel_widget(PanelKey.STAGE_EXPLORER) is None
    assert page.panel_dock(PanelKey.STAGE_EXPLORER) is None
    assert not page.panel_button(PanelKey.STAGE_EXPLORER).isChecked()
    assert page.panel_widget(PanelKey.CONSOLE) is None
    assert page.panel_dock(PanelKey.CONSOLE) is None
    assert not page.panel_button(PanelKey.CONSOLE).isChecked()

    assert sorted(dm.dockWidgetsMap()) == [
        "acquire_mda",
        "acquire_presets",
        "acquire_viewers",
    ]
    assert page._viewer_dock_manager.dockWidgetsMap() == {}
    assert not hasattr(page, "_channels")
    assert not {
        button.text() for button in page.toolbar.findChildren(QPushButton)
    }.intersection(mmcore.getAvailableConfigs(mmcore.getChannelGroup()))

    assert [section.title for section in tabs.sections] == [
        "Channels",
        "Positions",
        "Grid / Tile Scan",
        "Z Stack",
        "Time Series",
        "Camera ROI",
        "Saving",
        "Settings",
    ]
    assert tabs.section("c").content_widget is page._mda.channels
    assert tabs.section("p").content_widget is page._mda.stage_positions
    assert tabs.section("g").content_widget is page._mda.grid_plan
    assert tabs.section("z").content_widget is page._mda.z_plan
    assert tabs.section("t").content_widget is page._mda.time_plan
    assert tabs.saving_section.content_widget is page._mda.save_info
    assert tabs.saving_section is tabs.sections[-2]
    assert tabs.settings_section is tabs.sections[-1]
    tab_bar = tabs.tabBar()
    assert tab_bar is not None and tab_bar.isHidden()

    channels_section = tabs.section("c")
    assert channels_section.expanded
    assert channels_section.checked
    channels_section.set_expanded(False)
    assert channels_section.checked
    assert not channels_section.content_visible
    channels_section.set_checked(False)
    assert not channels_section.checked
    assert not channels_section.expanded
    channels_section.set_checked(True)

    assert tabs.saving_section.summary == "Memory only"
    assert not tabs.saving_section.checked
    tabs.saving_section.set_checked(True)
    assert page._mda.save_info.isChecked()
    tabs.saving_section.set_checked(False)
    assert not page._mda.save_info.isChecked()

    # Opening Groups & Presets builds it lazily.
    page.panel_button(PanelKey.PRESETS).setChecked(True)  # default-open already
    presets = page.panel_widget(PanelKey.PRESETS)
    presets_dock = page.panel_dock(PanelKey.PRESETS)
    assert isinstance(presets, AcquisitionPresetSelector)
    assert presets_dock is not None
    assert presets_dock.widget() is presets
    assert presets_dock.windowTitle() == "Groups and Presets"
    assert not presets_dock.isClosed()
    assert page.panel_button(PanelKey.PRESETS).isChecked()

    # Groups & Presets is the upstream GroupPresetTableWidget with its
    # editing/save/load controls hidden — editing groups already lives on the
    # Configurations tab, and saving/loading a .cfg on the Hardware tab.
    hidden_buttons = {
        presets.edit_groups_btn,
        presets.save_btn,
        presets.load_btn,
    }
    assert all(button.isHidden() for button in hidden_buttons)
    assert not presets.table_wdg.isHidden()

    # Opening Properties builds it lazily and tabs it beside Groups and Presets.
    page.panel_button(PanelKey.PROPERTIES).click()
    browser = page.panel_widget(PanelKey.PROPERTIES)
    props_dock = page.panel_dock(PanelKey.PROPERTIES)
    assert page.panel_button(PanelKey.PROPERTIES).isChecked()
    assert browser is not None and not browser.isWindow()
    assert props_dock is not None and not props_dock.isClosed()
    assert props_dock.widget() is browser
    assert props_dock.dockAreaWidget() is presets_dock.dockAreaWidget()

    # Toggling off closes the dock but keeps the (expensive) widget alive.
    page.panel_button(PanelKey.PROPERTIES).click()
    assert props_dock.isClosed()
    assert not page.panel_button(PanelKey.PROPERTIES).isChecked()
    assert page.panel_widget(PanelKey.PROPERTIES) is browser

    # Toggling back on reuses the same widget and dock rather than rebuilding.
    page.panel_button(PanelKey.PROPERTIES).click()
    assert not props_dock.isClosed()
    assert page.panel_widget(PanelKey.PROPERTIES) is browser
    assert page.panel_dock(PanelKey.PROPERTIES) is props_dock

    # Closing the dock from its own tab (as the ✕ button does) unchecks the
    # toolbar toggle, same as clicking it would.
    props_dock.closeDockWidget()
    assert props_dock.isClosed()
    assert not page.panel_button(PanelKey.PROPERTIES).isChecked()

    presets_dock.closeDockWidget()
    assert not page.panel_button(PanelKey.PRESETS).isChecked()
    page.panel_button(PanelKey.PRESETS).click()
    assert not presets_dock.isClosed()
    assert page.panel_button(PanelKey.PRESETS).isChecked()


def test_acquire_inactive_dock_tab_labels_are_visible(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Inactive dock tab labels use the theme's text color, not ADS's default.

    ADS's built-in stylesheet colors an inactive ``CDockWidgetTab`` label with
    ``palette(dark)`` -- a shadow role that renders near-black on the dark
    theme's near-black tab background, effectively invisible (see
    ``_apply_dock_style``'s docstring). The active tab must stay visually
    distinct from the inactive ones.
    """
    set_theme(DARK_THEME)
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    # a second tab, to be inactive alongside Presets
    page.panel_button(PanelKey.PROPERTIES).click()

    ss = page._dock_manager.styleSheet()
    viewer_ss = page._viewer_dock_manager.styleSheet()
    inactive = qcolor(theme().text_secondary).name()
    active = qcolor(theme().text_primary).name()
    assert f"ads--CDockWidgetTab QLabel {{\n            color: {inactive};" in ss
    assert (
        f'ads--CDockWidgetTab[activeTab="true"] QLabel {{\n            '
        f"color: {active};" in ss
    )
    # base ADS chrome (e.g. the qproperty-icon rules for title-bar buttons)
    # must survive -- this appends overrides, it doesn't replace the sheet.
    assert "qproperty-icon" in ss
    assert inactive in viewer_ss
    assert active in viewer_ss


def test_acquire_dock_icons_are_themed_on_initial_dark_startup(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """ADS's fixed black chrome is corrected without requiring a theme toggle."""
    set_theme(DARK_THEME)
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    buttons = {
        btn.objectName(): btn
        for btn in page._dock_manager.findChildren(QAbstractButton)
    }
    for name in ("tabsMenuButton", "detachGroupButton", "dockAreaAutoHideButton"):
        rgb = _icon_avg_rgb(buttons[name].icon(), QSize(24, 24))
        assert rgb is not None
        # Dark-theme chrome should be a light neutral, not ADS's black source.
        assert sum(rgb) / 3 > 180

    close_rgb = _icon_avg_rgb(buttons["tabCloseButton"].icon(), QSize(24, 24))
    assert close_rgb is not None
    red = qcolor(theme().status_red)
    assert all(
        abs(actual - expected) < 2
        for actual, expected in zip(
            close_rgb, (red.red(), red.green(), red.blue()), strict=True
        )
    )


def test_acquire_dock_style_and_fonts_follow_theme_toggle(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Switching theme re-derives the tab-label override colors."""
    set_theme(DARK_THEME)
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    set_theme(LIGHT_THEME)
    ss = page._dock_manager.styleSheet()
    assert qcolor(theme().text_secondary).name() in ss
    assert qcolor(theme().text_primary).name() in ss
    close = next(
        btn
        for btn in page._dock_manager.findChildren(QAbstractButton)
        if btn.objectName() == "tabCloseButton"
    )
    close_rgb = _icon_avg_rgb(close.icon(), QSize(24, 24))
    assert close_rgb is not None
    red = qcolor(theme().status_red)
    assert all(
        abs(actual - expected) < 2
        for actual, expected in zip(
            close_rgb, (red.red(), red.green(), red.blue()), strict=True
        )
    )
    set_theme(DARK_THEME)


def test_acquire_dock_close_icon_stays_red_after_tab_switch(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Switching which tab is current must not un-tint any tab's close X.

    Regression test: ADS's own stylesheet drives each tab close button's icon
    via a ``qproperty-icon`` rule keyed on the dynamic ``activeTab`` property
    (see ``_apply_dock_style``). Switching which tab is current in an area
    flips that property on both the tab losing and the tab gaining focus, and
    Qt re-polishing either one re-applies that rule -- silently overwriting
    the red tint ``_refresh_dock_icons`` set, on *both* tabs, not just the one
    that changed. This used to only be corrected again on the next theme
    toggle or newly-added dock; nothing re-ran it on a plain tab switch.
    """
    set_theme(DARK_THEME)
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    page.panel_button(PanelKey.PRESETS).setChecked(True)  # default-open already
    page.panel_button(PanelKey.PROPERTIES).click()
    presets_dock = page.panel_dock(PanelKey.PRESETS)
    properties_dock = page.panel_dock(PanelKey.PROPERTIES)
    assert presets_dock is not None and properties_dock is not None

    def close_buttons() -> list[QAbstractButton]:
        return [
            btn
            for btn in page._dock_manager.findChildren(QAbstractButton)
            if btn.objectName() == "tabCloseButton"
        ]

    red = qcolor(theme().status_red)
    expected = (red.red(), red.green(), red.blue())

    def assert_all_red() -> None:
        for btn in close_buttons():
            rgb = _icon_avg_rgb(btn.icon(), QSize(24, 24))
            assert rgb is not None
            assert all(abs(a - e) < 2 for a, e in zip(rgb, expected, strict=True))

    # Switching to the tab added most recently (Properties) must not un-tint
    # either its own close button or the one it stole focus from (Presets).
    # No manual refresh call here -- this exercises the real signal wiring
    # (CDockAreaWidget.currentChanged -> _queue_dock_icon_refresh) end to end.
    properties_dock.setAsCurrentTab()
    qtbot.waitUntil(assert_all_red, timeout=1000)

    # And switching back must not un-tint them either.
    presets_dock.setAsCurrentTab()
    qtbot.waitUntil(assert_all_red, timeout=1000)


def test_acquire_dock_contents_follow_zoom(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    """Widgets inside the dock manager rescale with Cmd+Shift+±, in both directions.

    ``CDockManager`` applies its own stylesheet in its constructor, which
    freezes the resolved font of everything inside it -- so without
    ``_refresh_dock_fonts``, dock contents (e.g. the MDA channel table's
    Exposure column) silently stop tracking ``set_zoom()`` while everything
    outside the dock area keeps rescaling normally.
    """
    set_theme(DARK_THEME)
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    def exposure_cell_size() -> float:
        table = page._mda.channels.table()
        cell = table.cellWidget(0, table.indexOf(page._mda.channels.EXPOSURE))
        assert cell is not None
        return cell.font().pointSizeF()

    try:
        set_zoom(1.5)
        assert exposure_cell_size() == pytest.approx(QApplication.font().pointSizeF())
        set_zoom(0.8)
        assert exposure_cell_size() == pytest.approx(QApplication.font().pointSizeF())
        set_zoom(1.25)
        assert exposure_cell_size() == pytest.approx(QApplication.font().pointSizeF())
    finally:
        set_theme(DARK_THEME)  # restores the default zoom too


def test_acquire_docks_are_movable_and_pinnable(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """All four docks declare themselves movable and pinnable to a side bar.

    This checks the *configuration* rather than performing a live redock:
    moving a dock so its source area empties (what dragging it elsewhere, or
    pinning it, actually does) reproducibly segfaults under
    ``QT_QPA_PLATFORM=offscreen`` + pytest-qt -- confirmed test-harness-only
    (independent of any app code, on both PyQt6Ads 4.4.0.post2 and the latest
    5.0.0), since interactive drag-and-drop on a real display does not
    reproduce it. See ``_configure_ads``'s docstring. Actual rearranging is a
    manual smoke-test item.
    """
    set_theme(DARK_THEME)
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    assert CDockManager.testAutoHideConfigFlag(
        CDockManager.eAutoHideFlag.DockAreaHasAutoHideButton
    )
    page.panel_button(PanelKey.PRESETS).setChecked(True)  # default-open already
    presets_dock = page.panel_dock(PanelKey.PRESETS)
    assert presets_dock is not None
    # Every non-viewer panel belongs to the same outer manager, so users can
    # move them across the full left/right/top/bottom perimeter. Viewers use a
    # nested manager and can only rearrange within the central workspace.
    assert page._mda_dock.dockManager() is page._dock_manager
    assert presets_dock.dockManager() is page._dock_manager
    assert page._viewers._dock_manager is page._viewer_dock_manager
    assert page._viewer_dock_manager is not page._dock_manager
    DF = CDockWidget.DockWidgetFeature
    for dock in (page._mda_dock, presets_dock):
        assert dock.features().value & DF.DockWidgetPinnable.value
        assert dock.features().value & DF.DockWidgetMovable.value
        assert not dock.features().value & DF.DockWidgetFloatable.value


def test_acquire_lazy_dock_tabs_into_existing_area(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Opening Properties tabs it into the Presets area without emptying either.

    Unlike moving a dock *out* of its area, adding a new dock *into* an
    existing (non-empty) one never destroys anything, so this is safe to
    exercise directly -- and it's the one area-membership change the app
    actually performs at runtime (see ``_add_side_dock``).
    """
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page.panel_button(PanelKey.PRESETS).setChecked(True)  # default-open already
    presets_dock = page.panel_dock(PanelKey.PRESETS)
    assert presets_dock is not None
    page.panel_button(PanelKey.PROPERTIES).click()
    props_dock = page.panel_dock(PanelKey.PROPERTIES)
    assert props_dock is not None
    assert props_dock.dockAreaWidget() is presets_dock.dockAreaWidget()
    assert not presets_dock.isClosed()
    assert not props_dock.isClosed()


def test_acquire_console_dock_is_lazy(
    mmcore: CMMCorePlus, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Console is only imported/constructed on first open, same as Properties."""

    class FakeConsole(QWidget):
        def __init__(self, mmcore: CMMCorePlus, parent: QWidget | None = None) -> None:
            super().__init__(parent)

    # the console panel's factory does a function-local import, re-reading
    # this attribute from the source module on every call -- patch there,
    # not on AcquirePage.
    monkeypatch.setattr("pymmcore_gui.widgets._mm_console.MMConsole", FakeConsole)

    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    assert page.panel_widget(PanelKey.CONSOLE) is None
    assert page.panel_dock(PanelKey.CONSOLE) is None

    page.panel_button(PanelKey.PRESETS).setChecked(True)  # default-open already
    presets_dock = page.panel_dock(PanelKey.PRESETS)
    assert presets_dock is not None
    page.panel_button(PanelKey.CONSOLE).click()
    console = page.panel_widget(PanelKey.CONSOLE)
    dock = page.panel_dock(PanelKey.CONSOLE)
    assert isinstance(console, FakeConsole)
    assert dock is not None and not dock.isClosed()
    assert dock.dockAreaWidget() is presets_dock.dockAreaWidget()

    page.panel_button(PanelKey.CONSOLE).click()
    assert dock.isClosed()
    assert not page.panel_button(PanelKey.CONSOLE).isChecked()
    assert page.panel_widget(PanelKey.CONSOLE) is console  # not rebuilt


def test_acquire_stage_explorer_is_a_lazy_toolbar_dock(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """The toolbar opens one themed Stage Explorer in the right dock area."""
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    button = page.panel_button(PanelKey.STAGE_EXPLORER)
    assert not button.isChecked()
    assert not button.icon().isNull()
    assert page.panel_widget(PanelKey.STAGE_EXPLORER) is None

    button.click()
    explorer = page.panel_widget(PanelKey.STAGE_EXPLORER)
    dock = page.panel_dock(PanelKey.STAGE_EXPLORER)
    assert isinstance(explorer, ThemedStageExplorer)
    assert explorer._stage_poller.isRunning()
    assert dock is not None and not dock.isClosed()
    assert dock.widget() is explorer
    assert dock.windowTitle() == "Stage Explorer"
    assert button.isChecked()

    button.click()
    assert dock.isClosed()
    assert page.panel_widget(PanelKey.STAGE_EXPLORER) is explorer

    page.close()
    assert not explorer._stage_poller.isRunning()


def test_acquire_stages_panel_starts_empty(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    """The Stages panel opens with no stage docks; Add lists every device."""
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    assert page.panel_widget(PanelKey.STAGES) is None
    # per-device flavor: the button defaults to XYZStageWidget (see StageKind)
    page._set_stage_kind(StageKind.PER_DEVICE)
    page.panel_button(PanelKey.STAGES).click()
    panel = page.panel_widget(PanelKey.STAGES)
    assert isinstance(panel, StagesPanel)
    assert panel._docks == {}
    assert not panel._empty_hint.isHidden()

    # test_config.cfg: one XY stage ("XY") and two Z stages ("Z", "Z1").
    menu = panel._build_add_menu()
    assert [a.text() for a in menu.actions()] == ["XY", "Z", "Z1"]


def test_acquire_stages_panel_add_and_remove_devices(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Adding stages opens independent docks; closing one frees it for re-adding."""
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page._set_stage_kind(StageKind.PER_DEVICE)
    page.panel_button(PanelKey.STAGES).click()
    panel = page.panel_widget(PanelKey.STAGES)
    assert isinstance(panel, StagesPanel)

    menu = panel._build_add_menu()
    next(a for a in menu.actions() if a.text() == "XY").trigger()
    assert panel._empty_hint.isHidden()
    xy_dock = panel._docks["XY"]
    xy_widget = xy_dock.widget()
    assert isinstance(xy_widget, StageWidget)
    assert xy_widget._device == "XY"
    assert not xy_dock.isClosed()

    # "XY" no longer offered; both Z stages still are.
    menu2 = panel._build_add_menu()
    assert [a.text() for a in menu2.actions()] == ["Z", "Z1"]

    # Adding a second, different device leaves the first untouched -- both
    # coexist, split into their own side-by-side column in the panel's own
    # nested dock area (not tabbed together).
    next(a for a in menu2.actions() if a.text() == "Z1").trigger()
    z1_dock = panel._docks["Z1"]
    assert isinstance(z1_dock.widget(), StageWidget)
    assert not xy_dock.isClosed()
    assert not z1_dock.isClosed()
    assert z1_dock.dockAreaWidget() is not xy_dock.dockAreaWidget()

    # Closing one (its own tab's X) removes it and frees the name to re-add.
    xy_dock.closeDockWidget()
    assert "XY" not in panel._docks
    menu3 = panel._build_add_menu()
    assert [a.text() for a in menu3.actions()] == ["XY", "Z"]
    assert panel._empty_hint.isHidden()  # Z1 is still open, hint not needed


def test_acquire_stages_panel_add_menu_disabled_when_nothing_left(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Once every loaded stage is open, the Add menu says so instead of being empty."""
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page._set_stage_kind(StageKind.PER_DEVICE)
    page.panel_button(PanelKey.STAGES).click()
    panel = page.panel_widget(PanelKey.STAGES)
    assert isinstance(panel, StagesPanel)

    for device in ("XY", "Z", "Z1"):
        panel._add_stage(device)

    menu = panel._build_add_menu()
    actions = menu.actions()
    assert len(actions) == 1
    assert actions[0].text() == "No stages available"
    assert not actions[0].isEnabled()


def test_stages_button_defaults_to_xyz_stage_widget(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """A plain (left-)click on Stages opens XYZStageWidget, not the per-device panel."""
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    assert page._stage_kind == StageKind.XYZ

    page.panel_button(PanelKey.STAGES).click()
    widget = page.panel_widget(PanelKey.STAGES)
    assert isinstance(widget, XYZStageWidget)

    # its move/halt buttons are unstyled like every other themed button --
    # unlike StagesPanel, which does this itself per added device
    buttons = widget.findChildren(QAbstractButton)
    assert buttons
    assert all(btn.property("variant") == "subtle" for btn in buttons)


def test_stages_kind_converts_in_place_without_losing_the_dock(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Switching flavors swaps the docked content but keeps the same CDockWidget.

    So a conversion doesn't disturb wherever the user has the panel arranged
    (position, pin state, tab) -- only its content widget changes.
    """
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page.panel_button(PanelKey.STAGES).click()
    dock = page.panel_dock(PanelKey.STAGES)
    assert isinstance(page.panel_widget(PanelKey.STAGES), XYZStageWidget)

    page._set_stage_kind(StageKind.PER_DEVICE)
    assert page.panel_dock(PanelKey.STAGES) is dock
    assert isinstance(page.panel_widget(PanelKey.STAGES), StagesPanel)

    page._set_stage_kind(StageKind.XYZ)
    assert page.panel_dock(PanelKey.STAGES) is dock
    assert isinstance(page.panel_widget(PanelKey.STAGES), XYZStageWidget)

    # selecting the kind already active is a no-op, not a rebuild
    widget = page.panel_widget(PanelKey.STAGES)
    page._set_stage_kind(StageKind.XYZ)
    assert page.panel_widget(PanelKey.STAGES) is widget


def test_stages_kind_change_before_first_open_is_deferred(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Picking a kind before the panel has ever been opened just sets the default."""
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    assert page.panel_dock(PanelKey.STAGES) is None

    page._set_stage_kind(StageKind.PER_DEVICE)
    assert page.panel_dock(PanelKey.STAGES) is None  # nothing built yet

    page.panel_button(PanelKey.STAGES).click()
    assert isinstance(page.panel_widget(PanelKey.STAGES), StagesPanel)


def test_xyz_stage_widget_snap_checkbox_ensures_preview(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Checking XYZStageWidget's own Snap checkbox must create the lazy Preview.

    Regression test: ``_on_stage_widget_added`` was only ever wired for the
    per-device StagesPanel's ``stageWidgetAdded`` signal, which XYZStageWidget
    doesn't emit -- checking its Snap and moving the stage (without having
    snapped/gone live first) snapped into a Preview that didn't exist yet, so
    nothing was shown.
    """
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page.panel_button(PanelKey.STAGES).click()
    widget = page.panel_widget(PanelKey.STAGES)
    assert isinstance(widget, XYZStageWidget)

    with patch.object(page._viewers, "ensure_preview") as ensure_preview:
        widget.snap_checkbox.setChecked(True)
    ensure_preview.assert_called_once()


def test_per_device_stage_widget_snap_checkbox_still_ensures_preview(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """The pre-existing per-device StageWidget wiring survives the XYZ addition."""
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page._set_stage_kind(StageKind.PER_DEVICE)
    page.panel_button(PanelKey.STAGES).click()
    panel = page.panel_widget(PanelKey.STAGES)
    assert isinstance(panel, StagesPanel)
    panel._add_stage("XY")
    stage_widget = panel._docks["XY"].widget()
    assert isinstance(stage_widget, StageWidget)

    with patch.object(page._viewers, "ensure_preview") as ensure_preview:
        stage_widget.snap_checkbox.setChecked(True)
    ensure_preview.assert_called_once()


def test_stage_explorer_sends_positions_to_mda(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Stage Explorer's Replace/Add choices update and reveal MDA Positions."""
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page.panel_button(PanelKey.STAGE_EXPLORER).click()
    explorer = page.panel_widget(PanelKey.STAGE_EXPLORER)
    assert isinstance(explorer, ThemedStageExplorer)

    old = useq.Position(name="Old", x=1, y=2)
    page._mda.stage_positions.setValue([old])
    replacement = [
        useq.Position(name="Explorer 1", x=10, y=20),
        useq.Position(name="Explorer 2", x=30, y=40),
    ]
    explorer.sendToMDARequested.emit(replacement, True)
    assert page._mda.stage_positions.value(exclude_unchecked=False) == tuple(
        replacement
    )

    added = useq.Position(name="Explorer 3", x=50, y=60)
    explorer.sendToMDARequested.emit([added], False)
    assert page._mda.stage_positions.value(exclude_unchecked=False) == (
        *replacement,
        added,
    )
    section = page._mda._collapsible_tabs().section("p")
    assert section.checked
    assert section.expanded
    assert section.summary == "On · 3 positions"
    assert page.panel_button(PanelKey.MDA).isChecked()


def test_acquire_panel_buttons_match_registry(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Every registry entry gets one icon-only, checkable toggle button."""
    set_theme(DARK_THEME)
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    for info in PANELS:
        btn = page.panel_button(info.key)
        assert btn.isCheckable()
        assert btn.text() == ""
        assert not btn.icon().isNull()
        assert btn.toolTip() == info.tooltip
        assert btn.property("variant") == "subtle"

    for info in PANELS:
        assert page.panel_button(info.key).isChecked() == info.default_open

    # Only the default-open panels build eagerly; the rest of the registry
    # stays lazy, matching the guarantee the other tests exercise.
    assert sorted(page._dock_manager.dockWidgetsMap()) == [
        "acquire_mda",
        "acquire_presets",
        "acquire_viewers",
    ]

    # The bar's contents (not its host) are what the registry contract
    # guarantees, so this test survives relocating it -- see
    # AcquirePage._place_panel_bar. This one assertion pins today's default
    # placement and is the only line to update if the bar moves.
    assert page._panel_bar.parent() is page.toolbar


def _panel_customize_menu(page: AcquirePage) -> QMenu:
    """The ⋯ customize menu, built but not exec'd (exec would block the test)."""
    return page._panel_bar.build_menu()


def _toggle_customize_menu_entry(page: AcquirePage, title: str, checked: bool) -> None:
    """Check/uncheck one entry of a freshly built customize menu, as a click would."""
    menu = _panel_customize_menu(page)
    action = next(a for a in menu.actions() if a.text() == title)
    action.setChecked(checked)
    menu.deleteLater()


def test_acquire_customize_menu_lists_hideable_panels(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """The ⋯ menu offers every panel except the always-visible MDA one."""
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    menu = _panel_customize_menu(page)
    toggles = [a for a in menu.actions() if a.isCheckable()]
    titles = [a.text() for a in toggles]
    assert titles == [info.title for info in PANELS if not info.always_visible]
    # Everything starts visible, so every entry starts checked.
    assert all(action.isChecked() for action in toggles)
    assert "MDA" not in titles
    assert not page.hidden_panels()

    # Nothing else: layout operations live in the layout menu next door.
    assert [a.text() for a in menu.actions() if not a.isCheckable() and a.text()] == []

    # Both affordances exist: the bar's own ⋯ button, and right-click on the
    # host toolbar row.
    assert not page._panel_bar._menu_btn.icon().isNull()
    assert page.toolbar.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu


def test_acquire_customize_menu_hides_button_and_closes_panel(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Unchecking a panel in the ⋯ menu removes its button and closes its dock."""
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    page.panel_button(PanelKey.PRESETS).setChecked(True)  # default-open already
    presets_dock = page.panel_dock(PanelKey.PRESETS)
    assert presets_dock is not None and not presets_dock.isClosed()

    _toggle_customize_menu_entry(page, "Groups and Presets", False)
    assert page.panel_button(PanelKey.PRESETS).isHidden()
    assert presets_dock.isClosed()
    assert page.hidden_panels() == {PanelKey.PRESETS}
    assert PanelKey.PRESETS not in page.open_panels()
    # The widget is kept alive, same as a plain close/reopen.
    assert page.panel_widget(PanelKey.PRESETS) is not None

    # Re-checking brings the button back *and* re-opens the panel -- that's
    # the point of picking it from the menu.
    _toggle_customize_menu_entry(page, "Groups and Presets", True)
    assert not page.panel_button(PanelKey.PRESETS).isHidden()
    assert not presets_dock.isClosed()
    assert page.hidden_panels() == set()


def test_acquire_customize_menu_cannot_hide_mda(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """MDA is always_visible, so even a direct request can't hide its button."""
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    page._set_panel_visible(PanelKey.MDA, False)
    assert not page.panel_button(PanelKey.MDA).isHidden()
    assert PanelKey.MDA not in page.hidden_panels()


def test_acquire_apply_hidden_panels_round_trips(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """apply_hidden_panels() is the inverse of hidden_panels(), incl. unknown keys."""
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    page.apply_hidden_panels({PanelKey.CONSOLE, PanelKey.EXCEPTION_LOG})
    assert page.hidden_panels() == {PanelKey.CONSOLE, PanelKey.EXCEPTION_LOG}
    assert page.panel_button(PanelKey.CONSOLE).isHidden()
    assert not page.panel_button(PanelKey.PRESETS).isHidden()
    # Hiding a never-opened panel must not build it (laziness is preserved).
    assert page.panel_widget(PanelKey.CONSOLE) is None

    # A key from a newer/older release is ignored rather than raising.
    page.apply_hidden_panels({PanelKey.CONSOLE, "not_a_registered_panel"})
    assert page.hidden_panels() == {PanelKey.CONSOLE}


def test_acquire_apply_hidden_panels_never_opens_panels(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Showing a button must not open its panel -- only the ⋯ menu does that.

    Regression test: ``apply_hidden_panels`` used to route through
    ``_set_panel_visible``, whose "re-adding a button opens its panel"
    behaviour is right for an interactive menu click but catastrophic on the
    restore path -- it force-opened *every* registered panel on launch,
    eagerly building all of them and burying the MDA dock.
    """
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    page.apply_hidden_panels(set())

    assert page.open_panels() == {PanelKey.MDA, PanelKey.PRESETS}
    assert sorted(page._dock_manager.dockWidgetsMap()) == [
        "acquire_mda",
        "acquire_presets",
        "acquire_viewers",
    ]
    for info in PANELS:
        assert not page.panel_button(info.key).isHidden()
        if not info.default_open:
            assert page.panel_widget(info.key) is None


def test_acquire_docked_panels_are_reparented_not_windows(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Docking reparents every panel widget, clearing any standalone window flags.

    ``PropertyBrowser`` is a QDialog upstream and ``create_exception_log``
    sets ``WindowStaysOnTopHint | Window``. ``dock.setWidget()`` reparents
    them, and ``QWidget.setParent()`` clears window flags -- this must keep
    working *without* a pre-emptive ``setWindowFlags()`` call, which Qt
    documents as hiding the widget.
    """
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page.resize(1400, 900)
    page.show()
    qtbot.waitExposed(page)

    for key in (PanelKey.PROPERTIES, PanelKey.EXCEPTION_LOG):
        page.panel_button(key).click()
        widget = page.panel_widget(key)
        dock = page.panel_dock(key)
        assert widget is not None and dock is not None
        assert not widget.isWindow()
        assert widget.parent() is not None
        assert not dock.isClosed()

    # The MDA panel, which is open from the start, must actually be on screen.
    assert page._mda.isVisible()


def test_acquire_new_panels_go_to_right_column_and_tab_together(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """A panel opened for the first time lands in the right sidebar, tabbed.

    The first one creates the right column; every subsequent one tabs into
    that same column rather than spawning a second one beside it. MDA is the
    only panel placed elsewhere (left).
    """
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    side_keys = [i.key for i in PANELS if i.key != PanelKey.MDA]
    page.panel_button(side_keys[0]).click()
    right_area = page._right_dock_area
    assert right_area is not None
    assert right_area is not page._central_dock_area
    assert page._mda_dock.dockAreaWidget() is not right_area

    for key in side_keys[1:]:
        page.panel_button(key).click()
        dock = page.panel_dock(key)
        assert dock is not None
        assert dock.dockAreaWidget() is right_area, f"{key} did not tab into the column"

    assert right_area.dockWidgetsCount() == len(side_keys)


def test_acquire_right_column_survives_being_emptied(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """After the right column is emptied, the next new panel rebuilds it.

    ADS destroys a dock area once its last dock leaves, so the cached
    ``_right_dock_area`` can end up wrapping a deleted C++ object. Docking
    *into* that would crash instead of tabbing, hence ``_resolve_right_dock_area``.
    """
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    page.panel_button(PanelKey.PRESETS).setChecked(True)  # default-open already
    assert page._right_dock_area is not None
    page.panel_button(PanelKey.PRESETS).setChecked(
        True
    )  # default-open already  # close it again

    # Must not raise, and must give the new panel a usable right column.
    page.panel_button(PanelKey.EXCEPTION_LOG).click()
    log_dock = page.panel_dock(PanelKey.EXCEPTION_LOG)
    assert log_dock is not None and not log_dock.isClosed()
    area = log_dock.dockAreaWidget()
    assert area is not None
    assert area is not page._central_dock_area
    assert page._mda_dock.dockAreaWidget() is not area


def test_acquire_reset_layout_restores_defaults(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Reset Layout re-opens only the defaults, un-hides buttons, re-pins widths."""
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page.resize(1400, 900)
    page.show()
    qtbot.waitExposed(page)
    qtbot.waitUntil(lambda: page._mda_width_locked_at_real_size, timeout=2000)

    # MDA and Groups and Presets are the defaults; open a third panel and
    # hide a button to get somewhere a reset has to undo.
    page.panel_button(PanelKey.EXCEPTION_LOG).click()
    page.apply_hidden_panels({PanelKey.CONSOLE})
    assert page.open_panels() == {
        PanelKey.MDA,
        PanelKey.PRESETS,
        PanelKey.EXCEPTION_LOG,
    }
    assert page.hidden_panels() == {PanelKey.CONSOLE}

    with qtbot.waitSignal(page.layoutReset):
        page.reset_layout()

    assert page.open_panels() == {PanelKey.MDA, PanelKey.PRESETS}
    assert page.hidden_panels() == set()
    for info in PANELS:
        assert not page.panel_button(info.key).isHidden()
    # Widgets survive, exactly as a normal close/reopen does.
    assert page.panel_widget(PanelKey.PRESETS) is not None

    mda_area = page._mda_dock.dockAreaWidget()
    assert mda_area is not None
    assert mda_area.width() == _MDA_DOCK_WIDTH


def test_acquire_layout_menu_reset_entry_is_wired(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Selecting "Default" from the layout menu reaches AcquirePage.reset_layout.

    Deliberately exercises reset via a *hidden button* rather than an open
    panel: this is a wiring test, and hiding a never-opened panel builds no
    dock, so nothing here empties a dock area -- the ADS operation that is
    fatal under the offscreen test platform (see ``_configure_ads``). The
    behavioural coverage lives in the reset tests above.
    """
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page.apply_hidden_panels({PanelKey.CONSOLE})
    assert page.panel_button(PanelKey.CONSOLE).isHidden()

    menu = page._layout_btn.build_menu()
    reset = next(a for a in menu.actions() if a.text() == DEFAULT_LAYOUT_NAME)
    with qtbot.waitSignal(page.layoutReset):
        reset.trigger()
    menu.deleteLater()

    assert not page.panel_button(PanelKey.CONSOLE).isHidden()
    assert page.panel_widget(PanelKey.CONSOLE) is None  # still never built
    assert page.open_panels() == {PanelKey.MDA, PanelKey.PRESETS}


def test_acquire_reset_layout_after_restore_repins_default_widths(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """A reset supersedes a restored layout, so the canonical widths come back."""
    page_a = AcquirePage(mmcore)
    qtbot.addWidget(page_a)
    page_a.resize(1400, 900)
    page_a.show()
    qtbot.waitExposed(page_a)
    qtbot.waitUntil(lambda: page_a._mda_width_locked_at_real_size, timeout=2000)

    mda_area = page_a._mda_dock.dockAreaWidget()
    assert mda_area is not None
    handle = next(h for h, a in page_a._width_locked_areas.items() if a is mda_area)
    qtbot.mousePress(handle, Qt.MouseButton.LeftButton)  # type: ignore[no-untyped-call]
    page_a._dock_manager.setSplitterSizes(
        mda_area, _resized_splitter_sizes(page_a, mda_area, 500)
    )
    qtbot.mouseRelease(handle, Qt.MouseButton.LeftButton)  # type: ignore[no-untyped-call]
    _park_real_cursor_away()
    saved = page_a.current_layout()
    state, keys = saved.dock_state, saved.panels
    assert state is not None

    page_b = AcquirePage(mmcore)
    qtbot.addWidget(page_b)
    page_b.resize(1400, 900)
    assert page_b.restore_layout(state, keys)
    page_b.show()
    qtbot.waitExposed(page_b)
    qtbot.waitUntil(lambda: page_b._mda_width_locked_at_real_size, timeout=2000)
    assert page_b._layout_restored

    page_b.reset_layout()
    assert not page_b._layout_restored
    mda_area_b = page_b._mda_dock.dockAreaWidget()
    assert mda_area_b is not None
    assert mda_area_b.width() == _MDA_DOCK_WIDTH


def test_acquire_camera_roi_is_embedded_and_exception_log_panel_opens(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Camera ROI lives in MDA; the exception log remains a docked panel."""
    from pymmcore_widgets import CameraRoiWidget

    from pymmcore_gui.widgets._exception_log import ExceptionLog

    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    assert isinstance(page._mda.camera_roi, CameraRoiWidget)
    assert page._mda.tabs.roi_section.title == "Camera ROI"
    assert page._mda.camera_roi.select_roi_btn.isVisibleTo(page._mda.camera_roi)
    assert not page._mda.camera_roi.snap_checkbox.isHidden()
    assert page._mda.camera_roi.snap_checkbox.isChecked()
    assert not page._mda.camera_roi.roiInfoVisible()
    assert page._mda.camera_roi._info_lbl_wdg.isHidden()
    select_rgb = _icon_avg_rgb(
        page._mda.camera_roi.select_roi_btn.icon(), QSize(24, 24)
    )
    crop_rgb = _icon_avg_rgb(page._mda.camera_roi.crop_btn.icon(), QSize(24, 24))
    assert select_rgb is not None and crop_rgb is not None
    assert all(abs(a - b) < 4 for a, b in zip(select_rgb, crop_rgb, strict=True))
    assert "camera_roi" not in page._panels

    page.panel_button(PanelKey.EXCEPTION_LOG).click()
    log_widget = page.panel_widget(PanelKey.EXCEPTION_LOG)
    log_dock = page.panel_dock(PanelKey.EXCEPTION_LOG)
    assert isinstance(log_widget, ExceptionLog)
    assert log_dock is not None and not log_dock.isClosed()
    # create_exception_log sets WindowStaysOnTopHint | Window upstream --
    # every registry panel must be normalized to a plain docked child.
    assert not log_widget.isWindow()

    assert log_dock.dockAreaWidget() is not page._mda_dock.dockAreaWidget()


def test_acquire_rejects_layout_with_removed_camera_roi_panel(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    assert not page.restore_layout(b"legacy ADS state", {PanelKey.MDA, "camera_roi"})
    assert page.open_panels() == {PanelKey.MDA, PanelKey.PRESETS}


def test_camera_roi_preset_restarts_live_toolbar(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    section = page._mda.tabs.roi_section
    section.set_checked(True)
    # Auto Snap's live stop/snap/restart order is covered upstream. Keep this
    # toolbar-state check headless; showing the full ADS window requires a real
    # macOS screen/graphics context.
    page._mda.camera_roi.snap_checkbox.setChecked(False)

    mmcore.startContinuousSequenceAcquisition()
    assert page._live_btn.isChecked()
    assert page._live_btn.toolTip() == "Stop"

    try:
        page._mda.camera_roi.camera_roi_combo.setCurrentText("64 x 64")

        assert not mmcore.isSequenceRunning()
        assert not page._live_btn.isChecked()
        assert page._live_btn.toolTip() == "Live"
        qtbot.waitUntil(mmcore.isSequenceRunning)
        qtbot.waitUntil(page._live_btn.isChecked)
        assert page._live_btn.toolTip() == "Stop"
    finally:
        if mmcore.isSequenceRunning():
            mmcore.stopSequenceAcquisition()


def test_camera_roi_auto_snap_creates_preview_before_snap(
    mmcore: CMMCorePlus, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    editor = page._mda.camera_roi
    page._mda.tabs.roi_section.set_checked(True)
    assert editor.snap_checkbox.isChecked()
    assert page._viewers.preview is None

    operations: list[str] = []
    monkeypatch.setattr(editor.snap_checkbox, "isVisible", lambda: True)
    monkeypatch.setattr(
        page._viewers, "ensure_preview", lambda: operations.append("preview")
    )
    mmcore.events.imageSnapped.connect(lambda: operations.append("snap"))

    editor.camera_roi_combo.setCurrentText("64 x 64")

    assert operations == ["preview", "snap"]


def test_camera_roi_live_view_sync_is_bidirectional(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    editor = page._mda.camera_roi
    section = page._mda.tabs.roi_section
    planned = {
        "camera": "Camera",
        "x": 20,
        "y": 30,
        "width": 200,
        "height": 160,
    }
    editor.setRoiValue(planned)
    section.set_checked(True)

    try:
        editor.select_roi_btn.click()
        qtbot.waitUntil(mmcore.isSequenceRunning, timeout=2_000)
        assert page._roi_sync.active
        assert tuple(mmcore.getROI("Camera")) == (0, 0, 512, 512)
        preview = page._viewers.preview
        assert preview is not None
        viewer = preview.viewer
        assert viewer.roi is not None
        assert viewer.roi.bounding_box == ((20.0, 30.0), (220.0, 190.0))
        assert not viewer.roi_selection_active()
        assert viewer.existing_roi_editing_active()
        assert viewer.roi_visual_visible()
        assert viewer.roi_visual_selected()

        # Viewer coordinates are normalized to integer camera pixels and only edit
        # the plan; hardware remains full-frame throughout the selection session.
        viewer.roi.bounding_box = ((10.2, 12.8), (111.1, 99.2))
        assert editor.roiValue() == {
            "camera": "Camera",
            "x": 10,
            "y": 12,
            "width": 102,
            "height": 88,
        }
        assert tuple(mmcore.getROI("Camera")) == (0, 0, 512, 512)

        typed = {
            "camera": "Camera",
            "x": 40,
            "y": 50,
            "width": 120,
            "height": 100,
        }
        editor.setRoiValue(typed)
        assert viewer.roi.bounding_box == ((40.0, 50.0), (160.0, 150.0))

        # Leaving the MDA selection session stops live and removes its overlay.
        editor.select_roi_btn.click()
        assert not page._roi_sync.active
        qtbot.waitUntil(lambda: not mmcore.isSequenceRunning(), timeout=2_000)
        assert viewer.roi is None
        assert not viewer.roi_selection_active()

        # ndv's own ROI button remains local: it must not start live or opt the
        # MDA camera ROI into a selection session. Drawing only copies the ROI
        # into the disabled editor as a Custom ROI plan.
        section.set_checked(False)
        viewer.set_roi_selection_active(True)
        assert not page._roi_sync.active
        assert not mmcore.isSequenceRunning()
        assert not section.checked
        assert viewer.roi is not None
        viewer.roi.bounding_box = ((5.2, 7.8), (105.1, 87.4))
        assert editor.roiValue() == {
            "camera": "Camera",
            "x": 5,
            "y": 7,
            "width": 101,
            "height": 81,
        }
        assert editor.camera_roi_combo.currentText() == "Custom ROI"
        assert not section.checked
        assert not page._roi_sync.active
        assert not mmcore.isSequenceRunning()
        viewer.set_roi_selection_active(False)

        # Only the MDA action starts the coordinated selection session.
        section.set_checked(True)
        editor.select_roi_btn.click()
        qtbot.waitUntil(mmcore.isSequenceRunning, timeout=2_000)
        assert page._roi_sync.active
        assert viewer.roi is not None
        assert viewer.roi_visual_visible()
        assert viewer.roi_visual_selected()

        editor.crop_btn.click()
        assert tuple(mmcore.getROI("Camera")) == (5, 7, 101, 81)
        assert not page._roi_sync.active
        qtbot.waitUntil(mmcore.isSequenceRunning, timeout=2_000)
    finally:
        if mmcore.isSequenceRunning():
            mmcore.stopSequenceAcquisition()


def test_camera_roi_session_configures_ndv_before_roi_and_stops_live(
    mmcore: CMMCorePlus, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression coverage that does not require constructing a GPU canvas."""

    class FakeSignal:
        def __init__(self) -> None:
            self.callbacks: list[object] = []

        def connect(self, callback: object) -> None:
            self.callbacks.append(callback)

        def disconnect(self, callback: object) -> None:
            self.callbacks.remove(callback)

        def emit(self, *args: object) -> None:
            for callback in list(self.callbacks):
                callback(*args)  # type: ignore[operator]

    class FakeRoi:
        def __init__(
            self, bbox: tuple[tuple[float, float], tuple[float, float]]
        ) -> None:
            self._bounding_box = bbox
            self.visible = True
            self.events = SimpleNamespace(bounding_box=FakeSignal())

        @property
        def bounding_box(self) -> tuple[tuple[float, float], tuple[float, float]]:
            return self._bounding_box

        @bounding_box.setter
        def bounding_box(
            self, value: tuple[tuple[float, float], tuple[float, float]]
        ) -> None:
            self._bounding_box = value
            self.events.bounding_box.emit(value)

    class FakeViewer:
        def __init__(self) -> None:
            self.operations: list[str] = []
            self._roi: FakeRoi | None = None
            self._mode_active = False
            self._mode_changed = FakeSignal()
            self._existing_edit_active = False
            self.visual_selected = False

        @property
        def roi(self) -> FakeRoi | None:
            return self._roi

        @roi.setter
        def roi(
            self,
            value: tuple[tuple[float, float], tuple[float, float]] | None,
        ) -> None:
            self.operations.append("roi")
            self._roi = None if value is None else FakeRoi(value)

        def set_roi_selection_active(self, active: bool) -> None:
            self.operations.append(f"mode:{active}")
            self._mode_active = active
            self._mode_changed.emit(active)

        def roi_selection_active(self) -> bool:
            return self._mode_active

        def set_roi_visual_selected(self, selected: bool) -> None:
            self.visual_selected = selected

        def set_existing_roi_editing_active(self, active: bool) -> None:
            self.operations.append(f"edit:{active}")
            self._existing_edit_active = active
            self._mode_active = False
            self.visual_selected = active

        def clear_roi(self) -> None:
            self.roi = None

        def connect_roi_selection_changed(self, callback: object) -> None:
            self._mode_changed.connect(callback)

        def disconnect_roi_selection_changed(self, callback: object) -> None:
            self._mode_changed.disconnect(callback)

    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    viewer = FakeViewer()
    preview = SimpleNamespace(viewer=viewer)
    monkeypatch.setattr(page._viewers, "ensure_preview", lambda: preview)
    monkeypatch.setattr(
        page._live_btn,
        "ensure_live",
        mmcore.startContinuousSequenceAcquisition,
    )

    editor = page._mda.camera_roi
    editor.setRoiValue(
        {"camera": "Camera", "x": 20, "y": 30, "width": 200, "height": 160}
    )
    page._mda.tabs.roi_section.set_checked(True)
    editor.select_roi_btn.click()

    assert viewer.operations[:2] == ["roi", "edit:True"]
    assert viewer.roi is not None
    assert viewer.roi.bounding_box == ((20, 30), (220, 190))
    assert viewer.visual_selected
    assert mmcore.isSequenceRunning()

    viewer.roi.bounding_box = ((30.25, 40.75), (230.25, 200.75))
    assert editor.roiValue() == {
        "camera": "Camera",
        "x": 30,
        "y": 40,
        "width": 200,
        "height": 160,
    }
    assert viewer.roi.bounding_box == ((30.25, 40.75), (230.25, 200.75))

    editor.select_roi_btn.click()

    assert not page._roi_sync.active
    assert viewer.roi is None
    assert not viewer.roi_selection_active()
    assert not mmcore.isSequenceRunning()

    # A standalone ndv ROI remains standalone. It updates only the editor
    # coordinates/mode, preserving both live state and the MDA opt-in checkbox.
    section = page._mda.tabs.roi_section
    section.set_checked(False)
    viewer.roi = ((0, 0), (1, 1))
    viewer.set_roi_selection_active(True)
    # Explicit annotation: pyright narrows `viewer.roi` to the tuple literal
    # just assigned above rather than the property's actual `FakeRoi | None`
    # getter type, so a bare `assert viewer.roi is not None` doesn't stick.
    roi: FakeRoi | None = viewer.roi
    assert roi is not None
    roi.bounding_box = ((5.2, 7.8), (105.1, 87.4))

    assert editor.roiValue() == {
        "camera": "Camera",
        "x": 5,
        "y": 7,
        "width": 101,
        "height": 81,
    }
    assert editor.camera_roi_combo.currentText() == "Custom ROI"
    assert not section.checked
    assert not page._roi_sync.active
    assert not mmcore.isSequenceRunning()

    # Translating both edges by the same fractional delta must preserve size;
    # independently flooring/ceiling them used to grow each dimension by one.
    roi.bounding_box = ((20.25, 22.75), (120.15, 102.35))
    assert editor.roiValue() == {
        "camera": "Camera",
        "x": 20,
        "y": 22,
        "width": 101,
        "height": 81,
    }
    roi.bounding_box = ((30.6, 32.4), (130.5, 112.0))
    assert editor.roiValue()["width"] == 101
    assert editor.roiValue()["height"] == 81

    # Acquired MDA viewers use the same passive one-way synchronization.
    mda_viewer = FakeViewer()
    page._viewers.mdaViewerCreated.emit(mda_viewer)
    mda_viewer.roi = ((0, 0), (1, 1))
    mda_viewer.set_roi_selection_active(True)
    mda_roi: FakeRoi | None = mda_viewer.roi
    assert mda_roi is not None
    mda_roi.bounding_box = ((25.4, 30.1), (125.2, 90.9))

    assert editor.roiValue() == {
        "camera": "Camera",
        "x": 25,
        "y": 30,
        "width": 101,
        "height": 61,
    }
    assert editor.camera_roi_combo.currentText() == "Custom ROI"
    assert not section.checked
    assert not mmcore.isSequenceRunning()

    page._viewers.mdaViewerClosed.emit(mda_viewer)


def test_acquire_panel_button_icons_follow_theme(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Panel-button icons re-derive their color on a theme toggle, in both directions.

    Regression test for using ``setIcon`` instead of ``set_source_icon``: a
    bare ``setIcon`` would leave the app-wide ``ensure_visible_icon`` sweep
    (which runs right after our ``StyleChange`` handler) re-deriving from the
    *previous* theme's icon, since it never sees our freshly-set one as the
    "original".
    """
    set_theme(DARK_THEME)
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    btn = page.panel_button(PanelKey.PRESETS)
    rgb = _icon_avg_rgb(btn.icon(), QSize(24, 24))
    assert rgb is not None
    dark = qcolor(theme().status_green)
    assert all(
        abs(a - b) < 4
        for a, b in zip(rgb, (dark.red(), dark.green(), dark.blue()), strict=True)
    )

    set_theme(LIGHT_THEME)
    rgb = _icon_avg_rgb(btn.icon(), QSize(24, 24))
    assert rgb is not None
    light = qcolor(theme().status_green)
    assert all(
        abs(a - b) < 4
        for a, b in zip(rgb, (light.red(), light.green(), light.blue()), strict=True)
    )
    set_theme(DARK_THEME)


def test_acquire_panel_buttons_follow_zoom(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    """Panel-button icons rescale with zoom, same as Snap/Live.

    Unlike a ``QToolBar``, a bare ``QPushButton`` isn't touched by
    ``set_zoom()``'s icon-size pass over every ``QToolBar`` -- so
    ``PanelButtonBar`` must re-apply its own icon size on every zoom change,
    same as ``SnapButton``/``LiveButton`` already do.
    """
    set_theme(DARK_THEME)
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    try:
        set_zoom(1.5)
        expected = acquire_toolbar_module._icon_size()
        for info in PANELS:
            assert page.panel_button(info.key).iconSize() == expected

        set_zoom(0.8)
        expected = acquire_toolbar_module._icon_size()
        for info in PANELS:
            assert page.panel_button(info.key).iconSize() == expected
    finally:
        set_theme(DARK_THEME)


def test_snap_opens_closable_preview(
    mmcore: CMMCorePlus,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePreview(QWidget):
        def __init__(
            self,
            mmcore: CMMCorePlus,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self._core = mmcore
            self.frames = 0
            self.detached = False
            mmcore.events.imageSnapped.connect(self._on_image_snapped)

        def _on_image_snapped(self) -> None:
            self._core.getImage()
            self.frames += 1

        def detach(self) -> None:
            self.detached = True
            self._core.events.imageSnapped.disconnect(self._on_image_snapped)

    def run_worker_now(func: Callable[[], None], **_: object) -> None:
        func()

    monkeypatch.setattr(acquire_viewers_module, "NDVPreview", FakePreview)
    monkeypatch.setattr(acquire_toolbar_module, "create_worker", run_worker_now)

    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    assert page._viewers._preview_dock is None

    page._snap_btn.click()
    preview = page._viewers.preview
    assert isinstance(preview, FakePreview)
    preview_dock = page._viewers._preview_dock
    assert preview_dock is not None
    assert preview_dock.windowTitle() == "Preview"
    assert preview.frames == 1

    preview_dock.closeDockWidget()
    assert page._viewers._preview_dock is None
    assert page._viewers.preview is None
    assert preview.detached

    page._snap_btn.click()
    replacement = page._viewers.preview
    assert isinstance(replacement, FakePreview)
    assert replacement is not preview
    assert replacement.frames == 1


def test_selecting_channel_row_applies_only_its_core_config(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    set_theme(DARK_THEME)
    mda = MemoryMDAWidget(mmcore)
    qtbot.addWidget(mda)
    mda.show()
    qtbot.waitExposed(mda)
    channels = mda.channels
    presets = tuple(mmcore.getAvailableConfigs("Channel"))
    assert len(presets) >= 4
    first = presets[0]
    other_preset = presets[2]
    third_preset = presets[3]
    mda.setValue(
        useq.MDASequence(
            channels=tuple(
                useq.Channel(
                    group="Channel",
                    config=config,
                    exposure=12.5 + index,
                )
                for index, config in enumerate(presets)
            )
        )
    )

    mmcore.setConfig("Channel", first)
    mmcore.setExposure(8)
    table = channels.table()
    model = table.model()
    assert model is not None
    config_column = table.indexOf(channels._config_column)

    # Clicking the ● column on each row activates that channel on the microscope.
    for row, preset in enumerate(presets):
        mda._on_channel_row_selected(model.index(row, 0))
        assert mmcore.getCurrentConfig("Channel") == preset

    # Row activation never pushes exposure to hardware (just-in-time rule).
    assert mmcore.getExposure() == pytest.approx(8)
    assert channels.value(exclude_unchecked=False)[1].exposure == pytest.approx(13.5)

    # Clicking the Exposure editor neither selects that row nor applies the
    # channel to the microscope: Exposure isn't a row-activating column, so
    # highlighting it would misrepresent which channel is actually active.
    exposure_column = table.indexOf(channels.EXPOSURE)
    exposure_editor = table.cellWidget(1, exposure_column)
    assert exposure_editor is not None
    table.setCurrentCell(0, config_column)  # set a known visual starting position
    mmcore.setConfig("Channel", first)
    qtbot.mouseClick(  # type: ignore[no-untyped-call]
        exposure_editor, Qt.MouseButton.LeftButton
    )
    assert table.currentRow() == 0  # unchanged
    assert mmcore.getCurrentConfig("Channel") == first  # unchanged by editor click

    # A genuine mouse click directly on the ● column (no cell widget there,
    # so this exercises table.clicked / _on_channel_cell_clicked, not
    # _on_channel_row_selected called programmatically) selects the row and
    # applies its channel.
    current_col = table.indexOf(CURRENT_CHANNEL_COLUMN)
    current_index = model.index(2, current_col)
    qtbot.mouseClick(  # type: ignore[no-untyped-call]
        table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=table.visualRect(current_index).center(),
    )
    assert channels.activeRow() == 2
    assert table.currentRow() == 2
    assert {index.row() for index in table.selectedIndexes()} == {2}
    assert mmcore.getCurrentConfig("Channel") == presets[2]
    mmcore.setConfig("Channel", first)  # reset for the next section

    # The Config combo is the other way (besides the ● column) to move
    # hardware -- but only for the row that's already active. Activating a
    # DIFFERENT row's combo just updates that row's stored channel; it never
    # applies anything or changes which row is active.
    assert channels.activeRow() == 0
    other_row_cell = table.cellWidget(1, config_column)
    assert other_row_cell is not None
    other_row_combo = other_row_cell.findChild(QComboBox)
    assert other_row_combo is not None
    other_row_combo.setCurrentText(third_preset)
    other_row_combo.activated.emit(other_row_combo.currentIndex())
    QApplication.processEvents()
    assert channels.activeRow() == 0  # unchanged -- row 1 isn't the active row
    assert table.currentRow() == 0  # unchanged
    assert mmcore.getCurrentConfig("Channel") == first  # unchanged

    # Make row 1 active first (as the user would, via the ● column) -- its
    # stored config (third_preset, set above) is applied on activation.
    mda._on_channel_row_selected(model.index(1, 0))
    assert channels.activeRow() == 1
    assert mmcore.getCurrentConfig("Channel") == third_preset

    # Now that row 1 IS the active row, a user activation of its own Config
    # combo (the `activated` signal) applies the newly picked preset
    # immediately, same as clicking the ● column. A plain programmatic text
    # change (currentTextChanged only, as during a sequence restore/refresh)
    # still stays hardware-neutral.
    config_cell = table.cellWidget(1, config_column)
    assert config_cell is not None
    config_combo = config_cell.findChild(QComboBox)
    assert config_combo is not None
    config_combo.setCurrentText(other_preset)
    QApplication.processEvents()
    assert mmcore.getCurrentConfig("Channel") == third_preset  # no hardware change yet
    config_combo.activated.emit(config_combo.currentIndex())
    QApplication.processEvents()
    assert table.currentRow() == 1
    assert channels.activeRow() == 1
    assert mmcore.getCurrentConfig("Channel") == other_preset  # now applied

    table.clearChecks()
    mda.refresh_channel_table()
    assert channels.value() == ()
    assert len(channels.value(exclude_unchecked=False)) == len(presets)
    table.checkAllRows()

    # After a table rebuild the ● row is preserved without re-applying hardware.
    # (Row 1's config combo was switched to `other_preset` above, so that --
    # not presets[1] -- is what row 1 now holds and what activating it applies.)
    mda._on_channel_row_selected(model.index(1, 0))
    assert channels.activeRow() == 1
    mda.refresh_channel_table()
    assert channels.activeRow() == 1
    assert mmcore.getCurrentConfig("Channel") == other_preset

    # Rebuilding/restoring the table via setValue also preserves the ● row.
    mda.setValue(mda.value())
    assert channels.activeRow() == 1
    assert mmcore.getCurrentConfig("Channel") == other_preset

    with patch.object(
        channels,
        "value",
        return_value=(useq.Channel(group="missing", config="unknown", exposure=1),),
    ):
        mda._on_channel_row_selected(model.index(0, config_column))
    assert mmcore.getCurrentConfig("Channel") == other_preset


def test_core_config_change_syncs_channel_row_selection(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    set_theme(DARK_THEME)
    mda = MemoryMDAWidget(mmcore)
    qtbot.addWidget(mda)
    channels = mda.channels
    table = channels.table()
    presets = tuple(mmcore.getAvailableConfigs("Channel"))
    assert len(presets) >= 3

    # Table holds the first two presets only.
    in_table = presets[:2]
    not_in_table = presets[2]
    mda.setValue(
        useq.MDASequence(
            channels=tuple(
                useq.Channel(group="Channel", config=config, exposure=100.0 + i)
                for i, config in enumerate(in_table)
            )
        )
    )

    # A preset activated on the core (e.g. from the Groups & Presets table, not
    # through this table at all) updates the ● indicator *and* visibly selects
    # the matching row -- not just the ● column.
    mmcore.setConfig("Channel", in_table[1])
    QApplication.processEvents()
    assert channels.activeRow() == 1
    assert table.currentRow() == 1
    assert {index.row() for index in table.selectedIndexes()} == {1}
    assert mmcore.getCurrentConfig("Channel") == in_table[1]

    mmcore.setConfig("Channel", in_table[0])
    QApplication.processEvents()
    assert channels.activeRow() == 0
    assert table.currentRow() == 0
    assert {index.row() for index in table.selectedIndexes()} == {0}

    # Idle sync does not push the row's exposure to the hardware
    # (just-in-time rule): only the ● indicator is updated.
    mmcore.setExposure(7.0)
    mmcore.setConfig("Channel", in_table[1])
    QApplication.processEvents()
    assert mmcore.getExposure() == pytest.approx(7.0)

    # Activating a channel that has no row clears the ● indicator and the
    # visible row selection.
    mmcore.setConfig("Channel", not_in_table)
    QApplication.processEvents()
    assert channels.activeRow() == -1
    assert table.currentRow() == -1
    assert not table.selectedIndexes()
    assert mmcore.getCurrentConfig("Channel") == not_in_table


def test_editing_active_row_exposure_or_intensity_applies_during_live(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    set_theme(DARK_THEME)
    mda = MemoryMDAWidget(mmcore)
    qtbot.addWidget(mda)
    channels = mda.channels
    table = channels.table()
    presets = tuple(mmcore.getAvailableConfigs("Channel"))
    assert len(presets) >= 2

    mda.setValue(
        useq.MDASequence(
            channels=tuple(
                useq.Channel(group="Channel", config=config, exposure=50.0 + i)
                for i, config in enumerate(presets[:2])
            )
        )
    )

    source_device, source_property = "Camera", "TestProperty1"
    source_group = next(
        label
        for label, pairs in channels.lightSources().items()
        if pairs == [(source_device, source_property)]
    )
    channels.setLightSourceVisible(True)
    channels.setChannelProperties(
        [
            {
                "channel_index": 0,
                "config": presets[0],
                "group": source_group,
                "device": source_device,
                "property": source_property,
                "value": mmcore.getPropertyLowerLimit(source_device, source_property),
            }
        ]
    )

    exposure_col = table.indexOf(channels.EXPOSURE)
    intensity_col = table.indexOf(channels.INTENSITY)
    active_exposure = table.cellWidget(0, exposure_col)
    active_intensity = table.cellWidget(0, intensity_col)
    other_exposure = table.cellWidget(1, exposure_col)
    assert isinstance(active_exposure, QDoubleSpinBox)
    assert isinstance(active_intensity, QDoubleSpinBox)
    assert isinstance(other_exposure, QDoubleSpinBox)

    mda._on_channel_row_selected(_row_index(table, 0))
    assert channels.activeRow() == 0

    # Idle: pressing Enter still follows the just-in-time rule -- no hardware
    # change until the next capture.
    active_exposure.setValue(77.0)
    active_exposure.editingFinished.emit()
    assert mmcore.getExposure() != pytest.approx(77.0)

    mmcore.startContinuousSequenceAcquisition()
    try:
        assert mmcore.isSequenceRunning()

        # Live: pressing Enter on the ACTIVE row's exposure applies immediately.
        active_exposure.setValue(123.0)
        active_exposure.editingFinished.emit()
        assert mmcore.getExposure() == pytest.approx(123.0)

        # Live: pressing Enter on the ACTIVE row's intensity applies immediately.
        new_intensity = mmcore.getPropertyUpperLimit(source_device, source_property)
        active_intensity.setValue(new_intensity)
        active_intensity.editingFinished.emit()
        assert float(
            mmcore.getProperty(source_device, source_property)
        ) == pytest.approx(new_intensity)

        # Live: editing a row that ISN'T active never moves hardware.
        other_exposure.setValue(999.0)
        other_exposure.editingFinished.emit()
        assert mmcore.getExposure() == pytest.approx(123.0)
    finally:
        mmcore.stopSequenceAcquisition()


def test_channel_property_selector_lists_all_runtime_numeric_sliders(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    mda = MemoryMDAWidget(mmcore)
    qtbot.addWidget(mda)
    channels = mda.channels

    expected = {
        (str(device), str(prop))
        for device, prop in mmcore.iterProperties(
            property_type=(PropertyType.Integer, PropertyType.Float),
            has_limits=True,
            is_read_only=False,
            as_object=False,
        )
        if not mmcore.isPropertyPreInit(device, prop)
    }
    choices = channels.lightSources()

    # every ranged property is offered as its own single-property source
    assert expected <= {pairs[0] for pairs in choices.values() if len(pairs) == 1}
    assert choices
    assert all(
        label == f"{pairs[0][0]} · {pairs[0][1]}"
        for label, pairs in choices.items()
        if len(pairs) == 1 and PROPERTY_SEPARATOR in label
    )
    assert channels.show_light_source.text() == "Show Light Source"

    table = channels.table()
    property_col = table.indexOf(channels._light_source_column)
    value_col = table.indexOf(channels.INTENSITY)
    property_header = table.horizontalHeaderItem(property_col)
    value_header = table.horizontalHeaderItem(value_col)
    assert property_header is not None
    assert value_header is not None
    assert property_header.text() == "Light Source"
    assert value_header.text() == "Intensity [%]"


# ------------------------- light source declarations -------------------------


def _load_cfg_with_light_sources(
    mmcore: CMMCorePlus,
    tmp_path: Path,
    declarations: Sequence[tuple[str, str, str, float]],
) -> Path:
    """Load a copy of the test cfg with ``(preset, device, property, value)`` lines."""
    dest = tmp_path / "declared.cfg"
    lines = Path(__file__).with_name("test_config.cfg").read_text().splitlines()
    lines += [
        f"#@LightSource,Channel,{preset},{device},{prop},{value}"
        for preset, device, prop, value in declarations
    ]
    dest.write_text("\n".join(lines) + "\n")
    mmcore.loadSystemConfiguration(dest)
    return dest


def _declared_mda(
    mmcore: CMMCorePlus, qtbot: QtBot, presets: Sequence[str]
) -> MemoryMDAWidget:
    """An MDA widget with one channel row per entry of ``presets``."""
    mda = MemoryMDAWidget(mmcore)
    qtbot.addWidget(mda)
    mda.setValue(
        useq.MDASequence(
            channels=tuple(
                useq.Channel(group="Channel", config=config, exposure=100.0)
                for config in presets
            )
        )
    )
    mda.channels.setLightSourceVisible(True)
    return mda


def _light_source_of(mda: MemoryMDAWidget, row: int) -> tuple[str, float]:
    """Return the (label, intensity) shown in ``row``'s light source columns."""
    channels = mda.channels
    table = channels.table()
    ls_col = table.indexOf(channels._light_source_column)
    int_col = table.indexOf(channels.INTENSITY)
    label = channels._light_source_column.get_cell_data(table, row, ls_col)
    intensity = channels.INTENSITY.get_cell_data(table, row, int_col)
    return str(label["light_source"]), float(intensity["intensity"])


def test_light_source_declarations_are_invisible_to_the_core(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    """The declarations are comments: loading a cfg must not act on them at all."""
    groups_before = set(mmcore.getAvailableConfigGroups())
    _load_cfg_with_light_sources(
        mmcore, tmp_path, [("DAPI", "Camera", "TestProperty1", 0.05)]
    )

    # no new config group, and the declared value was not pushed to hardware
    assert set(mmcore.getAvailableConfigGroups()) == groups_before
    assert float(mmcore.getProperty("Camera", "TestProperty1")) == pytest.approx(0.0)
    # and the channel group still identifies its presets
    mmcore.setConfig("Channel", "DAPI")
    assert mmcore.getCurrentConfig("Channel") == "DAPI"


def test_light_source_declaration_populates_channel_rows(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    _load_cfg_with_light_sources(
        mmcore, tmp_path, [("DAPI", "Camera", "TestProperty1", 0.05)]
    )
    mda = _declared_mda(mmcore, qtbot, ("DAPI", "FITC"))

    mda.refresh_channel_table()

    label, intensity = _light_source_of(mda, 0)
    assert mda.channels.lightSources()[label] == [("Camera", "TestProperty1")]
    assert intensity == pytest.approx(0.05)
    # FITC has no declaration, and nothing is inferred for it
    assert _light_source_of(mda, 1)[0] == ""

    # and the declaration reaches the acquisition-time properties
    props = mda.channels.channelProperties(exclude_unchecked=False)
    assert [(p["channel_index"], p["device"], p["property"]) for p in props] == [
        (0, "Camera", "TestProperty1")
    ]


def test_light_source_declaration_resolves_multi_property_group(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    """A declaration listing several properties restores as its group label.

    That is what makes one Intensity spin box drive every slider of a multi-channel
    illuminator, rather than the row landing on a single "Device · Property" entry.
    """
    pairs = [("Camera", "TestProperty2"), ("Camera", "TestProperty4")]
    _load_cfg_with_light_sources(
        mmcore, tmp_path, [("DAPI", device, prop, 12.0) for device, prop in pairs]
    )
    for device, prop in pairs:
        mmcore.defineConfig("_lida", "levels", device, prop, "0.0")
    mda = _declared_mda(mmcore, qtbot, ("DAPI",))

    mda.refresh_channel_table()

    label, intensity = _light_source_of(mda, 0)
    assert label == "_lida"
    assert mda.channels.lightSources()["_lida"] == pairs
    assert intensity == pytest.approx(12.0)
    # one row, but a property entry per underlying slider
    props = mda.channels.channelProperties(exclude_unchecked=False)
    assert [(p["device"], p["property"], p["value"]) for p in props] == [
        (device, prop, 12.0) for device, prop in pairs
    ]


def test_light_source_is_never_inferred_from_the_channel_preset(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """A ranged property inside a channel preset is not treated as a declaration.

    Storing the light source there would break preset identity, so it is only ever
    read from the cfg's comment block -- see LIGHT_SOURCE_COMMENT.
    """
    mmcore.defineConfig("Channel", "DAPI", "Camera", "TestProperty1", "0.05")
    mda = _declared_mda(mmcore, qtbot, ("DAPI",))

    mda.refresh_channel_table()

    assert _light_source_of(mda, 0)[0] == ""
    assert mda.channels.channelProperties(exclude_unchecked=False) == []


def test_stale_light_source_declaration_is_ignored(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    """A declaration naming a device this configuration doesn't have is dropped."""
    _load_cfg_with_light_sources(
        mmcore, tmp_path, [("DAPI", "NoSuchDevice", "Level", 5.0)]
    )
    mda = _declared_mda(mmcore, qtbot, ("DAPI",))

    mda.refresh_channel_table()

    assert _light_source_of(mda, 0)[0] == ""


def test_malformed_light_source_declarations_are_skipped(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    """The block is hand-editable, so junk must be ignored rather than raise."""
    dest = tmp_path / "declared.cfg"
    lines = Path(__file__).with_name("test_config.cfg").read_text().splitlines()
    lines += [
        "#@LightSource,Channel,DAPI,Camera,TestProperty1",  # too few fields
        "#@LightSource,Channel,FITC,Camera,TestProperty1,not-a-number",
        "#@LightSource,OtherGroup,Cy5,Camera,TestProperty1,0.05",  # other group
        "#@LightSource,Channel,Rhodamine,Camera,TestProperty1,0.05",  # the good one
    ]
    dest.write_text("\n".join(lines) + "\n")
    mmcore.loadSystemConfiguration(dest)
    mda = _declared_mda(mmcore, qtbot, ("DAPI", "FITC", "Cy5", "Rhodamine"))

    mda.refresh_channel_table()

    assert [_light_source_of(mda, row)[0] for row in range(3)] == ["", "", ""]
    assert _light_source_of(mda, 3)[1] == pytest.approx(0.05)


def test_changing_a_row_preset_follows_the_new_declaration(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    _load_cfg_with_light_sources(
        mmcore, tmp_path, [("DAPI", "Camera", "TestProperty1", 0.05)]
    )
    mda = _declared_mda(mmcore, qtbot, ("DAPI",))
    mda.refresh_channel_table()
    assert _light_source_of(mda, 0)[0] != ""

    table = mda.channels.table()
    config_col = table.indexOf(mda.channels._config_column)
    cell = table.cellWidget(0, config_col)
    assert cell is not None
    # the config cell is a CheckableCombo wrapper; the row's preset lives on the
    # QComboBox nested inside it, which is what _install_channel_editor_filters
    # connects to
    combo = cell.findChildren(QComboBox)[0]

    # switching to an undeclared preset clears the row's light source ...
    combo.setCurrentText("FITC")
    combo.activated.emit(combo.currentIndex())
    assert _light_source_of(mda, 0) == ("", 0.0)

    # ... and switching back restores it
    combo.setCurrentText("DAPI")
    combo.activated.emit(combo.currentIndex())
    label, intensity = _light_source_of(mda, 0)
    assert mda.channels.lightSources()[label] == [("Camera", "TestProperty1")]
    assert intensity == pytest.approx(0.05)


def test_saved_sequence_wins_over_light_source_declarations(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    """Restoring an MDASequence keeps its own channel properties."""
    _load_cfg_with_light_sources(
        mmcore, tmp_path, [("DAPI", "Camera", "TestProperty1", 0.05)]
    )
    mda = _declared_mda(mmcore, qtbot, ("DAPI",))
    channels = mda.channels
    label = next(
        lbl
        for lbl, pairs in channels.lightSources().items()
        if pairs == [("Camera", "TestProperty2")]
    )
    channels.setChannelProperties(
        [
            {
                "channel_index": 0,
                "config": "DAPI",
                "group": label,
                "device": "Camera",
                "property": "TestProperty2",
                "value": 33.0,
            }
        ]
    )

    mda.setValue(mda.value())

    restored_label, restored_intensity = _light_source_of(mda, 0)
    assert restored_label == label
    assert restored_intensity == pytest.approx(33.0)


def test_save_light_sources_writes_a_complete_cfg(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    mda = _declared_mda(mmcore, qtbot, ("DAPI", "FITC"))
    channels = mda.channels
    label = next(
        lbl
        for lbl, pairs in channels.lightSources().items()
        if pairs == [("Camera", "TestProperty1")]
    )
    channels.setChannelProperties(
        [
            {
                "channel_index": 0,
                "config": "DAPI",
                "group": label,
                "device": "Camera",
                "property": "TestProperty1",
                "value": 0.05,
            }
        ]
    )

    dest = tmp_path / "with_light_sources.cfg"
    with patch.object(QFileDialog, "getSaveFileName", return_value=(str(dest), "")):
        mda._save_light_sources_btn.click()

    lines = dest.read_text().splitlines()
    assert "#@LightSource,Channel,DAPI,Camera,TestProperty1,0.05" in lines
    # the whole configuration is written, not just the light sources
    assert any(line.startswith("Device,") for line in lines)
    assert any(line.startswith("PixelSize_um,") for line in lines)
    # FITC had no light source, so it gets no declaration
    assert not any(line.startswith("#@LightSource,Channel,FITC,") for line in lines)

    # nothing was written as real configuration: no new group, and the channel
    # preset itself is untouched, so preset identity still works
    assert not any(line.startswith("ConfigGroup,Channel-LightSource") for line in lines)
    assert not any(
        line.startswith("ConfigGroup,Channel,DAPI,Camera,TestProperty1")
        for line in lines
    )
    mmcore.setConfig("Channel", "DAPI")
    assert mmcore.getCurrentConfig("Channel") == "DAPI"


def test_saved_light_sources_round_trip_through_a_reloaded_cfg(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    mda = _declared_mda(mmcore, qtbot, ("DAPI",))
    channels = mda.channels
    label = next(
        lbl
        for lbl, pairs in channels.lightSources().items()
        if pairs == [("Camera", "TestProperty1")]
    )
    channels.setChannelProperties(
        [
            {
                "channel_index": 0,
                "config": "DAPI",
                "group": label,
                "device": "Camera",
                "property": "TestProperty1",
                "value": 0.05,
            }
        ]
    )

    dest = tmp_path / "round_trip.cfg"
    with patch.object(QFileDialog, "getSaveFileName", return_value=(str(dest), "")):
        mda._save_light_sources_btn.click()

    # a fresh widget over the reloaded file picks the declaration back up
    mmcore.loadSystemConfiguration(dest)
    reloaded = _declared_mda(mmcore, qtbot, ("DAPI",))
    reloaded.refresh_channel_table()

    new_label, intensity = _light_source_of(reloaded, 0)
    assert reloaded.channels.lightSources()[new_label] == [("Camera", "TestProperty1")]
    assert intensity == pytest.approx(0.05)


def test_save_light_sources_warns_when_the_feature_is_off(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    """Saving with light sources hidden would silently erase every declaration."""
    _load_cfg_with_light_sources(
        mmcore, tmp_path, [("DAPI", "Camera", "TestProperty1", 0.05)]
    )
    mda = _declared_mda(mmcore, qtbot, ("DAPI",))
    mda.channels.setLightSourceVisible(False)

    with (
        patch.object(QMessageBox, "warning") as warning,
        patch.object(QFileDialog, "getSaveFileName") as dialog,
    ):
        mda._save_light_sources_btn.click()

    assert warning.called
    assert not dialog.called
    # the loaded cfg's declarations are still intact
    assert mda._light_source_declarations == {
        "DAPI": [("Camera", "TestProperty1", 0.05)]
    }


def test_memory_mda_hides_estimated_duration(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    mda = MemoryMDAWidget(mmcore)
    qtbot.addWidget(mda)
    mda.show()

    assert mda._duration_label.isHidden()
    assert mda._time_warning.isHidden()

    mda.setValue(
        useq.MDASequence(
            time_plan=useq.TIntervalLoops(interval=timedelta(milliseconds=1), loops=3),
            channels=(useq.Channel(config="DAPI", exposure=100),),
        )
    )
    QApplication.processEvents()
    assert mda._duration_label.isHidden()
    assert mda._time_warning.isHidden()


def test_collapsible_mda_round_trips_all_original_widgets(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    set_theme(DARK_THEME)
    mmcore.setAutoFocusDevice("Autofocus")
    mda = MemoryMDAWidget(mmcore)
    qtbot.addWidget(mda)
    tabs = mda._collapsible_tabs()
    reference = UpstreamMDAWidget(mmcore=mmcore)
    qtbot.addWidget(reference)
    z_plan = useq.ZRangeAround(range=4, step=1)
    time_plan = useq.TIntervalLoops(interval=timedelta(seconds=2), loops=3)
    grid_plan = useq.GridRowsColumns(rows=2, columns=3)
    positions = (
        useq.AbsolutePosition(
            x=10,
            y=20,
            z=3,
            name="Site 1",
            sequence=useq.MDASequence(
                channels=(useq.Channel(config="Cy5", exposure=9),)
            ),
        ),
    )
    channels = (
        useq.Channel(
            group="Channel",
            config="DAPI",
            exposure=12.5,
            acquire_every=2,
            do_stack=False,
            z_offset=1.5,
        ),
        useq.Channel(group="Channel", config="FITC", exposure=27),
    )
    source = useq.MDASequence(
        axis_order=tuple("tpgzc"),
        channels=channels,
        stage_positions=positions,
        grid_plan=grid_plan,
        z_plan=z_plan,
        time_plan=time_plan,
        autofocus_plan=useq.AxesBasedAF(axes=("p", "t")),
        keep_shutter_open_across=("z", "t"),
        metadata={
            "pymmcore_widgets": {
                "save_dir": "/tmp",
                "save_name": "roundtrip.ome.tif",
                "format": "ome-tiff",
                "should_save": True,
            }
        },
    )

    reference.setValue(source)
    mda.setValue(source)
    reference_result = reference.value()
    result = mda.value()

    assert result.replace(metadata=reference_result.metadata) == reference_result
    assert result.metadata["pymmcore_widgets"]["camera_roi"]["enabled"] is False
    assert result.channels == channels
    assert result.stage_positions == positions
    assert isinstance(result.grid_plan, useq.GridRowsColumns)
    assert result.grid_plan.rows == grid_plan.rows
    assert result.grid_plan.columns == grid_plan.columns
    assert result.z_plan == z_plan
    assert result.time_plan == time_plan
    assert result.autofocus_plan is not None
    assert result.autofocus_plan.axes == ("p", "t")
    assert result.keep_shutter_open_across == ("z", "t")
    assert all(tabs.isChecked(axis) for axis in "cpgzt")
    assert tabs.saving_section.checked
    assert tabs.saving_section is tabs.sections[-2]
    assert mda.save_info.save_name.text() == "roundtrip.ome.tif"

    channel_table = mda.channels.table()
    assert not channel_table.isColumnHidden(
        channel_table.indexOf(mda.channels.ACQUIRE_EVERY)
    )
    # Do Stack is gated behind the upstream "advanced" toggle (and the Z axis,
    # which the round-tripped sequence above already activates).
    mda.channels.advanced.setChecked(True)
    assert not channel_table.isColumnHidden(
        channel_table.indexOf(mda.channels.DO_STACK)
    )

    source_group = "_slider_test"
    source_device, source_property = "Camera", "TestProperty1"
    intensity = mmcore.getPropertyUpperLimit(source_device, source_property)
    mda.channels.setLightSourceVisible(True)
    mda.channels.setChannelProperties(
        [
            {
                "channel_index": 0,
                "config": "DAPI",
                "group": source_group,
                "device": source_device,
                "property": source_property,
                "value": intensity,
            }
        ]
    )
    with_properties = mda.value()
    mda.channels.setLightSourceVisible(False)
    mda.setValue(with_properties)
    assert mda.channels.lightSourceVisible()
    restored_property = mda.channels.channelProperties()[0]
    # "_slider_test" is a single-preset group wrapping the same property that is
    # also offered as "Camera · TestProperty1"; a round trip must preserve
    # whichever of the two the sequence was saved under.
    assert restored_property["group"] == source_group
    assert restored_property["device"] == source_device
    assert restored_property["property"] == source_property
    assert restored_property["value"] == pytest.approx(intensity)

    tabs.setChecked("z", False)
    assert result.z_plan == z_plan
    assert mda.value().z_plan is None
    assert channel_table.isColumnHidden(channel_table.indexOf(mda.channels.DO_STACK))


def test_collapsible_mda_preserves_per_position_af_offsets(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Per-position autofocus offsets survive the sectioned Positions editor.

    The offset lives on each position's sub-sequence ``autofocus_plan`` and is
    edited through the upstream table's AF column + "Set AF Offset per Position"
    toggle. This guards that parity once the legacy widget is removed (Phase 6).
    """
    set_theme(DARK_THEME)
    mmcore.setAutoFocusDevice("Autofocus")
    mda = MemoryMDAWidget(mmcore)
    qtbot.addWidget(mda)

    positions = (
        useq.AbsolutePosition(
            x=1,
            y=2,
            z=3,
            name="P1",
            sequence=useq.MDASequence(
                autofocus_plan=useq.AxesBasedAF(
                    autofocus_motor_offset=42.0, axes=("p",)
                )
            ),
        ),
        useq.AbsolutePosition(
            x=5,
            y=6,
            z=7,
            name="P2",
            sequence=useq.MDASequence(
                autofocus_plan=useq.AxesBasedAF(
                    autofocus_motor_offset=-13.0, axes=("p",)
                )
            ),
        ),
    )
    mda.setValue(useq.MDASequence(stage_positions=positions))

    assert mda.stage_positions.af_per_position.isChecked()
    restored = mda.value().stage_positions
    assert restored == positions
    seq0 = restored[0].sequence
    assert seq0 is not None and seq0.autofocus_plan is not None
    assert seq0.autofocus_plan.autofocus_motor_offset == 42.0
    seq1 = restored[1].sequence
    assert seq1 is not None and seq1.autofocus_plan is not None
    assert seq1.autofocus_plan.autofocus_motor_offset == -13.0


def test_collapsible_mda_retains_saving_and_execution_controls(
    mmcore: CMMCorePlus,
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    set_theme(DARK_THEME)
    mda = MemoryMDAWidget(mmcore)
    qtbot.addWidget(mda)
    tabs = mda._collapsible_tabs()

    requested = tmp_path / "experiment.ome.tif"
    requested.touch()
    mda.save_info.setValue(
        {
            "save_dir": str(tmp_path),
            "save_name": requested.name,
            "format": "ome-tiff",
            "should_save": True,
        }
    )
    assert tabs.saving_section.checked
    assert tabs.saving_section is tabs.sections[-2]
    assert mda.prepare_mda() == tmp_path / "experiment_001.ome.tif"

    sequence = mda.value()
    for suffix in (".yaml", ".json"):
        settings_path = tmp_path / f"mda-settings{suffix}"
        mda.save(settings_path)
        restored = MemoryMDAWidget(mmcore)
        qtbot.addWidget(restored)
        restored.load(settings_path)
        assert restored.value().channels == sequence.channels
        assert restored.save_info.value() == mda.save_info.value()

    channel_checkbox = tabs.section("c").checkbox
    assert channel_checkbox is not None
    mmcore.mda.events.sequenceStarted.emit(sequence, {})
    assert not channel_checkbox.isEnabled()
    assert not mda.channels.isEnabled()
    assert mda.control_btns.cancel_btn.isEnabled()
    assert mda.control_btns.run_btn.isHidden()
    assert not mda.control_btns.pause_btn.isHidden()
    assert not mda.control_btns.cancel_btn.isHidden()

    mmcore.mda.events.sequenceFinished.emit(sequence)
    assert channel_checkbox.isEnabled()
    assert mda.channels.isEnabled()
    assert not mda.control_btns.run_btn.isHidden()
    assert mda.control_btns.pause_btn.isHidden()
    assert mda.control_btns.cancel_btn.isHidden()


def test_collapsible_mda_runs_disk_backed_acquisition(
    mmcore: CMMCorePlus,
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    set_theme(DARK_THEME)
    mda = MemoryMDAWidget(mmcore)
    qtbot.addWidget(mda)
    mda.setValue(
        useq.MDASequence(
            channels=(useq.Channel(group="Channel", config="DAPI", exposure=1),)
        )
    )
    destination = tmp_path / "acquisition.ome.tif"
    mda.save_info.setValue(destination)

    with qtbot.waitSignal(mmcore.mda.events.sequenceFinished, timeout=10_000):
        mda.run_mda()

    assert destination.is_file()

    mda.save_info.setChecked(False)
    with qtbot.waitSignal(mmcore.mda.events.sequenceFinished, timeout=10_000):
        mda.run_mda()
    assert mda.control_btns.run_btn.isEnabled()


def test_collapsible_mda_shows_store_creation_progress(
    mmcore: CMMCorePlus,
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disk-backed runs show feedback until the MDA startup signal arrives."""
    mda = MemoryMDAWidget(mmcore)
    qtbot.addWidget(mda)
    mda.show()
    mda.save_info.setValue(tmp_path / "acquisition.ome.tif")

    startup_observed = False

    def fake_execute(_output: object) -> None:
        nonlocal startup_observed
        startup_observed = mda._store_overlay.isVisible()

    monkeypatch.setattr(mda, "execute_mda", fake_execute)
    mda.run_mda()

    assert startup_observed
    assert mda._store_overlay.isVisible()
    assert mda._store_overlay._message == "Creating data store…"

    mmcore.mda.events.sequenceStarted.emit(mda.value(), {})
    qtbot.waitUntil(mda._store_overlay.isHidden)

    # Memory-backed runs are fast and should not flash a store message.
    mda.save_info.setChecked(False)
    mda.run_mda()
    assert mda._store_overlay.isHidden()


def test_collapsible_mda_disables_every_editor_during_acquisition(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """All sections — axes, Saving, Settings — lock during a run and unlock after.

    Extends the channels-only assertion: ``set_editor_enabled`` must sweep every
    axis section, the Settings body, and the Saving section so nothing stays
    editable while an acquisition is in flight.
    """
    set_theme(DARK_THEME)
    mda = MemoryMDAWidget(mmcore)
    qtbot.addWidget(mda)
    tabs = mda._collapsible_tabs()

    # Turn every optional axis on so their editors are expected to be enabled
    # while idle, and therefore disabled during the run.
    for axis in "pgzt":
        tabs.setChecked(axis, True)
    tabs.saving_section.set_checked(True)

    axis_widgets = {
        "c": mda.channels,
        "p": mda.stage_positions,
        "g": mda.grid_plan,
        "z": mda.z_plan,
        "t": mda.time_plan,
    }
    sequence = mda.value()

    mmcore.mda.events.sequenceStarted.emit(sequence, {})
    for axis, widget in axis_widgets.items():
        section = tabs.section(axis)
        assert section.checkbox is not None
        assert not section.checkbox.isEnabled(), f"{axis} checkbox stayed enabled"
        assert not widget.isEnabled(), f"{axis} editor stayed enabled"
    assert not tabs.settings_section._body.isEnabled()
    assert tabs.saving_section.checkbox is not None
    assert not tabs.saving_section.checkbox.isEnabled()
    assert not mda.save_info.isEnabled()
    # Footer: only Pause/Cancel are actionable while running.
    assert mda.control_btns.run_btn.isHidden()
    assert mda.control_btns.pause_btn.isEnabled()
    assert mda.control_btns.cancel_btn.isEnabled()

    mmcore.mda.events.sequenceFinished.emit(sequence)
    for axis, widget in axis_widgets.items():
        section = tabs.section(axis)
        assert section.checkbox is not None
        assert section.checkbox.isEnabled(), f"{axis} checkbox stayed disabled"
        assert widget.isEnabled(), f"{axis} editor stayed disabled"
    assert tabs.settings_section._body.isEnabled()
    assert tabs.saving_section.checkbox.isEnabled()
    assert mda.save_info.isEnabled()
    assert not mda.control_btns.run_btn.isHidden()
    assert mda.control_btns.cancel_btn.isHidden()


def test_live_opens_preview_before_streaming(
    mmcore: CMMCorePlus,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePreview(QWidget):
        def __init__(
            self,
            mmcore: CMMCorePlus,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self._core = mmcore
            self.streaming_started = 0
            mmcore.events.continuousSequenceAcquisitionStarted.connect(
                self._on_streaming_started
            )

        def _on_streaming_started(self) -> None:
            self.streaming_started += 1

        def detach(self) -> None:
            self._core.events.continuousSequenceAcquisitionStarted.disconnect(
                self._on_streaming_started
            )

    monkeypatch.setattr(acquire_viewers_module, "NDVPreview", FakePreview)
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    assert page._viewers.preview is None

    page._live_btn.click()
    preview = page._viewers.preview
    assert isinstance(preview, FakePreview)
    assert page._viewers._preview_dock is not None
    assert page._viewers._preview_dock.windowTitle() == "Preview"
    assert preview.streaming_started == 1
    assert mmcore.isSequenceRunning()

    page._live_btn.click()
    assert not mmcore.isSequenceRunning()


def test_snap_and_live_apply_the_active_channel_capture_settings(
    mmcore: CMMCorePlus,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePreview(QWidget):
        def __init__(
            self,
            mmcore: CMMCorePlus,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)

        def detach(self) -> None:
            pass

    def run_worker_now(func: Callable[[], None], **_: object) -> None:
        func()

    monkeypatch.setattr(acquire_viewers_module, "NDVPreview", FakePreview)
    monkeypatch.setattr(acquire_toolbar_module, "create_worker", run_worker_now)

    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    channels = page._mda.channels
    config = tuple(mmcore.getAvailableConfigs("Channel"))[-1]
    exposure = 23.5
    page._mda.setValue(
        useq.MDASequence(
            channels=(useq.Channel(group="Channel", config=config, exposure=exposure),)
        )
    )

    table = channels.table()
    table.indexOf(channels._config_column)
    channels.setActiveRow(0)
    source_device, source_property = "Camera", "TestProperty1"
    source_group = next(
        label
        for label, pairs in channels.lightSources().items()
        if pairs == [(source_device, source_property)]
    )
    intensity = mmcore.getPropertyUpperLimit(source_device, source_property)
    channels.setLightSourceVisible(True)
    channels.setChannelProperties(
        [
            {
                "channel_index": 0,
                "config": config,
                "group": source_group,
                "device": source_device,
                "property": source_property,
                "value": intensity,
            }
        ]
    )

    first_config = next(iter(mmcore.getAvailableConfigs("Channel")))
    baseline_intensity = mmcore.getPropertyLowerLimit(source_device, source_property)

    def reset_core() -> None:
        # Drift the core to a preset absent from the single-row table; this now
        # clears the ● indicator (core->table reverse sync). Re-activate the
        # row afterwards, as a user would, so the "active channel" is restored.
        # Idle activation re-applies only the preset, not exposure/intensity --
        # those must be re-applied at capture time, which is what this verifies.
        mmcore.setConfig("Channel", first_config)
        mmcore.setExposure(5)
        mmcore.setProperty(source_device, source_property, baseline_intensity)
        page._mda._on_channel_row_selected(_row_index(table, 0))

    def capture_settings() -> tuple[str, float, float]:
        return (
            mmcore.getCurrentConfig("Channel"),
            mmcore.getExposure(),
            float(mmcore.getProperty(source_device, source_property)),
        )

    snapped: list[tuple[str, float, float]] = []
    mmcore.events.imageSnapped.connect(lambda: snapped.append(capture_settings()))
    reset_core()
    page._snap_btn.click()
    assert snapped[0][0] == config
    assert snapped[0][1:] == pytest.approx((exposure, intensity))
    captured = capture_settings()
    assert captured[0] == config
    assert captured[1:] == pytest.approx((exposure, intensity))

    started: list[tuple[str, float, float]] = []
    mmcore.events.continuousSequenceAcquisitionStarted.connect(
        lambda: started.append(capture_settings())
    )
    reset_core()
    page._live_btn.click()
    assert started[0][0] == config
    assert started[0][1:] == pytest.approx((exposure, intensity))
    captured = capture_settings()
    assert captured[0] == config
    assert captured[1:] == pytest.approx((exposure, intensity))
    page._live_btn.click()
    assert not mmcore.isSequenceRunning()


def test_switching_channel_rows_during_live_applies_all_capture_settings(
    mmcore: CMMCorePlus,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePreview(QWidget):
        def __init__(
            self,
            mmcore: CMMCorePlus,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)

        def detach(self) -> None:
            pass

    monkeypatch.setattr(acquire_viewers_module, "NDVPreview", FakePreview)

    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    channels = page._mda.channels
    configs = tuple(mmcore.getAvailableConfigs("Channel"))
    selected_configs = (configs[0], configs[-1])
    exposures = (11.5, 37.5)
    page._mda.setValue(
        useq.MDASequence(
            channels=tuple(
                useq.Channel(group="Channel", config=config, exposure=exposure)
                for config, exposure in zip(selected_configs, exposures, strict=True)
            )
        )
    )

    source_device, source_property = "Camera", "TestProperty1"
    source_group = next(
        label
        for label, pairs in channels.lightSources().items()
        if pairs == [(source_device, source_property)]
    )
    property_values = (
        mmcore.getPropertyLowerLimit(source_device, source_property),
        mmcore.getPropertyUpperLimit(source_device, source_property),
    )
    channels.setLightSourceVisible(True)
    channels.setChannelProperties(
        [
            {
                "channel_index": row,
                "config": config,
                "group": source_group,
                "device": source_device,
                "property": source_property,
                "value": value,
            }
            for row, (config, value) in enumerate(
                zip(selected_configs, property_values, strict=True)
            )
        ]
    )

    def capture_settings() -> tuple[str, float, float]:
        return (
            mmcore.getCurrentConfig("Channel"),
            mmcore.getExposure(),
            float(mmcore.getProperty(source_device, source_property)),
        )

    table = channels.table()
    table.indexOf(channels._config_column)
    # Activate row 0 on the microscope before starting live.
    page._mda._on_channel_row_selected(_row_index(table, 0))
    mmcore.setExposure(5)
    mmcore.setProperty(source_device, source_property, property_values[1])

    live_started: list[None] = []
    live_stopped: list[None] = []
    mmcore.events.continuousSequenceAcquisitionStarted.connect(
        lambda *_: live_started.append(None)
    )
    mmcore.events.sequenceAcquisitionStopped.connect(
        lambda *_: live_stopped.append(None)
    )

    page._live_btn.click()
    try:
        assert mmcore.isSequenceRunning()
        assert len(live_started) == 1
        assert not live_stopped
        current = capture_settings()
        assert current[0] == selected_configs[0]
        assert current[1:] == pytest.approx((exposures[0], property_values[0]))

        # Switch the active channel during live by clicking the ● column.
        page._mda._on_channel_row_selected(_row_index(table, 1))
        QApplication.processEvents()

        assert channels.activeRow() == 1
        assert mmcore.isSequenceRunning()
        assert len(live_started) == 1
        assert not live_stopped
        current = capture_settings()
        assert current[0] == selected_configs[1]
        assert current[1:] == pytest.approx((exposures[1], property_values[1]))
    finally:
        if mmcore.isSequenceRunning():
            mmcore.stopSequenceAcquisition()


def test_acquire_page_adds_sink_backed_mda_tab(
    mmcore: CMMCorePlus,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Emitter:
        def emit(self) -> None:
            pass

    class FakeViewer:
        def __init__(self, data: object, /, **kwargs: object) -> None:
            self.data = data
            self.kwargs = kwargs
            self.display_model = SimpleNamespace(current_index={})
            self.data_wrapper = SimpleNamespace(
                dims_changed=Emitter(), data_changed=Emitter()
            )
            self._widget = QWidget()
            self.closed = False

        def widget(self) -> QWidget:
            return self._widget

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(acquire_viewers_module, "MMArrayViewer", FakeViewer)
    set_theme(DARK_THEME)
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    mmcore.mda.run(
        useq.MDASequence(channels=(useq.Channel(config="DAPI", exposure=10),)),
        output="memory",
    )
    qtbot.wait(20)

    assert page._viewers._active_dock is not None
    assert page._viewers._active_dock.windowTitle().startswith("MDA ")
    viewer = page._viewers.active_viewer
    assert isinstance(viewer, FakeViewer)
    assert viewer.data is not None

    dock = page._viewers._active_dock
    dock.closeDockWidget()
    assert page._viewers.active_viewer is None
    assert viewer.closed


def test_acquire_viewer_close_reclaims_space_without_moving_mda(
    mmcore: CMMCorePlus,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Viewer relayout stays inside the nested manager and reclaims dead space."""

    class Emitter:
        def emit(self) -> None:
            pass

    class FakeViewer:
        def __init__(self, data: object, /, **kwargs: object) -> None:
            self.data = data
            self.display_model = SimpleNamespace(current_index={})
            self.data_wrapper = SimpleNamespace(
                dims_changed=Emitter(), data_changed=Emitter()
            )
            self._widget = QWidget()
            self.closed = False

        def widget(self) -> QWidget:
            return self._widget

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(acquire_viewers_module, "MMArrayViewer", FakeViewer)
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page.resize(1400, 900)
    page.show()
    qtbot.waitExposed(page)
    qtbot.waitUntil(lambda: page._mda_width_locked_at_real_size, timeout=2000)
    page.panel_button(PanelKey.PRESETS).setChecked(True)  # default-open already

    mmcore.mda.run(
        useq.MDASequence(channels=(useq.Channel(config="DAPI", exposure=10),)),
        output="memory",
    )
    qtbot.wait(20)

    dock = page._viewers._active_dock
    assert dock is not None
    viewer = page._viewers.active_viewer
    assert isinstance(viewer, FakeViewer)
    DF = CDockWidget.DockWidgetFeature
    assert dock.features().value & DF.DockWidgetDeleteOnClose.value

    mda_area = page._mda_dock.dockAreaWidget()
    right_area = page._right_dock_area
    assert mda_area is not None and right_area is not None
    mda_width_before = mda_area.width()
    right_width_before = right_area.width()

    # Split another viewer beside the active one inside the nested manager.
    # Emptying that split area must not affect either outer tools column.
    second = CDockWidget(page._viewer_dock_manager, "Viewer2")
    second.setWidget(QWidget())
    second.setFeature(DF.DockWidgetDeleteOnClose, True)
    page._viewer_dock_manager.addDockWidget(
        DockWidgetArea.RightDockWidgetArea, second, dock.dockAreaWidget()
    )
    qtbot.wait(20)
    second.closeDockWidget()
    qtbot.wait(20)

    assert page._viewers.active_viewer is viewer
    assert not viewer.closed
    assert mda_area.width() == mda_width_before
    assert right_area.width() == right_width_before

    # Closing one side of an inner split should let its sibling reclaim the
    # space, and a later viewer should tab into that remaining area.
    third = CDockWidget(page._viewer_dock_manager, "Viewer3")
    third.setWidget(QWidget())
    third.setFeature(DF.DockWidgetDeleteOnClose, True)
    page._viewer_dock_manager.addDockWidget(
        DockWidgetArea.RightDockWidgetArea, third, dock.dockAreaWidget()
    )
    qtbot.wait(20)
    third_area = third.dockAreaWidget()
    assert third_area is not None
    third_width_before = third_area.width()

    dock.closeDockWidget()
    qtbot.wait(20)

    assert page._viewers.active_viewer is None
    assert viewer.closed
    assert third_area.width() > third_width_before * 1.5
    assert mda_area.width() == mda_width_before
    assert right_area.width() == right_width_before

    fourth = page._viewers._new_dock("Viewer4")
    fourth.setWidget(QWidget())
    assert fourth.dockAreaWidget() is third_area


def test_acquire_new_viewer_split_is_equal(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    """Dragging one of two viewer tabs aside creates an equal 50/50 split."""
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page.resize(1200, 800)
    page.show()
    qtbot.waitExposed(page)

    first = page._viewers._new_dock("Viewer1")
    first.setWidget(QWidget())
    second = page._viewers._new_dock("Viewer2")
    second.setWidget(QWidget())
    shared_area = first.dockAreaWidget()
    assert shared_area is not None
    assert second.dockAreaWidget() is shared_area

    # Disable ADS's programmatic-insertion helper so this specifically verifies
    # our relocation fallback instead of merely retesting the global flag.
    flag = CDockManager.eConfigFlag.EqualSplitOnInsertion
    CDockManager.setConfigFlag(flag, False)
    try:
        page._viewer_dock_manager.addDockWidget(
            DockWidgetArea.RightDockWidgetArea, second, shared_area
        )
        qtbot.wait(20)
    finally:
        CDockManager.setConfigFlag(flag, True)

    second_area = second.dockAreaWidget()
    assert second_area is not None and second_area is not shared_area
    splitter = second_area.parentWidget()
    assert isinstance(splitter, QSplitter)
    first_size, second_size = splitter.sizes()
    assert abs(first_size - second_size) <= 1


def test_acquire_viewer_records_frame_metadata_regardless_of_follow_lock(
    mmcore: CMMCorePlus,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-frame metadata capture must not be gated by the follow/lock toggle.

    ``AcquireViewersManager._on_frame_ready`` used to check ``_follow_acquisition``
    before doing anything at all; the metadata now appended to the viewer's
    ``AcquisitionRecord`` must happen regardless, or locking the slider (e.g. to
    inspect an earlier timepoint mid-run) would silently truncate whatever gets
    exported later from the viewer's Save button.
    """
    from pymmcore_plus.metadata import frame_metadata

    class Emitter:
        def emit(self) -> None:
            pass

    class FakeViewer:
        def __init__(self, data: object, /, **kwargs: object) -> None:
            self.data = data
            self.display_model = SimpleNamespace(current_index={})
            self.data_wrapper = SimpleNamespace(
                dims_changed=Emitter(), data_changed=Emitter()
            )
            self._widget = QWidget()

        def widget(self) -> QWidget:
            return self._widget

        def close(self) -> None:
            pass

    monkeypatch.setattr(acquire_viewers_module, "MMArrayViewer", FakeViewer)
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    sequence = useq.MDASequence(channels=(useq.Channel(config="DAPI", exposure=10),))
    mmcore.mda.run(sequence, output="memory")
    qtbot.wait(20)

    manager = page._viewers
    dock = manager._active_dock
    assert dock is not None
    record = manager._records[dock]
    assert record.acquisition is not None
    assert record.acquisition.settings.dtype
    assert record.acquisition.summary_meta is not None
    n_before = len(record.acquisition.frame_meta)
    assert n_before >= 1  # the one real frame from the run above

    frame = np.zeros((4, 4), dtype="uint16")
    event = next(iter(sequence))
    meta = frame_metadata(mmcore, runner_time_ms=1.0)

    manager._follow_acquisition = False
    manager._on_frame_ready(frame, event, meta)
    assert len(record.acquisition.frame_meta) == n_before + 1

    manager._follow_acquisition = True
    manager._on_frame_ready(frame, event, meta)
    assert len(record.acquisition.frame_meta) == n_before + 2


def _resized_splitter_sizes(
    page: AcquirePage, area: CDockAreaWidget, width: int
) -> list[int]:
    """Current splitter sizes for *area*'s splitter with *area*'s slot set to *width*.

    ``setSplitterSizes`` needs one entry per *current* child of that specific
    splitter (2 columns if only MDA/central exist, 3 once the right column is
    open too) -- built from the live sizes rather than assumed, so this stays
    correct regardless of how many columns are open.
    """
    sizes = list(page._dock_manager.splitterSizes(area))
    splitter = area.parentWidget()
    assert isinstance(splitter, QSplitter)
    idx = splitter.indexOf(area)
    freed = sizes[idx] - width
    sizes[idx] = width
    # Give the freed/needed space to another column so the total is unchanged.
    other = 0 if idx != 0 else 1
    sizes[other] += freed
    return sizes


def _park_real_cursor_away() -> None:
    """Move the real global cursor off of any splitter handle after a drag.

    On a real platform (confirmed inert under the offscreen test platform,
    which is exactly why this can't be verified locally), ``QTest`` mouse
    simulation warps the real system cursor, not just synthetic per-widget
    events -- so without this, wherever a drag left the real cursor can
    bleed into whatever ``AcquirePage`` gets constructed next. Two pages
    built with the same default geometry, as the restore-path tests below
    do, can easily place a handle at the exact screen position a previous
    one's drag left the real cursor at; ``_update_width_handle_hover``'s
    poll would then read that as a live hover on a column nobody is
    actually touching, unlock it, and never see anything move the
    (nonexistent) real pointer back off again to re-lock it.
    """
    QCursor.setPos(QPoint(-10_000, -10_000))


def _assert_column_resists_relayout_but_stays_draggable(
    page: AcquirePage, qtbot: QtBot, area: CDockAreaWidget, new_width: int
) -> None:
    """Shared body for the MDA / right-column width-lock regression tests below."""
    starting_width = area.width()
    assert area.minimumWidth() == area.maximumWidth() == starting_width

    # A direct, deliberate attempt to resize it away from the locked width
    # must be clamped straight back -- exercising the same splitter API ADS
    # itself uses internally when it recomputes proportions.
    page._dock_manager.setSplitterSizes(area, _resized_splitter_sizes(page, area, 50))
    assert area.width() == starting_width

    # A real drag: press the handle (unlocks), resize (as the splitter would
    # live, mid-drag), release (re-locks at wherever the user left it).
    handle = next(h for h, a in page._width_locked_areas.items() if a is area)
    assert isinstance(handle, QWidget)
    # A fixed-width neighbor can prevent the real platform from targeting the
    # handle at all. The cursor poll must unlock while nearby, then re-lock at
    # the current width after the pointer leaves.
    with patch("pymmcore_gui._modern_gui._acquire.QCursor") as cursor_cls:
        cursor_cls.pos.return_value = handle.mapToGlobal(handle.rect().center())
        page._update_width_handle_hover()
        assert area.minimumWidth() == 0
        cursor_cls.pos.return_value = QPoint(-10_000, -10_000)
        page._update_width_handle_hover()
        assert area.minimumWidth() == area.maximumWidth() == starting_width

    splitter = area.parentWidget()
    assert isinstance(splitter, QSplitter)
    area_idx = splitter.indexOf(area)
    delta = new_width - starting_width
    if area_idx > 0:
        delta = -delta
    start = handle.rect().center()
    end = start + QPoint(delta, 0)

    qtbot.mousePress(  # type: ignore[no-untyped-call]
        handle, Qt.MouseButton.LeftButton, pos=start
    )
    assert area.minimumWidth() == 0
    qtbot.mouseMove(handle, pos=end)  # type: ignore[no-untyped-call]
    assert area.width() == new_width
    qtbot.mouseRelease(  # type: ignore[no-untyped-call]
        handle, Qt.MouseButton.LeftButton, pos=end
    )
    _park_real_cursor_away()
    assert area.minimumWidth() == area.maximumWidth() == new_width
    assert area.width() == new_width

    # The new, user-chosen width is itself just as resistant to relayout.
    page._dock_manager.setSplitterSizes(area, _resized_splitter_sizes(page, area, 50))
    assert area.width() == new_width


def test_acquire_mda_dock_width_resists_relayout_but_stays_draggable(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """The MDA dock area's width resists relayout, but a real drag can change it.

    Viewer relayout is now isolated in the nested viewer manager, but changes
    to areas in the outer MDA/tools manager can still recompute that manager's
    splitter proportions. A *reactive* fix (re-applying
    ``setSplitterSizes`` after the fact, even across several deferred
    event-loop turns) proved unreliable in practice: it's a race against
    ADS's own relayout passes, which can be arbitrarily delayed. The actual
    fix (``AcquirePage._install_width_lock``) instead keeps
    ``minimumWidth == maximumWidth`` on the locked dock area at all times --
    a hard Qt layout constraint a splitter must respect in *any* layout
    pass, regardless of what triggers it or when -- except for the exact
    duration of a real mouse drag on the handle to its right, caught via
    ``eventFilter`` (mouse press unlocks; release re-locks at the new
    width). This verifies both halves directly: an explicit, deliberate attempt
    to resize the splitter away from the locked width has no effect while
    locked, and a simulated press-drag-release on the handle both unlocks it
    for the drag and re-locks it at the resulting (user-chosen) width
    afterward.
    """
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page.resize(1400, 900)
    page.show()
    qtbot.waitExposed(page)
    # AcquirePage is constructed standalone here, before it has real window
    # geometry, so its one-time initial pin/lock (in __init__) captured a too
    # small width; showEvent schedules a debounced re-pin (see
    # ``_schedule_width_settle``) that fires once geometry stops changing,
    # same as it would for the real app's startup. Wait for it to settle.
    qtbot.waitUntil(lambda: page._mda_width_locked_at_real_size, timeout=2000)

    mda_area = page._mda_dock.dockAreaWidget()
    assert mda_area is not None
    assert mda_area.width() == _MDA_DOCK_WIDTH
    _assert_column_resists_relayout_but_stays_draggable(page, qtbot, mda_area, 500)


def test_acquire_right_dock_is_always_resizable(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """The right sidebar never depends on hover detection to become resizable."""
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page.resize(1400, 900)
    page.show()
    qtbot.waitExposed(page)
    qtbot.waitUntil(lambda: page._mda_width_locked_at_real_size, timeout=2000)

    page.panel_button(PanelKey.PRESETS).setChecked(True)  # default-open already
    right_area = page._right_dock_area
    assert right_area is not None
    right_column = page._column_widget(right_area)
    expected_width = min(page._dock_manager.width() // 4, 500)
    qtbot.waitUntil(lambda: right_column.width() == expected_width, timeout=2000)
    starting_width = right_column.width()
    assert starting_width > 0
    assert right_column.minimumWidth() == 0
    assert right_column.maximumWidth() == QWIDGETSIZE_MAX
    assert right_column not in page._width_locked_areas.values()

    target_width = max(100, starting_width - 100)
    page._dock_manager.setSplitterSizes(
        right_area, _resized_splitter_sizes(page, right_area, target_width)
    )
    assert right_column.width() == target_width


def test_column_widget_locates_top_splitter_child_through_nesting(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """``_column_widget`` finds the outer splitter's direct child at any depth.

    Regression test for two bugs, both only reproducible on a real,
    non-offscreen display:

    1. Restoring a layout whose right column held more than one *stacked*
       (non-tabbed) area -- e.g. Camera ROI split out below Groups & Presets
       rather than tabbed alongside it, exactly what ADS does on a real
       drag-and-drop split -- left the whole column permanently
       un-resizable. ``_install_width_lock`` used to lock whichever single
       ``CDockAreaWidget`` ``_resolve_right_dock_area`` happened to find.
       That's fine while every right-side panel stays tabbed together in one
       area, but a vertical splitter forces every stacked child to share its
       width -- so locking just *one* of several stacked areas to a fixed
       width transitively locked the *whole* column to that width forever,
       with no boundary left to unlock through (the eventFilter was watching
       the divider between the stacked panels, not the actual column-width
       handle).

    2. The first fix for (1) assumed the MDA column could never itself be
       split this way, and used ``self._mda_dock.dockAreaWidget()`` as a
       fixed anchor to identify the outer splitter. That assumption doesn't
       hold -- ADS lets a user stack a panel onto MDA's column exactly like
       any other. The current version doesn't reference MDA (or any other
       specific dock) at all: it climbs until the next splitter up is no
       longer itself nested in another splitter, which identifies the outer
       (MDA / center / right) splitter structurally, symmetric across every
       column.

    Built from plain ``QSplitter``/``QWidget`` stand-ins rather than an
    actual ADS split: splitting a dock area for real (``addDockWidget`` into
    a *new* area, as opposed to tabbing into an existing one) reproducibly
    crashes under this automated harness -- see ``_configure_ads`` -- so
    real ADS rearranging stays a manual smoke-test item here too. This
    instead verifies the tree-walking logic directly.
    """
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    outer = QSplitter(Qt.Orientation.Horizontal)
    qtbot.addWidget(outer)

    # Left (MDA) column: split into two stacked areas -- case 2 above.
    left = QSplitter(Qt.Orientation.Vertical)
    left_area_a = QWidget()
    left_area_b = QWidget()
    left.addWidget(left_area_a)
    left.addWidget(left_area_b)

    # Center column: ADS wraps even an *unsplit* column in a chain of
    # single-child splitters in practice -- verify the walk isn't fooled
    # into stopping one level early by that extra wrapping.
    center_outer_wrap = QSplitter(Qt.Orientation.Vertical)
    center_inner_wrap = QSplitter(Qt.Orientation.Horizontal)
    center_area = QWidget()
    center_inner_wrap.addWidget(center_area)
    center_outer_wrap.addWidget(center_inner_wrap)

    # Right column: split into two stacked areas -- case 1 above.
    right = QSplitter(Qt.Orientation.Vertical)
    right_area_a = QWidget()
    right_area_b = QWidget()
    right.addWidget(right_area_a)
    right.addWidget(right_area_b)

    outer.addWidget(left)
    outer.addWidget(center_outer_wrap)
    outer.addWidget(right)

    # Every stacked area in a split column resolves to that column's
    # wrapping splitter -- left (MDA) and right alike, no special-casing.
    assert page._column_widget(cast("CDockAreaWidget", left_area_a)) is left
    assert page._column_widget(cast("CDockAreaWidget", left_area_b)) is left
    assert page._column_widget(cast("CDockAreaWidget", right_area_a)) is right
    assert page._column_widget(cast("CDockAreaWidget", right_area_b)) is right
    # An unsplit column resolves to the outer splitter's direct child, no
    # matter how many redundant single-child wrapper splitters ADS put
    # between it and the area itself.
    assert (
        page._column_widget(cast("CDockAreaWidget", center_area)) is center_outer_wrap
    )


def test_acquire_layout_round_trip(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    """save_layout()/restore_layout() round-trip the open panels onto a fresh page."""
    page_a = AcquirePage(mmcore)
    qtbot.addWidget(page_a)
    page_a.panel_button(PanelKey.PRESETS).setChecked(True)  # default-open already
    page_a.panel_button(PanelKey.EXCEPTION_LOG).click()

    saved = page_a.current_layout()
    state, keys = saved.dock_state, saved.panels
    assert state is not None
    assert keys == {PanelKey.MDA, PanelKey.PRESETS, PanelKey.EXCEPTION_LOG}

    page_b = AcquirePage(mmcore)
    qtbot.addWidget(page_b)
    assert page_b.restore_layout(state, keys)

    assert page_b.open_panels() == keys
    for key in keys:
        assert page_b.panel_button(key).isChecked()
    # Console was never opened on page_a, so restoring must not force every
    # registered panel open -- laziness survives a restore.
    assert page_b.panel_widget(PanelKey.CONSOLE) is None


def test_acquire_layout_round_trip_restores_open_stage_devices(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """The Stages panel's open devices are part of the saved layout.

    They live in the panel's own nested dock manager, which
    ``CDockManager.saveState()`` on the outer manager can't see -- so they
    need their own round trip through ``AcquireLayout.stage_devices``.
    """
    page_a = AcquirePage(mmcore)
    qtbot.addWidget(page_a)
    page_a._set_stage_kind(StageKind.PER_DEVICE)
    page_a.panel_button(PanelKey.STAGES).click()
    panel_a = page_a.panel_widget(PanelKey.STAGES)
    assert isinstance(panel_a, StagesPanel)
    panel_a._add_stage("XY")
    panel_a._add_stage("Z1")

    saved = page_a.current_layout()
    assert saved.stage_devices == {"XY", "Z1"}

    page_b = AcquirePage(mmcore)
    qtbot.addWidget(page_b)
    assert page_b._stage_kind == StageKind.XYZ  # a fresh page still defaults to xyz
    assert page_b.restore_layout(
        saved.dock_state, saved.panels, saved.stage_devices, saved.stage_kind
    )
    assert page_b._stage_kind == StageKind.PER_DEVICE

    panel_b = page_b.panel_widget(PanelKey.STAGES)
    assert isinstance(panel_b, StagesPanel)
    assert panel_b.open_devices() == {"XY", "Z1"}


def test_acquire_layout_restore_skips_stage_devices_missing_from_config(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """A saved stage device the current config doesn't have is skipped, not an error.

    E.g. restoring the same named layout after switching to a configuration
    with fewer stages than the one it was saved under.
    """
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page._set_stage_kind(StageKind.PER_DEVICE)
    page.panel_button(PanelKey.STAGES).click()

    saved = page.current_layout()
    assert saved.dock_state is not None
    assert page.restore_layout(
        saved.dock_state, saved.panels, {"XY", "NotALoadedStage"}, saved.stage_kind
    )

    panel = page.panel_widget(PanelKey.STAGES)
    assert isinstance(panel, StagesPanel)
    assert panel.open_devices() == {"XY"}


def test_acquire_restore_collapses_an_expanded_auto_hide_panel(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """A side-bar-pinned panel that was expanded when saved reopens collapsed.

    ``CDockManager.restoreState`` faithfully replays every detail of the
    saved arrangement, including an auto-hide flyout being expanded on
    screen at the instant the layout was saved. A pinned panel reads as
    "tucked away" -- reopening the app (or switching layouts) shouldn't pop
    it back open on its own.
    """
    page_a = AcquirePage(mmcore)
    page_a.resize(1400, 900)
    qtbot.addWidget(page_a)
    page_a.show()
    page_a.panel_button(PanelKey.EXCEPTION_LOG).click()
    dock = page_a.panel_dock(PanelKey.EXCEPTION_LOG)
    assert dock is not None
    dock.setAutoHide(True, SideBarLocation.SideBarTop)
    container = dock.autoHideDockContainer()
    assert container is not None
    container.collapseView(False)  # expand the flyout, as if the user opened it
    assert container.isVisible()

    saved = page_a.current_layout()
    assert saved.dock_state is not None

    page_b = AcquirePage(mmcore)
    page_b.resize(1400, 900)
    qtbot.addWidget(page_b)
    page_b.show()
    assert page_b.restore_layout(saved.dock_state, saved.panels)

    dock_b = page_b.panel_dock(PanelKey.EXCEPTION_LOG)
    assert dock_b is not None
    assert dock_b.isAutoHide()  # still pinned to the sidebar
    container_b = dock_b.autoHideDockContainer()
    assert container_b is not None
    assert not container_b.isVisible()  # ...but collapsed, not popped open


def test_acquire_restored_right_dock_is_resizable(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """A right sidebar restored on relaunch has no stale fixed-width constraint."""
    page_a = AcquirePage(mmcore)
    page_a.resize(1400, 900)
    page_a.panel_button(PanelKey.PRESETS).setChecked(True)  # default-open already
    saved = page_a.current_layout()
    state, keys = saved.dock_state, saved.panels
    assert state is not None

    page_a.close()
    page_a.deleteLater()

    page_b = AcquirePage(mmcore)
    qtbot.addWidget(page_b)
    page_b.resize(1400, 900)
    assert page_b.restore_layout(state, keys)
    page_b.show()
    qtbot.waitExposed(page_b)
    qtbot.waitUntil(lambda: page_b._mda_width_locked_at_real_size, timeout=2000)

    right_area = page_b._right_dock_area
    assert right_area is not None
    right_column = page_b._column_widget(right_area)
    assert right_column.minimumWidth() == 0
    assert right_column.maximumWidth() == QWIDGETSIZE_MAX
    starting_width = right_column.width()
    target_width = max(100, starting_width - 100)
    page_b._dock_manager.setSplitterSizes(
        right_area, _resized_splitter_sizes(page_b, right_area, target_width)
    )
    assert right_column.width() == target_width


def test_acquire_repairs_an_unusably_narrow_restored_right_dock(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """A previously persisted transient 60 px tools column is expanded.

    The pinned width itself is derived from whatever width the window
    actually ends up with (``min(dock_manager.width() // 4,
    _RIGHT_DOCK_MAX_WIDTH)``, matching ``_pin_dock_widths``) rather than
    hardcoded to the requested ``resize(1400, 900)`` -- CI's real (not
    virtual/configurable) macOS/Windows runner displays can silently clamp a
    top-level window smaller than requested, unlike a local dev machine.
    """
    page_a = AcquirePage(mmcore)
    page_a.resize(1400, 900)
    page_a.show()
    qtbot.waitExposed(page_a)
    qtbot.waitUntil(lambda: page_a._mda_width_locked_at_real_size, timeout=2000)
    page_a.panel_button(PanelKey.PRESETS).setChecked(True)  # default-open already
    right_area = page_a._right_dock_area
    assert right_area is not None
    right_column = page_a._column_widget(right_area)
    expected_pinned_width_a = min(
        page_a._dock_manager.width() // 4, _RIGHT_DOCK_MAX_WIDTH
    )
    qtbot.waitUntil(
        lambda: right_column.width() == expected_pinned_width_a, timeout=2000
    )
    page_a._dock_manager.setSplitterSizes(
        right_area, _resized_splitter_sizes(page_a, right_area, 60)
    )
    assert right_column.width() == 60
    saved = page_a.current_layout()
    state, keys = saved.dock_state, saved.panels
    assert state is not None

    page_a.close()
    page_a.deleteLater()

    page_b = AcquirePage(mmcore)
    qtbot.addWidget(page_b)
    page_b.resize(1400, 900)
    assert page_b.restore_layout(state, keys)
    page_b.show()
    qtbot.waitExposed(page_b)
    qtbot.waitUntil(lambda: page_b._mda_width_locked_at_real_size, timeout=2000)

    restored_area = page_b._right_dock_area
    assert restored_area is not None
    restored_column = page_b._column_widget(restored_area)
    expected_pinned_width_b = min(
        page_b._dock_manager.width() // 4, _RIGHT_DOCK_MAX_WIDTH
    )
    # A range, not an equality. The repair aims for the pinned width but can
    # only take what the donor column has to spare above its own minimum --
    # and on a runner that clamped the window below the requested size, that
    # target is smaller still. What must hold either way: the corrupt 60 px
    # was expanded, and the repair never overshoots the target.
    assert 60 < restored_column.width() <= expected_pinned_width_b
    assert restored_column.minimumWidth() == 0
    assert restored_column.maximumWidth() == QWIDGETSIZE_MAX


def test_acquire_opening_a_panel_widens_an_unusable_column(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Opening a tool panel must never leave the column at ADS's 47--60px sliver.

    Regression test for the reported "why is Groups and Presets tiny?".
    ``_add_side_dock`` pins the width only when it *creates* the column, so
    every other way a panel gets shown -- re-showing a dock that already
    exists (``_toggle_panel`` calls ``toggleView`` and never reaches
    ``_add_side_dock``), or tabbing into a column that is already a sliver --
    used to leave it unusable, with no way back except dragging it.

    The sliver is forced here rather than provoked through a real
    close/reopen: on the offscreen test platform ADS happens to lay a
    re-shown column out at a usable width anyway, so only a real display
    reproduces the original report. What this pins down is the behaviour that
    fixes it -- opening a panel widens a below-floor column -- which is
    platform independent.

    The target itself, not a hardcoded floor, is what the assertions compare
    against: CI's headless display (real, just small and fixed-resolution --
    see the other width tests in this file) can silently clamp the requested
    1400px window, in which case ``min(dm.width() // 4, _RIGHT_DOCK_MAX_WIDTH)``
    -- the same number ``_widen_unusable_right_column`` aims for -- can itself
    land under the usability floor. The widen still helps in that case; it
    just can't promise reaching a floor the window has no room for.
    """
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page.resize(1400, 900)
    page.show()
    qtbot.waitExposed(page)
    qtbot.waitUntil(lambda: page._mda_width_locked_at_real_size, timeout=2000)

    right_area = page._right_dock_area
    assert right_area is not None
    column = page._column_widget(right_area)
    target = min(page._dock_manager.width() // 4, _RIGHT_DOCK_MAX_WIDTH)

    # Pre-build the dock so the re-show path (toggleView, not addDockWidget)
    # is exercised: addDockWidget schedules an ADS layout pass that can race
    # with _widen_right_column_soon on the offscreen platform.
    page.panel_button(PanelKey.EXCEPTION_LOG).setChecked(True)
    assert page.panel_dock(PanelKey.EXCEPTION_LOG) is not None
    page.panel_button(PanelKey.EXCEPTION_LOG).setChecked(False)

    page._dock_manager.setSplitterSizes(
        right_area, _resized_splitter_sizes(page, right_area, 60)
    )
    assert column.width() == 60

    # Re-show via toggleView -- no addDockWidget, no competing ADS layout.
    page.panel_button(PanelKey.EXCEPTION_LOG).setChecked(True)
    qtbot.waitUntil(lambda: column.width() > 60, timeout=2000)
    assert column.width() <= target
    if target >= _RIGHT_DOCK_MIN_USABLE_WIDTH:
        assert column.width() >= _RIGHT_DOCK_MIN_USABLE_WIDTH

    # A width the user chose is still theirs -- widening only ever rescues a
    # column that is below the usability floor. Only meaningful once the
    # first widen actually reached that floor -- otherwise opening another
    # panel legitimately widens it further, toward the same unreachable
    # target the first one settled for.
    if column.width() >= _RIGHT_DOCK_MIN_USABLE_WIDTH:
        usable = column.width()
        page.panel_button(PanelKey.PROPERTIES).setChecked(True)
        qtbot.wait(50)
        assert column.width() == usable


def test_acquire_narrow_column_repair_gives_up_instead_of_retrying_forever(
    mmcore: CMMCorePlus, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A column the repair cannot widen far enough must still finish settling.

    Regression test for a hang caught only on CI: the repair reported "not
    ready, retry" both when QtADS hadn't finished rebuilding the dock tree
    *and* when it simply could not widen the column any further. The second
    case recomputes identical numbers forever, so a display that clamps the
    window below the requested size (or any donor column with nothing to
    spare) left the settle timer firing every 200ms for the life of the page
    with ``_mda_width_locked_at_real_size`` never set -- so the MDA width
    lock was never installed either.

    Raising the usability floor out of reach reproduces that state without
    depending on the display size, which can't be forced below the MDA
    column's own minimum anyway.
    """
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page.resize(1400, 900)
    page.show()
    qtbot.waitExposed(page)
    qtbot.waitUntil(lambda: page._mda_width_locked_at_real_size, timeout=2000)

    right_area = page._right_dock_area
    assert right_area is not None
    monkeypatch.setattr(acquire_module, "_RIGHT_DOCK_MIN_USABLE_WIDTH", 10_000)

    # Now behave as a restore does: the layout came from disk, so the settle
    # takes the repair path rather than the canonical pin.
    page._layout_restored = True
    page._mda_width_locked_at_real_size = False
    assert page._widen_unusable_right_column() is True

    page._settle_and_lock_widths()
    assert page._mda_width_locked_at_real_size


def test_acquire_restore_does_not_repin_column_widths(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """A restored layout keeps the user's widths; the one-shot pin is skipped.

    Mirrors ``test_acquire_mda_dock_width_resists_relayout_but_stays_draggable``'s
    drag simulation, but checks the *restore* path: the width a user dragged
    the MDA column to in a previous session must survive into a freshly
    constructed page, rather than being reset to the canonical
    ``_MDA_DOCK_WIDTH`` by AcquirePage's normal first-show pin.
    """
    page_a = AcquirePage(mmcore)
    qtbot.addWidget(page_a)
    page_a.resize(1400, 900)
    page_a.show()
    qtbot.waitExposed(page_a)
    qtbot.waitUntil(lambda: page_a._mda_width_locked_at_real_size, timeout=2000)

    mda_area = page_a._mda_dock.dockAreaWidget()
    assert mda_area is not None
    handle = next(h for h, a in page_a._width_locked_areas.items() if a is mda_area)
    qtbot.mousePress(handle, Qt.MouseButton.LeftButton)  # type: ignore[no-untyped-call]
    page_a._dock_manager.setSplitterSizes(
        mda_area, _resized_splitter_sizes(page_a, mda_area, 500)
    )
    qtbot.mouseRelease(handle, Qt.MouseButton.LeftButton)  # type: ignore[no-untyped-call]
    _park_real_cursor_away()
    assert mda_area.width() == 500

    saved = page_a.current_layout()
    state, keys = saved.dock_state, saved.panels
    assert state is not None

    # Mirrors MainWindow.restore_state(): geometry is applied, then the dock
    # layout is restored, then the window is shown.
    page_b = AcquirePage(mmcore)
    qtbot.addWidget(page_b)
    page_b.resize(1400, 900)
    assert page_b.restore_layout(state, keys)
    page_b.show()
    qtbot.waitExposed(page_b)
    qtbot.waitUntil(lambda: page_b._mda_width_locked_at_real_size, timeout=2000)
    # A stray real cursor position -- left over from *any* earlier real drag,
    # in this test or another one entirely -- can park itself on page_b's own
    # handle purely by screen-coordinate coincidence and read as a live hover
    # nobody is actually performing (see _park_real_cursor_away). Parking it
    # away and giving the ever-running hover poll a moment to see that is
    # exactly the self-correction that mechanism exists for, so do that
    # before asserting on the settled state rather than assuming nothing
    # could have nudged it since waitUntil returned.
    _park_real_cursor_away()
    qtbot.wait(50)

    mda_area_b = page_b._mda_dock.dockAreaWidget()
    assert mda_area_b is not None
    # The exact width the user dragged to -- not the canonical pin, and above
    # all not zero. Asserting the concrete value matters: `!= _MDA_DOCK_WIDTH`
    # plus `min == max == width` is also satisfied by a collapsed 0px column,
    # which is precisely the bug this guards (locking a restored layout before
    # the window is first shown froze every column at min == max == 0).
    assert mda_area_b.width() == 500
    assert page_b._mda.isVisible()
    assert page_b._width_locked_areas
    assert mda_area_b.minimumWidth() == mda_area_b.maximumWidth() == 500


def test_pin_dock_widths_preserves_a_locked_custom_mda_width(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """``_pin_dock_widths`` must reuse MDA's current locked width, not the default.

    Regression test for a reported bug: a layout saved with MDA resized
    narrower and *no* right column open (a "minimal" layout, loaded with
    ``-l``) looked fine on restore -- the visible size stayed clamped to the
    saved width -- but the moment the first tool panel was opened afterward
    (creating the right column for the first time, via ``_add_side_dock``),
    the very next hover near the MDA/viewer boundary snapped the column back
    to the canonical default width, and moving the mouse away then locked
    that wrong width in permanently. Verified by hand against a real running
    app (before the fix: opening the first side panel after restoring a
    narrower MDA left the splitter's cached slot for MDA at the canonical
    700px even though the visible/locked width was 400px; hovering then
    unlocked and snapped to the stale 700; after the fix, both stay 400).

    ``_pin_dock_widths`` unconditionally wrote the canonical
    ``_MDA_DOCK_WIDTH`` into the splitter's *cached* slot for MDA whenever it
    ran, regardless of what MDA was actually locked to. The lock clamped what
    was *visible*, so nothing looked wrong until later, when something
    (a hover-triggered unlock) let Qt fall through to that stale cached
    value. That downstream Qt relayout isn't reliably reproducible inside the
    synthetic pytest-qt harness even when driven through a full
    ``create_mmgui`` two-window restore -- confirmed by hand -- so this
    instead asserts on ``_pin_dock_widths``' own output (what it writes into
    the splitter for MDA's slot) the moment MDA is already locked to a custom
    width, which is the actual defect and is deterministic regardless of
    platform/timing.
    """
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page.resize(1400, 900)
    page.show()
    qtbot.waitExposed(page)
    qtbot.waitUntil(lambda: page._mda_width_locked_at_real_size, timeout=2000)

    mda_area = page._mda_dock.dockAreaWidget()
    assert mda_area is not None
    # A custom width already locked in -- from a user's own drag, or from a
    # restored/named layout -- is exactly what _pin_dock_widths sees whenever
    # it runs; how it got there doesn't matter to this function.
    mda_area.setMinimumWidth(400)
    mda_area.setMaximumWidth(400)
    qtbot.wait(20)
    assert mda_area.width() == 400

    splitter = mda_area.parentWidget()
    assert isinstance(splitter, QSplitter)
    idx = splitter.indexOf(mda_area)

    page._pin_dock_widths()
    assert splitter.sizes()[idx] == 400


def test_acquire_restore_preserves_nested_viewer_manager(
    mmcore: CMMCorePlus, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restoring the outer tools layout leaves the inner viewer tree intact."""

    class FakePreview(QWidget):
        def __init__(self, mmcore: CMMCorePlus, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._core = mmcore

    def run_worker_now(func: Callable[[], None], **_: object) -> None:
        func()

    monkeypatch.setattr(acquire_viewers_module, "NDVPreview", FakePreview)
    monkeypatch.setattr(acquire_toolbar_module, "create_worker", run_worker_now)

    page_a = AcquirePage(mmcore)
    qtbot.addWidget(page_a)
    page_a.panel_button(PanelKey.PRESETS).setChecked(True)  # default-open already
    saved = page_a.current_layout()
    state, keys = saved.dock_state, saved.panels
    assert state is not None
    # Tear the source page down before building the second one. Leaving two
    # live CDockManagers (each with its own docks and snap/live core
    # connections) around until pytest-qt's teardown segfaults the offscreen
    # platform while it repaints them -- the same harness-only ADS fragility
    # documented in ``_configure_ads``.
    page_a.close()
    page_a.deleteLater()

    page_b = AcquirePage(mmcore)
    qtbot.addWidget(page_b)
    inner_manager = page_b._viewer_dock_manager
    assert page_b.restore_layout(state, keys)
    assert page_b._viewers._dock_manager is inner_manager
    assert inner_manager.centralWidget() is None

    # Must not raise/crash: the outer manager's restore does not rebuild the
    # independent inner viewer area used by ensure_preview.
    page_b._snap_btn.click()
    assert page_b._viewers.preview is not None
    assert inner_manager.dockAreaCount() == 1


def test_mda_action_icons_use_theme_status_colors(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    set_theme(DARK_THEME)
    widget = MemoryMDAWidget(mmcore)
    qtbot.addWidget(widget)

    def assert_icon_color(icon: QIcon, expected: Color) -> None:
        rgb = _icon_avg_rgb(icon, QSize(24, 24))
        assert rgb is not None
        color = qcolor(expected)
        expected_rgb = color.red(), color.green(), color.blue()
        assert all(
            abs(actual - wanted) < 2
            for actual, wanted in zip(rgb, expected_rgb, strict=True)
        )

    green = theme().status_green
    red = theme().status_red
    assert_icon_color(widget.control_btns.run_btn.icon(), green)
    assert_icon_color(widget.channels.act_add_row.icon(), green)
    assert_icon_color(widget.control_btns.cancel_btn.icon(), red)
    assert_icon_color(widget.channels.act_remove_row.icon(), red)
    assert_icon_color(widget.channels.act_clear.icon(), red)

    # Upstream replaces this icon at runtime; our later signal handler must
    # replace its literal "lime" icon with the theme green again.
    mmcore.mda.events.sequencePauseToggled.emit(True)
    assert widget.control_btns.pause_btn.text() == "Resume"
    assert_icon_color(widget.control_btns.pause_btn.icon(), green)

    position_btn = widget.stage_positions.findChild(MDAButton)
    assert position_btn is not None
    position_btn.setValue(
        useq.MDASequence(channels=(useq.Channel(config="DAPI", exposure=10),))
    )
    assert_icon_color(position_btn.seq_btn.icon(), green)
    assert_icon_color(position_btn.clear_btn.icon(), red)

    set_theme(LIGHT_THEME)
    QApplication.processEvents()
    assert_icon_color(widget.control_btns.run_btn.icon(), theme().status_green)
    assert_icon_color(widget.channels.act_remove_row.icon(), theme().status_red)


def test_toolbar_action_separators_follow_theme(qtbot: QtBot) -> None:
    """Standard QToolBar separators stay visible in both application themes."""
    toolbar = QToolBar()
    qtbot.addWidget(toolbar)

    try:
        for app_theme in (DARK_THEME, LIGHT_THEME):
            set_theme(app_theme)
            option = QStyleOption()
            option.initFrom(toolbar)
            option.rect = QRect(0, 0, 9, 40)
            option.state |= QStyle.StateFlag.State_Horizontal
            image = QImage(option.rect.size(), QImage.Format.Format_ARGB32)
            background = option.palette.color(QPalette.ColorRole.Window)
            image.fill(background)

            painter = QPainter(image)
            style = QApplication.style()
            assert style is not None
            style.drawPrimitive(
                QStyle.PrimitiveElement.PE_IndicatorToolBarSeparator,
                option,
                painter,
                toolbar,
            )
            painter.end()

            center_x = option.rect.width() // 2
            separator = image.pixelColor(center_x, option.rect.height() // 2)
            assert separator == option.palette.color(QPalette.ColorRole.Mid)
            assert separator == qcolor(theme().border_default)
            assert separator != background
    finally:
        set_theme(DARK_THEME)


def test_mda_grid_bounds_icons_stay_visible_after_action_changes(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """The Bounds editor must theme each raw icon installed by Mark/Move."""
    set_theme(DARK_THEME)
    widget = MemoryMDAWidget(mmcore)
    qtbot.addWidget(widget)

    grid = widget.grid_plan
    bounds = grid._core_xy_bounds
    buttons = bounds.findChildren(QPushButton)
    expected = qcolor(theme().text_primary)
    expected_rgb = expected.red(), expected.green(), expected.blue()

    def assert_themed() -> None:
        for button in buttons:
            rgb = _icon_avg_rgb(button.icon(), QSize(24, 24))
            assert rgb is not None
            assert all(
                abs(actual - wanted) < 2
                for actual, wanted in zip(rgb, expected_rgb, strict=True)
            )

    # Entering the edge/bounds mode keeps the initially themed Mark glyphs.
    grid._mode_area_radio.setChecked(True)
    grid._mode_bounds_radio.setChecked(True)
    assert_themed()

    # Each action change replaces the glyph upstream; both replacements must
    # be re-themed rather than reverting to the nearly invisible black source.
    bounds.go_middle.setChecked(True)
    assert_themed()
    bounds.go_middle.setChecked(False)
    assert_themed()


def test_position_subsequence_popup_is_collapsed_and_themed(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Per-position editors start compact and use the main app's styling."""
    set_theme(DARK_THEME)
    widget = MemoryMDAWidget(mmcore)
    qtbot.addWidget(widget)
    position_btn = widget.stage_positions.findChild(MDAButton)
    assert position_btn is not None

    popup = _MDAPopup(parent=position_btn)
    qtbot.addWidget(popup)
    popup.show()
    QApplication.processEvents()

    # _MDAPopup.mda_tabs is typed as the base MDATabs since its concrete type
    # is chosen dynamically at runtime (see _MDAPopup.__init__); this popup is
    # opened from a Collapsible/CoreConnected tree, so it's always these here.
    tabs = cast("CollapsibleCoreMDATabs", popup.mda_tabs)
    assert all(not section.expanded for section in tabs.sections)
    assert all(
        not child.styleSheet()
        for child in (popup, *popup.findChildren(QWidget))
        if not isinstance(child, QAbstractSlider)
    )
    assert all(
        button.property("variant") for button in popup.findChildren(QAbstractButton)
    )

    grid = cast("CoreConnectedGridPlanWidget", tabs.grid_plan)
    bounds = grid._core_xy_bounds
    grid._mode_bounds_radio.setChecked(True)
    expected = qcolor(theme().text_primary)
    expected_rgb = expected.red(), expected.green(), expected.blue()

    def assert_bounds_icons_themed() -> None:
        for button in bounds.findChildren(QPushButton):
            rgb = _icon_avg_rgb(button.icon(), QSize(24, 24))
            assert rgb is not None
            assert all(
                abs(actual - wanted) < 2
                for actual, wanted in zip(rgb, expected_rgb, strict=True)
            )

    assert_bounds_icons_themed()
    bounds.go_middle.setChecked(True)
    assert_bounds_icons_themed()


def test_stage_explorer_style(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    """The shared Stage Explorer remains consistent in classic and modern GUIs."""
    set_theme(DARK_THEME)
    explorer = ThemedStageExplorer(mmcore=mmcore)
    qtbot.addWidget(explorer)
    toolbar = explorer.toolBar()

    # Matches the rest of the app's action buttons (Snap/Live/etc.), not the
    # native QStyle's (larger) PM_ToolBarIconSize.
    expected_size = theme().scaled(20)
    assert toolbar.iconSize() == QSize(expected_size, expected_size)
    tool_buttons = [
        button
        for action in toolbar.actions()
        if isinstance((button := toolbar.widgetForAction(action)), QToolButton)
    ]
    assert tool_buttons
    assert all(not button.autoRaise() for button in tool_buttons)
    assert all(button.property("variant") == "subtle" for button in tool_buttons)
    assert explorer._contrast_slider._slider.styleSheet() == ""
    assert not toolbar.stop_scan_action.icon().isNull()

    expected = qcolor(theme().text_primary)
    expected_rgb = expected.red(), expected.green(), expected.blue()
    marker_actions = (
        toolbar.poll_stage_action,
        *toolbar.marker_mode_action_group.actions(),
    )
    for action in marker_actions:
        rgb = _icon_avg_rgb(action.icon(), QSize(24, 24))
        assert rgb is not None
        assert all(
            abs(actual - wanted) < 2
            for actual, wanted in zip(rgb, expected_rgb, strict=True)
        )


def test_stage_explorer_refreshes_all_pixel_dependent_geometry(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    explorer = ThemedStageExplorer(mmcore=mmcore)
    qtbot.addWidget(explorer)
    marker = explorer._stage_pos_marker
    assert marker is not None

    with (
        patch.object(
            explorer.roi_manager,
            "update_fovs",
            wraps=explorer.roi_manager.update_fovs,
        ) as update_fovs,
        patch.object(marker, "set_rect_size", wraps=marker.set_rect_size) as set_size,
        patch.object(marker, "apply_transform", wraps=marker.apply_transform) as apply,
    ):
        explorer.refreshPixelGeometry()

    width, height = explorer._fov_w_h()
    update_fovs.assert_called_with((width, height))
    set_size.assert_called_with(mmcore.getImageWidth(), mmcore.getImageHeight())
    apply.assert_called_once()


def test_pixel_config_commit_refreshes_open_stage_explorer(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    window = MainWindow(mmcore=mmcore)
    qtbot.addWidget(window)
    window._acquire.panel_button(PanelKey.STAGE_EXPLORER).click()
    explorer = window._acquire.panel_widget(PanelKey.STAGE_EXPLORER)
    assert isinstance(explorer, ThemedStageExplorer)

    with patch.object(explorer, "refreshPixelGeometry") as refresh:
        window._configurations.pixelConfigurationsApplied.emit()
    refresh.assert_called_once_with()


def test_commit_to_core_does_not_re_enter(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    """A commit started while one is in flight must be dropped, not nested.

    ``BusyOverlay.start()`` flushes the event loop before the core work begins;
    a second "Save to core" arriving there would otherwise start a nested
    rewrite of a core the first pass is still halfway through.
    """
    page = ConfigurationsPage(mmcore)
    qtbot.addWidget(page)
    page._tabs.setCurrentWidget(page._group_tab)

    depth = 0
    max_depth = 0
    attempts = 0

    def reentrant_save() -> None:
        nonlocal depth, max_depth, attempts
        depth += 1
        max_depth = max(max_depth, depth)
        if not attempts:
            # only re-enter once, so that losing the guard fails this
            # assertion rather than recursing until the test times out
            attempts += 1
            page.commit_current_to_core()  # what a queued second click does
        depth -= 1

    with patch.object(page._group_tab, "save", side_effect=reentrant_save):
        page.commit_current_to_core()

    assert max_depth == 1


def test_save_buttons_are_disabled_during_a_commit(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    page = ConfigurationsPage(mmcore)
    qtbot.addWidget(page)
    page._tabs.setCurrentWidget(page._group_tab)
    assert page._save_core_btn.isEnabled()

    seen: list[bool] = []
    with patch.object(
        page._group_tab,
        "save",
        side_effect=lambda: seen.append(page._save_core_btn.isEnabled()),
    ):
        page.commit_current_to_core()

    assert seen == [False]
    # ...and restored afterwards
    assert page._save_core_btn.isEnabled()
    assert page._save_file_btn.isEnabled()


def test_unchanged_groups_are_not_rewritten(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    """Saving must only touch groups whose contents actually differ.

    deleteConfigGroup() is destructive — it drops the channel-group designation
    among other things — so a save that changed one preset has no business
    tearing down every other group.
    """
    page = ConfigurationsPage(mmcore)
    qtbot.addWidget(page)
    assert len(mmcore.getAvailableConfigGroups()) > 1

    # nothing edited: the editor already mirrors the core
    deleted: list[str] = []
    real_delete = mmcore.deleteConfigGroup

    def recording_delete(name: str) -> None:
        # wraps rather than replaces: a no-op would leave the core in a state
        # the rewrite then errors on, and save()'s error path is a modal box
        deleted.append(name)
        real_delete(name)

    with patch.object(mmcore, "deleteConfigGroup", side_effect=recording_delete):
        page._group_tab.save()
    assert deleted == []
    # the groups are all still there, untouched
    assert len(mmcore.getAvailableConfigGroups()) > 1


def test_embedded_pixel_apply_coalesces_pixel_size_events(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """The bulk rewrite must not emit pixelSizeChanged once per preset.

    pymmcore-plus emits it from deletePixelSizeConfig(), definePixelSizeConfig()
    and setPixelSizeUm() alike, and every listener answers by calling
    getPixelSizeUm(), which MMCore resolves by reading the objective's live
    property values -- a serial round trip per read on real hardware.
    """
    page = ConfigurationsPage(mmcore)
    qtbot.addWidget(page)
    assert len(mmcore.getAvailablePixelSizeConfigs()) > 1

    events: list[float] = []
    mmcore.events.pixelSizeChanged.connect(events.append)
    page._tabs.setCurrentWidget(page._pixel_config)
    page.commit_current_to_core()

    # one coalesced event with the final configuration in place, not a storm of
    # intermediate ones computed from a half-rewritten config
    assert events == [mmcore.getPixelSizeUm()]


def test_successful_embedded_pixel_apply_emits_geometry_refresh_signal(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    page = ConfigurationsPage(mmcore)
    qtbot.addWidget(page)
    applied = Mock()
    page.pixelConfigurationsApplied.connect(applied)

    with (
        patch.object(page._pixel_config, "_on_apply"),
        patch.object(page._pixel_config, "isClean", return_value=True),
    ):
        page._pixel_config.apply()
    applied.assert_called_once_with()


def test_hardware_toolbar_buttons_use_primary_style(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    set_theme(DARK_THEME)
    page = HardwareSetupPage(mmcore)
    qtbot.addWidget(page)

    buttons = page.toolbar.findChildren(QPushButton)
    assert [button.text() for button in buttons] == [
        "New",
        "Load…",
        "Save…",
    ]
    assert all(button.property("variant") == "primary" for button in buttons)
    assert page._available._type.itemText(0) == "All Types"
    assert not any(
        label.text() == "Type:" for label in page._available.findChildren(QLabel)
    )


def test_hardware_load_uses_native_config_semantics(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    config = Path(__file__).with_name("test_config.cfg")
    set_theme(DARK_THEME)
    page = HardwareSetupPage(mmcore)
    qtbot.addWidget(page)

    # Reproduce loading the same file again from the Hardware tab.
    with patch.object(QFileDialog, "getOpenFileName", return_value=(str(config), "")):
        page.load_config()

    # System/Startup and the Core roles are applied by MMCore's native loader,
    # not merely defined as regular presets.
    assert mmcore.getCurrentConfig("System") == "Startup"
    assert mmcore.getChannelGroup() == "Channel"
    assert mmcore.getCameraDevice() == "Camera"
    assert mmcore.getShutterDevice() == "Shutter"
    assert mmcore.getFocusDevice() == "Z"
    assert page.model.config_file == str(config)


def _hardware_page_over(
    mmcore: CMMCorePlus, qtbot: QtBot, cfg: Path
) -> HardwareSetupPage:
    """A hardware page whose model came from ``cfg``, as after loading it."""
    page = HardwareSetupPage(mmcore)
    qtbot.addWidget(page)
    with patch.object(QFileDialog, "getOpenFileName", return_value=(str(cfg), "")):
        page.load_config()
    return page


def test_saving_over_a_cfg_offers_to_keep_its_light_sources(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    """Microscope.save() rewrites the file, so the comment block must be re-added."""
    cfg = _load_cfg_with_light_sources(
        mmcore, tmp_path, [("DAPI", "Camera", "TestProperty1", 0.05)]
    )
    page = _hardware_page_over(mmcore, qtbot, cfg)

    with patch.object(
        QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes
    ) as question:
        assert page.save_to(str(cfg))

    assert question.called
    assert question.call_args.args[2] == (
        "Do you want to keep the light source info for the DAPI channel?"
    )
    lines = cfg.read_text().splitlines()
    assert "#@LightSource,Channel,DAPI,Camera,TestProperty1,0.05" in lines
    # the rest of the configuration was still rewritten
    assert any(line.startswith("ConfigGroup,Channel,DAPI,") for line in lines)


def test_keeping_light_sources_lists_every_affected_channel(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    cfg = _load_cfg_with_light_sources(
        mmcore,
        tmp_path,
        [
            ("FITC", "Camera", "TestProperty2", 3.0),
            ("Cy5", "Camera", "TestProperty1", 0.05),
        ],
    )
    page = _hardware_page_over(mmcore, qtbot, cfg)

    with patch.object(
        QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes
    ) as question:
        assert page.save_to(str(cfg))

    assert question.call_args.args[2] == (
        "Do you want to keep the light source info for the Cy5, FITC channels?"
    )
    kept = [line for line in cfg.read_text().splitlines() if "#@LightSource" in line]
    assert kept == [
        "#@LightSource,Channel,FITC,Camera,TestProperty2,3.0",
        "#@LightSource,Channel,Cy5,Camera,TestProperty1,0.05",
    ]


def test_declining_to_keep_light_sources_drops_them(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    cfg = _load_cfg_with_light_sources(
        mmcore, tmp_path, [("DAPI", "Camera", "TestProperty1", 0.05)]
    )
    page = _hardware_page_over(mmcore, qtbot, cfg)

    with patch.object(
        QMessageBox, "question", return_value=QMessageBox.StandardButton.No
    ):
        assert page.save_to(str(cfg))

    assert "#@LightSource" not in cfg.read_text()


def test_light_sources_for_removed_channels_are_dropped_without_asking(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    """A declaration for a channel the configuration no longer defines is dead."""
    cfg = _load_cfg_with_light_sources(
        mmcore, tmp_path, [("NoSuchChannel", "Camera", "TestProperty1", 0.05)]
    )
    page = _hardware_page_over(mmcore, qtbot, cfg)

    with patch.object(QMessageBox, "question") as question:
        assert page.save_to(str(cfg))

    assert not question.called
    assert "#@LightSource" not in cfg.read_text()


def test_saving_to_a_new_file_never_asks_about_light_sources(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    cfg = _load_cfg_with_light_sources(
        mmcore, tmp_path, [("DAPI", "Camera", "TestProperty1", 0.05)]
    )
    page = _hardware_page_over(mmcore, qtbot, cfg)

    dest = tmp_path / "somewhere_else.cfg"
    with patch.object(QMessageBox, "question") as question:
        assert page.save_to(str(dest))

    assert not question.called
    assert "#@LightSource" not in dest.read_text()
    # and the file that *does* hold them is untouched
    assert "#@LightSource" in cfg.read_text()


def test_configuration_save_buttons_are_in_toolbar(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    set_theme(DARK_THEME)
    page = ConfigurationsPage(mmcore)
    qtbot.addWidget(page)

    assert page._save_core_btn.text() == "Save to core"
    assert page._save_file_btn.text() == "Save to file…"
    assert page._save_file_btn.property("variant") == (
        page._save_core_btn.property("variant")
    )
    assert page.toolbar._layout.indexOf(page._save_core_btn) == 0
    assert page.toolbar._layout.indexOf(page._save_file_btn) == 1

    group_layout = page._group_tab.layout()
    assert group_layout is not None and group_layout.count() == 1
    embedded_buttons = {
        button.text(): button for button in page._pixel_config.findChildren(QPushButton)
    }
    assert embedded_buttons["Apply and Close"].isHidden()
    assert embedded_buttons["Cancel"].isHidden()

    with qtbot.waitSignal(page.saveToFileRequested):
        page._save_file_btn.click()


def test_configuration_editor_action_icons_follow_theme(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    page = ConfigurationsPage(mmcore)
    qtbot.addWidget(page)
    group_actions = {
        action.text(): action for action in page._group_editor._tb.actions()
    }

    def assert_color(action: QAction, expected: Color) -> None:
        rgb = _icon_avg_rgb(action.icon(), QSize(24, 24))
        assert rgb is not None
        color = qcolor(expected)
        assert all(
            abs(actual - wanted) < 2
            for actual, wanted in zip(
                rgb, (color.red(), color.green(), color.blue()), strict=True
            )
        )

    try:
        for app_theme in (DARK_THEME, LIGHT_THEME):
            set_theme(app_theme)
            QApplication.processEvents()
            for name in (
                "Add Group",
                "Add Preset",
                "Edit Properties",
                "Duplicate",
            ):
                assert_color(group_actions[name], theme().status_green)
            assert_color(group_actions["Remove"], theme().status_red)
            assert_color(
                page._pixel_config._value_table.act_edit_props,
                theme().status_green,
            )
            assert_color(
                page._pixel_config._value_table.act_remove_props,
                theme().status_red,
            )
            assert_color(
                page._pixel_config._px_table.act_remove_row,
                theme().status_red,
            )
            assert_color(
                page._pixel_config._px_table.act_clear,
                theme().status_red,
            )
    finally:
        set_theme(DARK_THEME)


def test_save_to_core_commits_only_selected_config_tab(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    set_theme(DARK_THEME)
    page = ConfigurationsPage(mmcore)
    qtbot.addWidget(page)

    with (
        patch.object(page._group_tab, "save") as group_save,
        patch.object(page._pixel_config, "apply") as pixel_save,
    ):
        page._tabs.setCurrentWidget(page._group_tab)
        page._save_core_btn.click()
        group_save.assert_called_once_with()
        pixel_save.assert_not_called()

        page._tabs.setCurrentWidget(page._pixel_config)
        page._save_core_btn.click()
        pixel_save.assert_called_once_with()


def test_saving_selected_tab_keeps_other_tab_dirty(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    set_theme(DARK_THEME)
    page = ConfigurationsPage(mmcore)
    qtbot.addWidget(page)
    page._mark_group_dirty()
    page._mark_pixel_dirty()

    page._tabs.setCurrentWidget(page._group_tab)
    page.mark_current_saved()
    assert page.is_dirty()

    page._tabs.setCurrentWidget(page._pixel_config)
    page.mark_current_saved()
    assert not page.is_dirty()


def _recording_hardware(calls: list[str], path: str = "cfg") -> SimpleNamespace:
    """A stand-in HardwareSetupPage that logs the order of its save steps."""

    def prompt() -> str:
        calls.append("ask")
        return path

    def save_to(_path: str) -> bool:
        calls.append("file")
        return True

    return SimpleNamespace(
        prompt_save_path=Mock(side_effect=prompt), save_to=Mock(side_effect=save_to)
    )


def test_save_to_file_asks_where_before_committing_to_core() -> None:
    calls: list[str] = []
    configurations = SimpleNamespace(
        commit_to_core=Mock(side_effect=lambda: calls.append("core")),
        mark_saved=Mock(side_effect=lambda: calls.append("clean")),
    )
    window = SimpleNamespace(
        _configurations=configurations,
        _hardware=_recording_hardware(calls),
    )

    assert MainWindow._save_all(window)  # type: ignore[arg-type]
    # the dialog comes first (committing blocks the GUI thread for seconds on
    # real hardware), and the core is still written before the file
    assert calls == ["ask", "core", "file", "clean"]


def test_toolbar_save_asks_where_before_committing_selected_tab() -> None:
    calls: list[str] = []
    configurations = SimpleNamespace(
        commit_current_to_core=Mock(side_effect=lambda: calls.append("selected")),
        mark_current_saved=Mock(side_effect=lambda: calls.append("clean-selected")),
    )
    window = SimpleNamespace(
        _configurations=configurations,
        _hardware=_recording_hardware(calls),
    )

    assert MainWindow._save_current_configuration(window)  # type: ignore[arg-type]
    assert calls == ["ask", "selected", "file", "clean-selected"]


@pytest.mark.parametrize(
    "method, commit",
    [
        (MainWindow._save_all, "commit_to_core"),
        (MainWindow._save_current_configuration, "commit_current_to_core"),
    ],
)
def test_cancelling_the_save_dialog_leaves_the_core_untouched(
    method: object, commit: str
) -> None:
    """Dismissing the dialog must not have already rewritten the live core."""
    configurations = SimpleNamespace(
        **{commit: Mock(), "mark_saved": Mock(), "mark_current_saved": Mock()}
    )
    hardware = SimpleNamespace(prompt_save_path=Mock(return_value=""), save_to=Mock())
    window = SimpleNamespace(_configurations=configurations, _hardware=hardware)

    assert not method(window)  # type: ignore[operator]
    getattr(configurations, commit).assert_not_called()
    hardware.save_to.assert_not_called()


def test_acquire_width_settle_waits_out_a_late_resize(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """The initial width lock waits for resizing to stop, not just start.

    Regression test for a bug where the Acquire tab's dock layout became
    permanently un-resizable after a restart. ``AcquirePage`` lives inside
    ``MainWindow``'s ``QStackedWidget`` and isn't the initially-active page,
    so it only gets its first real ``showEvent`` once the app switches to it
    -- by which point ``MainWindow`` has already requested
    ``WindowMaximized``. On a real window manager that maximize is applied
    *asynchronously*, often completing just after this tab's first
    showEvent. The old one-shot ``showEvent`` handler locked the MDA/right
    column widths (``minimumWidth == maximumWidth``) immediately, so a late
    maximize landing a moment later froze them at a transient, too-small
    pre-maximize size that nothing ever revisited -- explaining why
    resizing worked fine when opening a panel interactively (the window was
    already settled by then) but not after a restart (the lock could win the
    race against the tab's own first show). The fix debounces the one-time
    lock via ``resizeEvent``, so it only fires once the geometry has
    actually stopped changing.
    """
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    # Simulate the pre-maximize transient: a too-small initial geometry.
    page.resize(400, 400)
    page.show()
    qtbot.waitExposed(page)

    mda_area = page._mda_dock.dockAreaWidget()
    assert mda_area is not None
    assert not page._mda_width_locked_at_real_size

    # Simulate the async WindowMaximized completing just after the first
    # showEvent -- before the debounce timer has fired.
    page.resize(1400, 900)
    assert not page._mda_width_locked_at_real_size

    qtbot.waitUntil(lambda: page._mda_width_locked_at_real_size, timeout=2000)
    # Locked at the *final* (1400-wide) canonical width, not the transient
    # 400-wide one.
    assert mda_area.width() == _MDA_DOCK_WIDTH
    assert mda_area.minimumWidth() == mda_area.maximumWidth() == _MDA_DOCK_WIDTH


def test_acquire_settle_retries_until_lock_actually_takes(
    mmcore: CMMCorePlus, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The debounce settle retries until the lock actually applies, not just once.

    Regression test for a CI-only failure (never reproduced locally, on any
    platform or Qt binding) where a restored layout's MDA area ended up
    completely unlocked (``minimumWidth() == 0``, ``maximumWidth() ==
    16777215``) despite ``_mda_width_locked_at_real_size`` having already
    flipped True and the area's *width* already matching the restored
    value. The debounce used to treat "the MDA area has a nonzero width" as
    proof the whole restored tree was ready to lock -- but
    ``_install_width_lock`` also needs a specific splitter handle index,
    which only exists once every sibling column has *also* been re-added to
    the tree by ADS's own deferred restore pass. A width can already be
    real before that handle exists, so the old width-only check could let
    the one-shot settle declare victory while the lock silently failed
    underneath it, with nothing left to ever retry. The fix has the locking
    functions report whether they actually succeeded and retries the
    debounce on *that* signal instead.

    Simulated here by forcing ``_install_width_lock`` to fail a couple of
    times before delegating to the real implementation -- reproducing the
    missing-handle race directly, since it never showed up locally on its
    own.
    """
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page.resize(1400, 900)

    call_count = 0
    real_install = page._install_width_lock

    def flaky_install(area: CDockAreaWidget) -> bool:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            # Simulate ADS not having rebuilt the expected splitter handle
            # yet, even though the area may already have a real width.
            return False
        return real_install(area)

    monkeypatch.setattr(page, "_install_width_lock", flaky_install)

    page.show()
    qtbot.waitExposed(page)
    qtbot.waitUntil(lambda: page._mda_width_locked_at_real_size, timeout=2000)

    assert call_count > 2, "test didn't actually exercise a retry"
    mda_area = page._mda_dock.dockAreaWidget()
    assert mda_area is not None
    assert mda_area.width() == _MDA_DOCK_WIDTH
    assert mda_area.minimumWidth() == mda_area.maximumWidth() == _MDA_DOCK_WIDTH
