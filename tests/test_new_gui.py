from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import useq

import pymmcore_gui._gui._acquire_viewers as acquire_viewers_module
from pymmcore_gui._app import LoadConfigDialog, create_mmgui
from pymmcore_gui._gui._acquire import AcquirePage
from pymmcore_gui._gui._configurations import ConfigurationsPage
from pymmcore_gui._gui._hardware import HardwareSetupPage
from pymmcore_gui._gui._main_win import MainWindow
from pymmcore_gui._gui._tab_bar import ThemedTabBar
from pymmcore_gui._gui._theme import set_theme
from pymmcore_gui._gui._theme._dark import DARK_THEME
from pymmcore_gui._qt.QtGui import QPalette
from pymmcore_gui._qt.QtWidgets import (
    QApplication,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QWidget,
)

if TYPE_CHECKING:
    import pytest
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot

    from pymmcore_gui._settings import Settings


def test_accepting_startup_config_selects_acquire(
    mmcore: CMMCorePlus,
    settings: Settings,
    qtbot: QtBot,
) -> None:
    config = Path(__file__).with_name("test_config.cfg")
    settings.last_config = config
    settings.auto_load_last_config = None

    def accept_after_processing_events(
        _dialog: LoadConfigDialog,
    ) -> QMessageBox.StandardButton:
        # The real modal prompt runs a nested event loop before returning.
        QApplication.processEvents()
        return QMessageBox.StandardButton.Yes

    with patch.object(LoadConfigDialog, "exec", accept_after_processing_events):
        window = create_mmgui(
            mm_config=None,
            mmcore=mmcore,
            install_sys_excepthook=False,
            install_sentry=False,
            exec_app=False,
            window_cls=MainWindow,
        )
    assert isinstance(window, MainWindow)
    qtbot.addWidget(window)

    assert window._stack.currentWidget() is window._acquire
    assert window._mode_tabs._tabs[2].active

    # A config loaded later from inside the running UI must not force a tab switch.
    window._mode_tabs._select(1)
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
        window_cls=MainWindow,
    )
    assert isinstance(window, MainWindow)
    qtbot.addWidget(window)

    # `-c` reaches create_mmgui as an explicit mm_config before app.exec().
    assert window._stack.currentWidget() is window._acquire
    QApplication.processEvents()
    assert window._stack.currentWidget() is window._acquire


def test_acquire_page_sidebar_layout(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    set_theme(DARK_THEME)
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    assert page._viewers.count() == 1
    assert isinstance(page._viewers.tabBar(), ThemedTabBar)
    assert page._viewers.tabText(0) == "Preview"
    assert not page.left.isHidden()
    assert not page.right.isHidden()
    assert page._mda.prepare_mda() == "memory"
    assert page._right_tabs.count() == 1
    assert isinstance(page._right_tabs.tabBar(), ThemedTabBar)
    assert page._right_tabs.widget(0) is page._mda
    assert page._right_tabs.tabText(0) == "MDA"
    assert page._mda_btn.isChecked()
    assert not page._props_btn.isChecked()

    # Groups & Presets is the upstream GroupPresetTableWidget with its
    # editing/save/load controls hidden — editing groups already lives on the
    # Configurations tab, and saving/loading a .cfg on the Hardware tab.
    hidden_buttons = {
        page._presets.edit_groups_btn,
        page._presets.save_btn,
        page._presets.load_btn,
    }
    assert all(button.isHidden() for button in hidden_buttons)
    assert not page._presets.table_wdg.isHidden()

    page._props_btn.click()
    assert page._right_tabs.count() == 2
    assert page._props_btn.isChecked()
    assert page._right_tabs.currentWidget() is page._property_browser
    assert page._right_tabs.tabText(1) == "Properties"
    assert page._property_browser is not None
    assert not page._property_browser.isWindow()

    # Toggling a button removes and restores its corresponding tab.
    page._props_btn.click()
    assert page._right_tabs.count() == 1
    assert not page._props_btn.isChecked()

    page._props_btn.click()
    page._mda_btn.click()
    assert page._right_tabs.count() == 1
    assert page._right_tabs.currentWidget() is page._property_browser
    assert not page._mda_btn.isChecked()

    page._mda_btn.click()
    assert page._right_tabs.count() == 2
    assert page._right_tabs.tabText(0) == "MDA"
    assert page._right_tabs.tabText(1) == "Properties"

    page._close_right_tab(0)
    assert page._right_tabs.count() == 1
    assert not page._mda_btn.isChecked()
    assert page._props_btn.isChecked()

    page._close_right_tab(0)
    assert page._right_tabs.count() == 0
    assert not page._props_btn.isChecked()
    assert page.right.isHidden()

    page._mda_btn.click()
    assert page._right_tabs.count() == 1
    assert page._right_tabs.currentWidget() is page._mda
    assert not page.right.isHidden()


def test_themed_tab_bar_keeps_style_after_reinserting_tab(qtbot: QtBot) -> None:
    set_theme(DARK_THEME)
    tabs = QTabWidget()
    bar = ThemedTabBar(tabs)
    tabs.setTabBar(bar)
    tabs.setTabsClosable(True)
    page = QWidget()
    tabs.addTab(page, "MDA")
    tabs.resize(320, 180)
    tabs.show()
    qtbot.addWidget(tabs)

    palette = QApplication.palette()
    expected_colors = (
        palette.color(QPalette.ColorRole.Highlight),
        palette.color(QPalette.ColorRole.WindowText),
    )

    def rendered_colors() -> set[tuple[int, int, int]]:
        image = bar.grab().toImage()
        return {
            image.pixelColor(x, y).getRgb()[:3]
            for y in range(image.height())
            for x in range(image.width())
        }

    for _ in range(5):
        QApplication.processEvents()
        colors = rendered_colors()
        assert all(
            (color.red(), color.green(), color.blue()) in colors
            for color in expected_colors
        )
        tabs.removeTab(0)
        tabs.insertTab(0, page, "MDA")


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

    mmcore.mda.run(useq.MDASequence(channels=["DAPI"]), output="memory")
    qtbot.wait(20)

    assert page._viewers.count() == 2
    assert page._viewers.tabText(1).startswith("MDA ")
    viewer = page._viewers.active_viewer
    assert isinstance(viewer, FakeViewer)
    assert viewer.data is not None

    page._viewers._close_tab(1)
    assert page._viewers.count() == 1
    assert viewer.closed


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


def test_save_to_file_commits_to_core_before_writing() -> None:
    calls: list[str] = []

    def save_file() -> bool:
        calls.append("file")
        return True

    configurations = SimpleNamespace(
        commit_to_core=Mock(side_effect=lambda: calls.append("core")),
        mark_saved=Mock(side_effect=lambda: calls.append("clean")),
    )
    hardware = SimpleNamespace(save_config=Mock(side_effect=save_file))
    window = SimpleNamespace(
        _configurations=configurations,
        _hardware=hardware,
    )

    assert MainWindow._save_all(window)  # type: ignore[arg-type]
    assert calls == ["core", "file", "clean"]


def test_toolbar_save_commits_selected_tab_before_writing() -> None:
    calls: list[str] = []

    def save_file() -> bool:
        calls.append("file")
        return True

    configurations = SimpleNamespace(
        commit_current_to_core=Mock(side_effect=lambda: calls.append("selected")),
        mark_current_saved=Mock(side_effect=lambda: calls.append("clean-selected")),
    )
    hardware = SimpleNamespace(save_config=Mock(side_effect=save_file))
    window = SimpleNamespace(
        _configurations=configurations,
        _hardware=hardware,
    )

    assert MainWindow._save_current_configuration(window)  # type: ignore[arg-type]
    assert calls == ["selected", "file", "clean-selected"]
