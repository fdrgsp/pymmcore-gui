from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

from pymmcore_gui._gui._configurations import ConfigurationsPage
from pymmcore_gui._gui._hardware import HardwareSetupPage
from pymmcore_gui._gui._main_win import MainWindow
from pymmcore_gui._gui._theme import set_theme
from pymmcore_gui._gui._theme._dark import DARK_THEME
from pymmcore_gui._qt.QtWidgets import QPushButton

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot


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
        "Reload from core",
    ]
    assert all(button.property("variant") == "primary" for button in buttons)


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
