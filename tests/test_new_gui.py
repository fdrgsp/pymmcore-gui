from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock, patch

import useq
from pymmcore_widgets.useq_widgets._positions import MDAButton

import pymmcore_gui._gui._acquire_toolbar as acquire_toolbar_module
import pymmcore_gui._gui._acquire_viewers as acquire_viewers_module
from pymmcore_gui._app import LoadConfigDialog, create_mmgui
from pymmcore_gui._array_viewer import _icon_avg_rgb
from pymmcore_gui._gui._acquire import AcquirePage
from pymmcore_gui._gui._configurations import ConfigurationsPage
from pymmcore_gui._gui._hardware import HardwareSetupPage
from pymmcore_gui._gui._main_win import MainWindow
from pymmcore_gui._gui._tab_bar import ThemedTabBar
from pymmcore_gui._gui._theme import (
    UI_FONT_SIZE_PT,
    UI_FONT_WEIGHT,
    qcolor,
    set_theme,
    theme,
    ui_font,
)
from pymmcore_gui._gui._theme._dark import DARK_THEME
from pymmcore_gui._gui._theme._light import LIGHT_THEME
from pymmcore_gui._qt.QtCore import QSize
from pymmcore_gui._qt.QtGui import QIcon, QPalette
from pymmcore_gui._qt.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QStyle,
    QTabBar,
    QTabWidget,
    QToolButton,
    QWidget,
)
from pymmcore_gui.widgets._mda_widget import MemoryMDAWidget

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot

    from pymmcore_gui._app import WindowProtocol
    from pymmcore_gui._gui._theme import Color
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
            window_cls=cast("type[WindowProtocol]", MainWindow),
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
        window_cls=cast("type[WindowProtocol]", MainWindow),
    )
    assert isinstance(window, MainWindow)
    qtbot.addWidget(window)

    # `-c` reaches create_mmgui as an explicit mm_config before app.exec().
    assert window._stack.currentWidget() is window._acquire
    QApplication.processEvents()
    assert window._stack.currentWidget() is window._acquire


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


def test_acquire_page_sidebar_layout(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    set_theme(DARK_THEME)
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    assert page._viewers.count() == 0
    assert page._viewers.preview is None
    assert isinstance(page._viewers.tabBar(), ThemedTabBar)
    # Group/Preset selection lives in the right sidebar now (as a tab,
    # alongside MDA/Properties); the left sidebar has nothing left in it.
    assert page.left.isHidden()
    assert not page.right.isHidden()
    assert page._mda.prepare_mda() == "memory"
    assert page._right_tabs.count() == 2
    assert isinstance(page._right_tabs.tabBar(), ThemedTabBar)
    assert page._right_tabs.widget(0) is page._presets
    assert page._right_tabs.tabText(0) == "Groups and Presets"
    assert page._right_tabs.widget(1) is page._mda
    assert page._right_tabs.tabText(1) == "MDA"
    assert page._right_tabs.currentWidget() is page._mda
    assert page._presets_btn.isChecked()
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
    assert page._right_tabs.count() == 3
    assert page._props_btn.isChecked()
    assert page._right_tabs.currentWidget() is page._property_browser
    assert page._right_tabs.tabText(2) == "Properties"
    assert page._property_browser is not None
    assert not page._property_browser.isWindow()

    # Toggling a button removes and restores its corresponding tab.
    page._props_btn.click()
    assert page._right_tabs.count() == 2
    assert not page._props_btn.isChecked()

    page._props_btn.click()
    page._mda_btn.click()
    assert page._right_tabs.count() == 2
    assert page._right_tabs.currentWidget() is page._property_browser
    assert not page._mda_btn.isChecked()

    # Re-enabling inserts back at the front of the tab bar.
    page._mda_btn.click()
    assert page._right_tabs.count() == 3
    assert page._right_tabs.tabText(0) == "MDA"
    assert page._right_tabs.tabText(1) == "Groups and Presets"
    assert page._right_tabs.tabText(2) == "Properties"

    page._close_right_tab(0)
    assert page._right_tabs.count() == 2
    assert not page._mda_btn.isChecked()
    assert page._props_btn.isChecked()

    page._close_right_tab(0)
    assert page._right_tabs.count() == 1
    assert not page._presets_btn.isChecked()

    page._close_right_tab(0)
    assert page._right_tabs.count() == 0
    assert not page._props_btn.isChecked()
    assert not page._props_btn.isChecked()
    assert page.right.isHidden()

    page._mda_btn.click()
    assert page._right_tabs.count() == 1
    assert page._right_tabs.currentWidget() is page._mda
    assert not page.right.isHidden()


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
    assert page._viewers.count() == 0

    page._snap_btn.click()
    preview = page._viewers.preview
    assert isinstance(preview, FakePreview)
    assert page._viewers.count() == 1
    assert page._viewers.tabText(0) == "Preview"
    assert preview.frames == 1

    page._viewers.tabCloseRequested.emit(0)
    assert page._viewers.count() == 0
    assert page._viewers.preview is None
    assert preview.detached

    page._snap_btn.click()
    replacement = page._viewers.preview
    assert isinstance(replacement, FakePreview)
    assert replacement is not preview
    assert replacement.frames == 1


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
    assert page._viewers.tabText(0) == "Preview"
    assert preview.streaming_started == 1
    assert mmcore.isSequenceRunning()

    page._live_btn.click()
    assert not mmcore.isSequenceRunning()


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

        def pixel_rgb(x: int, y: int) -> tuple[int, int, int]:
            color = image.pixelColor(x, y)
            return color.red(), color.green(), color.blue()

        return {
            pixel_rgb(x, y) for y in range(image.height()) for x in range(image.width())
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


def test_themed_tab_close_button_scrolls_with_its_tab(qtbot: QtBot) -> None:
    tabs = QTabWidget()
    bar = ThemedTabBar(tabs)
    tabs.setTabBar(bar)
    tabs.setTabsClosable(True)
    for index in range(10):
        tabs.addTab(QWidget(), f"MDA acquisition {index}")
    tabs.resize(320, 180)
    tabs.show()
    qtbot.addWidget(tabs)
    QApplication.processEvents()

    last = tabs.count() - 1
    close_button = bar.tabButton(last, QTabBar.ButtonPosition.RightSide)
    if close_button is None:
        close_button = bar.tabButton(last, QTabBar.ButtonPosition.LeftSide)
    assert close_button is not None
    initial_x = close_button.x()

    bar.setCurrentIndex(last)
    QApplication.processEvents()
    bar.grab()  # force the custom paint path that positions tab buttons

    assert close_button.x() != initial_x
    assert bar.tabRect(last).contains(close_button.geometry().center())


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

    assert page._viewers.count() == 1
    assert page._viewers.tabText(0).startswith("MDA ")
    viewer = page._viewers.active_viewer
    assert isinstance(viewer, FakeViewer)
    assert viewer.data is not None

    page._viewers._close_tab(0)
    assert page._viewers.count() == 0
    assert viewer.closed


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


def test_stage_explorer_style_and_mda_link(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    set_theme(DARK_THEME)
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    explorer = page._explorer
    toolbar = explorer.toolBar()

    style = explorer.style()
    assert style is not None
    expected_size = style.pixelMetric(QStyle.PixelMetric.PM_ToolBarIconSize)
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

    first = useq.AbsolutePosition(x=10, y=20, name="ROI 1")
    explorer.sendToMDARequested.emit([first], True)
    assert list(page._mda.stage_positions.value()) == [first]
    assert page._right_tabs.currentWidget() is page._mda

    second = useq.AbsolutePosition(x=30, y=40, name="ROI 2")
    explorer.sendToMDARequested.emit([second], False)
    assert list(page._mda.stage_positions.value()) == [first, second]


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
