from __future__ import annotations

import math
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
from pymmcore_widgets.mda._core_channels import PROPERTY_SEPARATOR
from pymmcore_widgets.useq_widgets._positions import MDAButton

import pymmcore_gui._modern_gui._acquire_toolbar as acquire_toolbar_module
import pymmcore_gui._modern_gui._acquire_viewers as acquire_viewers_module
from pymmcore_gui._app import LoadConfigDialog, create_mmgui
from pymmcore_gui._array_viewer import _icon_avg_rgb
from pymmcore_gui._modern_gui._acquire import _MDA_DOCK_WIDTH, AcquirePage
from pymmcore_gui._modern_gui._acquire_presets import AcquisitionPresetSelector
from pymmcore_gui._modern_gui._configurations import ConfigurationsPage
from pymmcore_gui._modern_gui._hardware import HardwareSetupPage
from pymmcore_gui._modern_gui._main_win import MainWindow
from pymmcore_gui._modern_gui._panels import PANELS, PanelKey
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
from pymmcore_gui._qt.QtAds import CDockManager, CDockWidget
from pymmcore_gui._qt.QtCore import QSize, Qt
from pymmcore_gui._qt.QtWidgets import (
    QAbstractButton,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QToolButton,
    QWidget,
)
from pymmcore_gui.widgets._active_channel_table import CURRENT_CHANNEL_COLUMN
from pymmcore_gui.widgets._mda_widget import MemoryMDAWidget
from pymmcore_gui.widgets._stage_explorer import ThemedStageExplorer

if TYPE_CHECKING:
    from collections.abc import Callable

    from pymmcore_plus import CMMCorePlus
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


def test_acquire_page_dock_layout(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    set_theme(DARK_THEME)
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    tabs = page._mda._collapsible_tabs()

    assert page._viewers.preview is None
    assert page._central_dock_area.dockWidgetsCount() == 1  # just the blank placeholder
    # AcquirePage doesn't use TabPage's left sidebar or its (now-removed)
    # right/bottom regions -- the dock manager supplies all docking itself.
    assert page.left.isHidden()
    assert not hasattr(page, "right")
    assert not hasattr(page, "bottom")

    dm = page._dock_manager
    assert dm.centralWidget() is page._central
    DF = CDockWidget.DockWidgetFeature
    assert page._central.features().value & DF.NoTab.value
    assert not page._central.features().value & DF.DockWidgetClosable.value

    assert page._mda_dock.widget() is page._mda
    assert not page._mda_dock.isClosed()
    assert page.panel_button(PanelKey.MDA).isChecked()
    assert page._mda.prepare_mda() == "memory"

    # Secondary panels are lazy: no widget or dock before first open.
    assert page.panel_widget(PanelKey.PRESETS) is None
    assert page.panel_dock(PanelKey.PRESETS) is None
    assert not page.panel_button(PanelKey.PRESETS).isChecked()
    assert page.panel_widget(PanelKey.PROPERTIES) is None
    assert page.panel_dock(PanelKey.PROPERTIES) is None
    assert not page.panel_button(PanelKey.PROPERTIES).isChecked()
    assert page.panel_widget(PanelKey.CONSOLE) is None
    assert page.panel_dock(PanelKey.CONSOLE) is None
    assert not page.panel_button(PanelKey.CONSOLE).isChecked()

    assert sorted(dm.dockWidgetsMap()) == [
        "acquire_mda",
        "acquire_viewers",
    ]
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
    page.panel_button(PanelKey.PRESETS).click()
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
    inactive = qcolor(theme().text_secondary).name()
    active = qcolor(theme().text_primary).name()
    assert f"ads--CDockWidgetTab QLabel {{\n                color: {inactive};" in ss
    assert (
        f'ads--CDockWidgetTab[activeTab="true"] QLabel {{\n                '
        f"color: {active};" in ss
    )
    # base ADS chrome (e.g. the qproperty-icon rules for title-bar buttons)
    # must survive -- this appends overrides, it doesn't replace the sheet.
    assert "qproperty-icon" in ss


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
    page.panel_button(PanelKey.PRESETS).click()
    presets_dock = page.panel_dock(PanelKey.PRESETS)
    assert presets_dock is not None
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
    page.panel_button(PanelKey.PRESETS).click()
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

    page.panel_button(PanelKey.PRESETS).click()
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

    assert page.panel_button(PanelKey.MDA).isChecked()
    for info in PANELS:
        if info.key != PanelKey.MDA:
            assert not page.panel_button(info.key).isChecked()

    # Only MDA is open by default -- nothing else in the registry builds
    # eagerly, matching the lazy-panel guarantee the other tests exercise.
    assert sorted(page._dock_manager.dockWidgetsMap()) == [
        "acquire_mda",
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

    # ...followed by a separated, non-checkable Reset Layout entry.
    assert [a.text() for a in menu.actions() if not a.isCheckable() and a.text()] == [
        "Reset Layout"
    ]
    assert any(a.isSeparator() for a in menu.actions())

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

    page.panel_button(PanelKey.PRESETS).click()
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

    assert page.open_panels() == {PanelKey.MDA}
    assert sorted(page._dock_manager.dockWidgetsMap()) == [
        "acquire_mda",
        "acquire_viewers",
    ]
    for info in PANELS:
        assert not page.panel_button(info.key).isHidden()
        if info.key != PanelKey.MDA:
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

    for key in (PanelKey.PROPERTIES, PanelKey.EXCEPTION_LOG, PanelKey.CAMERA_ROI):
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

    page.panel_button(PanelKey.PRESETS).click()
    assert page._right_dock_area is not None
    page.panel_button(PanelKey.PRESETS).click()  # close it again

    # Must not raise, and must give the new panel a usable right column.
    page.panel_button(PanelKey.CAMERA_ROI).click()
    roi_dock = page.panel_dock(PanelKey.CAMERA_ROI)
    assert roi_dock is not None and not roi_dock.isClosed()
    area = roi_dock.dockAreaWidget()
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
    qtbot.wait(50)

    page.panel_button(PanelKey.PRESETS).click()
    page.panel_button(PanelKey.CAMERA_ROI).click()
    page.apply_hidden_panels({PanelKey.CONSOLE})
    assert page.open_panels() == {PanelKey.MDA, PanelKey.PRESETS, PanelKey.CAMERA_ROI}
    assert page.hidden_panels() == {PanelKey.CONSOLE}

    with qtbot.waitSignal(page.layoutReset):
        page.reset_layout()

    assert page.open_panels() == {PanelKey.MDA}
    assert page.hidden_panels() == set()
    for info in PANELS:
        assert not page.panel_button(info.key).isHidden()
    # Widgets survive, exactly as a normal close/reopen does.
    assert page.panel_widget(PanelKey.PRESETS) is not None

    mda_area = page._mda_dock.dockAreaWidget()
    assert mda_area is not None
    assert mda_area.width() == _MDA_DOCK_WIDTH
    assert page._right_dock_area is None


def test_acquire_customize_menu_reset_entry_is_wired(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Triggering the menu's Reset Layout entry reaches AcquirePage.reset_layout.

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

    menu = _panel_customize_menu(page)
    reset = next(a for a in menu.actions() if a.text() == "Reset Layout")
    with qtbot.waitSignal(page.layoutReset):
        reset.trigger()
    menu.deleteLater()

    assert not page.panel_button(PanelKey.CONSOLE).isHidden()
    assert page.panel_widget(PanelKey.CONSOLE) is None  # still never built
    assert page.open_panels() == {PanelKey.MDA}


def test_acquire_reset_layout_after_restore_repins_default_widths(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """A reset supersedes a restored layout, so the canonical widths come back."""
    page_a = AcquirePage(mmcore)
    qtbot.addWidget(page_a)
    page_a.resize(1400, 900)
    page_a.show()
    qtbot.waitExposed(page_a)
    qtbot.wait(50)

    mda_area = page_a._mda_dock.dockAreaWidget()
    assert mda_area is not None
    handle = next(h for h, a in page_a._width_locked_areas.items() if a is mda_area)
    qtbot.mousePress(handle, Qt.MouseButton.LeftButton)  # type: ignore[no-untyped-call]
    page_a._dock_manager.setSplitterSizes(
        mda_area, _resized_splitter_sizes(page_a, mda_area, 500)
    )
    qtbot.mouseRelease(handle, Qt.MouseButton.LeftButton)  # type: ignore[no-untyped-call]
    state, keys = page_a.save_layout()
    assert state is not None

    page_b = AcquirePage(mmcore)
    qtbot.addWidget(page_b)
    page_b.resize(1400, 900)
    assert page_b.restore_layout(state, keys)
    page_b.show()
    qtbot.waitExposed(page_b)
    qtbot.wait(50)
    assert page_b._layout_restored

    page_b.reset_layout()
    assert not page_b._layout_restored
    mda_area_b = page_b._mda_dock.dockAreaWidget()
    assert mda_area_b is not None
    assert mda_area_b.width() == _MDA_DOCK_WIDTH


def test_acquire_camera_roi_and_exception_log_panels_open(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """The two newly-registered panels dock inline, not as standalone windows."""
    from pymmcore_widgets import CameraRoiWidget

    from pymmcore_gui.widgets._exception_log import ExceptionLog

    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    page.panel_button(PanelKey.CAMERA_ROI).click()
    roi_widget = page.panel_widget(PanelKey.CAMERA_ROI)
    roi_dock = page.panel_dock(PanelKey.CAMERA_ROI)
    assert isinstance(roi_widget, CameraRoiWidget)
    assert roi_dock is not None and not roi_dock.isClosed()
    assert not roi_widget.isWindow()

    page.panel_button(PanelKey.EXCEPTION_LOG).click()
    log_widget = page.panel_widget(PanelKey.EXCEPTION_LOG)
    log_dock = page.panel_dock(PanelKey.EXCEPTION_LOG)
    assert isinstance(log_widget, ExceptionLog)
    assert log_dock is not None and not log_dock.isClosed()
    # create_exception_log sets WindowStaysOnTopHint | Window upstream --
    # every registry panel must be normalized to a plain docked child.
    assert not log_widget.isWindow()

    assert roi_dock.dockAreaWidget() is log_dock.dockAreaWidget()


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
    dark = qcolor(theme().text_primary)
    assert all(
        abs(a - b) < 4
        for a, b in zip(rgb, (dark.red(), dark.green(), dark.blue()), strict=True)
    )

    set_theme(LIGHT_THEME)
    rgb = _icon_avg_rgb(btn.icon(), QSize(24, 24))
    assert rgb is not None
    light = qcolor(theme().text_primary)
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

    assert result == reference_result
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
    """Closing a viewer actually destroys its dock area, but MDA still holds.

    Viewer docks use ADS's ``DockWidgetDeleteOnClose`` feature so a closed
    viewer's dock-area/splitter node is genuinely removed (freeing its Qt
    widget/canvas resources and letting any remaining side-by-side viewer
    reclaim the freed space) rather than left behind as permanently-empty
    dead space. Destroying a dock area makes ADS recompute splitter
    proportions for the *whole* manager, which used to also resize unrelated
    docks like the MDA panel -- that's now prevented by the MDA column's
    width lock (see ``AcquirePage._install_width_lock``), not by avoiding
    real deletion.

    This can't be exercised through the actual "split two viewers side by
    side, then close the one left alone in its area" scenario that motivated
    the width lock: per the note in ``_configure_ads`` (``_acquire.py``),
    emptying a dock area reproducibly segfaults under
    ``QT_QPA_PLATFORM=offscreen`` + pytest-qt, independent of app code -- the
    same reason the app's own tests stick to non-emptying dock moves (this
    viewer is tabbed alongside the always-present central placeholder, so
    closing it doesn't empty the area). Instead this verifies the two things
    that matter: the feature flag is set, and the MDA column doesn't move.
    """

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
    assert mda_area is not None
    mda_width_before = mda_area.width()

    dock.closeDockWidget()

    assert page._viewers.active_viewer is None
    assert viewer.closed
    assert mda_area.width() == mda_width_before


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
    qtbot.mousePress(handle, Qt.MouseButton.LeftButton)  # type: ignore[no-untyped-call]
    assert area.minimumWidth() == 0
    page._dock_manager.setSplitterSizes(
        area, _resized_splitter_sizes(page, area, new_width)
    )
    assert area.width() == new_width
    qtbot.mouseRelease(  # type: ignore[no-untyped-call]
        handle, Qt.MouseButton.LeftButton
    )
    assert area.minimumWidth() == area.maximumWidth() == new_width
    assert area.width() == new_width

    # The new, user-chosen width is itself just as resistant to relayout.
    page._dock_manager.setSplitterSizes(area, _resized_splitter_sizes(page, area, 50))
    assert area.width() == new_width


def test_acquire_mda_dock_width_resists_relayout_but_stays_draggable(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """The MDA dock area's width resists relayout, but a real drag can change it.

    Regression test for a bug where ADS's own splitter relayout -- triggered
    whenever any dock area's visibility changes anywhere in the manager, not
    just areas adjacent to the one that changed -- could silently resize the
    MDA (or right, Groups & Presets / Properties / Console) column when a
    central viewer opened or closed. A *reactive* fix (re-applying
    ``setSplitterSizes`` after the fact, even across several deferred
    event-loop turns) proved unreliable in practice: it's a race against
    ADS's own relayout passes, which can be arbitrarily delayed. The actual
    fix (``AcquirePage._install_width_lock``) instead keeps
    ``minimumWidth == maximumWidth`` on the locked dock area at all times --
    a hard Qt layout constraint a splitter must respect in *any* layout
    pass, regardless of what triggers it or when -- except for the exact
    duration of a real mouse drag on the handle to its right, caught via
    ``eventFilter`` (mouse press unlocks; release re-locks at the new
    width). This can't be exercised through the actual "split two viewers
    side by side, close the one left alone in its area" scenario that
    motivated the fix -- per the note in ``_configure_ads``
    (``_acquire.py``), emptying a dock area reproducibly segfaults under
    ``QT_QPA_PLATFORM=offscreen`` + pytest-qt, independent of app code. This
    instead verifies both halves directly: an explicit, deliberate attempt
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
    # small width; showEvent's one-time re-pin corrects that once the widget
    # has real geometry, same as it would for the real app's startup. Give it
    # a moment to settle.
    qtbot.wait(50)

    mda_area = page._mda_dock.dockAreaWidget()
    assert mda_area is not None
    assert mda_area.width() == _MDA_DOCK_WIDTH
    _assert_column_resists_relayout_but_stays_draggable(page, qtbot, mda_area, 500)


def test_acquire_right_dock_width_resists_relayout_but_stays_draggable(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """The right (Groups & Presets / Properties / Console) column gets the same lock.

    Same regression as the MDA-column test above, but for the right-side
    column -- opening it lazily is exactly the kind of dock-area-visibility
    change elsewhere in the manager that used to be able to silently resize
    it too, so it needs the same width lock installed once it exists (see
    ``_add_side_dock``).
    """
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page.resize(1400, 900)
    page.show()
    qtbot.waitExposed(page)
    qtbot.wait(50)

    page.panel_button(PanelKey.PRESETS).click()
    right_area = page._right_dock_area
    assert right_area is not None
    _assert_column_resists_relayout_but_stays_draggable(page, qtbot, right_area, 300)


def test_acquire_layout_round_trip(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    """save_layout()/restore_layout() round-trip the open panels onto a fresh page."""
    page_a = AcquirePage(mmcore)
    qtbot.addWidget(page_a)
    page_a.panel_button(PanelKey.PRESETS).click()
    page_a.panel_button(PanelKey.CAMERA_ROI).click()

    state, keys = page_a.save_layout()
    assert state is not None
    assert keys == {PanelKey.MDA, PanelKey.PRESETS, PanelKey.CAMERA_ROI}

    page_b = AcquirePage(mmcore)
    qtbot.addWidget(page_b)
    assert page_b.restore_layout(state, keys)

    assert page_b.open_panels() == keys
    for key in keys:
        assert page_b.panel_button(key).isChecked()
    # Console was never opened on page_a, so restoring must not force every
    # registered panel open -- laziness survives a restore.
    assert page_b.panel_widget(PanelKey.CONSOLE) is None


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
    qtbot.wait(50)

    mda_area = page_a._mda_dock.dockAreaWidget()
    assert mda_area is not None
    handle = next(h for h, a in page_a._width_locked_areas.items() if a is mda_area)
    qtbot.mousePress(handle, Qt.MouseButton.LeftButton)  # type: ignore[no-untyped-call]
    page_a._dock_manager.setSplitterSizes(
        mda_area, _resized_splitter_sizes(page_a, mda_area, 500)
    )
    qtbot.mouseRelease(handle, Qt.MouseButton.LeftButton)  # type: ignore[no-untyped-call]
    assert mda_area.width() == 500

    state, keys = page_a.save_layout()
    assert state is not None

    # Mirrors MainWindow.restore_state(): geometry is applied, then the dock
    # layout is restored, then the window is shown.
    page_b = AcquirePage(mmcore)
    qtbot.addWidget(page_b)
    page_b.resize(1400, 900)
    assert page_b.restore_layout(state, keys)
    page_b.show()
    qtbot.waitExposed(page_b)
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


def test_acquire_restore_repoints_viewer_central_area(
    mmcore: CMMCorePlus, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first snap after a restore must not tab into a deleted C++ area.

    Regression test for ``AcquireViewersManager`` caching
    ``_central_dock_area`` at construction: ``CDockManager.restoreState``
    tears down and rebuilds the whole dock-area tree, so without
    ``AcquireViewersManager.set_central_dock_area`` the manager would hold a
    dangling reference and crash on the first preview/viewer opened after a
    restore.
    """

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
    page_a.panel_button(PanelKey.PRESETS).click()
    state, keys = page_a.save_layout()
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
    assert page_b.restore_layout(state, keys)

    # Must not raise/crash: ensure_preview tabs into the *current* central
    # area, not the one captured when AcquireViewersManager was constructed.
    page_b._snap_btn.click()
    assert page_b._viewers.preview is not None


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


def test_stage_explorer_style(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    """AcquirePage no longer shows the Stage Explorer, but the classic GUI still
    ships it (``actions/widget_actions.py``'s ``stage_explorer_widget`` action),
    so its theming stays covered directly rather than through AcquirePage.
    """
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
