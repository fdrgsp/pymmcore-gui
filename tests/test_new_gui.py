from __future__ import annotations

import math
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock, patch

import pytest
import useq
from pymmcore_plus import PropertyType
from pymmcore_widgets import MDAWidget as UpstreamMDAWidget
from pymmcore_widgets.useq_widgets._positions import MDAButton

import pymmcore_gui._modern_gui._acquire_toolbar as acquire_toolbar_module
import pymmcore_gui._modern_gui._acquire_viewers as acquire_viewers_module
from pymmcore_gui._app import LoadConfigDialog, create_mmgui
from pymmcore_gui._array_viewer import _icon_avg_rgb
from pymmcore_gui._modern_gui._acquire import AcquirePage
from pymmcore_gui._modern_gui._configurations import ConfigurationsPage
from pymmcore_gui._modern_gui._hardware import HardwareSetupPage
from pymmcore_gui._modern_gui._main_win import MainWindow
from pymmcore_gui._modern_gui._tab_bar import ThemedTabBar
from pymmcore_gui._modern_gui._theme import (
    UI_FONT_SIZE_PT,
    UI_FONT_WEIGHT,
    qcolor,
    set_theme,
    theme,
    ui_font,
)
from pymmcore_gui._modern_gui._theme._dark import DARK_THEME
from pymmcore_gui._modern_gui._theme._light import LIGHT_THEME
from pymmcore_gui._qt.QtCore import QSize, Qt
from pymmcore_gui._qt.QtGui import QIcon, QPalette
from pymmcore_gui._qt.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabBar,
    QTabWidget,
    QToolButton,
    QWidget,
)
from pymmcore_gui.widgets._mda_widget import MemoryMDAWidget
from pymmcore_gui.widgets._ranged_property_channels import CURRENT_CHANNEL_COLUMN

if TYPE_CHECKING:
    from collections.abc import Callable

    from pymmcore_plus import CMMCorePlus
    from pymmcore_widgets.useq_widgets._data_table import DataTable
    from pytestqt.qtbot import QtBot
    from qtpy.QtCore import QModelIndex

    from pymmcore_gui._app import WindowProtocol
    from pymmcore_gui._modern_gui._theme import Color
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


def test_acquire_page_sidebar_layout(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    set_theme(DARK_THEME)
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    tabs = page._mda._collapsible_tabs()

    assert page._viewers.count() == 0
    assert page._viewers.preview is None
    assert isinstance(page._viewers.tabBar(), ThemedTabBar)
    assert not page.left.isHidden()
    assert page._mda.parentWidget() is page.left
    assert not page._mda.isHidden()
    assert not page.right.isHidden()
    assert page._mda.prepare_mda() == "memory"
    assert page._right_tabs.count() == 1
    assert isinstance(page._right_tabs.tabBar(), ThemedTabBar)
    assert page._right_tabs.widget(0) is page._presets
    assert page._right_tabs.tabText(0) == "Groups and Presets"
    assert page._right_tabs.currentWidget() is page._presets
    assert page._presets_btn.isChecked()
    assert not page._props_btn.isChecked()
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
    page._close_right_tab(0)
    assert page._right_tabs.count() == 1
    assert not page._presets_btn.isChecked()
    assert page._props_btn.isChecked()

    page._close_right_tab(0)
    assert page._right_tabs.count() == 0
    assert not page._props_btn.isChecked()
    assert page.right.isHidden()

    page._presets_btn.click()
    assert page._right_tabs.count() == 1
    assert page._right_tabs.currentWidget() is page._presets
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
    assert len(presets) >= 3
    first = presets[0]
    other_preset = presets[2]
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

    # The Config combo is the other way (besides the ● column) to move hardware:
    # a plain programmatic text change (currentTextChanged only, as during a
    # sequence restore/refresh) stays hardware-neutral, but a user activation
    # (the `activated` signal) applies the newly picked preset immediately.
    config_cell = table.cellWidget(1, config_column)
    assert config_cell is not None
    config_combo = config_cell.findChild(QComboBox)
    assert config_combo is not None
    config_combo.setCurrentText(other_preset)
    QApplication.processEvents()
    assert mmcore.getCurrentConfig("Channel") == first  # no hardware change yet
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
        for label, pair in channels.lightSources().items()
        if pair == (source_device, source_property)
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

    assert set(choices.values()) == expected
    assert choices
    assert all(
        label == f"{device} · {prop}" for label, (device, prop) in choices.items()
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
    assert value_header.text() == "Intensity"


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
    assert restored_property["group"] == "Camera · TestProperty1"
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
    assert page._viewers.tabText(0) == "Preview"
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
        for label, pair in channels.lightSources().items()
        if pair == (source_device, source_property)
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
        for label, pair in channels.lightSources().items()
        if pair == (source_device, source_property)
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

    first = useq.AbsolutePosition(x=10, y=20, name="ROI 1")
    explorer.sendToMDARequested.emit([first], True)
    assert list(page._mda.stage_positions.value()) == [first]
    assert page._mda._collapsible_tabs().section("p").expanded

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
