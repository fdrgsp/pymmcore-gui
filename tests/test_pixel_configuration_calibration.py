from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
from superqt.cmap import QColormapComboBox

from pymmcore_gui._modern_gui._acquire_toolbar import LiveButton, SnapButton
from pymmcore_gui._modern_gui._configurations import ConfigurationsPage
from pymmcore_gui._pixel_calibration import (
    CalibrationCaptureSettings,
    CaptureStateTransaction,
    HardwareFingerprint,
    PixelCalibrationResult,
    RegistrationResult,
    fit_affine,
)
from pymmcore_gui._pixel_calibration._models import CalibrationObservation
from pymmcore_gui._qt.QtCore import Qt
from pymmcore_gui._qt.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QPushButton,
)

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot


def _result_for_selected_resolution(
    page: ConfigurationsPage,
) -> PixelCalibrationResult:
    widget = page._pixel_config
    selected = widget._selected_preset()
    assert selected is not None
    _, preset = selected
    angle = np.deg2rad(21)
    matrix = 0.41234567 * np.asarray(
        [
            [np.cos(angle), np.sin(angle)],
            [np.sin(angle), -np.cos(angle)],
        ]
    )
    shifts = np.asarray(((-60, 0), (60, 0), (0, -55), (0, 55)), dtype=float)
    deltas = shifts @ matrix.T
    fit = fit_affine(shifts, deltas)
    registration = RegistrationResult(
        shift_xy=(0, 0),
        psr=20,
        peak_ratio=2,
        overlap=0.9,
        normalized_error=0,
    )

    def _observation(shift: np.ndarray, delta: np.ndarray) -> CalibrationObservation:
        return CalibrationObservation(
            stage_position_um=(float(delta[0]), float(delta[1])),
            stage_delta_um=(float(delta[0]), float(delta[1])),
            registration=registration,
            corrected_shift_xy=(float(shift[0]), float(shift[1])),
        )

    validation_shifts = np.asarray(((25, 20), (-25, 20), (0, -30)), dtype=float)
    validation_deltas = validation_shifts @ matrix.T
    core = page._core
    fingerprint = HardwareFingerprint(
        camera=str(core.getCameraDevice()),
        xy_stage=str(core.getXYStageDevice()),
        binning=int(core.getBinning(core.getCameraDevice())),
        magnification=float(core.getMagnificationFactor()),
        roi=tuple(int(v) for v in core.getROI(core.getCameraDevice())),  # type: ignore[arg-type]
        image_shape=(int(core.getImageHeight()), int(core.getImageWidth())),
        dtype="uint16",
        channel_count=int(core.getNumberOfCameraChannels()),
        pixel_size_config=str(preset.name),
        config_settings=widget._preset_settings(preset),
    )
    return PixelCalibrationResult(
        fit=fit,
        raw_matrix=matrix,
        raw_pixel_size_um=fit.pixel_size_um,
        fingerprint=fingerprint,
        observations=tuple(
            _observation(shift, delta)
            for shift, delta in zip(shifts, deltas, strict=True)
        ),
        validation_observations=tuple(
            _observation(shift, delta)
            for shift, delta in zip(validation_shifts, validation_deltas, strict=True)
        ),
        stage_returned=True,
    )


def test_success_auto_populates_selected_row_and_marks_page_dirty(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    page = ConfigurationsPage(mmcore)
    qtbot.addWidget(page)
    page._tabs.setCurrentWidget(page._pixel_config)
    widget = page._pixel_config
    selected = widget._selected_preset()
    assert selected is not None
    row, preset = selected
    resolution_id = str(preset.name)
    old_core_size = float(mmcore.getPixelSizeUmByID(resolution_id))
    result = _result_for_selected_resolution(page)

    assert widget.isClean()
    with qtbot.waitSignal(widget.cleanChanged) as emitted:
        widget._calibration_panel._on_result(result)

    assert emitted.args == [False]
    assert page.is_dirty()
    assert page._pixel_dirty
    assert preset.pixel_size_um == pytest.approx(result.raw_pixel_size_um)
    assert preset.affine == pytest.approx(
        (
            result.raw_matrix[0, 0],
            result.raw_matrix[0, 1],
            0,
            result.raw_matrix[1, 0],
            result.raw_matrix[1, 1],
            0,
        )
    )
    spin = widget._px_table.table().cellWidget(row, 1)
    assert isinstance(spin, QDoubleSpinBox)
    assert spin.decimals() == 8
    assert spin.value() == pytest.approx(result.raw_pixel_size_um)
    assert widget._affine_table.value() == pytest.approx(preset.affine)
    # Successful measurement only edits the widget; the normal Save-to-core action
    # remains the persistence boundary.
    assert mmcore.getPixelSizeUmByID(resolution_id) == old_core_size


def test_calibration_running_locks_configuration_writes_and_other_tab(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    page = ConfigurationsPage(mmcore)
    qtbot.addWidget(page)
    page._tabs.setCurrentWidget(page._pixel_config)

    page._pixel_config.calibrationRunningChanged.emit(True)

    assert not page._save_core_btn.isEnabled()
    assert not page._save_file_btn.isEnabled()
    assert not page._tabs.isTabEnabled(page._tabs.indexOf(page._group_tab))

    page._pixel_config.calibrationRunningChanged.emit(False)
    assert page._save_core_btn.isEnabled()
    assert page._save_file_btn.isEnabled()
    assert page._tabs.isTabEnabled(page._tabs.indexOf(page._group_tab))


def test_calibration_panel_has_form_viewer_and_information_layout(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    page = ConfigurationsPage(mmcore)
    qtbot.addWidget(page)
    panel = page._pixel_config._calibration_panel
    widget = page._pixel_config

    assert not hasattr(panel, "_channels")
    assert widget._content_splitter.widget(1) is panel
    assert widget._content_splitter.widget(2) is widget._props_selector
    assert panel._settings_group.title() == "Settings"
    assert not panel._settings_group.isCheckable()
    assert not hasattr(panel, "_advanced_button")
    assert panel._motion_separator.frameShape() is QFrame.Shape.HLine
    for control, label in (
        (panel._safe_radius, panel._safe_radius_label),
        (panel._settle_time, panel._settle_time_label),
        (panel._return_tolerance, panel._return_tolerance_label),
    ):
        assert control.isVisibleTo(panel)
        assert control.toolTip()
        assert label.toolTip() == control.toolTip()
    assert panel._top_splitter.indexOf(panel._settings_group) == 0
    assert panel._top_splitter.indexOf(panel._viewer_widget) == 1
    assert panel._info_group.title() == "Calibration information"
    assert panel._info_splitter.orientation() is Qt.Orientation.Horizontal
    assert panel._info_splitter.widget(0) is panel._result_widget
    assert panel._result_text.frameShape() is QFrame.Shape.NoFrame
    assert panel._result_text.wordWrap()
    assert panel._info_splitter.widget(1) is panel._diagnostics
    assert panel._settle_time.value() == 0
    assert panel._camera_combo.currentText() == mmcore.getCameraDevice()
    assert panel._xy_stage_combo.currentText() == mmcore.getXYStageDevice()
    assert panel._xy_stage_combo.toolTip()
    assert panel._channel_group_combo.currentText() == mmcore.getChannelGroup()
    assert panel._channel_combo.currentText() in mmcore.getAvailableConfigs(
        mmcore.getChannelGroup()
    )
    assert isinstance(panel._snap_button, SnapButton)
    assert isinstance(panel._live_button, LiveButton)
    assert panel._snap_button.toolTip() == "Snap"
    assert panel._live_button.toolTip() == "Live"
    assert not panel._snap_button.icon().isNull()
    assert not panel._live_button.icon().isNull()
    assert not panel._start_button.icon().isNull()


def test_xy_stage_can_be_selected_when_core_stage_is_unassigned(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    page = ConfigurationsPage(mmcore)
    qtbot.addWidget(page)
    panel = page._pixel_config._calibration_panel

    with patch.object(mmcore, "getXYStageDevice", return_value=""):
        panel.refreshHardware()
        panel._xy_stage_combo.setCurrentText("XY")

        assert panel._xy_stage() == "XY"
        assert not panel._unavailable_reason()


def test_error_and_success_summary_use_same_information_area(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    page = ConfigurationsPage(mmcore)
    qtbot.addWidget(page)
    panel = page._pixel_config._calibration_panel
    result_field = panel._result_text

    panel._on_failure("Stage did not settle")
    assert panel._result_text is result_field
    # Text includes an inline icon (see _render_result_text); the tooltip is
    # always the clean, un-annotated message.
    assert "Stage did not settle" in result_field.text()
    assert result_field.toolTip() == "Stage did not settle"

    panel._on_result(_result_for_selected_resolution(page))
    assert panel._result_text is result_field
    assert "Independent validation passed" in result_field.text()
    assert panel._info_splitter.widget(0) is panel._result_widget
    assert panel._info_splitter.widget(1) is panel._diagnostics
    panel._diagnostics.resize(500, 240)
    assert not panel._diagnostics.grab().isNull()
    assert "green points" in panel._diagnostics.toolTip()


def test_no_resolution_selection_disables_entire_calibration_panel(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    page = ConfigurationsPage(mmcore)
    qtbot.addWidget(page)
    widget = page._pixel_config
    panel = widget._calibration_panel
    assert panel.isEnabled()

    widget._px_table.table().clearSelection()
    assert not panel.isEnabled()

    widget._px_table.table().selectRow(0)
    assert panel.isEnabled()


def test_calibration_preview_hides_nonessential_controls_and_defaults_to_gray(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    page = ConfigurationsPage(mmcore)
    qtbot.addWidget(page)
    # The live-frame viewer is built lazily (on first Snap / Live / Start
    # calibration) to avoid paying its OpenGL-context realization cost at
    # startup for sessions that never open this panel.
    preview = page._pixel_config._calibration_panel._ensure_preview()
    viewer = preview.viewer
    viewer_widget = viewer.widget()

    assert viewer.display_model.channel_mode.value == "grayscale"
    assert viewer_widget.add_roi_btn.isHidden()
    assert viewer_widget.ndims_btn.isHidden()
    assert viewer_widget.channel_mode_combo.isHidden()
    assert all(
        combo.isHidden() for combo in viewer_widget.findChildren(QColormapComboBox)
    )
    tooltips = {button.toolTip() for button in viewer_widget.findChildren(QPushButton)}
    assert "Save as OME-TIFF / OME-Zarr" not in tooltips
    assert "Cycle orthogonal views" not in tooltips


def test_multi_camera_utility_identification(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    page = ConfigurationsPage(mmcore)
    qtbot.addWidget(page)
    panel = page._pixel_config._calibration_panel

    with (
        patch.object(mmcore, "getDeviceLibrary", return_value="Utilities"),
        patch.object(mmcore, "getDeviceName", return_value="Multi Camera"),
        patch.object(
            mmcore,
            "getDeviceDescription",
            return_value="Combine multiple physical cameras",
        ),
    ):
        assert panel._is_multi_camera_utility("VirtualCamera")

    with (
        patch.object(mmcore, "getDeviceLibrary", return_value="DemoCamera"),
        patch.object(mmcore, "getDeviceName", return_value="DCam"),
        patch.object(mmcore, "getDeviceDescription", return_value="Demo camera"),
    ):
        assert not panel._is_multi_camera_utility("Camera")


def test_settings_form_builds_single_channel_capture_settings(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    page = ConfigurationsPage(mmcore)
    qtbot.addWidget(page)
    panel = page._pixel_config._calibration_panel
    source = next(
        label
        for label, pairs in panel._light_sources.items()
        if pairs == (("Camera", "TestProperty1"),)
    )
    panel._light_source_combo.setCurrentText(source)
    panel._light_intensity.setValue(0.05)
    panel._exposure.setValue(37.5)

    settings = panel._capture_settings()

    assert settings.camera == panel._camera_combo.currentText()
    assert settings.channel_group == panel._channel_group_combo.currentText()
    assert settings.channel_config == panel._channel_combo.currentText()
    assert settings.exposure_ms == pytest.approx(37.5)
    assert settings.light_properties == (("Camera", "TestProperty1", 0.05),)


def test_snap_frame_worker_restores_and_finishes_cleanly(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    page = ConfigurationsPage(mmcore)
    qtbot.addWidget(page)
    panel = page._pixel_config._calibration_panel
    assert not panel._unavailable_reason()

    # Keep this a worker/thread lifecycle test; NDV rendering is covered by its
    # own widget and is not available on every headless OpenGL test backend.
    with patch.object(panel._ensure_preview(), "append"):
        panel.snapFrame()
        qtbot.waitUntil(lambda: panel._thread is None, timeout=5000)

    assert not panel.isRunning()
    assert panel._phase_label.text() == "Frame snapped; hardware state restored"


def test_live_preview_applies_settings_restores_them_and_retains_fov(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    page = ConfigurationsPage(mmcore)
    qtbot.addWidget(page)
    panel = page._pixel_config._calibration_panel
    preview = panel._ensure_preview()
    old_exposure = float(mmcore.getExposure())
    start_x, start_y = mmcore.getXYPosition(panel._xy_stage())
    panel._exposure.setValue(old_exposure + 13)

    with patch.object(preview, "append"):
        panel._live_button.click()
        qtbot.waitUntil(mmcore.isSequenceRunning, timeout=5000)

        assert panel._live_button.isChecked()
        assert panel._live_button.toolTip() == "Stop"
        assert panel.isRunning()
        assert mmcore.getExposure() == pytest.approx(old_exposure + 13)
        assert not panel._exposure.isEnabled()
        assert panel._snap_button.isEnabled()
        assert panel._start_button.isEnabled()

        mmcore.setXYPosition(panel._xy_stage(), start_x + 2, start_y + 3)
        mmcore.waitForDevice(panel._xy_stage())
        panel._live_button.click()
        qtbot.waitUntil(lambda: not mmcore.isSequenceRunning(), timeout=5000)

    assert not panel._live_button.isChecked()
    assert panel._live_button.toolTip() == "Live"
    assert not panel.isRunning()
    assert mmcore.getExposure() == pytest.approx(old_exposure)
    assert mmcore.getXYPosition(panel._xy_stage()) == pytest.approx(
        (start_x + 2, start_y + 3), abs=0.01
    )
    assert panel._exposure.isEnabled()
    assert "starting field of view retained" in panel._phase_label.text()


def test_capture_transaction_applies_target_and_restores_exact_state(
    mmcore: CMMCorePlus,
) -> None:
    old_resolution = str(mmcore.getCurrentPixelSizeConfig())
    old_objective = str(mmcore.getProperty("Objective", "Label"))
    old_exposure = float(mmcore.getExposure())
    old_intensity = str(mmcore.getProperty("Camera", "TestProperty1"))
    resolution_settings = tuple(
        (str(device), str(prop), str(value))
        for device, prop, value in mmcore.getPixelSizeConfigData("Res20x")
    )
    settings = CalibrationCaptureSettings(
        resolution_settings=resolution_settings,
        channel_group="Channel",
        channel_config="DAPI",
        exposure_ms=old_exposure + 7,
        light_properties=(("Camera", "TestProperty1", 0.05),),
    )
    transaction = CaptureStateTransaction(mmcore, settings, resolution_id="Res20x")

    try:
        transaction.apply()
        assert mmcore.getCurrentPixelSizeConfig() == "Res20x"
        assert mmcore.getProperty("Objective", "Label") == "Nikon 20X Plan Fluor ELWD"
        assert mmcore.getExposure() == pytest.approx(old_exposure + 7)
        assert float(mmcore.getProperty("Camera", "TestProperty1")) == pytest.approx(
            0.05
        )
    finally:
        transaction.restore()

    assert mmcore.getCurrentPixelSizeConfig() == old_resolution
    assert mmcore.getProperty("Objective", "Label") == old_objective
    assert mmcore.getExposure() == pytest.approx(old_exposure)
    assert mmcore.getProperty("Camera", "TestProperty1") == old_intensity


def test_capture_transaction_temporarily_selects_and_restores_camera(
    mmcore: CMMCorePlus,
) -> None:
    mmcore.loadDevice("Camera2", "DemoCamera", "DCam")
    mmcore.initializeDevice("Camera2")
    old_camera = str(mmcore.getCameraDevice())
    old_exposure = float(mmcore.getExposure(old_camera))
    mmcore.setCameraDevice("Camera2")
    mmcore.setExposure(23.0)
    mmcore.setCameraDevice(old_camera)
    resolution_id = str(mmcore.getCurrentPixelSizeConfig())
    settings = CalibrationCaptureSettings(
        resolution_settings=tuple(
            (str(device), str(prop), str(value))
            for device, prop, value in mmcore.getPixelSizeConfigData(resolution_id)
        ),
        channel_group="Channel",
        channel_config="DAPI",
        exposure_ms=71.0,
        camera="Camera2",
    )
    transaction = CaptureStateTransaction(mmcore, settings, resolution_id=resolution_id)

    transaction.apply()
    assert mmcore.getCameraDevice() == "Camera2"
    assert mmcore.getExposure("Camera2") == pytest.approx(71.0)
    transaction.restore()

    assert mmcore.getCameraDevice() == old_camera
    assert mmcore.getExposure(old_camera) == pytest.approx(old_exposure)
    mmcore.setCameraDevice("Camera2")
    assert mmcore.getExposure("Camera2") == pytest.approx(23.0)
    mmcore.setCameraDevice(old_camera)
