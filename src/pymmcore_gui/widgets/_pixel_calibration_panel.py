from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Event
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from pymmcore_plus import CMMCorePlus, DeviceType, PropertyType
from superqt.iconify import QIconifyIcon
from superqt.utils import signals_blocked

from pymmcore_gui._light_sources import parse_light_source_comments
from pymmcore_gui._modern_gui._acquire_toolbar import LiveButton, SnapButton
from pymmcore_gui._modern_gui._theme import qcolor, theme
from pymmcore_gui._pixel_calibration import (
    CalibrationCancelled,
    CalibrationCaptureSettings,
    CalibrationOptions,
    CaptureStateTransaction,
    PixelCalibrationResult,
    run_pixel_calibration,
)
from pymmcore_gui._qt.QtCore import (
    QEvent,
    QObject,
    QPointF,
    QRectF,
    QSize,
    Qt,
    QThread,
    Signal,
)
from pymmcore_gui._qt.QtGui import QColor, QPainter, QPaintEvent, QPen
from pymmcore_gui._qt.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from pymmcore_gui.widgets.image_preview._ndv_preview import NDVPreview

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


logger = logging.getLogger(__name__)


def _describe_error(exc: BaseException) -> str:
    """Return a short, user-facing message for ``exc``.

    Calibration failures should read as plain sentences in the UI, not Python
    stack traces -- the full traceback is still logged (see call sites) for
    anyone who needs it. ``PixelCalibrationError`` and friends already carry a
    human-readable ``str()``; unexpected exceptions (bugs, hardware I/O
    errors) often do too, but fall back to the exception type name if their
    message is empty.
    """
    message = str(exc).strip()
    return message or type(exc).__name__


PHASE_LABELS = {
    "reference": "Acquiring stable reference images",
    "probe-x": "Finding a useful X-stage displacement",
    "probe-y": "Finding a useful Y-stage displacement",
    "measure": "Acquiring compass measurements",
    "validate": "Acquiring independent holdout measurements",
    "restore": "Restoring the original stage position",
    "complete": "Calibration complete",
}

_LIGHT_SOURCE_SEPARATOR = " · "


@dataclass(frozen=True)
class CalibrationTarget:
    """Resolution row currently bound to the calibration panel."""

    resolution_id: str
    settings: tuple[tuple[str, str, str], ...]
    binding_is_saved: bool


def _validation_residuals_px(result: PixelCalibrationResult) -> np.ndarray:
    """Return independent-validation prediction errors in image pixels."""
    inverse = np.linalg.inv(result.fit.matrix)
    residuals = []
    for observation in result.validation_observations:
        shift = np.asarray(observation.corrected_shift_xy)
        delta = np.asarray(observation.stage_delta_um)
        residuals.append(
            float(np.linalg.norm(inverse @ (delta - result.fit.matrix @ shift)))
        )
    return np.asarray(residuals, dtype=np.float64)


class CalibrationDiagnosticsWidget(QWidget):
    """Compare measured stage moves with affine predictions without matplotlib."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: PixelCalibrationResult | None = None
        self.setMinimumHeight(165)
        self.setToolTip(
            "Each blue arrow is a measured XY-stage move. Each green point is "
            "the endpoint predicted from the corresponding image shift. A good "
            "calibration places the green points on the blue arrow tips."
        )

    def setResult(self, result: PixelCalibrationResult | None) -> None:
        """Set the result rendered by the diagnostic graph."""
        self._result = result
        self.update()

    @staticmethod
    def _point(point: Sequence[float], rect: QRectF, extent: float) -> QPointF:
        return QPointF(
            rect.center().x() + float(point[0]) * rect.width() * 0.40 / extent,
            rect.center().y() - float(point[1]) * rect.height() * 0.40 / extent,
        )

    def _draw_vectors(self, painter: QPainter, rect: QRectF) -> None:
        assert self._result is not None
        observations = [
            obs
            for obs in (
                *self._result.observations,
                *self._result.validation_observations,
            )
            if obs.accepted
        ]
        painter.setPen(QColor("#aab3bd"))
        painter.drawText(
            rect.adjusted(3, 3, -3, -3),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            "Blue arrows: measured moves · Green dots: predicted endpoints",
        )
        painter.drawText(
            rect.adjusted(3, 21, -3, -3),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            "Good calibration: green dots coincide with arrow tips",
        )
        plot = rect.adjusted(8, 43, -8, -18)
        if not observations:
            painter.drawText(
                plot,
                Qt.AlignmentFlag.AlignCenter,
                "No diagnostic observations available",
            )
            return

        measured = np.asarray([obs.stage_delta_um for obs in observations])
        shifts = np.asarray([obs.corrected_shift_xy for obs in observations])
        predicted = shifts @ self._result.fit.matrix.T
        extent = max(
            float(np.max(np.abs(measured))),
            float(np.max(np.abs(predicted))),
            1e-9,
        )
        origin = self._point((0, 0), plot, extent)

        painter.setPen(QPen(QColor("#59636e"), 1))
        painter.drawLine(
            QPointF(plot.left(), plot.center().y()),
            QPointF(plot.right(), plot.center().y()),
        )
        painter.drawLine(
            QPointF(plot.center().x(), plot.top()),
            QPointF(plot.center().x(), plot.bottom()),
        )
        for actual, expected in zip(measured, predicted, strict=True):
            actual_point = self._point(actual, plot, extent)
            expected_point = self._point(expected, plot, extent)
            painter.setPen(QPen(QColor("#58a6ff"), 1.5))
            painter.drawLine(origin, actual_point)
            delta = actual_point - origin
            length = float(np.hypot(delta.x(), delta.y()))
            if length > 8:
                along = delta / length
                normal = QPointF(-along.y(), along.x())
                painter.drawLine(
                    actual_point,
                    actual_point - along * 7 + normal * 3.5,
                )
                painter.drawLine(
                    actual_point,
                    actual_point - along * 7 - normal * 3.5,
                )

            painter.setPen(QPen(QColor("#3fb950"), 1.5))
            painter.setBrush(QColor("#3fb950"))
            painter.drawEllipse(expected_point, 3.5, 3.5)

        painter.setPen(QColor("#7d8590"))
        painter.drawText(
            rect.adjusted(3, 3, -3, -3),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
            f"extent ±{extent:.3g} µm",
        )

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.palette().window())
        if self._result is None:
            painter.setPen(self.palette().text().color())
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Calibration diagnostics will appear here",
            )
            return
        self._draw_vectors(painter, QRectF(self.rect()).adjusted(5, 5, -5, -5))


class _FrameCore:
    """Forward MMCore calls while copying snapped frames into a Qt signal."""

    def __init__(
        self,
        core: CMMCorePlus,
        xy_stage: str,
        callback: Callable[[object], None],
    ) -> None:
        self._core = core
        self._xy_stage = xy_stage
        self._callback = callback
        self._number = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._core, name)

    def getImage(self) -> np.ndarray:
        """Return a copy and publish it with the current stage position."""
        image = np.asarray(self._core.getImage()).copy()
        self._number += 1
        position = tuple(float(v) for v in self._core.getXYPosition(self._xy_stage))
        self._callback(
            {
                "image": image,
                "number": self._number,
                "position": position,
            }
        )
        return image


class _CalibrationWorker(QObject):
    """Apply temporary capture state and run calibration off the GUI thread."""

    progress = Signal(str, float)
    frameReady = Signal(object)
    resultReady = Signal(object)
    previewReady = Signal()
    failed = Signal(str)
    cancelled = Signal(str)
    finished = Signal()

    def __init__(
        self,
        core: CMMCorePlus,
        target: CalibrationTarget,
        capture: CalibrationCaptureSettings,
        options: CalibrationOptions,
        xy_stage: str,
        cancel_event: Event,
        *,
        preview_only: bool,
    ) -> None:
        super().__init__()
        self._core = core
        self._target = target
        self._capture = capture
        self._options = options
        self._xy_stage = xy_stage
        self._cancel_event = cancel_event
        self._preview_only = preview_only

    def run(self) -> None:
        """Execute the transaction and report one terminal signal."""
        transaction = CaptureStateTransaction(
            cast("Any", self._core),
            self._capture,
            resolution_id=self._target.resolution_id,
        )
        failure = ""
        cancelled = False
        result: PixelCalibrationResult | None = None
        try:
            transaction.apply()
            proxy = _FrameCore(self._core, self._xy_stage, self.frameReady.emit)
            if self._preview_only:
                self._core.snapImage()
                proxy.getImage()
            else:
                result = run_pixel_calibration(
                    cast("Any", proxy),
                    self._options,
                    resolution_id=self._target.resolution_id,
                    xy_stage=self._xy_stage,
                    cancel_event=self._cancel_event,
                    progress=self.progress.emit,
                )
        except CalibrationCancelled as exc:
            # A deliberate Cancel click, not a bug -- log at info, and only
            # treated as a "cancelled" outcome below if restoration succeeds.
            cancelled = True
            logger.info("Pixel calibration cancelled: %s", exc)
            failure = _describe_error(exc)
        except BaseException as exc:
            # Full traceback goes to the log for developers; the GUI only
            # ever shows the short, human-readable message below.
            logger.exception("Pixel calibration failed")
            failure = _describe_error(exc)
        try:
            transaction.restore()
        except BaseException as exc:
            logger.exception("Capture-state restoration failed")
            cancelled = False  # a failed restoration always needs attention
            restore_failure = _describe_error(exc)
            failure = (
                f"{failure}\nCapture-state restoration also failed: {restore_failure}"
                if failure
                else f"Capture-state restoration failed: {restore_failure}"
            )

        if cancelled:
            self.cancelled.emit(failure)
        elif failure:
            self.failed.emit(failure)
        elif self._preview_only:
            self.previewReady.emit()
        elif result is not None:
            self.resultReady.emit(result)
        else:  # pragma: no cover
            self.failed.emit("Calibration completed without returning a result")
        self.finished.emit()


class PixelCalibrationPanel(QWidget):
    """Capture controls, ndv feedback, and diagnostics for one resolution row."""

    calibrationRunningChanged = Signal(bool)
    resultReady = Signal(object, str)

    def __init__(
        self,
        mmcore: CMMCorePlus,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._mmc = mmcore
        self._target: CalibrationTarget | None = None
        self._thread: QThread | None = None
        self._worker: _CalibrationWorker | None = None
        self._cancel_event: Event | None = None
        self._preview_only = False
        self._live_transaction: CaptureStateTransaction | None = None
        self._light_sources: dict[str, tuple[tuple[str, str], ...]] = {}
        self._channel_group_last = ""

        self.setMinimumWidth(700)
        self.setEnabled(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 0, 0, 0)
        layout.setSpacing(5)

        self._top_splitter = QSplitter(Qt.Orientation.Horizontal)

        self._settings_group = QGroupBox("Settings")
        self._settings_group.setCheckable(False)
        settings_layout = QVBoxLayout(self._settings_group)
        settings_form = QFormLayout()
        settings_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self._resolution_label = QLabel("No resolution selected")
        self._camera_combo = QComboBox()
        self._camera_combo.setToolTip(
            "Physical camera used for calibration. The Micro-Manager Multi Camera "
            "utility is intentionally excluded."
        )
        self._xy_stage_combo = QComboBox()
        self._xy_stage_combo.setToolTip(
            "XY stage moved during calibration. The current core XY stage is "
            "selected automatically; if none is assigned, choose a loaded stage."
        )
        self._channel_group_combo = QComboBox()
        self._channel_combo = QComboBox()
        self._exposure = QDoubleSpinBox()
        self._exposure.setRange(0, 86_400_000)
        self._exposure.setDecimals(3)
        self._exposure.setSuffix(" ms")
        self._light_source_combo = QComboBox()
        self._light_intensity = QDoubleSpinBox()
        self._light_intensity.setEnabled(False)
        self._light_intensity.setToolTip(
            "Value in the selected device property's native units"
        )
        settings_form.addRow("Resolution ID:", self._resolution_label)
        settings_form.addRow("Camera:", self._camera_combo)
        settings_form.addRow("XY stage:", self._xy_stage_combo)
        settings_form.addRow("Channel group:", self._channel_group_combo)
        settings_form.addRow("Channel:", self._channel_combo)
        settings_form.addRow("Exposure time:", self._exposure)
        settings_form.addRow("Light source:", self._light_source_combo)
        settings_form.addRow("Light intensity:", self._light_intensity)

        self._motion_separator = QFrame()
        self._motion_separator.setFrameShape(QFrame.Shape.HLine)
        self._motion_separator.setFrameShadow(QFrame.Shadow.Sunken)
        settings_form.addRow(self._motion_separator)

        self._safe_radius = QDoubleSpinBox()
        self._safe_radius.setRange(0.1, 100_000)
        self._safe_radius.setDecimals(2)
        self._safe_radius.setValue(100)
        self._safe_radius.setSuffix(" µm")
        safe_radius_tip = (
            "Maximum distance the XY stage may travel from its starting position "
            "during calibration. Set this to a distance that is safe for the "
            "objective, specimen, and stage."
        )
        self._safe_radius.setToolTip(safe_radius_tip)
        self._settle_time = QDoubleSpinBox()
        self._settle_time.setRange(0, 30)
        self._settle_time.setDecimals(3)
        self._settle_time.setValue(0)
        self._settle_time.setSuffix(" s")
        settle_time_tip = (
            "Time to wait after every XY-stage move before acquiring an image. "
            "Increase this if vibration or stage lag causes blurred frames."
        )
        self._settle_time.setToolTip(settle_time_tip)
        self._return_tolerance = QDoubleSpinBox()
        self._return_tolerance.setRange(0, 100)
        self._return_tolerance.setDecimals(3)
        self._return_tolerance.setValue(0.5)
        self._return_tolerance.setSuffix(" µm")
        return_tolerance_tip = (
            "Maximum allowed error between the final and original XY position. "
            "Calibration reports a failure if the stage cannot return within this "
            "distance."
        )
        self._return_tolerance.setToolTip(return_tolerance_tip)

        self._safe_radius_label = QLabel("Safe radius:")
        self._safe_radius_label.setToolTip(safe_radius_tip)
        self._settle_time_label = QLabel("Settle after move:")
        self._settle_time_label.setToolTip(settle_time_tip)
        self._return_tolerance_label = QLabel("Return tolerance:")
        self._return_tolerance_label.setToolTip(return_tolerance_tip)
        settings_form.addRow(self._safe_radius_label, self._safe_radius)
        settings_form.addRow(self._settle_time_label, self._settle_time)
        settings_form.addRow(self._return_tolerance_label, self._return_tolerance)
        settings_layout.addLayout(settings_form)
        settings_layout.addStretch(1)

        button_row = QHBoxLayout()
        self._snap_button = SnapButton(
            mmcore=self._mmc, parent=self, control_core=False
        )
        self._live_button = LiveButton(
            mmcore=self._mmc, parent=self, control_core=False
        )
        self._start_button = QPushButton("Start calibration")
        self._start_button.setProperty("variant", "primary")
        self._cancel_button = QPushButton("Cancel calibration safely")
        self._apply_action_icons()
        self._cancel_button.setEnabled(False)
        self._cancel_button.hide()
        button_row.addWidget(self._snap_button)
        button_row.addWidget(self._live_button)
        button_row.addWidget(self._start_button)
        settings_layout.addLayout(button_row)
        settings_layout.addWidget(self._cancel_button)

        self._viewer_widget = QWidget()
        viewer_layout = QVBoxLayout(self._viewer_widget)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(4)

        preview_header = QHBoxLayout()
        preview_header.addWidget(QLabel("Calibration images"))
        preview_header.addStretch(1)
        self._frame_label = QLabel("No frames")
        preview_header.addWidget(self._frame_label)
        viewer_layout.addLayout(preview_header)

        # The ndv viewer/vispy canvas realizes an OpenGL context on
        # construction (~0.2s, one-time per process) and stays resident for
        # the app's lifetime. Most sessions never open this panel, so it is
        # built lazily on first Snap / Live / Start calibration (see
        # _ensure_preview) instead of paying that cost at startup for every
        # user, whether or not they ever calibrate.
        self._viewer_layout = viewer_layout
        self._preview: NDVPreview | None = None
        self._preview_placeholder = QLabel(
            "Image preview appears here after Snap, Live, or Start calibration."
        )
        self._preview_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_placeholder.setWordWrap(True)
        self._preview_placeholder.setMinimumSize(320, 260)
        viewer_layout.addWidget(self._preview_placeholder, 1)

        self._top_splitter.addWidget(self._settings_group)
        self._top_splitter.addWidget(self._viewer_widget)
        self._top_splitter.setStretchFactor(0, 0)
        self._top_splitter.setStretchFactor(1, 1)
        self._top_splitter.setSizes([300, 500])
        layout.addWidget(self._top_splitter, 3)

        self._info_group = QGroupBox("Calibration information")
        info_layout = QVBoxLayout(self._info_group)
        self._phase_label = QLabel("Select a saved resolution to calibrate")
        self._phase_label.setWordWrap(True)
        info_layout.addWidget(self._phase_label)
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        info_layout.addWidget(self._progress)
        self._result_text = QLabel()
        self._result_text.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._result_text.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._result_text.setWordWrap(True)
        self._result_text.setMargin(4)
        self._result_text.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._set_result_message("No validated result")
        self._result_widget = QWidget()
        result_layout = QVBoxLayout(self._result_widget)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.addWidget(self._result_text, 1)
        self._diagnostics = CalibrationDiagnosticsWidget()
        self._info_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._info_splitter.setChildrenCollapsible(False)
        self._info_splitter.addWidget(self._result_widget)
        self._info_splitter.addWidget(self._diagnostics)
        self._info_splitter.setStretchFactor(0, 1)
        self._info_splitter.setStretchFactor(1, 2)
        self._info_splitter.setSizes([300, 500])
        info_layout.addWidget(self._info_splitter, 1)
        layout.addWidget(self._info_group, 2)

        self._snap_button.clicked.connect(self.snapFrame)
        self._live_button.clicked.connect(self.toggleLivePreview)
        self._start_button.clicked.connect(self.startCalibration)
        self._cancel_button.clicked.connect(self.cancelCalibration)
        self._camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        self._xy_stage_combo.currentTextChanged.connect(self._update_availability)
        self._channel_group_combo.currentTextChanged.connect(
            self._on_channel_group_changed
        )
        self._channel_combo.currentTextChanged.connect(self._on_channel_changed)
        self._light_source_combo.currentTextChanged.connect(
            self._on_light_source_changed
        )
        self._mmc.events.systemConfigurationLoaded.connect(self.refreshHardware)
        self._mmc.events.channelGroupChanged.connect(
            self._on_core_channel_group_changed
        )
        self._mmc.events.configGroupDeleted.connect(self.refreshHardware)
        self._mmc.events.configDefined.connect(self.refreshHardware)
        self._mmc.events.configDeleted.connect(self.refreshHardware)
        self._mmc.events.continuousSequenceAcquisitionStarted.connect(
            self._on_live_started
        )
        self._mmc.events.sequenceAcquisitionStopped.connect(self._on_live_stopped)
        self._mmc.events.XYStagePositionChanged.connect(
            self._on_xy_stage_position_changed
        )
        self.destroyed.connect(self._disconnect)
        self.refreshHardware()

    def _apply_action_icons(self) -> None:
        """Color Start/Cancel icons to match the app's Snap/Live convention.

        ``QIconifyIcon`` bakes its color in at construction, so it must be
        rebuilt (not just re-tinted) on every theme change -- see
        ``changeEvent`` below, and ``SnapButton``/``LiveButton`` in
        ``_acquire_toolbar.py`` for the same pattern.
        """
        green = qcolor(theme().status_green).name()
        red = qcolor(theme().status_red).name()
        icon_size = QSize(theme().scaled(20), theme().scaled(20))
        self._start_button.setIcon(QIconifyIcon("mdi:play-circle-outline", color=green))
        self._start_button.setIconSize(icon_size)
        self._cancel_button.setIcon(QIconifyIcon("mdi:stop-circle-outline", color=red))
        self._cancel_button.setIconSize(icon_size)

    def changeEvent(self, a0: QEvent | None) -> None:
        super().changeEvent(a0)
        if (
            a0 is not None
            and a0.type() == QEvent.Type.StyleChange
            and hasattr(self, "_start_button")
        ):
            self._apply_action_icons()

    def _disconnect(self) -> None:
        if self._live_transaction is not None:
            self.stopLivePreview()
        for signal, callback in (
            (self._mmc.events.systemConfigurationLoaded, self.refreshHardware),
            (
                self._mmc.events.channelGroupChanged,
                self._on_core_channel_group_changed,
            ),
            (self._mmc.events.configGroupDeleted, self.refreshHardware),
            (self._mmc.events.configDefined, self.refreshHardware),
            (self._mmc.events.configDeleted, self.refreshHardware),
            (
                self._mmc.events.continuousSequenceAcquisitionStarted,
                self._on_live_started,
            ),
            (self._mmc.events.sequenceAcquisitionStopped, self._on_live_stopped),
            (
                self._mmc.events.XYStagePositionChanged,
                self._on_xy_stage_position_changed,
            ),
        ):
            try:
                signal.disconnect(callback)
            except Exception:
                pass

    def setTarget(self, target: CalibrationTarget | None) -> None:
        """Bind calibration output to one selected resolution row."""
        if target != self._target and self._live_transaction is not None:
            self.stopLivePreview()
        self._target = target
        self.setEnabled(target is not None)
        self._resolution_label.setText(
            target.resolution_id if target is not None else "No resolution selected"
        )
        self._diagnostics.setResult(None)
        self._set_result_message("No validated result")
        self._update_availability()

    def refreshHardware(self, *_: object) -> None:
        """Refresh selectable devices, channels, light sources, and availability."""
        selected_stage = self._xy_stage_combo.currentText()
        stages = tuple(
            str(stage) for stage in self._mmc.getLoadedDevicesOfType(DeviceType.XYStage)
        )
        core_stage = str(self._mmc.getXYStageDevice())
        preferred_stage = core_stage or selected_stage
        with signals_blocked(self._xy_stage_combo):
            self._xy_stage_combo.clear()
            self._xy_stage_combo.addItems(stages)
            if preferred_stage in stages:
                self._xy_stage_combo.setCurrentText(preferred_stage)

        selected_camera = self._camera_combo.currentText()
        cameras = tuple(
            str(camera)
            for camera in self._mmc.getLoadedDevicesOfType(DeviceType.Camera)
            if not self._is_multi_camera_utility(str(camera))
        )
        preferred_camera = selected_camera or str(self._mmc.getCameraDevice())
        with signals_blocked(self._camera_combo):
            self._camera_combo.clear()
            self._camera_combo.addItems(cameras)
            if preferred_camera in cameras:
                self._camera_combo.setCurrentText(preferred_camera)

        selected_group = self._channel_group_combo.currentText()
        groups = tuple(str(group) for group in self._mmc.getAvailableConfigGroups())
        preferred_group = selected_group or str(self._mmc.getChannelGroup())
        with signals_blocked(self._channel_group_combo):
            self._channel_group_combo.clear()
            self._channel_group_combo.addItems(groups)
            if preferred_group in groups:
                self._channel_group_combo.setCurrentText(preferred_group)

        self._light_sources = self._find_light_sources()
        selected_source = self._light_source_combo.currentText()
        with signals_blocked(self._light_source_combo):
            self._light_source_combo.clear()
            self._light_source_combo.addItem("None")
            self._light_source_combo.addItems(self._light_sources)
            if selected_source in self._light_sources:
                self._light_source_combo.setCurrentText(selected_source)

        self._on_camera_changed()
        self._on_channel_group_changed(self._channel_group_combo.currentText())
        self._update_availability()

    def _is_multi_camera_utility(self, camera: str) -> bool:
        """Return whether `camera` is Micro-Manager's virtual Multi Camera."""
        try:
            library = str(self._mmc.getDeviceLibrary(camera)).casefold()
            adapter = str(self._mmc.getDeviceName(camera)).casefold()
            description = str(self._mmc.getDeviceDescription(camera)).casefold()
        except Exception:
            return False
        identity = f"{adapter} {description}".replace("-", " ").replace("_", " ")
        is_multi_camera = "multi camera" in identity or "multicamera" in identity
        return is_multi_camera and (library == "utilities" or "multi" in adapter)

    def _on_camera_changed(self, *_: object) -> None:
        camera = self._camera_combo.currentText()
        if camera:
            try:
                exposure = float(cast("Any", self._mmc).getExposure(camera))
            except Exception:
                exposure = float(self._mmc.getExposure())
            self._exposure.setValue(exposure)
        self._update_availability()

    def _channel_identity(self) -> tuple[str, str] | None:
        group = self._channel_group_combo.currentText()
        config = self._channel_combo.currentText()
        return (group, config) if group and config else None

    def _on_core_channel_group_changed(self, group: str, *_: object) -> None:
        if self._channel_group_combo.findText(str(group)) >= 0:
            self._channel_group_combo.setCurrentText(str(group))
        else:
            self.refreshHardware()

    def _on_channel_group_changed(self, group: str) -> None:
        previous_channel = self._channel_combo.currentText()
        configs = (
            tuple(str(config) for config in self._mmc.getAvailableConfigs(group))
            if group
            else ()
        )
        preferred = ""
        if group == self._channel_group_last and previous_channel in configs:
            preferred = previous_channel
        elif group:
            try:
                current = str(self._mmc.getCurrentConfig(group))
                preferred = current if current in configs else ""
            except Exception:
                pass
        with signals_blocked(self._channel_combo):
            self._channel_combo.clear()
            self._channel_combo.addItems(configs)
            if preferred:
                self._channel_combo.setCurrentText(preferred)
        self._channel_group_last = group
        self._on_channel_changed()

    def _on_channel_changed(self, *_: object) -> None:
        self._apply_declared_light_source()
        self._update_availability()

    def _find_light_sources(self) -> dict[str, tuple[tuple[str, str], ...]]:
        """Find writable ranged numeric properties usable as illumination."""
        properties = self._mmc.iterProperties(
            property_type=(PropertyType.Integer, PropertyType.Float),
            has_limits=True,
            is_read_only=False,
            as_object=False,
        )
        pairs = sorted(
            (
                (str(device), str(prop))
                for device, prop in properties
                if not self._mmc.isPropertyPreInit(device, prop)
            ),
            key=lambda pair: (pair[0].casefold(), pair[1].casefold()),
        )
        sources: dict[str, tuple[tuple[str, str], ...]] = {
            f"{device}{_LIGHT_SOURCE_SEPARATOR}{prop}": ((device, prop),)
            for device, prop in pairs
        }

        # A single-preset group whose entries are all ranged numeric properties
        # represents a multi-property source controlled by one intensity value.
        for group in self._mmc.getAvailableConfigGroups():
            presets = self._mmc.getAvailableConfigs(group)
            if len(presets) != 1:
                continue
            device_properties: list[tuple[str, str]] = []
            valid = True
            try:
                for device, prop, _ in self._mmc.getConfigData(group, presets[0]):
                    if (
                        self._mmc.getPropertyType(device, prop)
                        not in (PropertyType.Integer, PropertyType.Float)
                        or not self._mmc.hasPropertyLimits(device, prop)
                        or self._mmc.isPropertyReadOnly(device, prop)
                        or self._mmc.isPropertyPreInit(device, prop)
                    ):
                        valid = False
                        break
                    device_properties.append((str(device), str(prop)))
            except RuntimeError:
                valid = False
            if valid and device_properties and str(group) not in sources:
                sources[str(group)] = tuple(device_properties)
        return dict(sorted(sources.items(), key=lambda item: item[0].casefold()))

    def _apply_declared_light_source(self) -> None:
        identity = self._channel_identity()
        entries = (
            parse_light_source_comments(
                self._mmc.systemConfigurationFile() or "", identity[0]
            ).get(identity[1], [])
            if identity is not None
            else []
        )
        label = ""
        if entries:
            declared = frozenset((device, prop) for device, prop, _ in entries)
            label = next(
                (
                    source
                    for source, pairs in self._light_sources.items()
                    if frozenset(pairs) == declared
                ),
                "",
            )
        with signals_blocked(self._light_source_combo):
            self._light_source_combo.setCurrentText(label or "None")
        self._on_light_source_changed(label or "None")
        if entries and label:
            self._light_intensity.setValue(float(entries[0][2]))

    def _on_light_source_changed(self, source: str) -> None:
        pairs = self._light_sources.get(source, ())
        if not pairs:
            self._light_intensity.setEnabled(False)
            self._light_intensity.setRange(0, 0)
            self._light_intensity.setValue(0)
            self._update_availability()
            return
        lower = max(
            float(self._mmc.getPropertyLowerLimit(device, prop))
            for device, prop in pairs
        )
        upper = min(
            float(self._mmc.getPropertyUpperLimit(device, prop))
            for device, prop in pairs
        )
        is_integer = all(
            self._mmc.getPropertyType(device, prop) is PropertyType.Integer
            for device, prop in pairs
        )
        self._light_intensity.setDecimals(0 if is_integer else 3)
        self._light_intensity.setRange(lower, upper)
        self._light_intensity.setEnabled(lower <= upper)
        if lower <= upper:
            try:
                current = float(self._mmc.getProperty(*pairs[0]))
            except Exception:
                current = lower
            self._light_intensity.setValue(current)
        self._update_availability()

    def _unavailable_reason(self, *, allow_live: bool = False) -> str:
        if self._target is None or not self._target.resolution_id:
            return "Select one resolution ID"
        if not self._target.binding_is_saved:
            return (
                "Save this resolution ID and its identifying properties to core first"
            )
        if not self._camera_combo.currentText():
            return "Load and select a physical camera"
        if not self._xy_stage():
            return "Load and select an XY stage"
        if self._channel_identity() is None:
            return "Select a channel group and preset"
        try:
            if self._mmc.mda.is_running():
                return "Stop the running acquisition"
            if not allow_live and self._mmc.isSequenceRunning():
                return "Stop live mode"
        except Exception:
            pass
        return ""

    def _update_availability(self, *_: object) -> None:
        reason = self._unavailable_reason()
        action_reason = self._unavailable_reason(allow_live=True)
        idle = self._thread is None
        action_enabled = idle and not action_reason
        self._start_button.setEnabled(action_enabled)
        self._snap_button.setEnabled(action_enabled)
        self._live_button.setEnabled(action_enabled)
        if idle:
            if self._live_transaction is not None:
                self._phase_label.setText(
                    "Live preview active; choose the starting field of view"
                )
            else:
                self._phase_label.setText(reason or "Ready to preview or calibrate")

    def _set_inputs_enabled(self, enabled: bool) -> None:
        for widget in (
            self._camera_combo,
            self._xy_stage_combo,
            self._channel_group_combo,
            self._channel_combo,
            self._exposure,
            self._light_source_combo,
            self._safe_radius,
            self._settle_time,
            self._return_tolerance,
        ):
            widget.setEnabled(enabled)
        self._light_intensity.setEnabled(
            enabled and self._light_source_combo.currentText() in self._light_sources
        )

    def _capture_settings(self) -> CalibrationCaptureSettings:
        assert self._target is not None
        identity = self._channel_identity()
        if identity is None:
            raise RuntimeError("No calibration channel is selected")
        pairs = self._light_sources.get(self._light_source_combo.currentText(), ())
        intensity = float(self._light_intensity.value())
        properties = tuple((device, prop, intensity) for device, prop in pairs)
        return CalibrationCaptureSettings(
            resolution_settings=self._target.settings,
            channel_group=identity[0],
            channel_config=identity[1],
            exposure_ms=float(self._exposure.value()),
            camera=self._camera_combo.currentText(),
            light_properties=properties,
        )

    def _xy_stage(self) -> str:
        return self._xy_stage_combo.currentText()

    def _options(self) -> CalibrationOptions:
        return CalibrationOptions(
            safe_radius_um=self._safe_radius.value(),
            settle_time_s=self._settle_time.value(),
            stage_return_tolerance_um=self._return_tolerance.value(),
        )

    def snapFrame(self) -> None:
        """Snap with calibration capture settings, then restore hardware."""
        self._start_worker(preview_only=True)

    def startCalibration(self) -> None:
        """Start automatic calibration for the selected resolution."""
        self._start_worker(preview_only=False)

    def toggleLivePreview(self) -> None:
        """Start or stop live viewing with the selected calibration settings."""
        if self._mmc.isSequenceRunning():
            self.stopLivePreview()
            return
        reason = self._unavailable_reason(allow_live=True)
        if self._thread is not None or reason:
            self._live_button.setChecked(self._mmc.isSequenceRunning())
            if reason:
                self._phase_label.setText(reason)
            return
        assert self._target is not None
        preview = self._ensure_preview()
        transaction = CaptureStateTransaction(
            cast("Any", self._mmc),
            self._capture_settings(),
            resolution_id=self._target.resolution_id,
        )
        try:
            transaction.apply()
            self._live_transaction = transaction
            preview.attach(self._mmc)
            self._set_inputs_enabled(False)
            self.calibrationRunningChanged.emit(True)
            self._set_result_message(
                "Live preview uses the selected calibration settings. Move to the "
                "desired starting field of view, then stop Live or start calibration."
            )
            self._mmc.startContinuousSequenceAcquisition()
        except BaseException as exc:
            logger.exception("Could not start calibration live preview")
            self._live_transaction = None
            self._live_button.setChecked(False)
            if self._mmc.isSequenceRunning():
                self._mmc.stopSequenceAcquisition()
            if self._preview is not None:
                self._preview.detach()
            try:
                transaction.restore()
            except BaseException as restore_exc:
                logger.exception(
                    "Capture-state restoration failed after live-preview error"
                )
                exc = RuntimeError(
                    f"{_describe_error(exc)}; capture-state restoration also "
                    f"failed: {_describe_error(restore_exc)}"
                )
            self._set_inputs_enabled(True)
            self.calibrationRunningChanged.emit(False)
            self._update_availability()
            self._on_failure(_describe_error(exc))

    def stopLivePreview(self) -> None:
        """Stop live mode and restore capture state while retaining the stage FOV."""
        if self._mmc.isSequenceRunning():
            self._mmc.stopSequenceAcquisition()
        # Some cores emit sequenceAcquisitionStopped asynchronously. Finish here
        # as well so a following Snap/Start cannot race the restoration.
        if self._live_transaction is not None:
            self._finish_live_preview()

    def _finish_live_preview(self) -> None:
        transaction = self._live_transaction
        if transaction is None:
            return
        self._live_transaction = None
        if self._preview is not None:
            self._preview.detach()
        failure = ""
        try:
            transaction.restore()
        except BaseException as exc:
            logger.exception("Capture-state restoration failed after live preview")
            failure = _describe_error(exc)
        self._set_inputs_enabled(True)
        self.calibrationRunningChanged.emit(False)
        self._update_availability()
        if failure:
            self._on_failure(f"Capture-state restoration failed: {failure}")
        else:
            self._phase_label.setText(
                "Live preview stopped; starting field of view retained"
            )
            self._set_result_message(
                "Live preview stopped and optical settings restored. The selected "
                "XY stage position is the calibration starting field of view."
            )

    def _on_live_started(self, *_: object) -> None:
        if self._live_transaction is None:
            self._update_availability()
            return
        self._phase_label.setText(
            "Live preview active; choose the starting field of view"
        )
        try:
            x, y = self._mmc.getXYPosition(self._xy_stage())
            self._frame_label.setText(f"Live · XY {x:.3f}, {y:.3f} µm")
        except Exception:
            self._frame_label.setText("Live")
        self._update_availability()

    def _on_live_stopped(self, *_: object) -> None:
        if self._live_transaction is not None:
            self._finish_live_preview()
        else:
            self._update_availability()

    def _on_xy_stage_position_changed(self, device: str, x: float, y: float) -> None:
        if self._live_transaction is not None and device == self._xy_stage():
            self._frame_label.setText(f"Live · XY {x:.3f}, {y:.3f} µm")

    def _ensure_preview(self) -> NDVPreview:
        """Build the live-frame viewer on first use, replacing the placeholder."""
        if self._preview is None:
            preview = NDVPreview(
                self._mmc,
                self,
                viewer_options={
                    "show_3d_button": False,
                    "show_roi_button": False,
                    "show_channel_mode_selector": False,
                },
                show_save_button=False,
                show_roll_axes_button=False,
                show_colormap_selector=False,
            )
            # Snap/calibration frames arrive through the worker's queued
            # frameReady signal. Live preview temporarily attaches this viewer
            # to MMCore and detaches it before restoring capture state.
            preview.detach()
            preview.process_events_on_update = False
            preview.setMinimumSize(320, 260)
            self._viewer_layout.removeWidget(self._preview_placeholder)
            self._preview_placeholder.deleteLater()
            # Calibration is always interpreted in grayscale. Keep the useful
            # LUT range/auto controls while removing the colormap chooser.
            self._viewer_layout.addWidget(preview, 1)
            self._preview = preview
        return self._preview

    def _start_worker(self, *, preview_only: bool) -> None:
        reason = self._unavailable_reason(allow_live=True)
        if self._thread is not None or reason:
            if reason:
                self._phase_label.setText(reason)
            return
        self.stopLivePreview()
        reason = self._unavailable_reason()
        if reason:
            self._phase_label.setText(reason)
            return
        assert self._target is not None
        self._ensure_preview()
        self._preview_only = preview_only
        self._progress.setValue(0)
        self._diagnostics.setResult(None)
        self._set_result_message("Running…" if not preview_only else "Snapping…")
        self._phase_label.setText(
            "Acquiring preview frame" if preview_only else "Preparing capture settings"
        )
        self._cancel_event = Event()
        self._thread = QThread(self)
        self._worker = _CalibrationWorker(
            self._mmc,
            self._target,
            self._capture_settings(),
            self._options(),
            self._xy_stage(),
            self._cancel_event,
            preview_only=preview_only,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.frameReady.connect(self._on_frame)
        self._worker.resultReady.connect(self._on_result)
        self._worker.previewReady.connect(self._on_preview_ready)
        self._worker.failed.connect(self._on_failure)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.finished.connect(self._worker.deleteLater)
        # QThread lives in the GUI thread, so an automatic connection to quit()
        # would be queued there.  A direct connection lets shutdown() safely wait
        # for restoration without depending on the GUI event loop to quit the
        # worker thread first.
        cast("Any", self._worker.finished).connect(
            self._thread.quit, Qt.ConnectionType.DirectConnection
        )
        self._thread.finished.connect(self._on_thread_finished)
        self._cancel_button.setEnabled(not preview_only)
        self._cancel_button.setVisible(not preview_only)
        self._start_button.setEnabled(False)
        self._snap_button.setEnabled(False)
        self._live_button.setEnabled(False)
        self._set_inputs_enabled(False)
        self.calibrationRunningChanged.emit(True)
        self._thread.start()

    def cancelCalibration(self) -> None:
        """Request cancellation; the worker still restores stage and capture state."""
        if self._cancel_event is not None:
            self._cancel_event.set()
            self._cancel_button.setEnabled(False)
            self._phase_label.setText("Cancelling and restoring hardware…")

    def isRunning(self) -> bool:
        """Return whether calibration or calibration preview owns hardware."""
        worker_running = self._thread is not None and self._thread.isRunning()
        return worker_running or self._live_transaction is not None

    def shutdownCalibration(self) -> None:
        """Cancel and synchronously wait until hardware restoration is complete."""
        self.stopLivePreview()
        thread = self._thread
        if thread is None or not thread.isRunning():
            return
        self.cancelCalibration()
        # quit() is thread-safe.  The worker's synchronous run method continues
        # through both restoration paths before its thread event loop exits.
        thread.quit()
        thread.wait()

    def _on_progress(self, phase: str, fraction: float) -> None:
        self._phase_label.setText(PHASE_LABELS.get(phase, phase))
        self._progress.setValue(round(fraction * 1000))

    def _on_frame(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        image = np.asarray(payload["image"])
        self._ensure_preview().append(image)
        x, y = payload["position"]
        self._frame_label.setText(
            f"frame {int(payload['number'])} · XY {x:.3f}, {y:.3f} µm"
        )

    def _on_preview_ready(self) -> None:
        self._phase_label.setText("Frame snapped; hardware state restored")
        self._set_result_message(
            "Inspect image texture, focus, exposure, and saturation before starting."
        )

    def _on_result(self, payload: object) -> None:
        if not isinstance(payload, PixelCalibrationResult) or self._target is None:
            self._on_failure("Calibration returned an invalid result")
            return
        result = payload
        self._diagnostics.setResult(result)
        fit = result.fit
        warning_text = " · ".join(
            f"Warning: {warning.message}" for warning in result.warnings
        )
        validation_residuals = _validation_residuals_px(result)
        if validation_residuals.size:
            validation_summary = (
                "Independent validation passed: "
                f"RMS {float(np.sqrt(np.mean(validation_residuals**2))):.4f} px, "
                f"worst {float(np.max(validation_residuals)):.4f} px"
            )
        else:
            validation_summary = (
                f"Fit RMS/worst: {fit.rms_residual_px:.4f}/{fit.max_residual_px:.4f} px"
            )
        lines = [
            f"Pixel size: {fit.pixel_size_um:.8f} µm/px "
            f"(stored raw: {result.raw_pixel_size_um:.8f})",
            f"Rotation: {fit.rotation_deg:.3f}° · "
            f"{'mirrored' if fit.determinant < 0 else 'not mirrored'}",
            validation_summary,
        ]
        if warning_text:
            lines.append(warning_text)
        lines.append("")
        lines.append(
            f"Applied automatically to {self._target.resolution_id!r}. Use "
            '"Save to core" to update the current core instance, or '
            '"Save to file" to also save it to the .cfg file.'
        )
        self._set_result_message("\n".join(lines), preserve_newlines=True)
        self._phase_label.setText("Validated result applied; configuration is dirty")
        self._progress.setValue(1000)
        self.resultReady.emit(result, self._target.resolution_id)

    def _on_failure(self, message: str) -> None:
        """Show a clean, user-facing failure message.

        ``message`` is already a short human-readable sentence (see
        ``_describe_error``); the full traceback is logged, not shown here.
        """
        self._phase_label.setText("Calibration failed; hardware restoration attempted")
        self._set_result_message(message or "Unknown error")

    def _on_cancelled(self, message: str) -> None:
        """Report a user-requested cancellation distinctly from a failure."""
        self._phase_label.setText("Calibration cancelled; hardware restored")
        self._set_result_message(message or "Calibration cancelled")

    def _set_result_message(
        self, message: str, *, preserve_newlines: bool = False
    ) -> None:
        r"""Show result text, with the full text available on hover.

        Most messages are a single collapsed line (``\\n`` joined with
        " · ") to stay compact; a structured result (see ``_on_result``)
        instead keeps its line breaks so pixel size / rotation / the
        "applied to" note read as separate lines rather than one run-on
        sentence.
        """
        text = message if preserve_newlines else message.replace("\n", " · ")
        self._result_text.setText(text)
        self._result_text.setToolTip(text)

    def _on_thread_finished(self) -> None:
        terminal_message = self._phase_label.text()
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
        self._worker = None
        self._cancel_event = None
        self._cancel_button.setEnabled(False)
        self._cancel_button.hide()
        self._set_inputs_enabled(True)
        self.calibrationRunningChanged.emit(False)
        self._update_availability()
        # _update_availability normally owns the idle-state message.  Keep the
        # terminal result visible after a completed test, success, or failure.
        self._phase_label.setText(terminal_message)


__all__ = [
    "CalibrationDiagnosticsWidget",
    "CalibrationTarget",
    "PixelCalibrationPanel",
]
