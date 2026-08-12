"""Interactive example for the headless pixel-calibration routine.

Try the complete workflow without hardware first::

    uv run python example.py --demo

Run against a real Micro-Manager configuration::

    uv run python example.py \
        --config /path/to/MMConfig.cfg \
        --resolution-id 20X \
        --safe-radius-um 100

The calibration executes in a worker thread.  The window displays every snapped
image, stage position, calibration phase, affine predictions, and residuals.  It
does not write calibration data unless the separate Commit button is pressed and
confirmed.

Before using real hardware, choose a safe stage radius for the objective, sample,
and stage.  Make sure live mode and acquisitions are stopped and that a textured,
stationary specimen is in focus.
"""

# The demo core deliberately mirrors MMCore's camelCase API. Requiring a docstring on
# every one-line protocol shim would obscure the example's calibration workflow.
# ruff: noqa: D102, D103, D105

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from pymmcore_plus import CMMCorePlus

from pymmcore_gui._pixel_calibration import (
    CalibrationCommitError,
    CalibrationOptions,
    PixelCalibrationResult,
    commit_pixel_calibration,
    run_pixel_calibration,
)
from pymmcore_gui._qt.QtCore import QObject, QPointF, QRectF, QSize, Qt, QThread, Signal
from pymmcore_gui._qt.QtGui import (
    QCloseEvent,
    QColor,
    QImage,
    QPainter,
    QPaintEvent,
    QPen,
)
from pymmcore_gui._qt.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

PHASE_LABELS = {
    "reference": "Acquiring stable reference images",
    "probe-x": "Finding a useful X-stage displacement",
    "probe-y": "Finding a useful Y-stage displacement",
    "measure": "Acquiring compass measurements",
    "validate": "Acquiring independent holdout measurements",
    "restore": "Restoring the original stage position",
    "complete": "Calibration complete",
}


def _normalize_image(image: np.ndarray) -> QImage:
    """Convert an arbitrary camera image to an auto-leveled 8-bit QImage."""
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        array = array[..., :3]
        color = True
    elif array.ndim == 2:
        color = False
    else:
        raise ValueError(f"Cannot display camera image with shape {array.shape}")

    finite = np.isfinite(array)
    if not np.any(finite):
        display = np.zeros(array.shape, dtype=np.uint8)
    else:
        values = np.asarray(array[finite], dtype=np.float64)
        low, high = np.percentile(values, (0.5, 99.5))
        if high <= low:
            high = low + 1
        display = np.clip((array.astype(np.float64) - low) / (high - low), 0, 1)
        display = np.asarray(np.nan_to_num(display) * 255, dtype=np.uint8)
    display = np.ascontiguousarray(display)
    height, width = (int(display.shape[0]), int(display.shape[1]))
    image_bytes = display.tobytes()
    qimage: QImage
    if color:
        qimage = QImage(
            image_bytes,
            width,
            height,
            int(display.strides[0]),
            QImage.Format.Format_RGB888,
        )
    else:
        qimage = QImage(
            image_bytes,
            width,
            height,
            int(display.strides[0]),
            QImage.Format.Format_Grayscale8,
        )
    # Detach from the NumPy buffer, which belongs to the worker signal payload.
    return qimage.copy()


class ImagePanel(QWidget):
    """Small auto-scaling camera image widget with a title overlay."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self._image = QImage()
        self._details = "Waiting for an image"
        self.setMinimumSize(320, 240)

    def sizeHint(self) -> QSize:
        return QSize(480, 340)

    def set_array(self, image: np.ndarray, details: str) -> None:
        self._image = _normalize_image(image)
        self._details = details
        self.setToolTip(details)
        self.update()

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        del a0
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#15181d"))
        painter.setPen(QColor("#e6edf3"))
        painter.drawText(10, 21, self._title)
        painter.setPen(QColor("#9ba7b4"))
        painter.drawText(10, 40, self._details)
        target = QRectF(self.rect().adjusted(8, 48, -8, -8))
        if self._image.isNull():
            painter.drawText(target, Qt.AlignmentFlag.AlignCenter, "No image yet")
            return
        image_ratio = self._image.width() / self._image.height()
        target_ratio = target.width() / max(target.height(), 1)
        if image_ratio > target_ratio:
            height = target.width() / image_ratio
            target.setTop(target.center().y() - height / 2)
            target.setHeight(height)
        else:
            width = target.height() * image_ratio
            target.setLeft(target.center().x() - width / 2)
            target.setWidth(width)
        painter.drawImage(target, self._image)
        painter.setPen(QPen(QColor("#5d6b7a"), 1))
        painter.drawRect(target)
        painter.setPen(QPen(QColor(255, 255, 255, 90), 1))
        painter.drawLine(
            QPointF(target.center().x() - 8, target.center().y()),
            QPointF(target.center().x() + 8, target.center().y()),
        )
        painter.drawLine(
            QPointF(target.center().x(), target.center().y() - 8),
            QPointF(target.center().x(), target.center().y() + 8),
        )


class DiagnosticsPlot(QWidget):
    """Qt-painted measured/predicted displacement and residual plots."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: PixelCalibrationResult | None = None
        self.setMinimumHeight(240)

    def set_result(self, result: PixelCalibrationResult | None) -> None:
        self._result = result
        self.update()

    @staticmethod
    def _map_point(point: Sequence[float], rect: QRectF, extent: float) -> QPointF:
        return QPointF(
            rect.center().x() + float(point[0]) * rect.width() * 0.43 / extent,
            rect.center().y() - float(point[1]) * rect.height() * 0.43 / extent,
        )

    def _draw_vectors(self, painter: QPainter, rect: QRectF) -> None:
        assert self._result is not None
        observations = [obs for obs in self._result.observations if obs.accepted]
        measured = np.asarray([obs.stage_delta_um for obs in observations])
        shifts = np.asarray([obs.corrected_shift_xy for obs in observations])
        predicted = shifts @ self._result.fit.matrix.T
        extent = max(float(np.max(np.abs(measured))), 1e-9)

        painter.setPen(QPen(QColor("#52606d"), 1))
        painter.drawLine(
            QPointF(rect.left(), rect.center().y()),
            QPointF(rect.right(), rect.center().y()),
        )
        painter.drawLine(
            QPointF(rect.center().x(), rect.top()),
            QPointF(rect.center().x(), rect.bottom()),
        )
        origin = self._map_point((0, 0), rect, extent)
        for index, (actual, expected) in enumerate(
            zip(measured, predicted, strict=True), start=1
        ):
            actual_point = self._map_point(actual, rect, extent)
            expected_point = self._map_point(expected, rect, extent)
            painter.setPen(QPen(QColor("#58a6ff"), 1.5))
            painter.drawLine(origin, actual_point)
            painter.setBrush(QColor("#58a6ff"))
            painter.drawEllipse(actual_point, 4, 4)
            painter.setPen(QPen(QColor("#3fb950"), 2))
            painter.drawLine(
                QPointF(expected_point.x() - 5, expected_point.y() - 5),
                QPointF(expected_point.x() + 5, expected_point.y() + 5),
            )
            painter.drawLine(
                QPointF(expected_point.x() - 5, expected_point.y() + 5),
                QPointF(expected_point.x() + 5, expected_point.y() - 5),
            )
            painter.setPen(QColor("#c9d1d9"))
            painter.drawText(actual_point + QPointF(6, -4), str(index))
        painter.setPen(QColor("#9ba7b4"))
        painter.drawText(
            rect.adjusted(4, 4, -4, -4),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            "Stage Δ (µm):  ● measured   green cross = affine prediction",
        )
        painter.drawText(
            rect.adjusted(4, 4, -4, -4),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
            f"±{extent:.3g} µm",
        )

    def _draw_residuals(self, painter: QPainter, rect: QRectF) -> None:
        assert self._result is not None
        fit_residuals = np.linalg.norm(self._result.fit.residuals_px, axis=1)
        inverse = np.linalg.inv(self._result.fit.matrix)
        holdout_residuals = []
        for observation in self._result.validation_observations:
            shift = np.asarray(observation.corrected_shift_xy)
            delta = np.asarray(observation.stage_delta_um)
            error_um = delta - self._result.fit.matrix @ shift
            holdout_residuals.append(float(np.linalg.norm(inverse @ error_um)))
        residuals = np.concatenate((fit_residuals, holdout_residuals))
        colors = [QColor("#58a6ff")] * len(fit_residuals) + [QColor("#d29922")] * len(
            holdout_residuals
        )
        maximum = max(float(np.max(residuals)), 0.1)
        plot = rect.adjusted(8, 28, -8, -24)
        painter.setPen(QPen(QColor("#52606d"), 1))
        painter.drawLine(
            QPointF(plot.left(), plot.bottom()),
            QPointF(plot.right(), plot.bottom()),
        )
        slot = plot.width() / max(len(residuals), 1)
        for index, (value, color) in enumerate(zip(residuals, colors, strict=True)):
            height = float(value) / maximum * plot.height()
            bar = QRectF(
                plot.left() + index * slot + slot * 0.15,
                plot.bottom() - height,
                slot * 0.70,
                height,
            )
            painter.fillRect(bar, color)
            painter.setPen(QColor("#c9d1d9"))
            painter.drawText(
                QRectF(plot.left() + index * slot, plot.bottom() + 2, slot, 18),
                Qt.AlignmentFlag.AlignCenter,
                str(index + 1),
            )
        painter.setPen(QColor("#9ba7b4"))
        painter.drawText(
            rect.adjusted(4, 4, -4, -4),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            "Residual (px):  blue fit   amber holdout",
        )
        painter.drawText(
            rect.adjusted(4, 4, -4, -4),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
            f"max {float(np.max(residuals)):.3f} px",
        )

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        del a0
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#15181d"))
        if self._result is None:
            painter.setPen(QColor("#9ba7b4"))
            painter.drawText(
                QRectF(self.rect()),
                Qt.AlignmentFlag.AlignCenter,
                "Affine prediction and residual plots appear after calibration",
            )
            return
        outer = QRectF(self.rect()).adjusted(8, 8, -8, -8)
        gap = 16
        width = (outer.width() - gap) / 2
        left = QRectF(outer.left(), outer.top(), width, outer.height())
        right = QRectF(left.right() + gap, outer.top(), width, outer.height())
        self._draw_vectors(painter, left.adjusted(0, 24, 0, 0))
        self._draw_residuals(painter, right)


class FeedbackCore:
    """Forward MMCore calls while emitting copies of images for the GUI."""

    def __init__(self, core: Any, image_callback: Callable[[object], None]) -> None:
        self._core = core
        self._image_callback = image_callback
        self._frame_number = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._core, name)

    def getImage(self) -> np.ndarray:
        image = np.asarray(self._core.getImage()).copy()
        self._frame_number += 1
        try:
            stage = str(self._core.getXYStageDevice())
            position = tuple(float(v) for v in self._core.getXYPosition(stage))
        except Exception:
            position = None
        self._image_callback(
            {
                "image": image,
                "number": self._frame_number,
                "position": position,
            }
        )
        return image


class CalibrationWorker(QObject):
    """Run synchronous hardware operations away from the GUI thread."""

    progress = Signal(str, float)
    image_ready = Signal(object)
    succeeded = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        core: Any,
        options: CalibrationOptions,
        resolution_id: str | None,
        cancel_event: Event,
    ) -> None:
        super().__init__()
        self._core = core
        self._options = options
        self._resolution_id = resolution_id
        self._cancel_event = cancel_event

    def run(self) -> None:
        proxy = FeedbackCore(self._core, self.image_ready.emit)
        try:
            result = run_pixel_calibration(
                cast("Any", proxy),
                self._options,
                resolution_id=self._resolution_id,
                cancel_event=self._cancel_event,
                progress=self.progress.emit,
            )
        except BaseException:
            self.failed.emit(traceback.format_exc())
        else:
            self.succeeded.emit(result)
        finally:
            self.finished.emit()


class CalibrationWindow(QMainWindow):
    """Visual harness around the headless calibration and commit APIs."""

    def __init__(
        self,
        core: Any,
        options: CalibrationOptions,
        resolution_id: str | None,
        *,
        demo: bool,
        auto_start: bool,
    ) -> None:
        super().__init__()
        self._core = core
        self._options = options
        self._resolution_id = resolution_id
        self._demo = demo
        self._thread: QThread | None = None
        self._worker: CalibrationWorker | None = None
        self._cancel_event: Event | None = None
        self._result: PixelCalibrationResult | None = None
        self._reference_array: np.ndarray | None = None
        self._last_phase = ""

        self.setWindowTitle("Pixel Calibration Example")
        self.resize(1180, 900)
        central = QWidget(self)
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)

        self.info = QLabel(self._hardware_summary())
        self.info.setWordWrap(True)
        layout.addWidget(self.info)

        image_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.reference_image = ImagePanel("Reference frame")
        self.latest_image = ImagePanel("Latest frame")
        image_splitter.addWidget(self.reference_image)
        image_splitter.addWidget(self.latest_image)
        image_splitter.setSizes([580, 580])
        layout.addWidget(image_splitter, 4)

        self.plot = DiagnosticsPlot()
        layout.addWidget(self.plot, 3)

        status_row = QHBoxLayout()
        self.phase_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        status_row.addWidget(self.phase_label)
        status_row.addWidget(self.progress_bar, 1)
        layout.addLayout(status_row)

        button_row = QHBoxLayout()
        self.start_button = QPushButton("Start calibration")
        self.cancel_button = QPushButton("Cancel safely")
        self.commit_button = QPushButton("Commit to MMCore…")
        self.save_button = QPushButton("Save window image…")
        self.cancel_button.setEnabled(False)
        self.commit_button.setEnabled(False)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.cancel_button)
        button_row.addWidget(self.commit_button)
        button_row.addStretch(1)
        button_row.addWidget(self.save_button)
        layout.addLayout(button_row)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        self.log.setMaximumHeight(180)
        layout.addWidget(self.log)

        self.start_button.clicked.connect(self.start_calibration)
        self.cancel_button.clicked.connect(self.cancel_calibration)
        self.commit_button.clicked.connect(self.commit_result)
        self.save_button.clicked.connect(self.save_window_image)
        self._append(
            "No calibration values will be written during measurement. "
            "Commit is a separate confirmed action."
        )
        if auto_start:
            from pymmcore_gui._qt.QtCore import QTimer

            QTimer.singleShot(0, self.start_calibration)

    def _hardware_summary(self) -> str:
        try:
            camera = self._core.getCameraDevice() or "<none>"
            stage = self._core.getXYStageDevice() or "<none>"
            current = self._core.getCurrentPixelSizeConfig() or "<none>"
        except Exception as error:
            return f"Could not inspect hardware: {error}"
        mode = "synthetic demo" if self._demo else "real hardware"
        return (
            f"Mode: {mode}  |  Camera: {camera}  |  XY stage: {stage}  |  "
            f"Current pixel-size config: {current}  |  Commit target: "
            f"{self._resolution_id or '<disabled>'}  |  Safe radius: "
            f"{self._options.safe_radius_um:g} µm"
        )

    def _append(self, text: str) -> None:
        self.log.appendPlainText(text)

    def start_calibration(self) -> None:
        if self._thread is not None:
            return
        self._result = None
        self._reference_array = None
        self._last_phase = ""
        self.plot.set_result(None)
        self.progress_bar.setValue(0)
        self.phase_label.setText("Starting calibration")
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.commit_button.setEnabled(False)
        self._cancel_event = Event()
        self._thread = QThread(self)
        self._worker = CalibrationWorker(
            self._core,
            self._options,
            self._resolution_id,
            self._cancel_event,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.image_ready.connect(self._on_image)
        self._worker.succeeded.connect(self._on_result)
        self._worker.failed.connect(self._on_failure)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_finished)
        self._append("Calibration started.")
        self._thread.start()

    def cancel_calibration(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
            self.cancel_button.setEnabled(False)
            self.phase_label.setText("Cancellation requested; restoring stage…")
            self._append("Cancellation requested. Waiting for safe stage restoration.")

    def _on_progress(self, phase: str, fraction: float) -> None:
        self.progress_bar.setValue(round(fraction * 1000))
        self.phase_label.setText(PHASE_LABELS.get(phase, phase))
        if phase != self._last_phase:
            self._append(f"Phase: {PHASE_LABELS.get(phase, phase)}")
            self._last_phase = phase

    def _on_image(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        image = np.asarray(payload["image"])
        number = int(payload["number"])
        position = payload.get("position")
        if position is None:
            location = "stage position unavailable"
        else:
            location = f"XY=({position[0]:.3f}, {position[1]:.3f}) µm"
        details = (
            f"snap {number} · {image.shape} · {image.dtype} · {location} · "
            f"range [{float(np.min(image)):.3g}, {float(np.max(image)):.3g}]"
        )
        if self._reference_array is None:
            self._reference_array = image.copy()
            self.reference_image.set_array(image, details)
        self.latest_image.set_array(image, details)

    def _on_result(self, payload: object) -> None:
        if not isinstance(payload, PixelCalibrationResult):
            self._append("Worker returned an unexpected result type.")
            return
        self._result = payload
        self.plot.set_result(payload)
        fit = payload.fit
        matrix = fit.matrix
        inverse = np.linalg.inv(matrix)
        holdout_residuals = []
        for observation in payload.validation_observations:
            shift = np.asarray(observation.corrected_shift_xy)
            delta = np.asarray(observation.stage_delta_um)
            holdout_residuals.append(
                float(np.linalg.norm(inverse @ (delta - matrix @ shift)))
            )
        self._append("Calibration succeeded; MMCore has not been modified.")
        self._append(
            "Pixel→stage matrix (µm/px):\n"
            f"  [{matrix[0, 0]: .8f}  {matrix[0, 1]: .8f}]\n"
            f"  [{matrix[1, 0]: .8f}  {matrix[1, 1]: .8f}]"
        )
        self._append(
            f"Pixel size: {fit.pixel_size_um:.8f} µm/px; "
            f"rotation: {fit.rotation_deg:.3f}°; determinant: {fit.determinant:.8g}"
        )
        self._append(
            f"Fit RMS/max: {fit.rms_residual_px:.4f}/"
            f"{fit.max_residual_px:.4f} px; holdout max: "
            f"{max(holdout_residuals):.4f} px"
        )
        for warning in payload.warnings:
            self._append(f"Warning [{warning.code}]: {warning.message}")

    def _on_failure(self, details: str) -> None:
        self.phase_label.setText("Calibration failed")
        self._append(details.rstrip())
        summary = (
            details.rstrip().splitlines()[-1] if details.strip() else "Unknown error"
        )
        QMessageBox.critical(self, "Calibration failed", summary)

    def _on_thread_finished(self) -> None:
        assert self._thread is not None
        self._thread.deleteLater()
        self._thread = None
        self._worker = None
        self._cancel_event = None
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.commit_button.setEnabled(
            self._result is not None and self._resolution_id is not None
        )
        if self._result is not None:
            self.phase_label.setText("Validated result ready; not yet committed")

    def commit_result(self) -> None:
        if self._result is None or self._resolution_id is None:
            return
        resolution_id = self._resolution_id
        try:
            old_size = float(self._core.getPixelSizeUmByID(resolution_id))
        except Exception as error:
            QMessageBox.critical(self, "Cannot commit", str(error))
            return
        new_size = self._result.raw_pixel_size_um
        difference = (
            abs(new_size - old_size) / old_size if old_size > 0 else float("inf")
        )
        answer = QMessageBox.warning(
            self,
            "Commit pixel calibration?",
            f"Write the validated result to {resolution_id!r}?\n\n"
            f"Stored raw size: {old_size:.8g} µm/px\n"
            f"New raw size: {new_size:.8g} µm/px\n\n"
            "This changes the live MMCore state but does not save the .cfg file.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        allow_large_difference = False
        if difference > 0.10:
            answer = QMessageBox.warning(
                self,
                "Large calibration difference",
                f"The new value differs from the stored value by {difference:.1%}. "
                "Override the 10% commit guard?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            allow_large_difference = True
        try:
            commit_pixel_calibration(
                self._core,
                resolution_id,
                self._result,
                allow_large_difference=allow_large_difference,
            )
        except CalibrationCommitError as error:
            self._append(f"Commit failed: {error}")
            QMessageBox.critical(self, "Commit failed", str(error))
            return
        self._append(
            f"Committed calibration to {resolution_id!r} and verified readback."
        )
        self.commit_button.setEnabled(False)
        QMessageBox.information(
            self,
            "Calibration committed",
            "MMCore readback matched. Save the hardware configuration through the "
            "normal configuration workflow if this result should persist.",
        )

    def save_window_image(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save calibration diagnostics",
            "pixel-calibration.png",
            "PNG image (*.png)",
        )
        if path:
            if self.grab().save(path, "PNG"):
                self._append(f"Saved diagnostic window to {path}")
            else:
                QMessageBox.warning(self, "Save failed", f"Could not write {path}")

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        if a0 is None:
            return
        if self._thread is not None:
            self.cancel_calibration()
            QMessageBox.information(
                self,
                "Calibration is still stopping",
                "Cancellation was requested. Keep this window open until the stage "
                "has been restored.",
            )
            a0.ignore()
            return
        a0.accept()


class _DemoSetting:
    def getDeviceLabel(self) -> str:
        return "Objective"

    def getPropertyName(self) -> str:
        return "Label"

    def getPropertyValue(self) -> str:
        return "Demo-20X"


class _DemoConfiguration:
    def size(self) -> int:
        return 1

    def getSetting(self, index: int) -> _DemoSetting:
        if index != 0:
            raise IndexError(index)
        return _DemoSetting()


class _DemoMDA:
    def is_running(self) -> bool:
        return False


def _demo_texture(shape: tuple[int, int] = (320, 420)) -> np.ndarray:
    rng = np.random.default_rng(42)
    image = rng.normal(size=shape)
    for _ in range(5):
        image = (
            image
            + np.roll(image, 1, 0)
            + np.roll(image, -1, 0)
            + np.roll(image, 1, 1)
            + np.roll(image, -1, 1)
        ) / 5
    yy, xx = np.indices(shape)
    for _ in range(35):
        x = rng.uniform(0, shape[1])
        y = rng.uniform(0, shape[0])
        sigma = rng.uniform(2, 11)
        amplitude = rng.uniform(0.5, 2.5)
        image += amplitude * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    image -= np.min(image)
    image /= np.max(image)
    return 500 + 60_000 * image


def _fourier_shift(image: np.ndarray, shift_rc: Sequence[float]) -> np.ndarray:
    row_frequency = np.fft.fftfreq(image.shape[0])[:, None]
    column_frequency = np.fft.fftfreq(image.shape[1])[None, :]
    phase = np.exp(
        -2j
        * np.pi
        * (row_frequency * float(shift_rc[0]) + column_frequency * float(shift_rc[1]))
    )
    return np.asarray(np.real(np.fft.ifftn(np.fft.fftn(image) * phase)))


class SyntheticCalibrationCore:
    """Deterministic camera/stage model used by ``--demo``."""

    def __init__(self) -> None:
        angle = np.deg2rad(23)
        self.matrix = np.asarray(
            [
                [0.42 * np.cos(angle), 0.40 * np.sin(angle)],
                [0.42 * np.sin(angle), -0.40 * np.cos(angle)],
            ]
        )
        self.image = _demo_texture()
        self.origin = np.asarray((2500.0, 1400.0))
        self.position = self.origin.copy()
        self.mda = _DemoMDA()
        self.pixel_size = float(np.sqrt(abs(np.linalg.det(self.matrix))))
        self.affine: tuple[float, ...] = (
            float(self.matrix[0, 0]),
            float(self.matrix[0, 1]),
            0.0,
            float(self.matrix[1, 0]),
            float(self.matrix[1, 1]),
            0.0,
        )

    def getCameraDevice(self) -> str:
        return "SyntheticCamera"

    def getXYStageDevice(self) -> str:
        return "SyntheticXY"

    def getBinning(self, label: str) -> int:
        return 1

    def getMagnificationFactor(self) -> float:
        return 1.0

    def getROI(self, label: str) -> tuple[int, int, int, int]:
        return (0, 0, self.image.shape[1], self.image.shape[0])

    def getImageWidth(self) -> int:
        return int(self.image.shape[1])

    def getImageHeight(self) -> int:
        return int(self.image.shape[0])

    def getNumberOfCameraChannels(self) -> int:
        return 1

    def getCurrentPixelSizeConfig(self) -> str:
        return "Demo-20X"

    def getPixelSizeConfigData(self, config_name: str) -> _DemoConfiguration:
        if config_name != "Demo-20X":
            raise KeyError(config_name)
        return _DemoConfiguration()

    def getProperty(self, device: str, prop: str) -> str:
        if (device, prop) != ("Objective", "Label"):
            raise KeyError((device, prop))
        return "Demo-20X"

    def getXYPosition(self, label: str) -> tuple[float, float]:
        return (float(self.position[0]), float(self.position[1]))

    def setXYPosition(self, label: str, x: float, y: float) -> None:
        self.position[:] = (x, y)

    def waitForDevice(self, label: str) -> None:
        return None

    def isSequenceRunning(self, label: str) -> bool:
        return False

    def snapImage(self) -> None:
        return None

    def getImage(self) -> np.ndarray:
        stage_delta = self.position - self.origin
        pixel_shift = np.linalg.solve(self.matrix, stage_delta)
        apparent_shift = (-pixel_shift[1], -pixel_shift[0])
        shifted = _fourier_shift(self.image, apparent_shift)
        return np.asarray(np.clip(shifted, 0, 65535), dtype=np.uint16)

    def getPixelSizeUm(self) -> float:
        return self.pixel_size

    def getAvailablePixelSizeConfigs(self) -> tuple[str, ...]:
        return ("Demo-20X",)

    def getPixelSizeUmByID(self, resolution_id: str) -> float:
        return self.pixel_size

    def getPixelSizeAffineByID(self, resolution_id: str) -> tuple[float, ...]:
        return self.affine

    def setPixelSizeUm(self, resolution_id: str, value: float) -> None:
        self.pixel_size = value

    def setPixelSizeAffine(self, resolution_id: str, value: tuple[float, ...]) -> None:
        self.affine = value

    def getPixelSizeAffine(self) -> tuple[float, ...]:
        return self.affine


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visually exercise the headless pixel-calibration routine."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use a deterministic synthetic camera and XY stage.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Micro-Manager .cfg file. Required unless --demo is used.",
    )
    parser.add_argument(
        "--resolution-id",
        help="Existing pixel-size configuration to validate and optionally update. "
        "Defaults to MMCore's current matching configuration.",
    )
    parser.add_argument(
        "--safe-radius-um",
        type=float,
        default=100.0,
        help=(
            "Maximum radial stage displacement from the starting point (default: 100)."
        ),
    )
    parser.add_argument(
        "--settle-time-s",
        type=float,
        default=0.1,
        help="Delay after each completed stage move (default: 0.1).",
    )
    parser.add_argument(
        "--upsample-factor",
        type=int,
        default=20,
        help="Subpixel registration upsampling factor (default: 20).",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Start when the window opens instead of waiting for the button.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.demo and args.config is not None:
        parser.error("--demo and --config are mutually exclusive")
    if not args.demo and args.config is None:
        parser.error("--config is required for real hardware; use --demo otherwise")

    if args.demo:
        core: Any = SyntheticCalibrationCore()
    else:
        assert args.config is not None
        config = args.config.expanduser().resolve()
        if not config.is_file():
            parser.error(f"configuration file does not exist: {config}")
        core = CMMCorePlus()
        try:
            core.loadSystemConfiguration(str(config))
        except Exception as error:
            print(f"Failed to load {config}: {error}", file=sys.stderr)
            return 2

    resolution_id = args.resolution_id
    if resolution_id is None:
        try:
            resolution_id = str(core.getCurrentPixelSizeConfig()) or None
        except Exception:
            resolution_id = None
    options = CalibrationOptions(
        safe_radius_um=args.safe_radius_um,
        settle_time_s=args.settle_time_s,
        upsample_factor=args.upsample_factor,
    )

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Pixel Calibration Example")
    window = CalibrationWindow(
        core,
        options,
        resolution_id,
        demo=args.demo,
        auto_start=args.auto_start,
    )
    window.show()
    exit_code = app.exec()
    if not args.demo:
        try:
            core.unloadAllDevices()
        except Exception as error:
            print(f"Failed to unload devices cleanly: {error}", file=sys.stderr)
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
