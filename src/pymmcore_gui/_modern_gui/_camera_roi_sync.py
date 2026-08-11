"""Synchronize the embedded camera ROI editor with ndv viewers."""

from __future__ import annotations

import math
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pymmcore_gui._qt.QtCore import QObject, Signal

if TYPE_CHECKING:
    from collections.abc import Callable

    from pymmcore_plus import CMMCorePlus
    from pymmcore_widgets import CameraRoiValue

    from pymmcore_gui._array_viewer import MMArrayViewer
    from pymmcore_gui.widgets._mda_widget import MemoryMDAWidget
    from pymmcore_gui.widgets.image_preview._ndv_preview import NDVPreview

    from ._acquire_toolbar import LiveButton
    from ._acquire_viewers import AcquireViewersManager


@dataclass
class _ViewerObservation:
    """Callbacks and camera-coordinate context for one ndv viewer."""

    mode_callback: Callable[..., None]
    camera: str | None = None
    hardware_roi: tuple[int, int, int, int] | None = None
    dynamic_hardware_roi: bool = False
    roi_model: Any = None
    bbox_callback: Callable[..., None] | None = None
    last_bbox: tuple[tuple[float, float], tuple[float, float]] | None = None


class CameraRoiSyncController(QObject):
    """Observe viewer ROIs and mediate one full-frame live editing session.

    Camera ROI values use absolute camera coordinates.  The live preview uses the
    coordinates of whatever hardware crop is currently active, so entering a
    session briefly stops live mode, restores full chip, and then restarts live.
    The planned ROI is preserved in the MDA editor and is only committed by Crop
    or MDA preflight.
    """

    _mdaStarted = Signal()

    def __init__(
        self,
        core: CMMCorePlus,
        mda: MemoryMDAWidget,
        viewers: AcquireViewersManager,
        live_button: LiveButton,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._core = core
        self._mda = mda
        self._viewers = viewers
        self._live_button = live_button
        self._viewer: MMArrayViewer | None = None
        self._observed_viewers: dict[MMArrayViewer, _ViewerObservation] = {}
        self._active = False
        self._transitioning = False
        self._syncing = False
        self._connected = True
        self._restart_live_after_crop = False

        roi = mda.camera_roi
        roi.setRoiSelectionAvailable(True)
        mda.roiSelectionRequested.connect(self.set_active)
        roi.roiChanged.connect(self._on_widget_roi_changed)
        # Hide ndv's ROI visual before CameraRoiWidget's clicked handler changes
        # hardware and causes the live preview to rebuild its data model.
        roi.crop_btn.pressed.connect(self._on_crop_pressed)
        roi.crop_btn.clicked.connect(self._on_crop_committed)

        viewers.previewCreated.connect(self._on_preview_created)
        viewers.previewClosed.connect(self._on_preview_closed)
        viewers.mdaViewerCreated.connect(self._on_mda_viewer_created)
        viewers.mdaViewerClosed.connect(self._on_mda_viewer_closed)
        if viewers.preview is not None:
            self._attach_preview(viewers.preview)

        core.events.sequenceAcquisitionStopped.connect(self._on_live_stopped)
        core.events.systemConfigurationLoaded.connect(self._on_configuration_changed)
        core.events.propertyChanged.connect(self._on_property_changed)
        self._mda_started_callback = lambda *_args: self._mdaStarted.emit()
        core.mda.events.sequenceStarted.connect(self._mda_started_callback)
        self._mdaStarted.connect(self.stop)
        self.destroyed.connect(self._disconnect)

    @property
    def active(self) -> bool:
        """Whether a live full-frame ROI editing session is active."""
        return self._active

    def set_active(self, active: bool) -> None:
        """Start or stop the shared MDA/ndv ROI selection session."""
        if not active:
            self.stop()
            return
        if self._active:
            if self._viewer is not None:
                self._viewer.set_existing_roi_editing_active(True)
            return
        if self._core.mda.is_running():
            self._mda.camera_roi.setLiveSelectionActive(False)
            return

        editor = self._mda.camera_roi
        planned = editor.roiValue()

        self._transitioning = True
        try:
            # Hardware ROI changes require a fresh stream/buffer. Preserve whether
            # live was already running by always returning to live at the end.
            if self._core.isSequenceRunning():
                self._core.stopSequenceAcquisition()
            editor.applyFullFrame()

            # roiSet reflects the temporary full frame into the editor. Restore the
            # plan without touching hardware. Starting a selection session means
            # this ROI should participate in the next MDA run.
            editor.setRoiValue(planned)
            self._mda.tabs.roi_section.set_checked(True)

            preview = self._viewers.ensure_preview()
            self._attach_preview(preview)
            self._active = True
            self._set_viewer_roi(planned)
            if self._viewer is not None:
                self._viewer.set_existing_roi_editing_active(True)
            editor.setLiveSelectionActive(True)
            self._live_button.ensure_live()
        except Exception:
            self._active = False
            editor.setLiveSelectionActive(False)
            raise
        finally:
            self._transitioning = False

    def stop(self, *_args: object) -> None:
        """Leave ROI interaction mode without changing the planned/hardware ROI."""
        was_active = self._active
        if not was_active:
            return
        self._active = False
        self._transitioning = True
        try:
            self._disconnect_viewer_roi_model(self._viewer)
            if self._viewer is not None:
                if self._viewer.roi is not None:
                    # Removing the visual avoids leaving ndv/vispy with an
                    # uninitialized hidden marker while hardware ROI changes
                    # rebuild the preview's data model.
                    self._viewer.clear_roi()
                self._viewer.set_existing_roi_editing_active(False)
            self._mda.camera_roi.setLiveSelectionActive(False)
            if was_active and self._core.isSequenceRunning():
                self._core.stopSequenceAcquisition()
        finally:
            self._transitioning = False

    def _on_preview_created(self, preview: NDVPreview) -> None:
        self._attach_preview(preview)

    def _on_mda_viewer_created(self, viewer: MMArrayViewer) -> None:
        """Observe ROI drawings in an acquired-data viewer."""
        camera = self._core.getCameraDevice()
        hardware_roi: tuple[int, int, int, int] | None = None
        if camera:
            with suppress(RuntimeError):
                x, y, width, height = self._core.getROI(camera)
                hardware_roi = x, y, width, height
        self._observe_viewer(
            viewer,
            camera=camera or None,
            hardware_roi=hardware_roi,
            dynamic_hardware_roi=False,
        )

    def _on_mda_viewer_closed(self, viewer: MMArrayViewer) -> None:
        self._unobserve_viewer(viewer)

    def _on_crop_pressed(self) -> None:
        # Stop the stream before CameraRoiWidget's clicked handler changes the
        # hardware ROI. The clicked callback below resumes it on the new shape.
        self._restart_live_after_crop = self._core.isSequenceRunning()
        self.stop()
        if self._core.isSequenceRunning():
            self._core.stopSequenceAcquisition()

    def _on_crop_committed(self) -> None:
        restart = self._restart_live_after_crop
        self._restart_live_after_crop = False
        if restart and not self._core.mda.is_running():
            self._live_button.ensure_live()

    def _on_preview_closed(self) -> None:
        self.stop()
        self._detach_preview()

    def _attach_preview(self, preview: NDVPreview) -> None:
        viewer = getattr(preview, "viewer", None)
        if viewer is None:
            return
        if viewer is self._viewer:
            return
        self._detach_preview()
        self._viewer = viewer
        self._observe_viewer(viewer, dynamic_hardware_roi=True)

    def _detach_preview(self) -> None:
        if self._viewer is not None:
            self._unobserve_viewer(self._viewer)
        self._viewer = None

    def _observe_viewer(
        self,
        viewer: MMArrayViewer,
        *,
        camera: str | None = None,
        hardware_roi: tuple[int, int, int, int] | None = None,
        dynamic_hardware_roi: bool = False,
    ) -> None:
        if viewer in self._observed_viewers:
            return
        connect_mode = getattr(viewer, "connect_roi_selection_changed", None)
        if not callable(connect_mode) or not hasattr(viewer, "roi"):
            return

        def _mode_changed(*_args: object) -> None:
            self._on_ndv_mode_changed(viewer)

        self._observed_viewers[viewer] = _ViewerObservation(
            mode_callback=_mode_changed,
            camera=camera,
            hardware_roi=hardware_roi,
            dynamic_hardware_roi=dynamic_hardware_roi,
        )
        connect_mode(_mode_changed)
        self._connect_viewer_roi_model(viewer)

    def _unobserve_viewer(self, viewer: MMArrayViewer) -> None:
        observation = self._observed_viewers.get(viewer)
        if observation is None:
            return
        self._disconnect_viewer_roi_model(viewer)
        with suppress(Exception):
            viewer.disconnect_roi_selection_changed(observation.mode_callback)
        self._observed_viewers.pop(viewer, None)

    def _connect_viewer_roi_model(self, viewer: MMArrayViewer) -> None:
        """Observe the current ndv ROI without changing ndv interaction state."""
        observation = self._observed_viewers.get(viewer)
        if observation is None:
            return
        model = viewer.roi
        if model is observation.roi_model:
            return
        self._disconnect_viewer_roi_model(viewer)
        observation.roi_model = model
        observation.last_bbox = model.bounding_box if model is not None else None
        if model is not None:

            def _bbox_changed(
                bbox: tuple[tuple[float, float], tuple[float, float]],
            ) -> None:
                self._on_viewer_roi_changed(viewer, bbox)

            observation.bbox_callback = _bbox_changed
            model.events.bounding_box.connect(_bbox_changed)

    def _set_viewer_roi(self, value: CameraRoiValue) -> None:
        if self._viewer is None:
            return
        bbox = _roi_bbox(value)
        if self._viewer.roi is None:
            self._viewer.roi = bbox
        else:
            self._viewer.roi.bounding_box = bbox
            self._viewer.roi.visible = True
        self._connect_viewer_roi_model(self._viewer)
        self._viewer.set_roi_visual_selected(self._active)

    def _disconnect_viewer_roi_model(self, viewer: MMArrayViewer | None) -> None:
        observation = self._observed_viewers.get(viewer) if viewer is not None else None
        if observation is None:
            return
        if observation.roi_model is not None and observation.bbox_callback is not None:
            with suppress(Exception):
                observation.roi_model.events.bounding_box.disconnect(
                    observation.bbox_callback
                )
        observation.roi_model = None
        observation.bbox_callback = None
        observation.last_bbox = None

    def _on_widget_roi_changed(
        self, x: int, y: int, width: int, height: int, mode: str
    ) -> None:
        if not self._active or self._syncing:
            return
        if mode == "Full Chip":
            self.stop()
            return
        value: CameraRoiValue = {
            "camera": self._mda.camera_roi.camera,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        }
        self._syncing = True
        try:
            self._set_viewer_roi(value)
        finally:
            self._syncing = False

    def _on_viewer_roi_changed(
        self,
        viewer: MMArrayViewer,
        bbox: tuple[tuple[float, float], tuple[float, float]],
    ) -> None:
        observation = self._observed_viewers.get(viewer)
        if observation is None:
            return
        previous_bbox = observation.last_bbox
        observation.last_bbox = bbox
        if self._syncing:
            return
        editor = self._mda.camera_roi
        camera = (
            self._core.getCameraDevice()
            if observation.dynamic_hardware_roi
            else observation.camera
        )
        if not camera:
            return
        hardware_roi = observation.hardware_roi
        if observation.dynamic_hardware_roi:
            try:
                x, y, width, height = self._core.getROI(camera)
                hardware_roi = x, y, width, height
            except RuntimeError:
                return
        if hardware_roi is None:
            return
        offset_x, offset_y, image_width, image_height = hardware_roi
        if image_width <= 0 or image_height <= 0:
            return
        (left, top), (right, bottom) = bbox
        current = editor.roiValue()
        same_camera = current["camera"] == camera
        preserve_width = previous_bbox is not None and math.isclose(
            right - left,
            previous_bbox[1][0] - previous_bbox[0][0],
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
        preserve_height = previous_bbox is not None and math.isclose(
            bottom - top,
            previous_bbox[1][1] - previous_bbox[0][1],
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
        x0, x1 = _normalize_viewer_interval(
            left,
            right,
            image_width,
            current["width"] if same_camera and preserve_width else None,
        )
        y0, y1 = _normalize_viewer_interval(
            top,
            bottom,
            image_height,
            current["height"] if same_camera and preserve_height else None,
        )
        value: CameraRoiValue = {
            "camera": camera,
            "x": offset_x + x0,
            "y": offset_y + y0,
            "width": x1 - x0,
            "height": y1 - y0,
        }

        self._syncing = True
        try:
            try:
                editor.setRoiValue(value)
            except ValueError:
                # The camera used by an older acquisition may have since been
                # removed by a configuration change.
                return
            # An ndv-drawn rectangle is always an explicitly custom plan, even
            # when its dimensions happen to match a camera preset.  This only
            # updates the editor; opting the ROI into MDA remains the user's
            # choice via the section checkbox.
            editor.camera_roi_combo.setCurrentText("Custom ROI")
        finally:
            self._syncing = False

    def _on_ndv_mode_changed(self, viewer: MMArrayViewer) -> None:
        """Attach to an ROI created by ndv without controlling ndv or live mode."""
        self._connect_viewer_roi_model(viewer)

    def _on_live_stopped(self, *_args: object) -> None:
        if self._active and not self._transitioning:
            self.stop()

    def _on_configuration_changed(self, *_args: object) -> None:
        if self._active:
            self.stop()

    def _on_property_changed(self, device: str, prop: str, _value: object) -> None:
        if self._active and device == "Core" and prop == "Camera":
            self.stop()

    def _disconnect(self, *_args: object) -> None:
        if not self._connected:
            return
        self._connected = False
        self.stop()
        self._detach_preview()
        for viewer in tuple(self._observed_viewers):
            self._unobserve_viewer(viewer)
        self._mda.camera_roi.setRoiSelectionAvailable(False)
        connections = (
            (self._mda.roiSelectionRequested, self.set_active),
            (self._mda.camera_roi.roiChanged, self._on_widget_roi_changed),
            (self._mda.camera_roi.crop_btn.pressed, self._on_crop_pressed),
            (self._mda.camera_roi.crop_btn.clicked, self._on_crop_committed),
            (self._viewers.previewCreated, self._on_preview_created),
            (self._viewers.previewClosed, self._on_preview_closed),
            (self._viewers.mdaViewerCreated, self._on_mda_viewer_created),
            (self._viewers.mdaViewerClosed, self._on_mda_viewer_closed),
            (self._core.events.sequenceAcquisitionStopped, self._on_live_stopped),
            (
                self._core.events.systemConfigurationLoaded,
                self._on_configuration_changed,
            ),
            (self._core.events.propertyChanged, self._on_property_changed),
            (
                self._core.mda.events.sequenceStarted,
                self._mda_started_callback,
            ),
        )
        for signal, callback in connections:
            with suppress(Exception):
                signal.disconnect(callback)


def _roi_tuple(value: CameraRoiValue) -> tuple[int, int, int, int]:
    return value["x"], value["y"], value["width"], value["height"]


def _roi_bbox(
    value: CameraRoiValue,
) -> tuple[tuple[int, int], tuple[int, int]]:
    x, y, width, height = _roi_tuple(value)
    return (x, y), (x + width, y + height)


def _normalize_viewer_interval(
    start: float,
    end: float,
    limit: int,
    current_size: int | None,
) -> tuple[int, int]:
    """Convert one viewer interval to pixels without growing translations.

    ``current_size`` is provided when successive raw viewer extents are equal,
    meaning this dimension is being translated rather than resized. Keep that
    integer size and clamp the interval as a unit. A genuinely changed extent
    continues to use enclosing floor/ceil pixel bounds.
    """
    if current_size is not None:
        size = min(current_size, limit)
        normalized_start = max(0, min(math.floor(start), limit - size))
        return normalized_start, normalized_start + size

    normalized_start = max(0, min(math.floor(start), limit - 1))
    normalized_end = max(normalized_start + 1, min(math.ceil(end), limit))
    return normalized_start, normalized_end


__all__ = ["CameraRoiSyncController"]
