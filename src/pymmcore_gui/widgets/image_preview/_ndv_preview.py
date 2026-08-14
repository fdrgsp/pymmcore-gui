from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ndv.models
from ndv.models import RingBuffer
from superqt.cmap import QColormapComboBox

from pymmcore_gui._array_viewer import MMArrayViewer
from pymmcore_gui._qt.QtCore import QTimer
from pymmcore_gui._qt.QtWidgets import QApplication, QVBoxLayout, QWidget
from pymmcore_gui.widgets.image_preview._preview_base import ImagePreviewBase

if TYPE_CHECKING:
    import numpy as np
    import rendercanvas.qt
    from pymmcore_plus import CMMCorePlus

    class QRenderWidget(rendercanvas.qt.QRenderWidget, QWidget): ...  # pyright: ignore [reportIncompatibleMethodOverride]


# Live preview only needs the most recent frame.
BUFFER_SIZE = 1


class NDVPreview(ImagePreviewBase):
    def __init__(
        self,
        mmcore: CMMCorePlus,
        parent: QWidget | None = None,
        *,
        use_with_mda: bool = False,
        viewer_options: dict[str, Any] | None = None,
        show_save_button: bool = True,
        show_roll_axes_button: bool = True,
        show_colormap_selector: bool = True,
    ):
        super().__init__(parent, mmcore, use_with_mda=use_with_mda)
        px = (self._mmc.getPixelSizeUm() or None) if self._mmc else None
        self._viewer = MMArrayViewer(
            scales=({"x": px, "y": px} if px else {}),
            viewer_options=viewer_options,
            show_save_button=show_save_button,
            show_roll_axes_button=show_roll_axes_button,
        )
        self._viewer_options = dict(viewer_options or {})
        self._show_colormap_selector = show_colormap_selector
        self._buffer: RingBuffer | None = None
        self._buffer_applied = False
        self._core_dtype: tuple[str, tuple[int, ...]] | None = None
        self._is_rgb = False
        self.process_events_on_update = True
        qwdg = self._viewer.widget()
        qwdg.setParent(self)
        self._apply_control_visibility()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(qwdg)

    @property
    def viewer(self) -> MMArrayViewer:
        """Return the embedded viewer controller."""
        return self._viewer

    def append(self, data: np.ndarray) -> None:
        incoming_dtype_shape = (data.dtype.name, tuple(data.shape))
        if self._buffer is None or self._core_dtype != incoming_dtype_shape:
            # Auto Snap may be emitted synchronously from an earlier roiSet
            # listener, before this preview receives its own roiSet callback.
            # Trust the actual frame shape instead of appending it to a stale
            # pre-crop buffer.
            self._init_buffer(incoming_dtype_shape)
            self._buffer_applied = False
        if self._buffer is not None:
            self._buffer.append(data)
            if not self._buffer_applied:
                self._apply_viewer_settings()
                # Replacing the image with a differently shaped camera ROI does
                # not change ndv's dimensionality, so ndv preserves the old
                # camera range. Fit after the populated image has been rendered.
                QTimer.singleShot(0, self._viewer.reset_zoom)
            self._viewer.display_model.current_index.update({0: len(self._buffer) - 1})
            self._viewer.data_wrapper.data_changed.emit()
            if self.process_events_on_update:
                QApplication.processEvents()

    @property
    def dtype_shape(self) -> tuple[str, tuple[int, ...]] | None:
        return self._core_dtype

    def _get_core_dtype_shape(self) -> tuple[str, tuple[int, ...]] | None:
        if (core := self._mmc) is not None:
            if bits := core.getImageBitDepth():
                img_width = core.getImageWidth()
                img_height = core.getImageHeight()
                if core.getNumberOfComponents() > 1:
                    shape: tuple[int, ...] = (img_height, img_width, 3)
                else:
                    shape = (img_height, img_width)
                # coerce packed bits to byte-aligned numpy dtype
                # (this is how the data will actually come from pymmcore)
                if bits <= 8:
                    bits = 8
                elif bits <= 16:
                    bits = 16
                elif bits <= 32:
                    bits = 32
                return (f"uint{bits}", shape)
        return None

    def _init_buffer(
        self, core_dtype: tuple[str, tuple[int, ...]] | None = None
    ) -> None:
        """Create a single-frame buffer without assigning it to the viewer."""
        if core_dtype is None and (core_dtype := self._get_core_dtype_shape()) is None:
            return  # pragma: no cover

        self._core_dtype = core_dtype
        self._is_rgb = core_dtype[1][-1] == 3
        self._buffer = RingBuffer(max_capacity=BUFFER_SIZE, dtype=core_dtype)

    def _apply_viewer_settings(self) -> None:
        """Assign the buffer and configure grayscale or RGB display."""
        self._viewer.data = self._buffer
        self._buffer_applied = True
        self._viewer.display_model.visible_axes = (1, 2)
        if self._is_rgb:
            self._viewer.display_model.channel_axis = 3
            self._viewer.display_model.channel_mode = ndv.models.ChannelMode.RGBA
        else:
            self._viewer.display_model.channel_mode = ndv.models.ChannelMode.GRAYSCALE
            self._viewer.display_model.channel_axis = None
        self._apply_control_visibility()

    def _apply_control_visibility(self) -> None:
        """Apply initial NDV control options and optional colormap hiding.

        Some NDV Qt controls currently only react when their option changes after
        construction, rather than honoring the model's initial value. Applying
        the same options to the concrete controls keeps embedded previews
        deterministic while retaining the public viewer-options model.
        """
        widget = self._viewer.widget()
        by_option = {
            "show_3d_button": "ndims_btn",
            "show_roi_button": "add_roi_btn",
            "show_channel_mode_selector": "channel_mode_combo",
            "show_reset_zoom_button": "set_range_btn",
        }
        for option, attribute in by_option.items():
            if option in self._viewer_options:
                visible = bool(self._viewer_options[option])
                getattr(widget, attribute).setVisible(visible)
        if not self._show_colormap_selector:
            for combo in widget.findChildren(QColormapComboBox):
                combo.hide()

    def _setup_viewer(self) -> None:
        """Prepare a buffer that is swapped in after its first frame arrives.

        Assigning an empty buffer makes ndv auto-fit a canvas containing no image.
        With a rectangular ROI visual present, Vispy then attempts to measure its
        still-uninitialized marker handles and raises from ``Markers.bounds``.
        """
        core_dtype = self._get_core_dtype_shape()
        if core_dtype is None:
            return
        # An Auto Snap emitted synchronously during roiSet may already have
        # installed a populated buffer for this exact new shape. Do not replace
        # it with a second, empty buffer when roiSet propagation resumes.
        if self._buffer_applied and self._core_dtype == core_dtype:
            return
        self._init_buffer(core_dtype)
        self._buffer_applied = False

    def _update_pixel_scales(self) -> None:
        if self._mmc and (px := self._mmc.getPixelSizeUm()):
            self._viewer.display_model.scales.update({"x": px, "y": px})

    def _on_system_config_loaded(self) -> None:
        self._setup_viewer()
        self._update_pixel_scales()

    def _on_roi_set(self) -> None:
        """Reconfigure the viewer when a Camera ROI is set."""
        self._setup_viewer()
        self._update_pixel_scales()
