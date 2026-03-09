from __future__ import annotations

from typing import TYPE_CHECKING

import ndv
import numpy as np

from pymmcore_gui._qt.QtWidgets import QApplication, QVBoxLayout, QWidget
from pymmcore_gui.widgets.image_preview._preview_base import ImagePreviewBase

if TYPE_CHECKING:
    import rendercanvas.qt
    from pymmcore_plus import CMMCorePlus

    class QRenderWidget(rendercanvas.qt.QRenderWidget, QWidget): ...  # pyright: ignore


class NDVPreviewNoBuffer(ImagePreviewBase):
    def __init__(
        self,
        mmcore: CMMCorePlus,
        parent: QWidget | None = None,
        *,
        use_with_mda: bool = False,
    ):
        super().__init__(parent, mmcore, use_with_mda=use_with_mda)
        self._viewer = ndv.ArrayViewer()
        self.process_events_on_update = True
        qwdg = self._viewer.widget()
        qwdg.setParent(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(qwdg)

    def append(self, data: np.ndarray) -> None:
        if self._viewer.data is None:
            self._setup_viewer()
        self._viewer.data = data
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

    def _setup_viewer(self) -> None:
        # TODO: maybe we should let this continue if core_dtype != self.dtype_shape
        if (core_dtype := self._get_core_dtype_shape()) is None:
            return  # pragma: no cover

        self._core_dtype = core_dtype
        dtype_str, shape = core_dtype
        self._viewer.data = np.zeros(shape, dtype=dtype_str)
        self._viewer.display_model.visible_axes = (0, 1)
        if core_dtype[1][-1] == 3:  # RGB
            self._viewer.display_model.channel_axis = 2
            self._viewer.display_model.channel_mode = ndv.models.ChannelMode.RGBA
        else:
            self._viewer.display_model.channel_mode = ndv.models.ChannelMode.GRAYSCALE
            self._viewer.display_model.channel_axis = None

    def _on_system_config_loaded(self) -> None:
        self._viewer.data = None
        self._setup_viewer()

    def _on_roi_set(self) -> None:
        """Reconfigure the viewer when a Camera ROI is set."""
        self._viewer.data = None
        self._setup_viewer()
