"""Tabbed image viewers for the Acquire page."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pymmcore_gui._array_viewer import MMArrayViewer
from pymmcore_gui._ndv_viewers import _add_follow_lock_button, _extract_scales
from pymmcore_gui._qt.QtCore import QObject, QTimer, Signal
from pymmcore_gui._qt.QtWidgets import QTabBar, QTabWidget, QWidget
from pymmcore_gui.widgets.image_preview._ndv_preview import NDVPreview

from ._tab_bar import ThemedTabBar

if TYPE_CHECKING:
    from collections.abc import Callable

    import ndv
    import numpy as np
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.metadata import FrameMetaV1, SummaryMetaV1
    from useq import MDAEvent, MDASequence


class _StreamSignalBridge(QObject):
    """Marshal ome-writers stream notifications onto the Qt GUI thread."""

    dimsChanged = Signal()


@dataclass
class _ViewerRecord:
    viewer: ndv.ArrayViewer
    bridge: _StreamSignalBridge | None = None
    coords_signal: Any = None
    coords_callback: Callable[[], None] | None = None

    def disconnect(self) -> None:
        """Disconnect the live stream from a viewer that is being closed."""
        if self.coords_signal is not None and self.coords_callback is not None:
            with suppress(Exception):
                self.coords_signal.disconnect(self.coords_callback)
        self.coords_signal = None
        self.coords_callback = None


class AcquireViewers(QTabWidget):
    """Preview plus one sink-backed viewer tab for each MDA run."""

    _sequenceStarted = Signal(object, object)
    _frameReady = Signal(object, object, object)
    _sequenceFinished = Signal(object)

    def __init__(
        self, mmcore: CMMCorePlus, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._core = mmcore
        self._records: dict[QWidget, _ViewerRecord] = {}
        self._active_viewer: ndv.ArrayViewer | None = None
        self._active_widget: QWidget | None = None
        self._follow_acquisition = True
        self._connected = True

        self.setTabBar(ThemedTabBar(self))
        self.setDocumentMode(True)
        self.setTabsClosable(True)
        self.tabCloseRequested.connect(self._close_tab)

        self.preview = NDVPreview(mmcore=self._core, parent=self)
        preview_idx = self.addTab(self.preview, "Preview")
        # The preview is a permanent workspace; only completed/active MDA tabs close.
        if tab_bar := self.tabBar():
            tab_bar.setTabButton(
                preview_idx, QTabBar.ButtonPosition.RightSide, None
            )

        # pymmcore-plus MDA events may be emitted by the acquisition thread.
        # Re-emitting through QObject signals guarantees that all QWidget and ndv
        # mutations below happen on this object's GUI thread.
        self._sequenceStarted.connect(self._on_sequence_started)
        self._frameReady.connect(self._on_frame_ready)
        self._sequenceFinished.connect(self._on_sequence_finished)
        self._sequence_started_callback = self._sequenceStarted.emit
        self._frame_ready_callback = self._frameReady.emit
        self._sequence_finished_callback = self._sequenceFinished.emit

        events = self._core.mda.events
        events.sequenceStarted.connect(self._sequence_started_callback)
        events.frameReady.connect(self._frame_ready_callback)
        events.sequenceFinished.connect(self._sequence_finished_callback)
        self.destroyed.connect(self._disconnect)

    @property
    def active_viewer(self) -> ndv.ArrayViewer | None:
        """Return the viewer following the current MDA, if any."""
        return self._active_viewer

    def _on_sequence_started(
        self, sequence: MDASequence, meta: SummaryMetaV1
    ) -> None:
        """Create a viewer backed by the acquisition's live sink view."""
        self._active_viewer = None
        self._active_widget = None
        view = self._core.mda.get_view()
        if view is None:
            # Runs without a path, AcquisitionSettings, or "memory" output have
            # no sink to display.  The embedded MDA widget prevents this case by
            # supplying "memory" whenever file saving is disabled.
            return

        viewer = MMArrayViewer(view, scales=_extract_scales(sequence, meta))
        widget = viewer.widget()
        widget.setObjectName(f"ndv-{str(sequence.uid)[:8]}")

        record = _ViewerRecord(viewer)
        wrapper = viewer.data_wrapper
        coords_signal = getattr(view, "coords_changed", None)
        if coords_signal is not None and wrapper is not None:
            bridge = _StreamSignalBridge(widget)
            bridge.dimsChanged.connect(wrapper.dims_changed.emit)
            callback = bridge.dimsChanged.emit
            coords_signal.connect(callback)
            record.bridge = bridge
            record.coords_signal = coords_signal
            record.coords_callback = callback

        self._follow_acquisition = True
        with suppress(Exception):
            _add_follow_lock_button(viewer, self)
        self._records[widget] = record
        self._active_viewer = viewer
        self._active_widget = widget
        idx = self.addTab(widget, f"MDA {str(sequence.uid)[:8]}")
        self.setCurrentIndex(idx)

    def _on_frame_ready(
        self, frame: np.ndarray, event: MDAEvent, meta: FrameMetaV1
    ) -> None:
        """Follow the latest acquired index and redraw once its write settles."""
        viewer = self._active_viewer
        if viewer is None or not self._follow_acquisition:
            return

        current_index = viewer.display_model.current_index
        wrapper = viewer.data_wrapper
        index = {
            ("p" if str(axis) == "g" else str(axis)): value
            for axis, value in event.index.items()
        }

        def _update() -> None:
            try:
                current_index.update(index.items())
                if wrapper is not None:
                    wrapper.data_changed.emit()
            except Exception:  # viewer may have closed during the async write
                pass

        # Sink writes may complete asynchronously after frameReady.
        QTimer.singleShot(10, _update)

    def _on_sequence_finished(self, sequence: MDASequence) -> None:
        """Stop treating the most recent viewer as an active acquisition."""

    def _close_tab(self, index: int) -> None:
        widget = self.widget(index)
        if widget is None or widget is self.preview:
            return

        record = self._records.pop(widget, None)
        if record is not None:
            record.disconnect()
        if widget is self._active_widget:
            self._active_widget = None
            self._active_viewer = None

        self.removeTab(index)
        if record is not None:
            with suppress(Exception):
                record.viewer.close()
        widget.deleteLater()

    def _disconnect(self, obj: QObject | None = None) -> None:
        if not self._connected:
            return
        self._connected = False
        events = self._core.mda.events
        with suppress(Exception):
            events.sequenceStarted.disconnect(self._sequence_started_callback)
        with suppress(Exception):
            events.frameReady.disconnect(self._frame_ready_callback)
        with suppress(Exception):
            events.sequenceFinished.disconnect(self._sequence_finished_callback)
        for record in self._records.values():
            record.disconnect()
        self._records.clear()
        self._active_viewer = None
        self._active_widget = None
        self.preview.detach()
