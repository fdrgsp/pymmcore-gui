"""Dockable image viewers for the Acquire page."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pymmcore_plus.mda import OmeWritersSink, frame_meta_to_ome

from pymmcore_gui._array_viewer import MMArrayViewer
from pymmcore_gui._mda_export import AcquisitionRecord
from pymmcore_gui._ndv_viewers import _add_follow_lock_button, _extract_scales
from pymmcore_gui._qt.QtAds import CDockWidget
from pymmcore_gui._qt.QtCore import QObject, QTimer, Signal
from pymmcore_gui.widgets.image_preview._ndv_preview import NDVPreview

if TYPE_CHECKING:
    from collections.abc import Callable

    import ndv
    import numpy as np
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.metadata import FrameMetaV1, SummaryMetaV1
    from useq import MDAEvent, MDASequence

    from pymmcore_gui._qt.QtAds import CDockAreaWidget, CDockManager
    from pymmcore_gui._qt.QtWidgets import QWidget


class _StreamSignalBridge(QObject):
    """Marshal ome-writers stream notifications onto the Qt GUI thread."""

    dimsChanged = Signal()


@dataclass
class _ViewerRecord:
    viewer: ndv.ArrayViewer
    bridge: _StreamSignalBridge | None = None
    coords_signal: Any = None
    coords_callback: Callable[[], None] | None = None
    acquisition: AcquisitionRecord | None = None

    def disconnect(self) -> None:
        """Disconnect the live stream from a viewer that is being closed."""
        if self.coords_signal is not None and self.coords_callback is not None:
            with suppress(Exception):
                self.coords_signal.disconnect(self.coords_callback)
        self.coords_signal = None
        self.coords_callback = None


class AcquireViewersManager(QObject):
    """Lazy snap preview plus one dock-tabbed viewer for each MDA run.

    Every Preview/MDA-viewer instance is wrapped in its own ``CDockWidget`` and
    tabbed into the shared ``central_dock_area`` via ``addDockWidgetTabToArea``,
    mirroring the classic GUI's ``NDVViewersManager`` docking mechanics.

    Closed viewer docks use ADS's ``DockWidgetDeleteOnClose`` feature so a
    closed viewer's dock-area/splitter node is actually removed (freeing its
    Qt widget/canvas resources via the normal parent-child cascade) rather
    than left behind as a permanently-empty, still-space-occupying shell --
    otherwise splitting several viewers side by side and closing some of them
    leaves unreclaimable dead space that the remaining ones can't expand
    into. Destroying a dock area makes ADS recompute splitter proportions
    for the *whole* manager, which used to also resize unrelated docks (e.g.
    the MDA panel) just because a viewer tab was closed -- that's now
    prevented at the source (see ``AcquirePage._install_width_lock``, which
    hard-locks the MDA/right columns against any relayout regardless of
    cause), so it's safe to let ADS actually reclaim the space here.
    """

    _sequenceStarted = Signal(object, object)
    _frameReady = Signal(object, object, object)
    _sequenceFinished = Signal(object)
    previewCreated = Signal(object)
    previewClosed = Signal()
    mdaViewerCreated = Signal(object)
    mdaViewerClosed = Signal(object)

    def __init__(
        self,
        dock_manager: CDockManager,
        central_dock_area: CDockAreaWidget,
        mmcore: CMMCorePlus,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._parent_widget = parent
        self._dock_manager = dock_manager
        self._central_dock_area = central_dock_area
        self._core = mmcore
        self._records: dict[CDockWidget, _ViewerRecord] = {}
        self._active_viewer: ndv.ArrayViewer | None = None
        self._active_dock: CDockWidget | None = None
        self._follow_acquisition = True
        self._connected = True

        self.preview: NDVPreview | None = None
        self._preview_dock: CDockWidget | None = None

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

    def set_central_dock_area(self, area: CDockAreaWidget) -> None:
        """Re-point at the central area after a layout restore rebuilt it.

        ``CDockManager.restoreState()`` tears down and rebuilds the entire
        dock-area tree, so the area captured at construction time becomes a
        dangling wrapper around a destroyed C++ object -- without this, the
        first viewer opened after a restore would tab into that deleted area.
        """
        self._central_dock_area = area

    def _new_dock(self, title: str) -> CDockWidget:
        """Create a dock widget for a viewer/preview, tabbed into the central area."""
        dw = CDockWidget(self._dock_manager, title, self._parent_widget)
        dw.setFeature(CDockWidget.DockWidgetFeature.DockWidgetFloatable, False)
        dw.setFeature(CDockWidget.DockWidgetFeature.DockWidgetDeleteOnClose, True)
        self._dock_manager.addDockWidgetTabToArea(dw, self._central_dock_area)
        return dw

    def ensure_preview(self) -> NDVPreview:
        """Create and select the snap preview if it is not already open."""
        if self.preview is None:
            preview = self.preview = NDVPreview(
                mmcore=self._core, parent=self._parent_widget
            )
            dw = self._new_dock("Preview")
            dw.setWidget(preview, CDockWidget.eInsertMode.ForceNoScrollArea)
            dw.closed.connect(self._on_preview_closed)
            self._preview_dock = dw
            self.previewCreated.emit(preview)
        assert self._preview_dock is not None
        self._preview_dock.setAsCurrentTab()
        assert self.preview is not None
        return self.preview

    def _on_preview_closed(self) -> None:
        if (preview := self.preview) is not None:
            preview.detach()
        self.preview = None
        self._preview_dock = None
        self.previewClosed.emit()

    @property
    def active_viewer(self) -> ndv.ArrayViewer | None:
        """Return the viewer following the current MDA, if any."""
        return self._active_viewer

    def _on_sequence_started(self, sequence: MDASequence, meta: SummaryMetaV1) -> None:
        """Create a viewer backed by the acquisition's live sink view."""
        self._active_viewer = None
        self._active_dock = None
        view = self._core.mda.get_view()
        if view is None:
            # Runs without a path, AcquisitionSettings, or "memory" output have
            # no sink to display.  The embedded MDA widget prevents this case by
            # supplying "memory" whenever file saving is disabled.
            return

        viewer = MMArrayViewer(view, scales=_extract_scales(sequence, meta))
        widget = viewer.widget()
        sha = str(sequence.uid)[:8]
        widget.setObjectName(f"ndv-{sha}")

        record = _ViewerRecord(viewer)
        # Snapshot the sink's resolved settings + summary metadata now: the
        # sink is replaced wholesale on the *next* run, so a viewer left open
        # across two acquisitions must hold its own copy to export correctly
        # later. Per-frame metadata is appended live, in _on_frame_ready.
        sink = self._core.mda.get_sink()
        if isinstance(sink, OmeWritersSink):
            acquisition = AcquisitionRecord(
                settings=sink.settings, summary_meta=sink.summary_meta, view=view
            )
            record.acquisition = acquisition
            viewer._acquisition_record = acquisition  # read by MMArrayViewer._save_data
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

        dw = self._new_dock(f"MDA {sha}")
        dw.setWidget(widget, CDockWidget.eInsertMode.ForceNoScrollArea)
        dw.closed.connect(lambda: self._on_viewer_closed(dw))
        dw.setAsCurrentTab()

        self._records[dw] = record
        self._active_viewer = viewer
        self._active_dock = dw
        self.mdaViewerCreated.emit(viewer)

    def _on_frame_ready(
        self, frame: np.ndarray, event: MDAEvent, meta: FrameMetaV1
    ) -> None:
        """Record frame metadata for export, then follow the latest acquired index.

        Metadata capture happens unconditionally, *before* the follow-lock
        check below: the lock only controls whether the displayed slider
        position tracks new frames, and must not also silently truncate the
        metadata used later by the viewer's Save button.
        """
        if (dw := self._active_dock) is not None:
            record = self._records.get(dw)
            if record is not None and record.acquisition is not None:
                record.acquisition.frame_meta.append(frame_meta_to_ome(meta))

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

    def _on_viewer_closed(self, dw: CDockWidget) -> None:
        record = self._records.pop(dw, None)
        if record is not None:
            self.mdaViewerClosed.emit(record.viewer)
            record.disconnect()
        if dw is self._active_dock:
            self._active_dock = None
            self._active_viewer = None
        if record is not None:
            with suppress(Exception):
                record.viewer.close()

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
            self.mdaViewerClosed.emit(record.viewer)
            record.disconnect()
        self._records.clear()
        self._active_viewer = None
        self._active_dock = None
        if self.preview is not None:
            self.preview.detach()
            self.preview = None
