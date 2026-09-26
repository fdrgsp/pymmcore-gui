"""Dockable image viewers for the Acquire page."""

from __future__ import annotations

import gc
import hashlib
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ome_writers import ScratchFormat
from pymmcore_plus.mda import OmeWritersSink, frame_meta_to_ome

from pymmcore_gui._acquisition_loader import open_acquisition as _open_acquisition
from pymmcore_gui._array_viewer import MMArrayViewer
from pymmcore_gui._grid_axis import (
    GridAxisDataWrapper,
    GridAxisLayout,
    GridAxisLayoutKind,
)
from pymmcore_gui._mda_export import AcquisitionRecord
from pymmcore_gui._ndv_viewers import (
    _add_follow_lock_button,
    _DimsChangeGate,
    _extract_scales,
    _follow_index,
    _LiveRefresh,
    _RaggedFallbackCounter,
    _runner_sink,
    _StreamSignalBridge,
)
from pymmcore_gui._qt.QtAds import CDockWidget, DockWidgetArea
from pymmcore_gui._qt.QtCore import QObject, QTimer, Signal
from pymmcore_gui._qt.QtWidgets import QSplitter
from pymmcore_gui.widgets.image_preview._ndv_preview import NDVPreview

if TYPE_CHECKING:
    from collections.abc import Callable

    import ndv
    import numpy as np
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.mda import SinkProtocol
    from pymmcore_plus.metadata import FrameMetaV1, SummaryMetaV1
    from useq import MDAEvent, MDASequence

    from pymmcore_gui._qt.QtAds import CDockAreaWidget, CDockManager
    from pymmcore_gui._qt.QtWidgets import QWidget


def _release_runner_sink(runner: Any, sink: SinkProtocol) -> bool:
    """Release ``sink``, with the same safeguards as newer pymmcore-plus."""
    if callable(release_sink := getattr(runner, "release_sink", None)):
        return bool(release_sink(sink))
    # Compatibility for released pymmcore-plus versions that predate the
    # public method. Never mutate a running acquisition or a newer run's sink.
    if runner.is_running() or _runner_sink(runner) is not sink:
        return False
    runner._sink = None
    return True


@dataclass
class _ViewerRecord:
    viewer: ndv.ArrayViewer
    bridge: _StreamSignalBridge | None = None
    coords_signal: Any = None
    coords_callback: Callable[[], None] | None = None
    acquisition: AcquisitionRecord | None = None
    # The exact sink object behind this viewer's data, captured at
    # sequenceStarted -- see AcquireViewersManager._on_viewer_closed.
    sink: SinkProtocol | None = None
    # Planned p/g -> flattened-position mapping for this viewer, and the
    # coalesced-refresh state that follows it.
    layout: GridAxisLayout = field(default_factory=GridAxisLayout.none)
    refresh: _LiveRefresh | None = None
    pending_index: dict[str, int] | None = None
    dims_gate: _DimsChangeGate | None = None
    ragged_fallback: _RaggedFallbackCounter = field(
        default_factory=_RaggedFallbackCounter
    )
    # True only for a viewer created by _on_sequence_started (a live MDA
    # run). Gates whether mdaViewerCreated/mdaViewerClosed fire for it --
    # those drive CameraRoiSyncController's live-camera ROI observation,
    # which a reopened acquisition (arbitrary old pixel data, nothing to do
    # with the microscope's *current* camera/ROI) must never enter.
    is_live: bool = False
    # Releases a reopened acquisition's file handle(s)/zarr store. None for
    # a live viewer, whose data is owned by the runner/sink instead.
    loader_cleanup: Callable[[], None] | None = None

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
    tabbed into a dedicated nested ``CDockManager``, mirroring the classic
    GUI's ``NDVViewersManager`` docking mechanics.

    Closed viewer docks use ADS's ``DockWidgetDeleteOnClose`` feature so a
    closed viewer's dock-area/splitter node is actually removed (freeing its
    Qt widget/canvas resources via the normal parent-child cascade) rather
    than left behind as a permanently-empty, still-space-occupying shell --
    otherwise splitting several viewers side by side and closing some of them
    leaves unreclaimable dead space that the remaining ones can't expand
    into. The supplied dock manager is dedicated to viewers and nested inside
    the outer MDA/tools manager. Destroying or splitting a viewer area can
    therefore relayout only the viewer workspace, never the surrounding tool
    panels.
    """

    _sequenceStarted = Signal(object, object)
    _frameReady = Signal(object, object, object)
    _sequenceFinished = Signal(object)
    previewCreated = Signal(object)
    previewClosed = Signal()
    mdaViewerCreated = Signal(object)
    mdaViewerClosed = Signal(object)
    # (MDASequence, source_title) -- emitted when the user picks "Re-use
    # MDA…" on either a live MDA viewer or a reopened acquisition's viewer.
    reuseMDARequested = Signal(object, str)

    def __init__(
        self,
        dock_manager: CDockManager,
        mmcore: CMMCorePlus,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._parent_widget = parent
        self._dock_manager = dock_manager
        self._core = mmcore
        self._records: dict[CDockWidget, _ViewerRecord] = {}
        self._active_viewer: ndv.ArrayViewer | None = None
        self._active_dock: CDockWidget | None = None
        self._follow_acquisition = True
        self._connected = True
        # Set when a still-running run's viewer is closed: release_sink()
        # refuses to drop a sink while it's being written to, so the release
        # is retried once sequenceFinished confirms the run is done.
        self._pending_release: SinkProtocol | None = None

        self.preview: NDVPreview | None = None
        self._preview_dock: CDockWidget | None = None

        # PyQt6Ads 4.4 does not reliably honor EqualSplitOnInsertion when an
        # existing tab is dragged out into a new viewer area. Normalize only
        # the newly-created inner splitter after ADS finishes the relocation;
        # later user resizing of its handle remains untouched.
        self._dock_manager.dockAreaCreated.connect(self._equalize_new_split)

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

    def _new_dock(self, title: str) -> CDockWidget:
        """Create a viewer dock, tabbed with an existing viewer when possible."""
        dw = CDockWidget(self._dock_manager, title, self._parent_widget)
        dw.setFeature(CDockWidget.DockWidgetFeature.DockWidgetFloatable, False)
        dw.setFeature(CDockWidget.DockWidgetFeature.DockWidgetDeleteOnClose, True)
        if (target := self._viewer_target_area()) is None:
            self._dock_manager.addDockWidget(DockWidgetArea.CenterDockWidgetArea, dw)
        else:
            self._dock_manager.addDockWidgetTabToArea(dw, target)
        return dw

    @staticmethod
    def _disk_backed_title(sink: Any) -> str | None:
        """Return a real disk-backed sink's destination filename, else None.

        A memory/"scratch"-backed run (Saving unchecked in the MDA editor)
        has no meaningful destination to show -- `ScratchFormat.output_path`
        is just an identity/spill path, not something the user chose to
        save to.
        """
        if not isinstance(sink, OmeWritersSink):
            return None
        with suppress(Exception):
            settings = sink.settings
            if not isinstance(settings.format, ScratchFormat):
                return Path(settings.output_path).name
        return None

    @staticmethod
    def _rename_viewer_tab(dw: CDockWidget, viewer: MMArrayViewer, path: str) -> None:
        """Rename a viewer's dock/tab (and its `source_title`) to `path`'s filename.

        Used both when a live run's viewer is later saved and when a
        reopened acquisition's viewer is re-exported (e.g. as a different
        format) -- either way, the tab should reflect where the data now
        actually lives rather than its original generic/source label.
        """
        title = Path(path).name
        with suppress(RuntimeError):  # the dock may have been closed meanwhile
            dw.setWindowTitle(title)
        viewer.source_title = title

    def _viewer_target_area(self) -> CDockAreaWidget | None:
        """Return a visible viewer area to receive a newly-created viewer."""
        with suppress(RuntimeError):
            focused = self._dock_manager.focusedDockWidget()
            if focused is not None and (area := focused.dockAreaWidget()) is not None:
                return area
            for dock in self._dock_manager.openedDockWidgets():
                area = dock.dockAreaWidget()
                if area is not None and area.width() > 0:
                    return area
        return None

    def _equalize_new_split(self, area: CDockAreaWidget) -> None:
        """Share a newly-created viewer split equally between its siblings."""
        QTimer.singleShot(0, lambda: self._equalize_area_splitter(area))

    @staticmethod
    def _equalize_area_splitter(area: CDockAreaWidget) -> None:
        with suppress(RuntimeError):
            splitter = area.parentWidget()
            if isinstance(splitter, QSplitter) and splitter.count() > 1:
                splitter.setSizes([1] * splitter.count())

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

    def open_acquisition(self, path: str | Path) -> ndv.ArrayViewer:
        """Load a previously-acquired OME-TIFF/OME-Zarr dataset into a new tab.

        Unlike a live MDA run's viewer, the resulting viewer never becomes
        `active_viewer`/`_active_dock` (those track *only* the run currently
        being followed) and never fires `mdaViewerCreated` -- it isn't backed
        by the microscope's current camera/ROI, so it must stay invisible to
        `CameraRoiSyncController`'s live-camera ROI observation.

        Raises
        ------
        ValueError
            If `path` isn't a supported/openable acquisition -- propagated
            from `_acquisition_loader.open_acquisition` for the caller (e.g.
            a drag-and-drop handler) to report without crashing.
        """
        loaded = _open_acquisition(path)

        viewer = MMArrayViewer(loaded.wrapper)
        widget = viewer.widget()
        # Keyed to the resolved path (not a random id), so re-dropping the
        # exact same file is idempotent about naming; two different files
        # sharing a basename still get distinct object names.
        digest = hashlib.sha1(str(loaded.source_path).encode()).hexdigest()[:8]
        widget.setObjectName(f"ndv-loaded-{digest}")

        viewer.mda_sequence = loaded.sequence
        viewer.source_title = loaded.title
        if loaded.sequence is not None:
            sequence, title = loaded.sequence, loaded.title
            viewer._reuse_mda_callback = lambda: self.reuseMDARequested.emit(
                sequence, title
            )
        # So the Save button re-exports this acquisition's *real* recovered
        # metadata (dimensions, physical scale, channel names, summary
        # metadata) instead of falling back to stamping the microscope's
        # current state -- see MMArrayViewer._save_data.
        viewer._acquisition_record = loaded.record

        record = _ViewerRecord(viewer, loader_cleanup=loaded.close)

        dw = self._new_dock(loaded.title)
        dw.setWidget(widget, CDockWidget.eInsertMode.ForceNoScrollArea)
        dw.closed.connect(lambda: self._on_viewer_closed(dw))
        dw.setAsCurrentTab()
        viewer._on_saved = lambda path: self._rename_viewer_tab(dw, viewer, path)

        self._records[dw] = record
        return viewer

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

        # Snapshot the sink's resolved settings + summary metadata now: the
        # sink is replaced wholesale on the *next* run, so a viewer left open
        # across two acquisitions must hold its own copy to export correctly
        # later. Per-frame metadata is appended live, in _on_frame_ready.
        # The sink object itself is also kept (record.sink), so this specific
        # run's data can be released later by identity, even after the
        # runner's own `get_sink()` has moved on to a newer run.
        sink = _runner_sink(self._core.mda)
        acquisition: AcquisitionRecord | None = None
        layout = GridAxisLayout.none()
        if isinstance(sink, OmeWritersSink):
            layout = GridAxisLayout.build(sequence, sink.settings)
            acquisition = AcquisitionRecord(
                settings=sink.settings, summary_meta=sink.summary_meta, view=view
            )

        data: Any = view
        if layout.kind is not GridAxisLayoutKind.NONE:
            data = GridAxisDataWrapper(view, layout)

        viewer = MMArrayViewer(data, scales=_extract_scales(sequence, meta))
        widget = viewer.widget()
        sha = str(sequence.uid)[:8]
        widget.setObjectName(f"ndv-{sha}")

        title = self._disk_backed_title(sink) or f"MDA {sha}"
        viewer.mda_sequence = sequence
        viewer.source_title = title
        viewer._reuse_mda_callback = lambda: self.reuseMDARequested.emit(
            sequence, title
        )

        record = _ViewerRecord(viewer, sink=sink, layout=layout, is_live=True)
        if acquisition is not None:
            record.acquisition = acquisition
            viewer._acquisition_record = acquisition  # read by MMArrayViewer._save_data
        wrapper = viewer.data_wrapper
        coords_signal = getattr(view, "coords_changed", None)
        if coords_signal is not None and wrapper is not None:
            gate = _DimsChangeGate(wrapper)
            record.dims_gate = gate
            bridge = _StreamSignalBridge(gate.maybe_emit, widget)
            callback = bridge.dimsChanged.emit
            coords_signal.connect(callback)
            record.bridge = bridge
            record.coords_signal = coords_signal
            record.coords_callback = callback

        record.refresh = _LiveRefresh(
            lambda: self._apply_pending_index(record), parent=widget
        )

        self._follow_acquisition = True
        with suppress(Exception):
            _add_follow_lock_button(viewer, self)

        dw = self._new_dock(title)
        dw.setWidget(widget, CDockWidget.eInsertMode.ForceNoScrollArea)
        dw.closed.connect(lambda: self._on_viewer_closed(dw))
        dw.setAsCurrentTab()
        viewer._on_saved = lambda path: self._rename_viewer_tab(dw, viewer, path)

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
        record = None
        if (dw := self._active_dock) is not None:
            record = self._records.get(dw)
            if record is not None and record.acquisition is not None:
                record.acquisition.frame_meta.append(frame_meta_to_ome(meta))

        if (
            self._active_viewer is None
            or record is None
            or not self._follow_acquisition
        ):
            return

        record.pending_index = _follow_index(
            event, record.layout, record.ragged_fallback
        )
        if record.refresh is not None:
            record.refresh.request()

    def _apply_pending_index(self, record: _ViewerRecord) -> None:
        pending = record.pending_index
        if pending is None:
            return
        try:
            viewer = record.viewer
            current_index = viewer.display_model.current_index
            wrapper = viewer.data_wrapper
            before = dict(current_index)
            current_index.update(pending.items())
            if wrapper is not None and all(
                before.get(k) == v for k, v in pending.items()
            ):
                # current_index.update() was a full no-op (every requested key
                # already matched) -- force a redraw anyway, since the pixels
                # at this unchanged index may have just been written.
                wrapper.data_changed.emit()
        except Exception:  # viewer may have closed during the async write
            pass

    def _on_sequence_finished(self, sequence: MDASequence) -> None:
        """Flush the just-finished run's display, then retry releasing its data."""
        if (dw := self._active_dock) is not None:
            record = self._records.get(dw)
            if record is not None and record.refresh is not None:
                # The last frame must not be left stale behind a still-pending
                # coalesced refresh.
                record.refresh.flush_now()
        if (sink := self._pending_release) is not None:
            self._pending_release = None
            self._release_sink(sink)

    def _on_viewer_closed(self, dw: CDockWidget) -> None:
        record = self._records.pop(dw, None)
        if record is not None:
            # Only a live MDA run's viewer was ever announced via
            # mdaViewerCreated (CameraRoiSyncController's live-camera ROI
            # observation) -- a reopened acquisition never was, so it must
            # not raise the paired mdaViewerClosed either.
            if record.is_live:
                self.mdaViewerClosed.emit(record.viewer)
            record.disconnect()
            if record.refresh is not None:
                record.refresh.stop()
        if dw is self._active_dock:
            self._active_dock = None
            self._active_viewer = None
        if record is not None:
            with suppress(Exception):
                record.viewer.close()
            if record.loader_cleanup is not None:
                with suppress(Exception):
                    record.loader_cleanup()
            # Drop the runner's own reference to this run's data (freeing
            # scratch/memory-backed runs and their spill files), unless it's
            # still being written to -- release_sink() is a no-op if `sink`
            # is no longer the runner's current one (a newer run replaced it;
            # that run's data was already dropped by the runner itself, and
            # is kept alive only by this now-closed viewer's own references).
            if (sink := record.sink) is not None:
                if not self._release_sink(sink) and self._core.mda.is_running():
                    self._pending_release = sink

    def _release_sink(self, sink: SinkProtocol) -> bool:
        """Best-effort `release_sink`, followed by a GC pass to reclaim memory now.

        ndv/Qt objects tend to form reference cycles, so without an explicit
        collect the data can survive until the next cyclic-GC run instead of
        being freed the moment this viewer closes.
        """
        released = _release_runner_sink(self._core.mda, sink)
        if released:
            QTimer.singleShot(0, gc.collect)
        return released

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
            if record.refresh is not None:
                record.refresh.stop()
        self._records.clear()
        self._active_viewer = None
        self._active_dock = None
        self._pending_release = None
        if self.preview is not None:
            self.preview.detach()
            self.preview = None
