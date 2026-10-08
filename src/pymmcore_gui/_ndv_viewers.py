"""Dockable image viewers for the Acquire page."""

from __future__ import annotations

import gc
import hashlib
import math
import weakref
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ome_writers import ScratchFormat
from pymmcore_plus.autofocus import run_software_autofocus
from pymmcore_plus.mda import OmeWritersSink, frame_meta_to_ome

from pymmcore_gui._acquisition_loader import open_acquisition as _open_acquisition
from pymmcore_gui._array_viewer import MMArrayViewer
from pymmcore_gui._channel_luts import ChannelLUTMemory
from pymmcore_gui._mda_export import AcquisitionRecord
from pymmcore_gui._qt.QtAds import CDockWidget, DockWidgetArea
from pymmcore_gui._qt.QtCore import (
    QObject,
    QRunnable,
    Qt,
    QThread,
    QThreadPool,
    QTimer,
    Signal,
)
from pymmcore_gui._qt.QtWidgets import QSplitter
from pymmcore_gui._utils import autofocus_kind
from pymmcore_gui.widgets.image_preview._ndv_preview import NDVPreview

if TYPE_CHECKING:
    from collections.abc import Callable

    import ndv
    import numpy as np
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.autofocus import AutofocusResult
    from pymmcore_plus.mda import SinkProtocol
    from pymmcore_plus.metadata import FrameMetaV1, SummaryMetaV1
    from useq import MDAEvent, MDASequence

    from pymmcore_gui._acquisition_loader import LoadedAcquisition
    from pymmcore_gui._qt.QtAds import CDockAreaWidget, CDockManager
    from pymmcore_gui._qt.QtWidgets import QWidget


class _OpenAcquisitionSignals(QObject):
    """GUI-thread delivery point for one `_OpenAcquisitionTask`'s result."""

    finished = Signal(object)  # LoadedAcquisition
    failed = Signal(str)
    done = Signal()  # always, after finished/failed -- for cleanup


class _OpenAcquisitionTask(QRunnable):
    """Open one acquisition on a worker thread.

    Only the reader construction happens here; every Qt object is built on
    the GUI thread from `signals.finished`.
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self._path = path
        self.signals = _OpenAcquisitionSignals()

    def run(self) -> None:
        try:
            loaded = _open_acquisition(self._path)
        except Exception as e:
            self.signals.failed.emit(str(e))
        else:
            self.signals.finished.emit(loaded)
        finally:
            self.signals.done.emit()


def _runner_sink(runner: Any) -> SinkProtocol | None:
    """Return the runner's sink across released and development plus versions."""
    if callable(get_sink := getattr(runner, "get_sink", None)):
        return cast("SinkProtocol | None", get_sink())
    # get_sink() was added after pymmcore-plus 0.18.1. The runner has used
    # this same internal attribute since before our declared minimum version.
    return cast("SinkProtocol | None", getattr(runner, "_sink", None))


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
    tabbed into a dedicated nested ``CDockManager`` within the tools workspace.

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
    _eventStarted = Signal(object)
    _autofocusImageSnapped = Signal(object)
    _autofocusFinished = Signal(object, object)
    _testPreviewRequested = Signal()
    previewCreated = Signal(object)
    previewClosed = Signal()
    mdaViewerCreated = Signal(object)
    mdaViewerClosed = Signal(object)
    # str -- why a background `open_acquisition_async` could not open a path.
    acquisitionOpenFailed = Signal(str)
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
        self._channel_luts = ChannelLUTMemory()
        self._records: dict[CDockWidget, _ViewerRecord] = {}
        self._active_viewer: ndv.ArrayViewer | None = None
        self._active_dock: CDockWidget | None = None
        self._follow_acquisition = True
        self._connected = True
        # {(p, g): flattened "p"-slider slot}, reset per sequence -- see
        # _on_frame_ready for why this exists.
        self._shot_indices: dict[tuple[object, object], int] = {}
        # The tab to go back to once an autofocus routine that was asked to
        # show its images is done -- see _on_event_started.
        self._dock_before_autofocus: CDockWidget | None = None
        # Written on the acquisition thread, then read there by
        # _capture_unpreviewed_autofocus_image.  It lets us retain snaps made
        # before the GUI thread has had a chance to construct the lazy Preview.
        self._showing_autofocus_images = False
        # Set when a still-running run's viewer is closed: release_sink()
        # refuses to drop a sink while it's being written to, so the release
        # is retried once sequenceFinished confirms the run is done.
        self._pending_release: SinkProtocol | None = None

        # Background acquisition opening -- see open_acquisition_async. One
        # at a time, so a multi-file drop can't thrash the disk.
        self._open_pool = QThreadPool(self)
        self._open_pool.setMaxThreadCount(1)
        self._pending_opens: set[_OpenAcquisitionSignals] = set()

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
        self._eventStarted.connect(self._on_event_started)
        self._autofocusImageSnapped.connect(self._show_autofocus_image)
        self._autofocusFinished.connect(self._on_autofocus_finished)
        # The Qt stub omits connect's optional connection-type argument.
        cast("Any", self._testPreviewRequested).connect(
            self.ensure_preview,
            Qt.ConnectionType.BlockingQueuedConnection,
        )
        self._sequence_started_callback = self._sequenceStarted.emit
        self._frame_ready_callback = self._frameReady.emit
        self._sequence_finished_callback = self._sequenceFinished.emit
        self._event_started_callback = self._relay_event_started
        self._autofocus_finished_callback = self._relay_autofocus_finished
        self._unpreviewed_image_callback = self._capture_unpreviewed_autofocus_image

        events = self._core.mda.events
        events.sequenceStarted.connect(self._sequence_started_callback)
        events.frameReady.connect(self._frame_ready_callback)
        events.sequenceFinished.connect(self._sequence_finished_callback)
        # These three callbacks do only acquisition-thread work, then emit the
        # Qt signals above for every widget mutation.  They must be direct so
        # the first autofocus snap cannot outrun the show-images state.
        try:
            events.eventStarted.connect(
                self._event_started_callback, Qt.ConnectionType.DirectConnection
            )
        except TypeError:  # pragma: no cover - psygnal signaler
            events.eventStarted.connect(self._event_started_callback)
        # Keep the camera read on the thread that snapped.  A fast autofocus
        # may overwrite its buffer before the GUI thread creates the Preview.
        try:
            self._core.events.imageSnapped.connect(
                self._unpreviewed_image_callback,
                Qt.ConnectionType.DirectConnection,
            )
        except TypeError:  # pragma: no cover - psygnal signaler
            self._core.events.imageSnapped.connect(self._unpreviewed_image_callback)
        # ``autofocusFinished`` is newer than the rest of the runner's signals.
        self._af_finished_signal = getattr(events, "autofocusFinished", None)
        if self._af_finished_signal is not None:
            try:
                self._af_finished_signal.connect(
                    self._autofocus_finished_callback,
                    Qt.ConnectionType.DirectConnection,
                )
            except TypeError:  # pragma: no cover - psygnal signaler
                self._af_finished_signal.connect(self._autofocus_finished_callback)
        # A bound slot on this QObject may not run during its own destruction.
        # The owner's destroyed signal arrives before Qt deletes its children,
        # while this manager can still disconnect runner callbacks safely.
        if parent is not None:
            parent.destroyed.connect(self._disconnect)
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
            # Each run's viewer is tabbed over the Preview and takes the tab, so
            # without this a snap after an acquisition -- a stage move with "Snap"
            # checked, say -- would land in a Preview hidden behind it.
            preview.snapShown.connect(self._raise_preview)
            self.previewCreated.emit(preview)
        assert self._preview_dock is not None
        self._preview_dock.setAsCurrentTab()
        assert self.preview is not None
        return self.preview

    def run_software_autofocus_test(
        self,
        method: str,
        settings: Mapping[str, Any],
        should_cancel: Callable[[], bool],
    ) -> AutofocusResult:
        """Run the settings dialog's Test, showing its images when requested."""
        if self._core.mda.is_running():
            raise RuntimeError("An acquisition is running.")
        show_images = _shows_images(settings)
        if show_images:
            # The dialog runs this callable on a worker.  Wait until the GUI
            # thread has opened Preview before the routine takes its first snap.
            if QThread.currentThread() == self.thread():
                self.ensure_preview()
            else:
                self._testPreviewRequested.emit()
        result = run_software_autofocus(
            self._core, method, dict(settings), should_cancel=should_cancel
        )
        if (
            show_images
            and result.succeeded
            and math.isfinite(result.z_after)
            and not should_cancel()
        ):
            self._core.snapImage()
        return result

    def _on_event_started(self, event: MDAEvent) -> None:
        """Bring the Preview up for a routine that was asked to show its images.

        Asking to see them is a deliberate "let me watch this", so the Preview
        is opened if it is not even there yet and brought to the front -- and
        the tab that was in front goes back there afterwards, since what the
        user wants to watch for the rest of the run is the acquisition.
        """
        if autofocus_kind(event) != "software":
            return
        if not _shows_images(getattr(event.action, "settings", None)):
            return
        area = dock.dockAreaWidget() if (dock := self._preview_dock) else None
        with suppress(RuntimeError):
            current = area.currentDockWidget() if area is not None else None
            self._dock_before_autofocus = current or self._active_dock
        # The Preview may be created right here, after the event that opened
        # the autofocus window, so tell it where we are.
        self.ensure_preview().set_autofocus_running(True)

    def _relay_event_started(self, event: MDAEvent) -> None:
        """Record show-images state immediately, before relaying to the GUI."""
        settings = getattr(event.action, "settings", None)
        is_software = autofocus_kind(event) == "software"
        self._showing_autofocus_images = is_software and _shows_images(settings)
        self._eventStarted.emit(event)

    def _capture_unpreviewed_autofocus_image(self) -> None:
        """Retain an autofocus snap made before the lazy Preview exists."""
        if not self._showing_autofocus_images or self.preview is not None:
            return
        self._autofocusImageSnapped.emit(self._core.getImage())

    def _show_autofocus_image(self, image: object) -> None:
        """Display an autofocus image retained on the acquisition thread."""
        preview = self.ensure_preview()
        preview.set_autofocus_running(True)
        preview.append(image)  # type: ignore[arg-type]

    def _relay_autofocus_finished(self, event: MDAEvent, result: object) -> None:
        """Take the final focused snap, then relay completion to the GUI."""
        if (
            self._showing_autofocus_images
            and getattr(result, "kind", None) == "software"
            and getattr(result, "succeeded", False)
            and math.isfinite(getattr(result, "z_after", math.nan))
        ):
            # The routine has already moved the stage to result.z_after.  Snap
            # before closing the autofocus window so the Preview accepts it.
            with suppress(Exception):
                self._core.snapImage()
        self._showing_autofocus_images = False
        self._autofocusFinished.emit(event, result)

    def _on_autofocus_finished(
        self, _event: MDAEvent | None = None, _result: object = None
    ) -> None:
        if self.preview is not None:
            # A Preview created from a retained image may have missed the core's
            # autofocusFinished signal as well as its eventStarted signal.
            self.preview.set_autofocus_running(False)
        dock, self._dock_before_autofocus = self._dock_before_autofocus, None
        if dock is not None and dock is not self._preview_dock:
            with suppress(RuntimeError):  # the dock may have been closed
                dock.setAsCurrentTab()

    def _raise_preview(self) -> None:
        """Bring the Preview tab to the front, if it is still open."""
        if (dock := self._preview_dock) is not None:
            with suppress(RuntimeError):  # the dock may have been closed meanwhile
                dock.setAsCurrentTab()

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
        return self._viewer_for_loaded(_open_acquisition(path))

    def open_acquisition_async(self, path: str | Path) -> None:
        """Open `path` off the GUI thread, then add its tab when it's ready.

        Enumerating a multi-file acquisition, parsing its OME metadata and
        constructing readers is seconds of work for a large dataset -- doing
        it inline would freeze the window mid-drop. Failures arrive as
        `acquisitionOpenFailed` rather than an exception, since there is no
        longer a caller to raise into.

        Opens run one at a time: dropping ten datasets at once should not
        put ten readers into contention over the same disk, especially while
        an acquisition may be writing to it.
        """
        task = _OpenAcquisitionTask(path)
        signals = task.signals
        # Created here, so it belongs to the GUI thread and the worker's
        # emissions are delivered as queued events. Held onto because
        # QThreadPool frees the QRunnable as soon as run() returns.
        self._pending_opens.add(signals)
        signals.finished.connect(self._on_acquisition_loaded)
        signals.failed.connect(self.acquisitionOpenFailed)
        signals.done.connect(lambda s=signals: self._pending_opens.discard(s))
        self._open_pool.start(task)

    def _on_acquisition_loaded(self, loaded: LoadedAcquisition) -> None:
        """Build the viewer for a background-loaded acquisition (GUI thread)."""
        if not self._connected:
            # Torn down while this was loading: nothing will ever own the
            # reader, so release it here rather than leak the file handles.
            loaded.close()
            return
        self._viewer_for_loaded(loaded)

    def _viewer_for_loaded(self, loaded: LoadedAcquisition) -> ndv.ArrayViewer:
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
        self._shot_indices = {}
        view = self._core.mda.get_view()
        if view is None:
            # Runs without a path, AcquisitionSettings, or "memory" output have
            # no sink to display.  The embedded MDA widget prevents this case by
            # supplying "memory" whenever file saving is disabled.
            return

        viewer = MMArrayViewer(view, scales=_extract_scales(sequence, meta))
        self._channel_luts.bind_live_mda(viewer, sequence)
        widget = viewer.widget()
        sha = str(sequence.uid)[:8]
        widget.setObjectName(f"ndv-{sha}")

        # The sink object itself is fetched here (rather than only later) so
        # a disk-backed run's tab can be titled with its real destination
        # filename from the start, not just "MDA <sha>" until someone
        # manually saves it -- see _on_saved below for that latter case.
        sink = _runner_sink(self._core.mda)
        title = self._disk_backed_title(sink) or f"MDA {sha}"

        viewer.mda_sequence = sequence
        viewer.source_title = title
        viewer._reuse_mda_callback = lambda: self.reuseMDARequested.emit(
            sequence, title
        )

        record = _ViewerRecord(viewer, is_live=True)
        # Snapshot the sink's resolved settings + summary metadata now: the
        # sink is replaced wholesale on the *next* run, so a viewer left open
        # across two acquisitions must hold its own copy to export correctly
        # later. Per-frame metadata is appended live, in _on_frame_ready.
        # The sink object itself is also kept (record.sink), so this specific
        # run's data can be released later by identity, even after the
        # runner's own `get_sink()` has moved on to a newer run.
        record.sink = sink
        if isinstance(sink, OmeWritersSink):
            acquisition = AcquisitionRecord(
                settings=sink.settings, summary_meta=sink.summary_meta, view=view
            )
            record.acquisition = acquisition
            viewer._acquisition_record = acquisition  # read by MMArrayViewer._save_data
        wrapper = viewer.data_wrapper
        coords_signal = getattr(view, "coords_changed", None)
        if coords_signal is not None and wrapper is not None:
            bridge = _StreamSignalBridge(wrapper.dims_changed.emit, widget)
            callback = bridge.dimsChanged.emit
            coords_signal.connect(callback)
            record.bridge = bridge
            record.coords_signal = coords_signal
            record.coords_callback = callback

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
        if (dw := self._active_dock) is not None:
            record = self._records.get(dw)
            if record is not None and record.acquisition is not None:
                record.acquisition.frame_meta.append(frame_meta_to_ome(meta))

        viewer = self._active_viewer
        if viewer is None or not self._follow_acquisition:
            return

        current_index = viewer.display_model.current_index
        wrapper = viewer.data_wrapper
        index = {str(axis): value for axis, value in event.index.items()}
        if "p" in index or "g" in index:
            # A position's own grid sub-sequence yields both "p" (the real
            # position) and "g" (the tile within it) on the same event.
            # Naively renaming "g" -> "p" clobbered the real position value
            # and collided with another position's index. Route every
            # position-like event -- plain positions and position/tile pairs
            # alike -- through one shared counter instead, so every distinct
            # (position, tile) identity gets its own, strictly increasing
            # "p"-slider slot, assigned in acquisition order. That keeps a
            # plain position's slot from numerically colliding with a
            # flattened tile slot from another position's grid, and the
            # slider always ends on the true last frame. Revisiting the same
            # location later (e.g. the next timepoint) reuses its existing
            # slot rather than minting a new one.
            shot_key = (index.pop("p", None), index.pop("g", None))
            index["p"] = self._shot_indices.setdefault(
                shot_key, len(self._shot_indices)
            )

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
        """Retry releasing a just-finished run's data if its viewer already closed."""
        # a cancel mid-search means autofocusFinished may never arrive
        self._showing_autofocus_images = False
        self._on_autofocus_finished()
        if (sink := self._pending_release) is not None:
            self._pending_release = None
            self._release_sink(sink)

    def close_all_viewers(self) -> None:
        """Close every open viewer dock, releasing whatever each one holds.

        A reopened acquisition's viewer keeps an open file handle on its
        dataset for as long as it lives (see ``open_acquisition``), and only
        closing its dock releases it. Shutting the page down without this
        leaves those handles to garbage collection, which may not run until
        long after the window is gone.
        """
        for dw in list(self._records):
            with suppress(RuntimeError):
                dw.closeDockWidget()

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
        with suppress(Exception):
            events.eventStarted.disconnect(self._event_started_callback)
        with suppress(Exception):
            self._core.events.imageSnapped.disconnect(self._unpreviewed_image_callback)
        if self._af_finished_signal is not None:
            with suppress(Exception):
                self._af_finished_signal.disconnect(self._autofocus_finished_callback)
        for record in self._records.values():
            self.mdaViewerClosed.emit(record.viewer)
            record.disconnect()
        self._records.clear()
        self._active_viewer = None
        self._active_dock = None
        self._pending_release = None
        if self.preview is not None:
            self.preview.detach()
            self.preview = None


def _shows_images(settings: object) -> bool:
    """Whether `settings` asks a routine to show the images it scores.

    Walks nested settings, because `duo` carries a routine of its own under
    each of its two steps and either may be the one being watched.
    """
    if not isinstance(settings, Mapping):
        return False
    if settings.get("show_images"):
        return True
    return any(_shows_images(value) for value in settings.values())


class _StreamSignalBridge(QObject):
    """Marshal ome-writers dimension changes onto the Qt GUI thread."""

    dimsChanged = Signal()

    def __init__(self, callback: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._callback = callback
        # The receiver is deliberately a QObject-bound method.  Connecting the
        # signal straight to a regular Python callable lets PyQt invoke it in
        # the writer thread, which means ndv may create/hide sliders off the GUI
        # thread as live dimensions grow.
        self.dimsChanged.connect(self._notify)

    def _notify(self) -> None:
        self._callback()


def _add_follow_lock_button(ndv_viewer: ndv.ArrayViewer, manager: Any) -> None:
    """Add the Christina follow-acquisition toggle to an ndv viewer."""
    from superqt import QIconifyIcon

    from pymmcore_gui._array_viewer import ensure_visible_icon, set_source_icon
    from pymmcore_gui._qt.QtWidgets import QPushButton

    q_widget = ndv_viewer.widget()
    btn_layout = getattr(q_widget, "_btn_layout", None)
    if btn_layout is None:
        return

    btn = QPushButton(q_widget)
    btn.setCheckable(True)
    # this button is added after MMArrayViewer.__init__'s unstyle_widgets()
    # sweep (which is what gives the other viewer buttons their flat
    # "subtle" look), so it never picks up that variant on its own -- set it
    # explicitly or this renders with Qt's native (blue) checked style.
    btn.setProperty("variant", "subtle")

    def _set_icon(glyph: str) -> None:
        # QIconifyIcon renders these mdi glyphs near-black; the other viewer
        # buttons only look right because unstyle_widgets ran ensure_visible_icon
        # on them. This button missed that sweep, so recolor it ourselves (and
        # re-stash the source so theme changes re-derive correctly).
        set_source_icon(btn, QIconifyIcon(glyph))
        ensure_visible_icon(btn)

    _set_icon("mdi:lock-open-variant-outline")
    btn.setToolTip("Lock sliders (don't follow acquisition)")
    mgr_ref = weakref.ref(manager)

    def _toggled(locked: bool) -> None:
        if locked:
            _set_icon("mdi:lock-outline")
            btn.setToolTip("Unlock sliders (follow acquisition)")
        else:
            _set_icon("mdi:lock-open-variant-outline")
            btn.setToolTip("Lock sliders (don't follow acquisition)")
        if mgr := mgr_ref():
            mgr._follow_acquisition = not locked

    btn.toggled.connect(_toggled)
    btn_layout.addWidget(btn)


def _extract_scales(
    sequence: MDASequence | None = None, meta: SummaryMetaV1 | None = None
) -> dict[str, float]:
    """Build physical axis scales from MDA sequence and summary metadata."""
    scales: dict[str, float] = {}
    with suppress(Exception):
        if meta and (px := meta["image_infos"][0]["pixel_size_um"]):
            scales["x"] = float(px)
            scales["y"] = float(px)
    with suppress(Exception):
        if sequence and sequence.z_plan:
            from useq import ZAboveBelow, ZRangeAround, ZTopBottom

            if isinstance(sequence.z_plan, (ZTopBottom, ZRangeAround, ZAboveBelow)):
                scales["z"] = float(sequence.z_plan.step)
    return scales
