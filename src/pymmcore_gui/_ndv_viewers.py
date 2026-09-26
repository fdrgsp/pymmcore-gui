from __future__ import annotations

import weakref
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast
from weakref import WeakSet, WeakValueDictionary

import ndv
import useq
from pymmcore_plus.mda import OmeWritersSink, frame_meta_to_ome

from pymmcore_gui._array_viewer import MMArrayViewer
from pymmcore_gui._grid_axis import (
    GridAxisDataWrapper,
    GridAxisLayout,
    GridAxisLayoutKind,
)
from pymmcore_gui._mda_export import AcquisitionRecord
from pymmcore_gui._qt.QtAds import CDockWidget
from pymmcore_gui._qt.QtCore import QObject, QTimer, Signal
from pymmcore_gui._qt.QtWidgets import QWidget
from pymmcore_gui.widgets.image_preview._ndv_preview import NDVPreview

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    import numpy as np
    from ndv.models import DataWrapper
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.mda import SinkProtocol
    from pymmcore_plus.metadata import FrameMetaV1, SummaryMetaV1
    from useq import MDAEvent, MDASequence

    from pymmcore_gui.widgets.image_preview._preview_base import ImagePreviewBase


def _runner_sink(runner: Any) -> SinkProtocol | None:
    """Return the runner's sink across released and development plus versions."""
    if callable(get_sink := getattr(runner, "get_sink", None)):
        return cast("SinkProtocol | None", get_sink())
    # get_sink() was added after pymmcore-plus 0.18.1. The runner has used
    # this same internal attribute since before our declared minimum version.
    return cast("SinkProtocol | None", getattr(runner, "_sink", None))


class _UnplannedSlotCounter:
    """Arrival-order `(p, g) -> flattened-p` counter for unsupported layouts.

    Used only where `GridAxisLayout.build()` produced no planned mapping at
    all (`GridAxisLayoutKind.NONE` with `has_grid`) -- a sequence whose grid
    it could not derive or could not reconcile with the storage dimension.
    Genuinely ragged grids are *not* this case: they get a real planned
    mapping (`GridAxisLayoutKind.RAGGED`) and never reach here. Without a
    plan there is no way to compute the right flattened slot from `(p, g)`,
    so this reproduces the pre-existing arrival-order behavior -- which is
    not a source of storage truth, and (like the code it replaces) is not
    reliable across `sink.skip()`.
    """

    def __init__(self) -> None:
        self._slots: dict[tuple[object, object], int] = {}

    def reset(self) -> None:
        self._slots.clear()

    def resolve(self, p: object, g: object) -> int:
        return self._slots.setdefault((p, g), len(self._slots))


def _follow_index(
    event: MDAEvent, layout: GridAxisLayout, fallback: _UnplannedSlotCounter
) -> dict[str, int]:
    """Derive a viewer's display index for `event` from the planned layout.

    For a supported layout (GRID_ONLY/REGULAR), this is a pure function of
    the event and the (immutable, pre-validated) layout: it never desyncs
    under `sink.skip()`, and revisiting the same `(p, g)` at a later timepoint
    is naturally idempotent. For an unsupported/ragged layout (`layout.
    has_grid`), `fallback` provides the same arrival-order behavior as before:
    a grid exists somewhere in the sequence, so even a "p"-only event (an
    ungridded position) may not already name its correct flattened slot, once
    another position's grid tiles have consumed extra slots ahead of it. When
    the sequence has no grid at all, a "p" value already is the correct
    flattened slot and needs no counter.
    """
    index = {str(axis): value for axis, value in event.index.items()}
    if layout.kind is GridAxisLayoutKind.NONE:
        if layout.has_grid and ("p" in index or "g" in index):
            p = index.pop("p", None)
            g = index.pop("g", None)
            index["p"] = fallback.resolve(p, g)
        return index
    p = index.pop("p", None)
    g = index.pop("g", None)
    if layout.kind is GridAxisLayoutKind.GRID_ONLY:
        index["g"] = g if g is not None else 0
    else:
        index["p"] = p if p is not None else 0
        index["g"] = g if g is not None else 0
    return index


class _LiveRefresh:
    """Coalesce rapid frameReady/follow updates into one bounded-rate refresh.

    At most one `QTimer` is ever pending; `request()` may be called as often
    as frames arrive without growing a backlog of scheduled work.
    """

    def __init__(
        self,
        apply: Callable[[], None],
        *,
        interval_ms: int = 33,
        parent: QObject | None = None,
    ) -> None:
        self._apply = apply
        self._timer = QTimer(parent)
        self._timer.setSingleShot(True)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._apply)

    def request(self) -> None:
        """Mark a refresh as needed; schedule at most one pending timer."""
        if not self._timer.isActive():
            self._timer.start()

    def flush_now(self) -> None:
        """Apply immediately, cancelling any pending timer."""
        self._timer.stop()
        self._apply()

    def stop(self) -> None:
        """Cancel any pending timer without applying."""
        self._timer.stop()


class _DimsChangeGate:
    """Suppress a `dims_changed` emission when no exposed coordinate grew.

    Wraps the raw view's `coords_changed` -> wrapper `dims_changed` bridge:
    `sizes()` is available on every `DataWrapper` (grid-wrapped or the plain
    fallback), so this applies uniformly to both, not just grid viewers.
    """

    def __init__(self, wrapper: DataWrapper[Any]) -> None:
        self._wrapper = wrapper
        self._last_sizes: Mapping[Any, int] = dict(wrapper.sizes())

    def maybe_emit(self) -> None:
        sizes = dict(self._wrapper.sizes())
        if sizes != self._last_sizes:
            self._last_sizes = sizes
            self._wrapper.dims_changed.emit()


# NOTE: we make this a QObject mostly so that the lifetime of this object is tied to
# the lifetime of the parent QMainWindow.  If inheriting from QObject is removed in
# the future, make sure not to store a strong reference to this main_window
class NDVViewersManager(QObject):
    """Object that mediates a connection between the MDA experiment and ndv viewers.

    Parameters
    ----------
    parent : QWidget
        The parent widget.
    mmcore : CMMCorePlus
        The CMMCorePlus instance.
    """

    mdaViewerCreated = Signal(ndv.ArrayViewer, useq.MDASequence)
    previewViewerCreated = Signal(CDockWidget)
    viewerDestroyed = Signal(str)
    _sequenceStarted = Signal(object, object)
    _frameReady = Signal(object, object, object)
    _sequenceFinished = Signal(object)

    def __init__(self, parent: QWidget, mmcore: CMMCorePlus):
        super().__init__(parent)
        self._mmc = mmcore

        # weakref map of {sequence_uid: ndv.ArrayViewer}
        self._seq_viewers = WeakValueDictionary[str, ndv.ArrayViewer]()
        self._preview_dock_widgets = WeakSet[CDockWidget]()
        self._active_mda_viewer: ndv.ArrayViewer | None = None
        # Planned p/g -> flattened-position mapping for the active run, and the
        # coalesced-refresh state that follows it -- reset per sequence.
        self._layout: GridAxisLayout = GridAxisLayout.none()
        self._refresh: _LiveRefresh | None = None
        self._pending_index: dict[str, int] | None = None
        self._dims_gate: _DimsChangeGate | None = None
        self._unplanned_slots = _UnplannedSlotCounter()
        # Snapshot of the active run's sink settings/summary metadata, so
        # MMArrayViewer._save_data() can export canonical data even for the
        # classic GUI (mirrors AcquireViewersManager's AcquisitionRecord).
        self._current_acquisition: AcquisitionRecord | None = None

        # CONNECTIONS ---------------------------------------------------------

        self._is_mda_running = False
        self._follow_acquisition = True
        self._current_image_preview: CDockWidget | None = None

        ev = self._mmc.events
        ev.imageSnapped.connect(self._on_image_snapped)
        ev.sequenceAcquisitionStarted.connect(self._on_streaming_started)
        ev.continuousSequenceAcquisitionStarted.connect(self._on_streaming_started)
        ev.propertyChanged.connect(self._on_property_changed)

        self._runner = self._mmc.mda
        self._sequenceStarted.connect(self._on_sequence_started)
        self._frameReady.connect(self._on_frame_ready)
        self._sequenceFinished.connect(self._on_sequence_finished)
        self._sequence_started_callback = self._sequenceStarted.emit
        self._frame_ready_callback = self._frameReady.emit
        self._sequence_finished_callback = self._sequenceFinished.emit
        mda_ev = self._runner.events
        mda_ev.sequenceStarted.connect(self._sequence_started_callback)
        mda_ev.frameReady.connect(self._frame_ready_callback)
        mda_ev.sequenceFinished.connect(self._sequence_finished_callback)

        parent.destroyed.connect(self._cleanup)

    def _cleanup(self, obj: QObject | None = None) -> None:
        self._active_mda_viewer = None
        if self._refresh is not None:
            self._refresh.stop()
            self._refresh = None
        mda_ev = self._runner.events
        with suppress(Exception):
            mda_ev.sequenceStarted.disconnect(self._sequence_started_callback)
        with suppress(Exception):
            mda_ev.frameReady.disconnect(self._frame_ready_callback)
        with suppress(Exception):
            mda_ev.sequenceFinished.disconnect(self._sequence_finished_callback)

    def _on_sequence_started(
        self, sequence: useq.MDASequence, meta: SummaryMetaV1
    ) -> None:
        """Create a viewer backed by the MDA runner's live sink view."""
        self._is_mda_running = True
        self._layout = GridAxisLayout.none()
        self._pending_index = None
        self._dims_gate = None
        self._unplanned_slots.reset()
        self._current_acquisition = None
        if self._refresh is not None:
            self._refresh.stop()
            self._refresh = None
        view = self._runner.get_view()
        self._active_mda_viewer = (
            self._create_ndv_viewer(view, sequence, meta) if view is not None else None
        )

    def _on_frame_ready(
        self, frame: np.ndarray, event: useq.MDAEvent, meta: FrameMetaV1
    ) -> None:
        """Record frame metadata for export, then follow the latest acquired index.

        Metadata capture happens unconditionally, before the follow-lock check
        below: the lock only controls whether the displayed slider position
        tracks new frames, and must not also silently truncate the metadata
        used later by the viewer's Save button.
        """
        if (acquisition := self._current_acquisition) is not None:
            acquisition.frame_meta.append(frame_meta_to_ome(meta))

        if self._active_mda_viewer is None:
            return  # pragma: no cover
        if not self._follow_acquisition:
            return

        self._pending_index = _follow_index(event, self._layout, self._unplanned_slots)
        if self._refresh is not None:
            self._refresh.request()

    def _apply_pending_index(self) -> None:
        pending = self._pending_index
        if pending is None or (viewer := self._active_mda_viewer) is None:
            return
        try:
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

    def _on_sequence_finished(self, sequence: useq.MDASequence) -> None:
        """Called when a sequence has finished."""
        self._is_mda_running = False
        if self._refresh is not None:
            # The last frame must not be left stale behind a still-pending
            # coalesced refresh.
            self._refresh.flush_now()

    def _create_ndv_viewer(
        self,
        view: Any,
        sequence: MDASequence,
        meta: SummaryMetaV1 | None = None,
    ) -> ndv.ArrayViewer:
        """Create a shared MMArrayViewer backed by an ome-writers stream view."""
        sink = _runner_sink(self._runner)
        layout = GridAxisLayout.none()
        if isinstance(sink, OmeWritersSink):
            layout = GridAxisLayout.build(sequence, sink.settings)
            self._current_acquisition = AcquisitionRecord(
                settings=sink.settings, summary_meta=sink.summary_meta, view=view
            )
        self._layout = layout

        data: Any = view
        if layout.kind is not GridAxisLayoutKind.NONE:
            data = GridAxisDataWrapper(view, layout)

        ndv_viewer = MMArrayViewer(data, scales=_extract_scales(sequence, meta))
        if self._current_acquisition is not None:
            # read by MMArrayViewer._save_data
            ndv_viewer._acquisition_record = self._current_acquisition

        wrapper = ndv_viewer.data_wrapper
        if hasattr(view, "coords_changed") and wrapper is not None:
            gate = _DimsChangeGate(wrapper)
            self._dims_gate = gate
            bridge = _StreamSignalBridge(gate.maybe_emit, ndv_viewer.widget())
            view.coords_changed.connect(bridge.dimsChanged.emit)

        self._refresh = _LiveRefresh(
            self._apply_pending_index, parent=ndv_viewer.widget()
        )

        self._follow_acquisition = True
        with suppress(Exception):
            _add_follow_lock_button(ndv_viewer, self)
        self._seq_viewers[str(sequence.uid)] = ndv_viewer
        self.mdaViewerCreated.emit(ndv_viewer, sequence)
        return ndv_viewer

    def _create_or_show_img_preview(self) -> ImagePreviewBase | None:
        """Create or show the image preview widget, return True if created."""
        preview = None
        if self._current_image_preview is None:
            preview = NDVPreview(mmcore=self._mmc)
            if not isinstance((parent := self.parent()), QWidget):
                parent = None  # pragma: no cover

            # this is a hacky workaround:
            # Calling CDockWidget('title', parent) is deprecated
            # It is preferred to instantiate with a CDockManager.
            # parent will almost always be the MainWindow that dock_manager
            # (and in reality, will never be None)
            if dm := getattr(parent, "dock_manager", None):
                dw = CDockWidget(dm, "Preview", parent)
            else:  # pragma: no cover
                dw = CDockWidget("Preview", parent)

            self._current_image_preview = dw
            self._preview_dock_widgets.add(dw)
            dw.setWidget(preview)
            dw.setFeature(dw.DockWidgetFeature.DockWidgetFloatable, False)
            self.previewViewerCreated.emit(dw)
        else:
            self._current_image_preview.toggleView(True)

        return preview

    def _on_streaming_started(self) -> None:
        if not self._is_mda_running:
            if preview := self._create_or_show_img_preview():
                preview._on_streaming_start()

    def _on_image_snapped(self) -> None:
        if not self._is_mda_running:
            if preview := self._create_or_show_img_preview():
                preview.append(self._mmc.getImage())

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.__class__.__name__} {hex(id(self))} ({len(self)} viewer)>"

    def __len__(self) -> int:
        return len(self._seq_viewers)

    def viewers(self) -> Iterator[ndv.ArrayViewer]:
        yield from (self._seq_viewers.values())

    def _on_property_changed(self, dev: str, prop: str, value: str) -> None:
        if self._mmc is None:
            return  # pragma: no cover

        # if we change any camera property
        if dev == self._mmc.getCameraDevice() or (dev == "Core" and prop == "Camera"):
            if self._current_image_preview:
                # check if the existing viewer still has a valid shape and dtype
                # (dtype is actually tuple of (dtype, shape))
                preview = cast("NDVPreview", self._current_image_preview.widget())
                if preview._get_core_dtype_shape() != preview.dtype_shape:
                    preview.detach()
                    self._current_image_preview = None


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
