from __future__ import annotations

import weakref
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast
from weakref import WeakSet, WeakValueDictionary

import ndv
import useq

from pymmcore_gui._array_viewer import MMArrayViewer
from pymmcore_gui._qt.QtAds import CDockWidget
from pymmcore_gui._qt.QtCore import QObject, QTimer, Signal
from pymmcore_gui._qt.QtWidgets import QWidget
from pymmcore_gui.widgets.image_preview._ndv_preview import NDVPreview

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    from ndv.models._array_display_model import (
        IndexMap,  # pyright: ignore[reportPrivateImportUsage]
    )
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.metadata import FrameMetaV1, SummaryMetaV1
    from useq import MDASequence

    from pymmcore_gui.widgets.image_preview._preview_base import ImagePreviewBase


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
        view = self._runner.get_view()
        self._active_mda_viewer = (
            self._create_ndv_viewer(view, sequence, meta) if view is not None else None
        )

    def _on_frame_ready(
        self, frame: np.ndarray, event: useq.MDAEvent, meta: FrameMetaV1
    ) -> None:
        """Follow the latest acquired index and redraw the sink-backed viewer."""
        if (viewer := self._active_mda_viewer) is None:
            return  # pragma: no cover
        if not self._follow_acquisition:
            return

        current_index = viewer.display_model.current_index
        wrapper = viewer.data_wrapper
        index = {
            ("p" if str(axis) == "g" else axis): value
            for axis, value in event.index.items()
        }

        def _update(_idx: IndexMap = current_index) -> None:
            try:
                _idx.update(index.items())
                if wrapper is not None:
                    wrapper.data_changed.emit()
            except Exception:  # viewer may have closed during the async write
                pass

        QTimer.singleShot(10, _update)

    def _on_sequence_finished(self, sequence: useq.MDASequence) -> None:
        """Called when a sequence has finished."""
        self._is_mda_running = False

    def _create_ndv_viewer(
        self,
        view: Any,
        sequence: MDASequence,
        meta: SummaryMetaV1 | None = None,
    ) -> ndv.ArrayViewer:
        """Create a shared MMArrayViewer backed by an ome-writers stream view."""
        ndv_viewer = MMArrayViewer(view, scales=_extract_scales(sequence, meta))
        if hasattr(view, "coords_changed") and hasattr(
            ndv_viewer.data_wrapper, "dims_changed"
        ):
            bridge = _StreamSignalBridge(ndv_viewer.widget())
            view.coords_changed.connect(bridge.dimsChanged.emit)
            bridge.dimsChanged.connect(ndv_viewer.data_wrapper.dims_changed.emit)
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


def _add_follow_lock_button(ndv_viewer: ndv.ArrayViewer, manager: Any) -> None:
    """Add the Christina follow-acquisition toggle to an ndv viewer."""
    from superqt import QIconifyIcon

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
    btn.setIcon(QIconifyIcon("mdi:lock-open-variant-outline"))
    btn.setToolTip("Lock sliders (don't follow acquisition)")
    mgr_ref = weakref.ref(manager)

    def _toggled(locked: bool) -> None:
        if locked:
            btn.setIcon(QIconifyIcon("mdi:lock-outline"))
            btn.setToolTip("Unlock sliders (follow acquisition)")
        else:
            btn.setIcon(QIconifyIcon("mdi:lock-open-variant-outline"))
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
