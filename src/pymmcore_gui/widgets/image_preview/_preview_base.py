import warnings
from abc import abstractmethod
from contextlib import suppress

import numpy as np
from pymmcore_plus import CMMCorePlus

from pymmcore_gui._qt.QtCore import Qt, QTimerEvent, Signal
from pymmcore_gui._qt.QtWidgets import QWidget
from pymmcore_gui._utils import autofocus_kind, connect_direct
from pymmcore_gui.widgets.image_preview._snap_album import SnappedImage

_DEFAULT_WAIT = 10


class ImagePreviewBase(QWidget):
    # A snapped image was just displayed, outside any acquisition -- so whoever
    # owns this widget can bring it to the front.  Not emitted for the autofocus
    # images of a running acquisition: those are worth showing, but not worth
    # pulling the user off the viewer of the run in progress.
    snapShown = Signal()

    # An ordinary snap worth getting back later, as a `SnappedImage`.  Not emitted
    # for acquisition frames, or for the images an autofocus routine scores: those
    # would push the user's own snaps out of the album in a moment.  Emitted on the
    # thread that snapped.
    snapRecorded = Signal(object)

    # Snapped images are relayed through a Qt signal rather than displayed where
    # they arrive: autofocus snaps come from the acquisition thread, and `append`
    # drives the GPU canvas, which only the GUI thread may touch.
    _snapped = Signal(object, bool)

    def __init__(
        self,
        parent: QWidget | None,
        mmcore: CMMCorePlus,
        *,
        use_with_mda: bool = False,
    ):
        super().__init__(parent)
        self._timer_id: int | None = None  # timer for streaming

        self.use_with_mda = use_with_mda
        self._is_mda_running: bool = False
        # set while an autofocus event is in flight, during which a snapped
        # image is a routine's diagnostic image rather than acquisition data
        self._autofocus_running: bool = False
        self._mmc: CMMCorePlus | None = None
        self._snapped.connect(self._on_snapped)
        self.attach(mmcore)

    def attach(self, core: CMMCorePlus) -> None:
        """Attach this widget to events in `core`."""
        if self._mmc is not None:
            self.detach()

        ev = core.events
        # Deliberately a *direct* connection: an autofocus routine's next image
        # is already on the way, so the camera buffer has to be read on the
        # thread that snapped rather than whenever the GUI thread gets round to
        # a queued call -- otherwise half the images read back as duplicates of
        # a later one. `_on_image_snapped` relays the array it read to the GUI
        # thread.
        connect_direct(ev.imageSnapped, self._on_image_snapped)
        ev.continuousSequenceAcquisitionStarted.connect(self._on_streaming_start)
        ev.sequenceAcquisitionStarted.connect(self._on_streaming_start)
        ev.sequenceAcquisitionStopped.connect(self._on_streaming_stop)
        ev.exposureChanged.connect(self._on_exposure_changed)
        ev.systemConfigurationLoaded.connect(self._on_system_config_loaded)
        ev.roiSet.connect(self._on_roi_set)
        # A preview created mid-run missed sequenceStarted, and without this
        # would take every frame of the acquisition for an ordinary snap.
        self._is_mda_running = core.mda.is_running()
        # Update acquisition flags before the next snap reads them.
        events = core.mda.events
        connect_direct(events.sequenceStarted, self._on_mda_started)
        connect_direct(events.sequenceFinished, self._on_mda_finished)
        connect_direct(events.eventStarted, self._on_mda_event_started)
        connect_direct(events.autofocusFinished, self._on_autofocus_finished)

        self._mmc = core

    def detach(self) -> None:
        """Detach this widget from events in `core`."""
        if self._mmc is None:
            return
        core, self._mmc = self._mmc, None
        if self._timer_id is not None:
            self.killTimer(self._timer_id)
            self._timer_id = None

        ev = core.events
        connections = (
            (ev.imageSnapped, self._on_image_snapped),
            (ev.continuousSequenceAcquisitionStarted, self._on_streaming_start),
            (ev.sequenceAcquisitionStarted, self._on_streaming_start),
            (ev.sequenceAcquisitionStopped, self._on_streaming_stop),
            (ev.exposureChanged, self._on_exposure_changed),
            (ev.systemConfigurationLoaded, self._on_system_config_loaded),
            (ev.roiSet, self._on_roi_set),
            (core.mda.events.sequenceStarted, self._on_mda_started),
            (core.mda.events.sequenceFinished, self._on_mda_finished),
            (core.mda.events.eventStarted, self._on_mda_event_started),
            (core.mda.events.autofocusFinished, self._on_autofocus_finished),
        )
        for signal, callback in connections:
            with suppress(Exception):
                signal.disconnect(callback)

    @abstractmethod
    def append(self, data: np.ndarray) -> None:
        """Set texture data.

        The dtype must be compatible with wgpu texture formats.
        Will also apply contrast limits if _clims is "auto".
        """
        raise NotImplementedError

    # ----------------------------

    def _on_exposure_changed(self, device: str, value: str) -> None:
        # change timer interval
        if self._timer_id is not None:
            self.killTimer(self._timer_id)
            self._timer_id = self.startTimer(int(value), Qt.TimerType.PreciseTimer)

    def timerEvent(self, a0: QTimerEvent | None) -> None:
        if (core := self._mmc) and core.getRemainingImageCount() > 0:
            try:
                img = core.fixImage(core.getLastImage())
                self.append(img)
            except Exception as e:
                warnings.warn(
                    f"Failed to get image from core: {e}", RuntimeWarning, stacklevel=2
                )

    def _on_image_snapped(self) -> None:
        if (core := self._mmc) is None:
            return  # pragma: no cover
        during_mda = self._is_mda_running
        if during_mda and not (self.use_with_mda or self._autofocus_running):
            return  # pragma: no cover

        # Read the camera buffer here, on the thread that snapped: an autofocus
        # routine has the next image on the way, and would overwrite it before a
        # queued handler on the GUI thread ever got to look.
        image = core.getImage()
        # Recorded first: a snap made on the GUI thread is displayed synchronously,
        # and displaying processes events, which may well change the channel.
        if not (during_mda or self._autofocus_running):
            self.snapRecorded.emit(SnappedImage.capture(core, image))
        self._snapped.emit(image, during_mda)

    def _on_snapped(self, image: np.ndarray, during_mda: bool) -> None:
        self.append(image)
        if not during_mda:
            self.snapShown.emit()

    def set_autofocus_running(self, running: bool) -> None:
        """Say that an autofocus event is in progress.

        For a preview created *during* one, which missed its `eventStarted`;
        `autofocusFinished` clears it as usual. Also for a routine run outside any
        acquisition, whose owner has to clear it again.
        """
        self._autofocus_running = running

    def _on_mda_event_started(self, event: object) -> None:
        # Autofocus images are the only snaps worth showing mid-acquisition; the
        # next event ends the window, in case autofocusFinished never arrives.
        self._autofocus_running = autofocus_kind(event) is not None  # type: ignore[arg-type]

    def _on_autofocus_finished(self, *_: object) -> None:
        self._autofocus_running = False

    def _on_mda_finished(self) -> None:
        self._is_mda_running = False
        self._autofocus_running = False

    def _on_mda_started(self) -> None:
        self._is_mda_running = True

    def _on_streaming_start(self) -> None:
        if (core := self._mmc) is not None:
            wait = int(core.getExposure()) or _DEFAULT_WAIT
            self._timer_id = self.startTimer(wait, Qt.TimerType.PreciseTimer)

    def _on_streaming_stop(self) -> None:
        if self._timer_id is not None:
            self.killTimer(self._timer_id)
            self._timer_id = None

    def _on_system_config_loaded(self) -> None:
        pass

    def _on_roi_set(self) -> None:
        pass
