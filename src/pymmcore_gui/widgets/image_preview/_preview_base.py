import warnings
from abc import abstractmethod
from contextlib import suppress

import numpy as np
from pymmcore_plus import CMMCorePlus

from pymmcore_gui._qt.QtCore import Qt, QTimerEvent, Signal
from pymmcore_gui._qt.QtWidgets import QWidget
from pymmcore_gui._utils import autofocus_kind

_DEFAULT_WAIT = 10


class ImagePreviewBase(QWidget):
    # A snapped image was just displayed, outside any acquisition -- so whoever
    # owns this widget can bring it to the front.  Not emitted for the autofocus
    # images of a running acquisition: those are worth showing, but not worth
    # pulling the user off the viewer of the run in progress.
    snapShown = Signal()

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
        self._mmc: CMMCorePlus | None = mmcore
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
        # thread. A non-Qt signaler calls back on the emitting thread anyway.
        try:
            ev.imageSnapped.connect(
                self._on_image_snapped, Qt.ConnectionType.DirectConnection
            )
        except TypeError:  # pragma: no cover - psygnal signaler
            ev.imageSnapped.connect(self._on_image_snapped)
        ev.continuousSequenceAcquisitionStarted.connect(self._on_streaming_start)
        ev.sequenceAcquisitionStarted.connect(self._on_streaming_start)
        ev.sequenceAcquisitionStopped.connect(self._on_streaming_stop)
        ev.exposureChanged.connect(self._on_exposure_changed)
        ev.systemConfigurationLoaded.connect(self._on_system_config_loaded)
        ev.roiSet.connect(self._on_roi_set)
        ev.propertyChanged.connect(self._on_property_changed)
        # A preview created mid-run missed sequenceStarted, and without this
        # would take every frame of the acquisition for an ordinary snap.
        self._is_mda_running = core.mda.is_running()
        self._mda_started_callback = lambda: setattr(self, "_is_mda_running", True)
        self._mda_finished_callback = self._on_mda_finished
        core.mda.events.sequenceStarted.connect(self._mda_started_callback)
        core.mda.events.sequenceFinished.connect(self._mda_finished_callback)
        self._event_started_callback = self._on_mda_event_started
        core.mda.events.eventStarted.connect(self._event_started_callback)
        # ``autofocusFinished`` is newer than the rest of the runner's signals.
        self._af_finished_signal = getattr(core.mda.events, "autofocusFinished", None)
        self._af_finished_callback = self._on_autofocus_finished
        if self._af_finished_signal is not None:
            self._af_finished_signal.connect(self._af_finished_callback)

        self._mmc = core

    def detach(self) -> None:
        """Detach this widget from events in `core`."""
        if self._mmc is None:
            return  # pragma: no cover
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
            (ev.propertyChanged, self._on_property_changed),
            (
                core.mda.events.sequenceStarted,
                getattr(self, "_mda_started_callback", None),
            ),
            (
                core.mda.events.sequenceFinished,
                getattr(self, "_mda_finished_callback", None),
            ),
            (
                core.mda.events.eventStarted,
                getattr(self, "_event_started_callback", None),
            ),
            (
                getattr(self, "_af_finished_signal", None),
                getattr(self, "_af_finished_callback", None),
            ),
        )
        for signal, callback in connections:
            if signal is not None and callback is not None:
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
        self._snapped.emit(core.getImage(), during_mda)

    def _on_snapped(self, image: np.ndarray, during_mda: bool) -> None:
        self.append(image)
        if not during_mda:
            self.snapShown.emit()

    def set_autofocus_running(self, running: bool) -> None:
        """Say that an autofocus event is in progress.

        For a preview created *during* one, which missed its `eventStarted`;
        `autofocusFinished` clears it as usual.
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

    def _on_property_changed(self, dev: str, prop: str, value: str) -> None:
        pass
