import warnings
from abc import abstractmethod
from contextlib import suppress

import numpy as np
from pymmcore_plus import CMMCorePlus

from pymmcore_gui._qt.QtCore import Qt, QTimerEvent
from pymmcore_gui._qt.QtWidgets import QWidget

_DEFAULT_WAIT = 10


class ImagePreviewBase(QWidget):
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
        self._mmc: CMMCorePlus | None = mmcore
        self.attach(mmcore)

    def attach(self, core: CMMCorePlus) -> None:
        """Attach this widget to events in `core`."""
        if self._mmc is not None:
            self.detach()

        ev = core.events
        ev.imageSnapped.connect(self._on_image_snapped)
        ev.continuousSequenceAcquisitionStarted.connect(self._on_streaming_start)
        ev.sequenceAcquisitionStarted.connect(self._on_streaming_start)
        ev.sequenceAcquisitionStopped.connect(self._on_streaming_stop)
        ev.exposureChanged.connect(self._on_exposure_changed)
        ev.systemConfigurationLoaded.connect(self._on_system_config_loaded)
        ev.roiSet.connect(self._on_roi_set)
        ev.propertyChanged.connect(self._on_property_changed)
        self._mda_started_callback = lambda: setattr(
            self, "_is_mda_running", True
        )
        self._mda_finished_callback = lambda: setattr(
            self, "_is_mda_running", False
        )
        core.mda.events.sequenceStarted.connect(self._mda_started_callback)
        core.mda.events.sequenceFinished.connect(self._mda_finished_callback)

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
        )
        for signal, callback in connections:
            if callback is not None:
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
        if not self.use_with_mda and self._is_mda_running:
            return  # pragma: no cover

        last = core.getImage()
        self.append(last)

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
