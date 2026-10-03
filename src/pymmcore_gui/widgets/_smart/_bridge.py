"""Qt front for `pymmcore_plus.smart.SmartRunner`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pymmcore_plus.smart import SmartRunner

from pymmcore_gui._qt.QtCore import QObject, Signal

if TYPE_CHECKING:
    from pathlib import Path

    import useq
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.mda import SingleOutput
    from pymmcore_plus.smart import HookResult, SmartRunConfig


class SmartController(QObject):
    """Owns a `SmartRunner` and re-emits its signals as Qt signals.

    The runner emits from whichever thread produced an event -- the
    acquisition thread, an analysis callback thread, its finalizer. Re-emitting
    through this GUI-thread QObject queues every connected slot onto the GUI
    thread, so widgets can be updated directly.
    """

    runStarted = Signal(object)
    """dict: ``run_dir`` and the run's settings."""
    frameAcquired = Signal(object)
    """dict: the ``frames.jsonl`` record of each acquired frame."""
    analysisQueued = Signal(int)
    analysisFinished = Signal(object)
    """dict: the ``analysis.jsonl`` record of each completed hook call."""
    logMessage = Signal(str, str)
    analysisError = Signal(str, bool)
    runFinished = Signal(object)
    """dict summary: status, run_dir and counts."""

    def __init__(self, mmcore: CMMCorePlus, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.runner = SmartRunner(mmcore)
        events = self.runner.events
        # Bound methods of this object (held weakly by psygnal), not the Qt
        # signals' emit: a bound Qt signal is a short-lived wrapper.
        events.runStarted.connect(self._relay_run_started)
        events.frameAcquired.connect(self._relay_frame_acquired)
        events.analysisQueued.connect(self._relay_analysis_queued)
        events.analysisFinished.connect(self._relay_analysis_finished)
        events.logMessage.connect(self._relay_log_message)
        events.analysisError.connect(self._relay_analysis_error)
        events.runFinished.connect(self._relay_run_finished)

    # ------------------------------------------------------------ runner API

    @property
    def config(self) -> SmartRunConfig | None:
        return self.runner.config

    @property
    def run_dir(self) -> Path | None:
        return self.runner.run_dir

    def is_active(self) -> bool:
        return self.runner.is_active()

    def prepare(
        self,
        base: useq.MDASequence,
        config: SmartRunConfig,
        *,
        output: SingleOutput | None = None,
    ) -> HookResult:
        """Start the analysis worker (blocks: call off the GUI thread).

        The run's records always go to a folder: next to the data when saving,
        otherwise a temporary one the monitor points at.
        """
        return self.runner.prepare(
            base, config, output=output, run_dir="auto", packages=("pymmcore-gui",)
        )

    def start(self) -> None:
        self.runner.start()

    def request_stop(self) -> None:
        self.runner.request_stop()

    def cancel(self) -> None:
        self.runner.cancel()

    def abandon(self) -> None:
        self.runner.abandon()

    def shutdown(self) -> None:
        self.runner.shutdown()

    # ---------------------------------------------------------------- relays

    def _relay_run_started(self, info: dict[str, Any]) -> None:
        self.runStarted.emit(info)

    def _relay_frame_acquired(self, record: dict[str, Any]) -> None:
        self.frameAcquired.emit(record)

    def _relay_analysis_queued(self, frame_id: int) -> None:
        self.analysisQueued.emit(frame_id)

    def _relay_analysis_finished(self, record: dict[str, Any]) -> None:
        self.analysisFinished.emit(record)

    def _relay_log_message(self, level: str, message: str) -> None:
        self.logMessage.emit(level, message)

    def _relay_analysis_error(self, message: str, fatal: bool) -> None:
        self.analysisError.emit(message, fatal)

    def _relay_run_finished(self, summary: dict[str, Any]) -> None:
        self.runFinished.emit(summary)
