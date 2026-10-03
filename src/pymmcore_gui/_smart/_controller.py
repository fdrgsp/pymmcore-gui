"""Runs one smart acquisition: base events, analysis, and the feedback between them.

Threads involved, and what each one does here:

- **GUI thread**: `prepare`/`start`/`cancel`; receives every signal below.
- **Runner thread**: pulls events from `SmartEventIterator`, and calls
  `_on_frame_ready` for each frame. That handler is connected with a direct
  connection -- in this app the MDA signals are Qt signals, and a normal
  connection would queue it to the GUI thread, tying analysis latency to
  repaints (and blocking mode to the GUI's responsiveness). It must stay fast.
- **Executor callback thread**: `_on_result`, which turns a script's answer
  into queued events.
- **Finalizer thread**: tears the executor down after the run, so a slow or
  hung ``teardown`` never freezes the GUI.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Any, Final, Literal

from pymmcore_gui._qt.QtCore import QObject, Qt, Signal
from pymmcore_gui._smart._executors import (
    AnalysisExecutor,
    ExecutorStartError,
    create_executor,
)
from pymmcore_gui._smart._log import SmartRunLog, event_to_json
from pymmcore_gui._smart._scheduler import (
    SmartEventIterator,
    StopReason,
    provenance,
)
from pymmcore_gui.smart._api import FrameInfo

if TYPE_CHECKING:
    from concurrent.futures import Future
    from pathlib import Path

    import numpy as np
    import useq
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.mda import SingleOutput
    from pymmcore_plus.metadata import FrameMetaV1

    from pymmcore_gui._smart._loader import AnalyzeFilter, ScriptSpec
    from pymmcore_gui._smart._worker import WorkerResult
    from pymmcore_gui.smart._api import ExecutionMode, Response, SyncMode

OnError = Literal["stop", "skip"]

TEARDOWN_TIMEOUT_S: Final = 10.0


class SmartRunError(RuntimeError):
    """A smart run could not be prepared or started."""


@dataclass(frozen=True)
class SmartRunConfig:
    """Everything that determines how a smart run behaves."""

    spec: ScriptSpec
    params: dict[str, Any]
    execution: ExecutionMode
    sync: SyncMode
    filter: AnalyzeFilter
    on_error: OnError = "stop"
    analysis_timeout_s: float | None = None
    max_total_events: int = 10_000
    max_events_per_response: int = 1_000
    setup_timeout_s: float = 60.0

    def to_json(self) -> dict[str, Any]:
        return {
            "script": {
                "path": str(self.spec.path),
                "name": self.spec.name,
                "sha256": self.spec.sha256,
                "api_version": self.spec.api_version,
            },
            "params": self.params,
            "execution": self.execution,
            "sync": self.sync,
            "filter": self.filter.to_dict(),
            "on_error": self.on_error,
            "analysis_timeout_s": self.analysis_timeout_s,
            "max_total_events": self.max_total_events,
            "max_events_per_response": self.max_events_per_response,
        }


@dataclass
class SmartRunStats:
    """Running totals, readable from any thread (copied under a lock)."""

    frames: int = 0
    analyses_queued: int = 0
    analyses_done: int = 0
    errors: int = 0
    injected: int = 0
    dropped: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def response_summary(response: Response | None) -> dict[str, Any] | None:
    if response is None:
        return None
    return {
        # normalise_response always leaves a tuple of events
        "n_events": len(response.events)
        if isinstance(response.events, tuple)
        else None,
        "priority": response.priority,
        "timing": response.timing,
        "stop": response.stop,
        "drop_base": response.drop_base,
    }


class SmartController(QObject):
    """Prepares, runs and records one smart acquisition at a time."""

    runStarted = Signal(object)
    """`SmartRunLog` of the run that just started (its ``run_dir`` etc.)."""
    frameAcquired = Signal(object)
    """dict: the ``frames.jsonl`` record of each acquired frame."""
    analysisQueued = Signal(int)
    """frame_id sent to ``analyze``."""
    analysisFinished = Signal(object)
    """dict: the ``analysis.jsonl`` record of each completed call."""
    logMessage = Signal(str, str)
    """(level, message) logged by the script, or by the controller itself."""
    analysisError = Signal(str, bool)
    """(message, fatal). Fatal: the worker died; the run is stopping."""
    runFinished = Signal(object)
    """dict summary: status, run_dir and counts."""

    _finished = Signal(object)  # internal: finalizer thread -> GUI thread

    def __init__(self, mmcore: CMMCorePlus, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._mmc = mmcore
        self._lock = threading.Lock()
        self._config: SmartRunConfig | None = None
        self._executor: AnalysisExecutor | None = None
        self._setup_result: WorkerResult | None = None
        self._iterator: SmartEventIterator | None = None
        self._log: SmartRunLog | None = None
        self._frame_ids = itertools.count()
        self._response_ids = itertools.count()
        self._stats = SmartRunStats()
        self._active = False  # between start() and the end of finalization
        self._acquiring = False  # between start() and sequenceFinished
        self._user_cancelled = False
        self._connected = False
        self._finished.connect(self._on_finalized)

    # ------------------------------------------------------------ public API

    @property
    def config(self) -> SmartRunConfig | None:
        return self._config

    @property
    def run_log(self) -> SmartRunLog | None:
        return self._log

    @property
    def iterator(self) -> SmartEventIterator | None:
        return self._iterator

    def is_active(self) -> bool:
        """Whether a run is prepared, acquiring, or still finalizing."""
        return self._active or self._executor is not None

    def stats(self) -> SmartRunStats:
        with self._lock:
            return replace(self._stats, extra=dict(self._stats.extra))

    def prepare(self, config: SmartRunConfig, run_dir: Path) -> WorkerResult:
        """Start the analysis worker and run the script's ``setup``.

        Blocks until the worker is ready (seconds, in process mode): call it
        off the GUI thread or behind a busy indicator. Raises `SmartRunError`
        when the worker does not start or the script fails to load or set up;
        the worker is stopped again in that case.
        """
        if self.is_active():
            raise SmartRunError("A smart run is already in progress.")
        executor = create_executor(config.execution)
        try:
            result = executor.start(
                config.spec.path,
                config.params,
                run_dir,
                source=config.spec.source,
                max_events_per_response=config.max_events_per_response,
                timeout=config.setup_timeout_s,
            )
        except ExecutorStartError as e:
            executor.stop(timeout=1)
            raise SmartRunError(str(e)) from e
        if not result.ok:
            executor.stop(timeout=TEARDOWN_TIMEOUT_S)
            raise SmartRunError(
                f"The script failed to load or set up:\n\n{result.error}"
            )
        self._config = config
        self._executor = executor
        self._setup_result = result
        self._log = SmartRunLog(run_dir)
        return result

    def start(
        self,
        base: useq.MDASequence,
        output: SingleOutput | None = None,
        *,
        data_path: str | Path | None = None,
    ) -> None:
        """Start acquiring *base*, feeding frames to the prepared script."""
        config, executor, log = self._config, self._executor, self._log
        if config is None or executor is None or log is None:
            raise SmartRunError("Call prepare() before start().")
        if self._active:
            raise SmartRunError("A smart run is already in progress.")
        if next(iter(base), None) is None:
            # useq yields no events at all for a sequence with no axes; even a
            # purely reactive run needs one event to acquire the first frame.
            raise SmartRunError(
                "The base acquisition contains no events. Enable at least one "
                "channel (or position) so there is a first frame to analyze."
            )

        self._frame_ids = itertools.count()
        self._response_ids = itertools.count()
        self._stats = SmartRunStats()
        self._user_cancelled = False
        self._iterator = SmartEventIterator(
            base,
            self._mmc.mda,
            sync=config.sync,
            max_total_events=config.max_total_events,
            analysis_timeout_s=config.analysis_timeout_s,
            on_analysis_timeout=self._on_analysis_timeout,
        )
        log.open(
            {
                **config.to_json(),
                "base_sequence": base.model_dump(mode="json"),
                "data_path": None if data_path is None else str(data_path),
            },
            config.spec.source,
        )
        if (setup := self._setup_result) is not None:
            self._record_result(setup, injected=0, dropped=False)

        self._active = True
        self._acquiring = True
        self._connect()
        try:
            self._mmc.run_mda(self._iterator, output=output)
        except Exception:
            self._acquiring = False
            self._disconnect()
            self._finalize(status="error", stop_executor=True)
            raise
        self.runStarted.emit(log)

    def request_stop(self) -> None:
        """Finish after the event currently running; nothing more is queued."""
        if (iterator := self._iterator) is not None and self._acquiring:
            iterator.stop(StopReason.USER)
            self.logMessage.emit("info", "Stopping after the current event.")

    def cancel(self) -> None:
        """Cancel the run now (frames already acquired are kept)."""
        if not self._acquiring:
            return
        self._user_cancelled = True
        if (iterator := self._iterator) is not None:
            iterator.stop(StopReason.USER)
        self._mmc.mda.cancel()

    def abandon(self) -> None:
        """Release a prepared run that will not be started."""
        if self._executor is not None and not self._active:
            self._executor.stop(timeout=TEARDOWN_TIMEOUT_S)
            self._executor = None
            self._config = None
            self._log = None
            self._setup_result = None

    def shutdown(self) -> None:
        """Stop everything immediately (window closing). Idempotent."""
        if self._acquiring:
            self.cancel()
        self._disconnect()
        if (executor := self._executor) is not None:
            executor.stop(timeout=2)
            self._executor = None
        if (log := self._log) is not None:
            log.finish("aborted")

    # --------------------------------------------------------- runner thread

    def _connect(self) -> None:
        events = self._mmc.mda.events
        if isinstance(events, QObject):
            events.frameReady.connect(
                self._on_frame_ready, Qt.ConnectionType.DirectConnection
            )
        else:  # psygnal: already synchronous in the emitting (runner) thread
            events.frameReady.connect(self._on_frame_ready)
        events.sequenceFinished.connect(self._on_sequence_finished)
        self._connected = True

    def _disconnect(self) -> None:
        if not self._connected:
            return
        self._connected = False
        events = self._mmc.mda.events
        for signal, slot in (
            (events.frameReady, self._on_frame_ready),
            (events.sequenceFinished, self._on_sequence_finished),
        ):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError, ValueError):
                pass

    def _on_frame_ready(
        self, img: np.ndarray, event: useq.MDAEvent, meta: FrameMetaV1
    ) -> None:
        config, executor, log, iterator = (
            self._config,
            self._executor,
            self._log,
            self._iterator,
        )
        if config is None or executor is None or log is None or iterator is None:
            return
        frame_id = next(self._frame_ids)
        info = provenance(event)
        origin = info.get("origin", "base")
        parent = info.get("parent_frame_id")
        frame = FrameInfo(
            frame_id=frame_id,
            event=event.model_copy(update={"sequence": None}),
            metadata=dict(meta),
            origin=origin,
            parent_frame_id=parent,
        )
        record = {
            "frame_id": frame_id,
            "t_index": frame_id,
            "origin": origin,
            "parent_frame_id": parent,
            "response_id": info.get("response_id"),
            "event": event_to_json(event),
            "runner_time_ms": meta.get("runner_time_ms"),
            "camera": meta.get("camera_device"),
            "exposure_ms": meta.get("exposure_ms"),
            "position": meta.get("position"),
        }
        log.write_frame(record)
        with self._lock:
            self._stats.frames += 1
        self.frameAcquired.emit(record)

        if executor.broken or not config.filter.accepts(frame_id, event, origin):
            return
        iterator.analysis_submitted(frame_id)
        try:
            future = executor.submit(img, frame)
        except Exception as e:  # executor shut down / broken pool
            iterator.analysis_finished(frame_id)
            self._fatal(f"Could not send frame {frame_id} to analysis: {e}")
            return
        with self._lock:
            self._stats.analyses_queued += 1
        future.add_done_callback(partial(self._on_result, frame_id))
        self.analysisQueued.emit(frame_id)

    # ----------------------------------------------- executor callback thread

    def _on_result(self, frame_id: int, future: Future[WorkerResult]) -> None:
        iterator, config = self._iterator, self._config
        try:
            if future.cancelled():
                return
            if (exc := future.exception()) is not None:
                self._fatal(
                    f"The analysis worker died while analyzing frame {frame_id}: "
                    f"{exc!r}"
                )
                return
            result = future.result()
            injected, dropped = 0, False
            if result.ok and result.response is not None and iterator is not None:
                injected, dropped = self._apply(result, iterator)
            elif not result.ok:
                self._on_script_error(result, config)
            self._record_result(result, injected=injected, dropped=dropped)
        finally:
            if iterator is not None:
                iterator.analysis_finished(frame_id)

    def _apply(
        self, result: WorkerResult, iterator: SmartEventIterator
    ) -> tuple[int, bool]:
        """Act on a successful response; return (events injected, dropped?)."""
        response = result.response
        assert response is not None
        if response.drop_base:
            iterator.drop_base()
        injected = 0
        dropped = False
        if response.events:
            injected = iterator.inject(
                list(response.events),
                priority=response.priority,
                parent_frame_id=result.frame_id if result.frame_id is not None else -1,
                response_id=next(self._response_ids),
                relative_timing=response.timing == "relative",
            )
            dropped = injected == 0
        if response.stop:
            iterator.stop(StopReason.SCRIPT)
            self.logMessage.emit(
                "info", f"Frame {result.frame_id}: the script stopped the run."
            )
        with self._lock:
            self._stats.injected += injected
            self._stats.dropped += int(dropped)
        return injected, dropped

    def _on_script_error(
        self, result: WorkerResult, config: SmartRunConfig | None
    ) -> None:
        with self._lock:
            self._stats.errors += 1
        message = f"analyze() raised on frame {result.frame_id}:\n{result.error}"
        if config is not None and config.on_error == "skip":
            self.logMessage.emit("error", message)
            return
        if (iterator := self._iterator) is not None:
            iterator.stop(StopReason.ERROR)
        self.analysisError.emit(message, False)

    def _fatal(self, message: str) -> None:
        with self._lock:
            self._stats.errors += 1
        if (iterator := self._iterator) is not None:
            iterator.stop(StopReason.ERROR)
        self.analysisError.emit(message, True)

    def _on_analysis_timeout(self, frame_id: int) -> None:
        timeout = self._config.analysis_timeout_s if self._config else None
        self.analysisError.emit(
            f"Analysis of frame {frame_id} took longer than {timeout:g} s; "
            "stopping the run.",
            False,
        )

    def _record_result(
        self, result: WorkerResult, *, injected: int, dropped: bool
    ) -> None:
        record = {
            "frame_id": result.frame_id,
            "call": result.call,
            "ok": result.ok,
            "duration_ms": round(result.duration_ms, 3),
            "records": result.records,
            "logs": result.logs,
            "response": response_summary(result.response),
            "injected": injected,
            "dropped": dropped,
            "error": result.error,
        }
        if (log := self._log) is not None:
            log.write_analysis(record)
        if result.call == "analyze":
            with self._lock:
                self._stats.analyses_done += 1
        for level, message in result.logs:
            self.logMessage.emit(level, message)
        self.analysisFinished.emit(record)

    # ------------------------------------------------------- end of the run

    def _on_sequence_finished(self, *_: object) -> None:
        """GUI thread (queued): acquisition over; finish off the analysis side."""
        if not self._acquiring:
            return
        self._acquiring = False
        self._disconnect()
        self._finalize(status=self._final_status(), stop_executor=True)

    def _final_status(self) -> str:
        if self._user_cancelled:
            return "cancelled"
        reason = self._iterator.stop_reason if self._iterator else None
        if reason == StopReason.RUNNER:
            # FINISHING without a stop of ours: cancelled from elsewhere (the
            # Acquire page's Cancel button, a script in the console...).
            return "cancelled"
        finish_reason = self._mmc.mda.status.finish_reason
        if finish_reason is not None and str(finish_reason) == "errored":
            return "error"
        return reason or StopReason.COMPLETED

    def _finalize(self, *, status: str, stop_executor: bool) -> None:
        executor, log = self._executor, self._log

        def _run() -> None:
            teardown: WorkerResult | None = None
            if stop_executor and executor is not None:
                teardown = executor.stop(timeout=TEARDOWN_TIMEOUT_S)
            if teardown is not None:
                self._record_result(teardown, injected=0, dropped=False)
            stats = self.stats()
            summary = {
                "status": status,
                "run_dir": str(log.run_dir) if log else None,
                "frames": stats.frames,
                "analyses": stats.analyses_done,
                "errors": stats.errors,
                "injected_events": stats.injected,
                "dropped_responses": stats.dropped,
            }
            if log is not None:
                log.finish(
                    status,
                    counts={
                        k: v
                        for k, v in summary.items()
                        if k not in ("status", "run_dir")
                    },
                )
            self._finished.emit(summary)

        threading.Thread(target=_run, name="smart-finalize", daemon=True).start()

    def _on_finalized(self, summary: dict[str, Any]) -> None:
        self._executor = None
        self._setup_result = None
        self._active = False
        self.runFinished.emit(summary)
