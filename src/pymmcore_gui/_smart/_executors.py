"""Run a script's hooks on a background thread or in a separate process.

Both executors drive the same `_ScriptHost`, one call at a time and in order,
so a script behaves identically in either mode. They differ only in isolation:

- **thread**: no startup cost and no copying, but the script shares the GUI's
  interpreter -- pure-Python analysis competes with the GUI for the GIL, and a
  crash in a C extension takes the whole application down. A hung analysis
  cannot be interrupted.
- **process**: a spawned interpreter (never forked: forking a process that
  runs Qt and the core's threads is unsafe). Startup costs seconds and every
  frame is pickled across, but a crash or hang is contained and the process
  can be terminated.
"""

from __future__ import annotations

import multiprocessing
from abc import ABC, abstractmethod
from concurrent.futures import (
    CancelledError,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from concurrent.futures import TimeoutError as FutureTimeoutError
from concurrent.futures.process import BrokenProcessPool
from contextlib import suppress
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import psutil

from pymmcore_gui._smart._worker import (
    WorkerResult,
    _proc_analyze,
    _proc_init,
    _proc_setup,
    _proc_teardown,
    _ScriptHost,
    read_only_view,
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from pymmcore_gui.smart._api import ExecutionMode, FrameInfo


class ExecutorStartError(RuntimeError):
    """The analysis worker could not be started (timeout or crash)."""


class AnalysisExecutor(ABC):
    """Runs one script's hooks for the duration of one run."""

    mode: ExecutionMode

    @abstractmethod
    def start(
        self,
        path: Path,
        params: dict[str, Any],
        run_dir: Path,
        *,
        source: str | None = None,
        max_events_per_response: int = 1000,
        timeout: float = 60.0,
    ) -> WorkerResult:
        """Load the script and call its ``setup``; return that call's result.

        *source*, when given, is executed instead of re-reading *path* (which
        is still used for tracebacks and to import sibling modules).

        Raises `ExecutorStartError` if the worker does not come up in time.
        A script error (import or ``setup`` raising) is *not* raised: it is
        reported in the returned result, with ``ok=False``.
        """

    @abstractmethod
    def submit(self, image: np.ndarray, frame: FrameInfo) -> Future[WorkerResult]:
        """Queue ``analyze(image, frame, ctx)``; results arrive in submission order."""

    @abstractmethod
    def stop(self, timeout: float = 10.0) -> WorkerResult | None:
        """Call ``teardown`` (best effort) and release the worker. Idempotent.

        Returns the teardown result, or None if it could not run (already
        stopped, worker broken, or timed out).
        """

    @property
    @abstractmethod
    def broken(self) -> bool:
        """Whether the worker died; nothing more can be submitted."""


class ThreadAnalysisExecutor(AnalysisExecutor):
    mode: ExecutionMode = "thread"

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="smart-analysis"
        )
        self._host: _ScriptHost | None = None
        self._stopped = False

    def start(
        self,
        path: Path,
        params: dict[str, Any],
        run_dir: Path,
        *,
        source: str | None = None,
        max_events_per_response: int = 1000,
        timeout: float = 60.0,
    ) -> WorkerResult:
        def _load_and_setup() -> WorkerResult:
            # On the worker thread: the script's import-time side effects and
            # its setup() must not run on (or block) the GUI thread.
            self._host = _ScriptHost(
                path,
                params,
                run_dir,
                "thread",
                source=source,
                max_events_per_response=max_events_per_response,
            )
            return self._host.setup()

        try:
            return self._pool.submit(_load_and_setup).result(timeout)
        except FutureTimeoutError as e:
            raise ExecutorStartError(
                f"The analysis script did not finish loading within {timeout:g} s."
            ) from e

    def submit(self, image: np.ndarray, frame: FrameInfo) -> Future[WorkerResult]:
        host = self._host
        if host is None:
            raise RuntimeError("Executor not started.")
        return self._pool.submit(host.analyze, read_only_view(image), frame)

    def stop(self, timeout: float = 10.0) -> WorkerResult | None:
        if self._stopped:
            return None
        self._stopped = True
        result: WorkerResult | None = None
        if (host := self._host) is not None:

            def _teardown() -> WorkerResult:
                try:
                    return host.teardown()
                finally:
                    host.close()

            with suppress(FutureTimeoutError, CancelledError, RuntimeError):
                result = self._pool.submit(_teardown).result(timeout)
        # A hung analyze() cannot be interrupted: its thread lingers until the
        # call returns, but nothing else will ever be scheduled on it.
        self._pool.shutdown(wait=False, cancel_futures=True)
        return result

    @property
    def broken(self) -> bool:
        return False


class ProcessAnalysisExecutor(AnalysisExecutor):
    mode: ExecutionMode = "process"

    def __init__(self) -> None:
        self._pool: ProcessPoolExecutor | None = None
        self._pids: list[int] = []
        self._broken = False
        self._stopped = False

    def start(
        self,
        path: Path,
        params: dict[str, Any],
        run_dir: Path,
        *,
        source: str | None = None,
        max_events_per_response: int = 1000,
        timeout: float = 60.0,
    ) -> WorkerResult:
        self._pool = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_proc_init,
            initargs=(
                str(path),
                source,
                dict(params),
                str(run_dir),
                max_events_per_response,
            ),
        )
        try:
            future = self._pool.submit(_proc_setup)
            # Read from the pool rather than asked of the child: a child that
            # hangs while importing the script could never answer.
            self._pids = self._child_pids()
            return future.result(timeout)
        except FutureTimeoutError as e:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._kill_child()
            raise ExecutorStartError(
                f"The analysis process did not start within {timeout:g} s."
            ) from e
        except BrokenProcessPool as e:
            self._broken = True
            raise ExecutorStartError(
                "The analysis process exited while loading the script."
            ) from e

    def submit(self, image: np.ndarray, frame: FrameInfo) -> Future[WorkerResult]:
        if self._pool is None:
            raise RuntimeError("Executor not started.")
        # The parent sequence is irrelevant to the script and would be pickled
        # along with every single frame.
        frame = replace(frame, event=frame.event.model_copy(update={"sequence": None}))
        future = self._pool.submit(_proc_analyze, image, frame)
        future.add_done_callback(self._note_broken)
        return future

    def _note_broken(self, future: Future[WorkerResult]) -> None:
        if not future.cancelled() and isinstance(future.exception(), BrokenProcessPool):
            self._broken = True

    def stop(self, timeout: float = 10.0) -> WorkerResult | None:
        if self._stopped or self._pool is None:
            return None
        self._stopped = True
        result: WorkerResult | None = None
        if not self._broken:
            with suppress(
                FutureTimeoutError, BrokenProcessPool, CancelledError, RuntimeError
            ):
                result = self._pool.submit(_proc_teardown).result(timeout)
        self._pool.shutdown(wait=False, cancel_futures=True)
        # An idle child exits on its own once the pool is shut down; a hung
        # one (stuck in analyze or teardown) is terminated.
        self._kill_child(grace=0 if result is None else 2.0)
        return result

    def _child_pids(self) -> list[int]:
        # ProcessPoolExecutor starts its worker when the first call is
        # submitted and exposes it only privately; there is no public API.
        processes = getattr(self._pool, "_processes", None) or {}
        return [p.pid for p in processes.values() if p.pid is not None]

    def _kill_child(self, grace: float = 0) -> None:
        """Wait up to *grace* s for the worker to exit, then terminate it.

        A worker left running would also block interpreter exit, which joins
        the pool's management thread, which waits for the worker.
        """
        for pid in self._pids:
            with suppress(psutil.NoSuchProcess):
                child = psutil.Process(pid)
                try:
                    child.wait(timeout=grace)
                    continue
                except psutil.TimeoutExpired:
                    pass
                child.terminate()
                try:
                    child.wait(timeout=2)
                except psutil.TimeoutExpired:
                    child.kill()

    @property
    def broken(self) -> bool:
        return self._broken


def create_executor(mode: ExecutionMode) -> AnalysisExecutor:
    """Return a fresh, unstarted executor for *mode*."""
    if mode == "thread":
        return ThreadAnalysisExecutor()
    if mode == "process":
        return ProcessAnalysisExecutor()
    raise ValueError(f"Unknown execution mode {mode!r}")
