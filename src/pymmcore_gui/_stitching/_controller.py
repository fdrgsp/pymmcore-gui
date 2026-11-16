"""Queue worker processes, poll progress files and clean up temporary outputs.

Workers and background cleanup never block the GUI thread during a job.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any

from pymmcore_gui._qt.QtCore import (
    QObject,
    QProcess,
    QProcessEnvironment,
    QRunnable,
    QThreadPool,
    QTimer,
    Signal,
)
from pymmcore_gui._stitching._models import (
    GroupV1,
    ProgressEvent,
    StitchErrorV1,
    StitchJobV1,
    StitchResultV1,
)
from pymmcore_gui._stitching._paths import cleanup_paths, remove_path, unique_path
from pymmcore_gui._stitching._protocol import (
    JobDir,
    process_alive,
    registry_dir,
    write_model,
)
from pymmcore_gui._stitching._worker import THREAD_ENV_VARS

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np
    from pymmcore_plus import CMMCorePlus

logger = logging.getLogger(__name__)

POLL_INTERVAL_MS = 250
KEEP_FAILED_LOGS_S = 3 * 24 * 3600
MEMORY_CAP_SHARE = 0.25
"""Share of the RAM available at submit time that fusion may keep in flight."""


def worker_command(job_dir: Path) -> tuple[str, list[str]]:
    """Program and arguments starting the worker on `job_dir`.

    In development `sys.executable` is Python, so run the package; frozen,
    it is the app itself, whose entry point dispatches the same argument.
    """
    from pymmcore_gui.__main__ import STITCH_WORKER_ARG

    if getattr(sys, "frozen", False):
        return sys.executable, [STITCH_WORKER_ARG, str(job_dir)]
    return sys.executable, ["-m", "pymmcore_gui", STITCH_WORKER_ARG, str(job_dir)]


def worker_environment() -> QProcessEnvironment:
    env = QProcessEnvironment.systemEnvironment()
    for var in THREAD_ENV_VARS:
        env.insert(var, "1")
    env.insert("TQDM_DISABLE", "1")
    env.insert("PYTHONUNBUFFERED", "1")
    if getattr(sys, "frozen", False):
        # a fresh instance of a onefile bundle, not a child of this one
        env.insert("PYINSTALLER_RESET_ENVIRONMENT", "1")
    return env


def own_process_identity() -> tuple[int, float | None]:
    """Return the GUI process's PID and start time, for the worker's watchdog."""
    import psutil

    try:
        proc = psutil.Process()
        return proc.pid, proc.create_time()
    except psutil.Error:  # pragma: no cover
        return os.getpid(), None


def default_memory_cap() -> int:
    import psutil

    return int(psutil.virtual_memory().available * MEMORY_CAP_SHARE)


def capture_linear_part(core: CMMCorePlus) -> tuple[float, np.ndarray] | None:
    """Pixel size and `A` for the current camera state, as the Stage Explorer has it.

    Called at `sequenceStarted` for a fresh run. None without a pixel
    size: the grid could not have been planned then either.
    """
    import numpy as np
    from pymmcore_widgets.control._stage_explorer._stage_explorer import (
        AffineState,
    )

    try:
        state = AffineState(core)
    except Exception:  # pragma: no cover - no camera, no preset...
        logger.exception("Could not read the pixel affine")
        return None
    if not state.pixel_size_um > 0:
        return None
    return float(state.pixel_size_um), np.array(state.system_affine[:2, :2])


class _Task(QRunnable):
    def __init__(self, func: Callable[[], Any]) -> None:
        super().__init__()
        self._func = func

    def run(self) -> None:
        try:
            self._func()
        except Exception:  # pragma: no cover
            logger.exception("stitching cleanup failed")


def cleanup_stale_jobs(registry: Path, *, now: float | None = None) -> list[Path]:
    """Remove what dead jobs left behind; return the deleted paths.

    Only paths a job lists are touched -- the `.tmp` mosaics in its
    `job.json` and its own directory -- never a pattern in a user folder.
    A job whose worker or GUI still runs (another instance) is left alone,
    and a failed job keeps its log for a few days.
    """
    now = time.time() if now is None else now
    removed: list[Path] = []
    if not registry.is_dir():
        return removed
    for entry in registry.iterdir():
        if not entry.is_dir():
            continue
        jobdir = JobDir(entry)
        owner = jobdir.read_owner()
        if owner is not None and process_alive(owner.pid, owner.create_time):
            continue
        job = jobdir.read_job()
        try:
            age = now - entry.stat().st_mtime
        except OSError:
            continue
        if job is None and owner is None and age < KEEP_FAILED_LOGS_S:
            continue  # being created now, or a crashed submit: aged out below
        if (
            job is not None
            and job.parent_pid is not None
            and process_alive(job.parent_pid, job.parent_create_time)
        ):
            continue
        if job is not None:
            for tmp in job.tmp_outputs:
                if tmp.exists():
                    remove_path(tmp)
                    removed.append(tmp)
        failed = jobdir.error_file.exists() or (
            job is not None and not jobdir.result_file.exists() and owner is not None
        )
        if failed and age < KEEP_FAILED_LOGS_S:
            continue
        remove_path(entry)
        removed.append(entry)
    return removed


class JobState(Enum):
    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"

    @property
    def is_active(self) -> bool:
        return self in _ACTIVE

    @property
    def is_done(self) -> bool:
        return self in (JobState.FINISHED, JobState.FAILED, JobState.CANCELLED)


_ACTIVE = {JobState.QUEUED, JobState.STARTING, JobState.RUNNING, JobState.CANCELLING}

_TRANSITIONS: dict[JobState, set[JobState]] = {
    JobState.QUEUED: {JobState.STARTING, JobState.CANCELLED},
    JobState.STARTING: {JobState.RUNNING, JobState.FAILED, JobState.CANCELLING},
    JobState.RUNNING: {JobState.FINISHED, JobState.FAILED, JobState.CANCELLING},
    JobState.CANCELLING: {JobState.CANCELLED},
    JobState.FINISHED: set(),
    JobState.FAILED: set(),
    JobState.CANCELLED: set(),
}


@dataclass(eq=False)
class StitchJob:
    """One job as the GUI tracks it."""

    job: StitchJobV1
    dir: JobDir
    title: str
    state: JobState = JobState.QUEUED
    progress: ProgressEvent = field(default_factory=ProgressEvent)
    result: StitchResultV1 | None = None
    error: StitchErrorV1 | None = None
    exit_code: int | None = None
    process: QProcess | None = field(default=None, repr=False)

    @property
    def id(self) -> str:
        return self.job.job_id

    @property
    def status_text(self) -> str:
        if self.state == JobState.QUEUED:
            return "Waiting"
        if self.state == JobState.CANCELLING:
            return "Cancelling…"
        if self.state.is_done:
            return str(self.state.value).capitalize()
        return self.progress.text


class StitchingController(QObject):
    """Runs stitching jobs one at a time in a worker process.

    Signals
    -------
    changed()
        Any job was added, removed, or changed state or progress.
    jobFinished(StitchJob)
        A job wrote every mosaic.
    jobFailed(StitchJob)
        A job failed or its worker stopped unexpectedly.
    jobCancelled(StitchJob)
        A cancelled job's worker has exited (its partial output is being
        deleted in the background).
    """

    changed = Signal()
    jobFinished = Signal(object)
    jobFailed = Signal(object)
    jobCancelled = Signal(object)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        registry: Path | None = None,
        command: Callable[[Path], tuple[str, list[str]]] = worker_command,
        cleanup_on_start: bool = True,
    ) -> None:
        super().__init__(parent)
        self._registry = registry if registry is not None else registry_dir()
        self._command = command
        self._jobs: list[StitchJob] = []
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._poll)
        if cleanup_on_start:
            registry = self._registry
            self._pool.start(_Task(lambda: cleanup_stale_jobs(registry)))

    @property
    def registry(self) -> Path:
        return self._registry

    @property
    def jobs(self) -> list[StitchJob]:
        """Every job this session knows, oldest first."""
        return list(self._jobs)

    @property
    def active_jobs(self) -> list[StitchJob]:
        return [j for j in self._jobs if j.state.is_active]

    @property
    def running_job(self) -> StitchJob | None:
        return next(
            (
                j
                for j in self._jobs
                if j.state in (JobState.STARTING, JobState.RUNNING, JobState.CANCELLING)
            ),
            None,
        )

    def is_busy(self) -> bool:
        """Whether any job is queued or running."""
        return bool(self.active_jobs)

    def job(self, job_id: str) -> StitchJob | None:
        return next((j for j in self._jobs if j.id == job_id), None)

    def new_job_id(self) -> str:
        return uuid.uuid4().hex

    def submit(self, job: StitchJobV1, title: str | None = None) -> StitchJob:
        """Queue `job`; it starts when no other job is running."""
        jobdir = JobDir(self._registry / job.job_id)
        jobdir.path.mkdir(parents=True, exist_ok=False)
        if job.parent_pid is None:
            pid, created = own_process_identity()
            job = job.model_copy(
                update={"parent_pid": pid, "parent_create_time": created}
            )
        # Each output name is reserved by creating its `.tmp` directory next to
        # the acquisition. `job.json` lists it *before* it is created, so if the
        # GUI dies mid-submit, startup cleanup finds the reservation through
        # the dead job. Only names already reserved, plus the one being tried,
        # are listed: never one that may belong to another job.
        reserved: list[Path] = []
        try:
            groups: list[GroupV1] = []
            for group in job.groups:
                while True:
                    candidate = group.model_copy(
                        update={"output": unique_path(group.output)}
                    )
                    write_model(
                        jobdir.job_file,
                        job.model_copy(update={"groups": [*groups, candidate]}),
                    )
                    try:
                        candidate.tmp_output.mkdir()
                    except FileExistsError:
                        continue  # taken meanwhile: list the next name instead
                    reserved.append(candidate.tmp_output)
                    groups.append(candidate)
                    break
            job = job.model_copy(update={"groups": groups})
        except Exception:
            cleanup_paths([*reserved, jobdir.path])
            raise
        handle = StitchJob(job=job, dir=jobdir, title=title or job.source.name)
        self._jobs.append(handle)
        self.changed.emit()
        self._start_next()
        return handle

    def cancel(self, job: StitchJob | str) -> None:
        """Remove a queued job, or kill a running one and delete its output."""
        handle = self.job(job) if isinstance(job, str) else job
        if handle is None or not handle.state.is_active:
            return
        if handle.state == JobState.QUEUED:
            self._set_state(handle, JobState.CANCELLED)
            self._delete_later(handle, keep_dir=False)
            self.jobCancelled.emit(handle)
            return
        if handle.state == JobState.CANCELLING:
            return
        self._set_state(handle, JobState.CANCELLING)
        if handle.process is not None:
            handle.process.kill()

    def cancel_all(self) -> None:
        """Clear the queue and cancel the running job."""
        for handle in reversed(self.active_jobs):
            self.cancel(handle)

    def wait_for_idle(self, timeout_ms: int = 10_000) -> bool:
        """Block until the worker has exited and cleanup finished (app exit only)."""
        deadline = time.monotonic() + timeout_ms / 1000
        if (running := self.running_job) is not None and running.process is not None:
            running.process.waitForFinished(timeout_ms)
        remaining = max(int((deadline - time.monotonic()) * 1000), 0)
        done = self._pool.waitForDone(remaining)
        return done and self.running_job is None

    def forget(self, job: StitchJob) -> None:
        """Drop a finished, failed or cancelled job from the list."""
        if job.state.is_done and job in self._jobs:
            self._jobs.remove(job)
            self.changed.emit()

    def _set_state(self, handle: StitchJob, state: JobState) -> None:
        if state not in _TRANSITIONS[handle.state]:
            raise RuntimeError(
                f"stitch job {handle.id}: {handle.state.value} -> {state.value}"
            )
        handle.state = state
        if self.running_job is None:
            self._timer.stop()
        self.changed.emit()

    def _start_next(self) -> None:
        if self.running_job is not None:
            return
        handle = next((j for j in self._jobs if j.state == JobState.QUEUED), None)
        if handle is None:
            return
        self._set_state(handle, JobState.STARTING)
        proc = QProcess(self)
        handle.process = proc
        program, args = self._command(handle.dir.path)
        proc.setProgram(program)
        proc.setArguments(args)
        proc.setProcessEnvironment(worker_environment())
        # The protocol is files only; don't let a full pipe block the worker.
        proc.setStandardOutputFile(QProcess.nullDevice())
        proc.setStandardErrorFile(QProcess.nullDevice())
        proc.setStandardInputFile(QProcess.nullDevice())
        proc.started.connect(partial(self._on_started, handle))
        proc.finished.connect(partial(self._on_finished, handle))
        proc.errorOccurred.connect(partial(self._on_error, handle))
        proc.start()

    def _on_started(self, handle: StitchJob) -> None:
        if handle.state == JobState.STARTING:
            self._set_state(handle, JobState.RUNNING)
        self._timer.start()

    def _on_error(self, handle: StitchJob, error: QProcess.ProcessError) -> None:
        if error != QProcess.ProcessError.FailedToStart:
            return  # a crash is reported by `finished`
        handle.error = StitchErrorV1(
            message=f"Could not start the stitching worker: "
            f"{handle.process.errorString() if handle.process else error}"
        )
        self._finish(handle, JobState.FAILED)

    def _poll(self) -> None:
        handle = self.running_job
        if handle is None:
            self._timer.stop()
            return
        if (event := handle.dir.read_progress()) is not None and (
            event != handle.progress
        ):
            handle.progress = event
            self.changed.emit()

    def _on_finished(self, handle: StitchJob, exit_code: int, *_: object) -> None:
        handle.exit_code = exit_code
        if handle.state == JobState.CANCELLING:
            self._finish(handle, JobState.CANCELLED)
            return
        if (result := handle.dir.read_result()) is not None and exit_code == 0:
            handle.result = result
            self._finish(handle, JobState.FINISHED)
            return
        handle.error = handle.dir.read_error() or StitchErrorV1(
            message=f"The stitching worker stopped unexpectedly "
            f"(exit code {exit_code}).",
            traceback=handle.dir.log_tail(),
        )
        self._finish(handle, JobState.FAILED)

    def _finish(self, handle: StitchJob, state: JobState) -> None:
        if handle.state.is_done:
            return
        if (proc := handle.process) is not None:
            handle.process = None
            # Its slots hold this controller: left connected until the deferred
            # delete, they could make it the last owner, destroying the
            # controller from inside its own child's destructor (a crash).
            for signal in (proc.started, proc.finished, proc.errorOccurred):
                signal.disconnect()
            proc.deleteLater()
        self._set_state(handle, state)
        if state == JobState.FINISHED:
            self._delete_later(handle, keep_dir=False, outputs=False)
            self.jobFinished.emit(handle)
        elif state == JobState.FAILED:
            self._delete_later(handle, keep_dir=True)
            self.jobFailed.emit(handle)
        else:
            self._delete_later(handle, keep_dir=False)
            self.jobCancelled.emit(handle)
        self._start_next()

    def _delete_later(
        self, handle: StitchJob, *, keep_dir: bool, outputs: bool = True
    ) -> None:
        """Delete the partial mosaics (and the job dir) off the GUI thread."""
        tmp = handle.job.tmp_outputs if outputs else []
        path = handle.dir.path

        def work() -> None:
            cleanup_paths(tmp if keep_dir else [*tmp, path])

        self._pool.start(_Task(work))
