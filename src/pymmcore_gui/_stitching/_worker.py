"""The stitching worker process, started with the `_stitch-worker DIR` argument.

It never imports Qt (a test checks `sys.modules`). Its whole interface is the
job directory (see `_protocol`): it reads `job.json`, rewrites
`progress.json`, writes `result.json` or `error.json` last, and exits 0 or 1.
Everything libraries print goes to `worker.log`.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from pymmcore_gui._stitching._models import ProgressEvent, StitchJobV1
    from pymmcore_gui._stitching._protocol import JobDir

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_ORPHANED = 3

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
"""Math libraries stay single-threaded: our own pools spend the budget (§4.4).
The controller sets these in the worker's environment; `worker_main` sets them
again (without overriding) for a worker started by hand."""

PROGRESS_INTERVAL_S = 0.2
WATCHDOG_INTERVAL_S = 2.0


def _redirect_stdio(log_file: Path) -> TextIO:
    """Point stdout and stderr at the log, whether or not they existed.

    PyInstaller's windowed build can leave them as None, and a library that
    prints would then crash the worker.
    """
    log = open(log_file, "a", encoding="utf-8", buffering=1)
    sys.stdout = log
    sys.stderr = log
    with contextlib.suppress(Exception):
        sys.stdin = open(os.devnull, encoding="utf-8")
    return log


def _close_splash() -> None:
    """The bundle's bootloader shows its splash for every launch, ours too."""
    if "_PYI_SPLASH_IPC" not in os.environ:
        return
    with contextlib.suppress(Exception):
        import pyi_splash  # pyright: ignore [reportMissingModuleSource]

        pyi_splash.close()


def _configure_logging() -> None:
    import logging

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
    logging.getLogger("pymmcore_gui").setLevel(logging.INFO)
    logging.captureWarnings(True)


def _lower_priority() -> None:
    with contextlib.suppress(Exception):
        import psutil

        proc = psutil.Process()
        if sys.platform == "win32":
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            proc.nice(10)


def _start_watchdog(job: StitchJobV1, on_orphaned: Any = None) -> None:
    """Exit when the GUI process is gone (or its PID was reused)."""
    if job.parent_pid is None:
        return
    from pymmcore_gui._stitching._protocol import process_alive

    pid, created = job.parent_pid, job.parent_create_time

    def watch() -> None:
        while True:
            time.sleep(WATCHDOG_INTERVAL_S)
            if not process_alive(pid, created):
                print("GUI process is gone; stopping.", flush=True)
                if on_orphaned is not None:
                    on_orphaned()
                os._exit(EXIT_ORPHANED)
                return  # only reached when tests replace os._exit

    threading.Thread(target=watch, name="stitch-watchdog", daemon=True).start()


class _ProgressWriter:
    """Rewrites progress.json at most 5 times a second (and on phase changes)."""

    def __init__(self, jobdir: JobDir) -> None:
        self._jobdir = jobdir
        self._lock = threading.Lock()
        self._last_write = 0.0
        self._last_phase: object = None
        self._pending: ProgressEvent | None = None

    def __call__(self, event: ProgressEvent) -> None:
        from pymmcore_gui._stitching._protocol import write_model

        with self._lock:
            now = time.monotonic()
            changed = event.phase != self._last_phase
            if not changed and now - self._last_write < PROGRESS_INTERVAL_S:
                self._pending = event
                return
            self._pending = None
            self._last_write, self._last_phase = now, event.phase
            with contextlib.suppress(OSError):
                write_model(self._jobdir.progress_file, event)

    def flush(self) -> None:
        if self._pending is not None:
            self._last_phase = None
            self(self._pending)


def _remove_tmp_outputs(job: StitchJobV1) -> None:
    import shutil

    for tmp in job.tmp_outputs:
        if tmp.is_dir():
            shutil.rmtree(tmp, ignore_errors=True)
        elif tmp.exists():
            tmp.unlink(missing_ok=True)


def worker_main(argv: list[str]) -> int:
    """Run the job in directory `argv[0]`; return the process exit code."""
    for var in THREAD_ENV_VARS:
        os.environ.setdefault(var, "1")
    os.environ["TQDM_DISABLE"] = "1"
    _close_splash()

    if len(argv) != 1:
        return EXIT_USAGE
    from pymmcore_gui._stitching._protocol import JobDir

    jobdir = JobDir(Path(argv[0]))
    if not jobdir.path.is_dir():
        return EXIT_USAGE
    log = _redirect_stdio(jobdir.log_file)
    try:
        return _run(jobdir)
    finally:
        with contextlib.suppress(Exception):
            log.flush()


def _run(jobdir: JobDir) -> int:
    import psutil

    from pymmcore_gui._stitching._models import (
        SCHEMA_VERSION,
        OwnerV1,
        StitchErrorV1,
        StitchJobV1,
    )
    from pymmcore_gui._stitching._protocol import write_model

    print(f"stitching worker pid {os.getpid()} started", flush=True)
    me = psutil.Process()
    write_model(jobdir.owner_file, OwnerV1(pid=me.pid, create_time=me.create_time()))
    _configure_logging()

    def fail(message: str, tb: str = "") -> int:
        print(f"FAILED: {message}\n{tb}", flush=True)
        write_model(jobdir.error_file, StitchErrorV1(message=message, traceback=tb))
        return EXIT_FAILED

    try:
        raw = jobdir.job_file.read_text(encoding="utf-8")
    except OSError as e:
        return fail(f"Could not read the job: {e}")
    import json

    try:
        version = json.loads(raw).get("schema_version")
    except (ValueError, AttributeError):
        version = None
    if version != SCHEMA_VERSION:
        return fail(
            f"This worker understands job version {SCHEMA_VERSION}, not {version!r}."
        )
    try:
        job = StitchJobV1.model_validate_json(raw)
    except ValueError as e:
        return fail(f"Invalid job: {e}")

    _start_watchdog(job, on_orphaned=lambda: _remove_tmp_outputs(job))
    if job.settings.low_priority:
        _lower_priority()

    from pymmcore_gui._stitching import is_available

    if not is_available():
        return fail(
            "Stitching needs the optional 'multiview-stitcher' package: "
            "pip install 'pymmcore-gui[stitching]'"
        )
    from pymmcore_gui._stitching._engine import StitchError, run_stitch

    progress = _ProgressWriter(jobdir)
    try:
        result = run_stitch(job, progress)
    except StitchError as e:
        _remove_tmp_outputs(job)
        return fail(str(e))
    except BaseException as e:
        _remove_tmp_outputs(job)
        return fail(f"{type(e).__name__}: {e}", traceback.format_exc())
    progress.flush()
    write_model(jobdir.result_file, result)
    print(f"finished in {result.elapsed_s:.1f} s", flush=True)
    return EXIT_OK
