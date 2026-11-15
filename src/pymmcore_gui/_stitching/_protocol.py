"""Atomic job-directory communication between the GUI and worker.

Files replace stdio, which can be unavailable in a Windows windowed bundle.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from pymmcore_gui._stitching._models import (
    OwnerV1,
    ProgressEvent,
    StitchErrorV1,
    StitchJobV1,
    StitchResultV1,
)

M = TypeVar("M", bound=BaseModel)


def atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` so readers see the old or the new file, never half."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def write_model(path: Path, model: BaseModel) -> None:
    atomic_write_text(path, model.model_dump_json(indent=2))


def read_model(path: Path, cls: type[M]) -> M | None:
    """Return the model stored at `path`, or None if missing or unreadable."""
    try:
        return cls.model_validate_json(path.read_bytes())
    except (OSError, ValueError):
        return None


@dataclass(frozen=True)
class JobDir:
    """Paths of one job's files, under the registry (see `registry_dir`)."""

    path: Path

    @property
    def job_id(self) -> str:
        return self.path.name

    @property
    def job_file(self) -> Path:
        return self.path / "job.json"

    @property
    def progress_file(self) -> Path:
        return self.path / "progress.json"

    @property
    def result_file(self) -> Path:
        return self.path / "result.json"

    @property
    def error_file(self) -> Path:
        return self.path / "error.json"

    @property
    def owner_file(self) -> Path:
        return self.path / "owner.json"

    @property
    def log_file(self) -> Path:
        return self.path / "worker.log"

    def read_job(self) -> StitchJobV1 | None:
        return read_model(self.job_file, StitchJobV1)

    def read_progress(self) -> ProgressEvent | None:
        return read_model(self.progress_file, ProgressEvent)

    def read_result(self) -> StitchResultV1 | None:
        return read_model(self.result_file, StitchResultV1)

    def read_error(self) -> StitchErrorV1 | None:
        return read_model(self.error_file, StitchErrorV1)

    def read_owner(self) -> OwnerV1 | None:
        return read_model(self.owner_file, OwnerV1)

    def log_tail(self, max_lines: int = 20) -> str:
        """The end of the worker's log, for reporting an abnormal exit."""
        try:
            with self.log_file.open("rb") as fh:
                fh.seek(0, os.SEEK_END)
                fh.seek(max(fh.tell() - 16_384, 0))
                text = fh.read().decode("utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-max_lines:])


REGISTRY_ENV_VAR = "MMGUI_STITCH_JOBS_DIR"


def registry_dir() -> Path:
    """`<user data dir>/stitch-jobs`, where every job directory lives.

    `MMGUI_STITCH_JOBS_DIR` overrides the location (the tests use it).
    """
    if override := os.environ.get(REGISTRY_ENV_VAR):
        return Path(override)
    from platformdirs import user_data_dir

    return Path(user_data_dir(appname="pymmcore-gui")) / "stitch-jobs"


def process_alive(pid: int, create_time: float | None) -> bool:
    """Whether process `pid` runs and is the one started at `create_time`.

    Comparing the start time catches a PID the OS has since given to another
    process.
    """
    import psutil

    try:
        proc = psutil.Process(pid)
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return False
        return create_time is None or abs(proc.create_time() - create_time) < 1e-2
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False
