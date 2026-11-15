"""The worker process: Qt-free, stdio-independent, file protocol (plan §4.2-4.3)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import useq

from pymmcore_gui._stitching import _worker, is_available
from pymmcore_gui._stitching._inputs import build_job
from pymmcore_gui._stitching._protocol import JobDir, write_model

if TYPE_CHECKING:
    from pathlib import Path

    from pymmcore_gui._stitching._models import StitchJobV1

requires_mvs = pytest.mark.skipif(
    not is_available(), reason="needs the optional multiview-stitcher package"
)


def _job(tmp_path: Path, source: Path, **kw: Any) -> StitchJobV1:
    seq = useq.MDASequence(
        grid_plan=useq.GridRowsColumns(
            rows=2,
            columns=2,
            overlap=(15, 15),
            fov_width=64 * 0.5,
            fov_height=64 * 0.5,
        )
    )
    job = build_job(
        job_id="job",
        sequence=seq,
        source=source,
        pixel_size_um=0.5,
        linear_part=np.diag([0.5, 0.5]),
        **kw,
    )
    assert job is not None
    return job


def _jobdir(tmp_path: Path, job: StitchJobV1 | None = None, raw: str = "") -> JobDir:
    jd = JobDir(tmp_path / "jobs" / "job")
    jd.path.mkdir(parents=True)
    if job is not None:
        write_model(jd.job_file, job)
    else:
        jd.job_file.write_text(raw)
    return jd


def _run_worker(
    jd: JobDir, prelude: str = "", timeout: float = 120
) -> subprocess.CompletedProcess[str]:
    """Run the worker in a fresh interpreter, as `python -m pymmcore_gui` does."""
    code = (
        "import sys\n"
        f"{prelude}\n"
        "from pymmcore_gui.__main__ import main\n"
        f"sys.argv = ['pymm', '_stitch-worker', {str(jd.path)!r}]\n"
        "main()\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_worker_modules_import_without_qt() -> None:
    code = (
        "import sys\n"
        "from pymmcore_gui.__main__ import STITCH_WORKER_ARG\n"
        "from pymmcore_gui._stitching import _worker, _engine, _inputs\n"
        "import pymmcore_gui._ome_zarr_wrapper, pymmcore_gui._ome_tiff_wrapper\n"
        "qt = sorted(m for m in sys.modules if m.split('.')[0] in "
        "('PyQt6', 'PySide6', 'qtpy', 'superqt'))\n"
        "print(qt)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_unknown_job_version_is_refused(tmp_path: Path) -> None:
    job = _job(tmp_path, tmp_path / "a.ome.zarr")
    raw = json.loads(job.model_dump_json())
    raw["schema_version"] = 99
    jd = _jobdir(tmp_path, raw=json.dumps(raw))
    proc = _run_worker(jd)
    assert proc.returncode == _worker.EXIT_FAILED
    error = jd.read_error()
    assert error is not None
    assert "version" in error.message
    assert jd.read_result() is None
    assert jd.read_owner() is not None
    # nothing goes to the real stdio
    assert proc.stdout == ""
    assert "FAILED" in jd.log_file.read_text()


def test_invalid_job_is_an_error(tmp_path: Path) -> None:
    jd = _jobdir(tmp_path, raw='{"schema_version": 1, "groups": []}')
    proc = _run_worker(jd)
    assert proc.returncode == _worker.EXIT_FAILED
    error = jd.read_error()
    assert error is not None
    assert error.message.startswith("Invalid job")


def test_missing_job_dir_is_a_usage_error(tmp_path: Path) -> None:
    assert _worker.worker_main([str(tmp_path / "nope")]) == _worker.EXIT_USAGE
    assert _worker.worker_main([]) == _worker.EXIT_USAGE


@requires_mvs
def test_unreadable_source_reports_error_with_none_stdio(tmp_path: Path) -> None:
    """Like PyInstaller's windowed build: no stdout, stderr or stdin at all."""
    jd = _jobdir(tmp_path, _job(tmp_path, tmp_path / "missing.ome.zarr"))
    proc = _run_worker(jd, "sys.stdout = sys.stderr = sys.stdin = None")
    assert proc.returncode == _worker.EXIT_FAILED, proc.stderr
    error = jd.read_error()
    assert error is not None
    assert "Could not read" in error.message
    assert error.traceback == ""  # a user-facing error, not a bug


@requires_mvs
def test_real_job_writes_progress_and_result_without_qt(tmp_path: Path) -> None:
    from _stitch_synth import grid_sequence, write_grid

    seq = grid_sequence(2, 2, 64, 0.5)
    acq = write_grid(tmp_path / "run.ome.zarr", seq, shape=64, pixel_size=0.5)
    job = build_job(
        job_id="job",
        sequence=seq,
        source=acq.path,
        pixel_size_um=0.5,
        linear_part=np.diag([0.5, 0.5]),
    )
    assert job is not None
    jd = _jobdir(tmp_path, job)
    check_qt = (
        "import atexit\n"
        "atexit.register(lambda: open("
        f"{str(tmp_path / 'qt.txt')!r}, 'w').write("
        "str(sorted(m for m in sys.modules if m.startswith('PyQt')))))"
    )
    proc = _run_worker(jd, check_qt)
    assert proc.returncode == _worker.EXIT_OK, jd.log_tail()
    result = jd.read_result()
    assert result is not None
    assert result.groups[0].output.exists()
    progress = jd.read_progress()
    assert progress is not None
    assert progress.fraction > 0.9
    assert (tmp_path / "qt.txt").read_text() == "[]"


def test_watchdog_exits_when_the_gui_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    job = _job(tmp_path, tmp_path / "a.ome.zarr").model_copy(
        update={"parent_pid": gone.pid, "parent_create_time": time.time() - 5}
    )
    exited = threading.Event()
    codes: list[int] = []

    def fake_exit(code: int) -> None:
        codes.append(code)
        exited.set()

    monkeypatch.setattr(_worker, "WATCHDOG_INTERVAL_S", 0.01)
    monkeypatch.setattr(os, "_exit", fake_exit)
    _worker._start_watchdog(job)
    assert exited.wait(5)
    assert codes == [_worker.EXIT_ORPHANED]


def test_progress_writer_throttles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pymmcore_gui._stitching._protocol as protocol
    from pymmcore_gui._stitching._models import Phase, ProgressEvent

    jd = JobDir(tmp_path)
    writes: list[Any] = []
    original = protocol.write_model

    def record(path: Path, model: Any) -> None:
        writes.append(model)
        original(path, model)

    monkeypatch.setattr(protocol, "write_model", record)
    writer = _worker._ProgressWriter(jd)
    for i in range(50):
        writer(ProgressEvent(phase=Phase.FUSING, done=i, total=50))
    writer(ProgressEvent(phase=Phase.PYRAMID, done=0, total=1))
    writer(ProgressEvent(phase=Phase.PYRAMID, done=1, total=1))
    writer.flush()
    # the first event and each phase change are written at once, the rest
    # at most every PROGRESS_INTERVAL_S; the last one is flushed
    assert 3 <= len(writes) < 10
    assert writes[-1].phase == Phase.PYRAMID
    assert writes[-1].done == 1
    last = jd.read_progress()
    assert last is not None
    assert last.done == 1
