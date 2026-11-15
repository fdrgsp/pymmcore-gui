"""The GUI side: queue, process, progress, cancel, cleanup (plan §4.3, §10).

A fake worker script stands in for the real one, so these tests need neither
multiview-stitcher nor real data -- except the last, which runs one real job.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import useq

from _stitch_fakes import fake_controller, write_fake_worker
from pymmcore_gui._stitching import _controller as ctrl_mod
from pymmcore_gui._stitching import is_available
from pymmcore_gui._stitching._controller import (
    JobState,
    StitchingController,
    StitchJob,
    cleanup_stale_jobs,
    worker_command,
)
from pymmcore_gui._stitching._inputs import build_job
from pymmcore_gui._stitching._models import (
    OwnerV1,
    StitchErrorV1,
    StitchJobV1,
    tmp_path_for,
)
from pymmcore_gui._stitching._protocol import JobDir, write_model

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


GRID = useq.GridRowsColumns(
    rows=2, columns=2, overlap=(10, 10), fov_width=10, fov_height=10
)


@pytest.fixture
def fake_worker(tmp_path: Path) -> Path:
    return write_fake_worker(tmp_path)


def _job(tmp_path: Path, job_id: str = "j1", n_positions: int = 0) -> StitchJobV1:
    seq = useq.MDASequence.model_validate(
        {
            "grid_plan": GRID,
            "stage_positions": [(i * 1000, 0) for i in range(n_positions)],
        }
    )
    src = tmp_path / "data" / f"{job_id}.ome.zarr"
    src.parent.mkdir(exist_ok=True)
    job = build_job(
        job_id=job_id,
        sequence=seq,
        source=src,
        pixel_size_um=1.0,
        linear_part=np.eye(2),
    )
    assert job is not None
    return job


def _make_controller(
    qtbot: QtBot, tmp_path: Path, script: Path, modes: dict[str, str] | str
) -> StitchingController:
    return fake_controller(tmp_path, script, modes)


def _state(handle: StitchJob) -> JobState:
    """Read the state afresh (keeps mypy from narrowing it across a wait)."""
    return handle.state


def _wait_done(qtbot: QtBot, ctrl: StitchingController, timeout: int = 20_000) -> None:
    qtbot.waitUntil(lambda: not ctrl.is_busy(), timeout=timeout)
    assert ctrl.wait_for_idle(timeout)


def test_success(qtbot: QtBot, tmp_path: Path, fake_worker: Path) -> None:
    ctrl = _make_controller(qtbot, tmp_path, fake_worker, "ok")
    states: list[JobState] = []
    ctrl.changed.connect(lambda: states.append(ctrl.jobs[0].state))
    with qtbot.waitSignal(ctrl.jobFinished, timeout=20_000) as blocker:
        handle = ctrl.submit(_job(tmp_path), title="run 1")
    assert blocker.args == [handle]
    assert handle.state == JobState.FINISHED
    assert handle.result is not None
    out = handle.result.groups[0].output
    assert out.exists()
    assert not tmp_path_for(out).exists()
    # Queued -> Starting -> Running -> Finished, in that order
    seen = [s for i, s in enumerate(states) if i == 0 or s != states[i - 1]]
    assert seen == [
        JobState.QUEUED,
        JobState.STARTING,
        JobState.RUNNING,
        JobState.FINISHED,
    ]
    _wait_done(qtbot, ctrl)
    assert not handle.dir.path.exists()  # nothing to keep after success


def test_progress_is_polled(qtbot: QtBot, tmp_path: Path, fake_worker: Path) -> None:
    ctrl = _make_controller(qtbot, tmp_path, fake_worker, "hang")
    handle = ctrl.submit(_job(tmp_path))
    qtbot.waitUntil(lambda: handle.progress.total == 4, timeout=20_000)
    assert handle.status_text == "Fusing 1/4"
    ctrl.cancel(handle)
    _wait_done(qtbot, ctrl)


def test_one_worker_at_a_time(qtbot: QtBot, tmp_path: Path, fake_worker: Path) -> None:
    ctrl = _make_controller(qtbot, tmp_path, fake_worker, {"a": "hang", "b": "ok"})
    a = ctrl.submit(_job(tmp_path, "a"))
    b = ctrl.submit(_job(tmp_path, "b"))
    qtbot.waitUntil(lambda: a.state == JobState.RUNNING, timeout=20_000)
    assert _state(b) == JobState.QUEUED
    assert b.status_text == "Waiting"
    assert ctrl.running_job is a
    with qtbot.waitSignal(ctrl.jobFinished, timeout=20_000):
        ctrl.cancel(a)
    assert _state(a) == JobState.CANCELLED
    assert _state(b) == JobState.FINISHED
    _wait_done(qtbot, ctrl)


def test_failure_reports_error_and_keeps_log(
    qtbot: QtBot, tmp_path: Path, fake_worker: Path
) -> None:
    ctrl = _make_controller(qtbot, tmp_path, fake_worker, "fail")
    with qtbot.waitSignal(ctrl.jobFailed, timeout=20_000):
        handle = ctrl.submit(_job(tmp_path))
    assert handle.state == JobState.FAILED
    assert handle.error is not None
    assert handle.error.message == "no overlap at all"
    _wait_done(qtbot, ctrl)
    assert handle.dir.path.exists()  # its log stays for a while
    assert not tmp_path_for(handle.job.groups[0].output).exists()


def test_abnormal_exit_reports_log_tail(
    qtbot: QtBot, tmp_path: Path, fake_worker: Path
) -> None:
    ctrl = _make_controller(qtbot, tmp_path, fake_worker, "crash")
    with qtbot.waitSignal(ctrl.jobFailed, timeout=20_000):
        handle = ctrl.submit(_job(tmp_path))
    assert handle.error is not None
    assert "stopped unexpectedly (exit code 7)" in handle.error.message
    assert "Segmentation fault" in handle.error.traceback
    _wait_done(qtbot, ctrl)
    assert not tmp_path_for(handle.job.groups[0].output).exists()


def test_failed_to_start(qtbot: QtBot, tmp_path: Path) -> None:
    ctrl = StitchingController(
        registry=tmp_path / "registry",
        command=lambda d: (str(tmp_path / "no-such-program"), []),
        cleanup_on_start=False,
    )
    with qtbot.waitSignal(ctrl.jobFailed, timeout=20_000):
        handle = ctrl.submit(_job(tmp_path))
    assert handle.error is not None
    assert "Could not start" in handle.error.message


def test_cancel_waits_for_exit_then_deletes_off_the_gui_thread(
    qtbot: QtBot,
    tmp_path: Path,
    fake_worker: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctrl = _make_controller(qtbot, tmp_path, fake_worker, "hang")
    handle = ctrl.submit(_job(tmp_path, n_positions=2))
    tmps = [tmp_path_for(g.output) for g in handle.job.groups]
    qtbot.waitUntil(lambda: all(t.exists() for t in tmps), timeout=20_000)

    deleted_on: list[tuple[Path, bool]] = []
    original = ctrl_mod._rmtree

    def spy(path: Path) -> None:
        # the worker must be gone before anything is deleted
        deleted_on.append((path, threading.current_thread() is threading.main_thread()))
        original(path)

    monkeypatch.setattr(ctrl_mod, "_rmtree", spy)
    with qtbot.waitSignal(ctrl.jobCancelled, timeout=20_000):
        ctrl.cancel(handle)
        assert _state(handle) == JobState.CANCELLING
        assert handle.status_text == "Cancelling…"
    assert _state(handle) == JobState.CANCELLED
    assert handle.exit_code is not None
    _wait_done(qtbot, ctrl)
    assert not any(t.exists() for t in tmps)
    assert not handle.dir.path.exists()
    assert deleted_on
    assert not any(on_main for _, on_main in deleted_on)


def test_cancel_queued_job(qtbot: QtBot, tmp_path: Path, fake_worker: Path) -> None:
    ctrl = _make_controller(qtbot, tmp_path, fake_worker, "hang")
    a = ctrl.submit(_job(tmp_path, "a"))
    b = ctrl.submit(_job(tmp_path, "b"))
    ctrl.cancel(b)
    assert b.state == JobState.CANCELLED
    ctrl.cancel_all()
    _wait_done(qtbot, ctrl)
    assert a.state == JobState.CANCELLED
    ctrl.forget(a)
    assert ctrl.jobs == [b]


def test_invalid_transition_raises(tmp_path: Path, qtbot: QtBot) -> None:
    ctrl = StitchingController(
        registry=tmp_path / "registry",
        command=lambda d: (sys.executable, ["-c", "pass"]),
        cleanup_on_start=False,
    )
    handle = ctrl.submit(_job(tmp_path))
    with pytest.raises(RuntimeError):
        ctrl._set_state(handle, JobState.FINISHED)  # from Starting
    _wait_done(qtbot, ctrl)


def test_worker_command() -> None:
    program, args = worker_command(Path("/jobs/x"))
    assert program == sys.executable
    assert args == ["-m", "pymmcore_gui", "_stitch-worker", str(Path("/jobs/x"))]


def test_worker_command_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    program, args = worker_command(Path("/jobs/x"))
    assert program == sys.executable
    assert args == ["_stitch-worker", str(Path("/jobs/x"))]
    env = ctrl_mod.worker_environment()
    assert env.value("PYINSTALLER_RESET_ENVIRONMENT") == "1"
    assert env.value("OMP_NUM_THREADS") == "1"


# ------------------------------ startup cleanup ------------------------------


def _registry_entry(
    registry: Path,
    tmp_path: Path,
    name: str,
    *,
    owner: OwnerV1 | None = None,
    parent: tuple[int, float] | None = None,
    failed: bool = False,
    age_s: float = 0,
) -> tuple[JobDir, Path]:
    job = _job(tmp_path, name)
    if parent is not None:
        job = job.model_copy(
            update={"parent_pid": parent[0], "parent_create_time": parent[1]}
        )
    jd = JobDir(registry / name)
    jd.path.mkdir(parents=True)
    write_model(jd.job_file, job)
    if owner is not None:
        write_model(jd.owner_file, owner)
    if failed:
        write_model(jd.error_file, StitchErrorV1(message="x"))
    tmp = tmp_path_for(job.groups[0].output)
    tmp.mkdir(parents=True)
    (tmp / "chunk").write_bytes(b"0")
    t = time.time() - age_s
    os.utime(jd.path, (t, t))
    return jd, tmp


def test_cleanup_stale_jobs(tmp_path: Path) -> None:
    import psutil

    registry = tmp_path / "registry"
    me = psutil.Process()
    dead = subprocess_pid_that_exited()
    alive_owner = OwnerV1(pid=me.pid, create_time=me.create_time())
    dead_owner = OwnerV1(pid=dead, create_time=1.0)

    crashed, crashed_tmp = _registry_entry(
        registry, tmp_path, "crashed", owner=dead_owner
    )
    running, running_tmp = _registry_entry(
        registry, tmp_path, "running", owner=alive_owner
    )
    failed, failed_tmp = _registry_entry(
        registry, tmp_path, "failed", owner=dead_owner, failed=True
    )
    old, old_tmp = _registry_entry(
        registry, tmp_path, "old", owner=dead_owner, failed=True, age_s=10 * 86400
    )
    other_gui, other_tmp = _registry_entry(
        registry,
        tmp_path,
        "queued",
        parent=(me.ppid(), psutil.Process(me.ppid()).create_time()),
    )
    # a user's own file that merely looks like a partial mosaic
    lookalike = tmp_path / "data" / "mine_stitched.ome.zarr.tmp"
    lookalike.mkdir(parents=True)

    removed = cleanup_stale_jobs(registry)

    # a worker that died without a result failed: its log stays a while
    assert not crashed_tmp.exists() and crashed.path.exists()
    assert running_tmp.exists() and running.path.exists()
    assert not failed_tmp.exists() and failed.path.exists()  # log kept
    assert not old_tmp.exists() and not old.path.exists()
    assert other_tmp.exists() and other_gui.path.exists()
    assert lookalike.exists()
    assert set(removed) == {crashed_tmp, failed_tmp, old_tmp, old.path}


def subprocess_pid_that_exited() -> int:
    import subprocess

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


# -------------------------------- integration --------------------------------


@pytest.mark.skipif(not is_available(), reason="needs multiview-stitcher")
def test_real_worker_end_to_end(qtbot: QtBot, tmp_path: Path) -> None:
    from _stitch_synth import grid_sequence, write_grid
    from pymmcore_gui._acquisition_loader import open_acquisition

    seq = grid_sequence(2, 2, 64, 0.5)
    acq = write_grid(tmp_path / "run.ome.zarr", seq, shape=64, pixel_size=0.5)
    pid, created = ctrl_mod.own_process_identity()
    job = build_job(
        job_id="real",
        sequence=seq,
        source=acq.path,
        pixel_size_um=0.5,
        linear_part=np.diag([0.5, 0.5]),
        tile_shape_yx=(64, 64),
        parent_pid=pid,
        parent_create_time=created,
    )
    assert job is not None
    ctrl = StitchingController(registry=tmp_path / "registry", cleanup_on_start=False)
    with (
        qtbot.waitSignals([ctrl.jobFinished], raising=True, timeout=120_000),
        qtbot.assertNotEmitted(ctrl.jobFailed),
    ):
        handle = ctrl.submit(job)
    assert handle.result is not None
    out = handle.result.groups[0].output
    loaded = open_acquisition(out)
    try:
        assert loaded.wrapper.dims[-2:] == ("y", "x")
    finally:
        loaded.close()
    _wait_done(qtbot, ctrl)


def test_cleanup_removes_partial_ome_tiff_mosaics(tmp_path: Path) -> None:
    """An OME-TIFF job leaves a file and a fused OME-Zarr; both go."""
    seq = useq.MDASequence(grid_plan=GRID)
    job = build_job(
        job_id="tiff",
        sequence=seq,
        source=tmp_path / "data" / "run.ome.tiff",
        pixel_size_um=1.0,
        linear_part=np.eye(2),
    )
    assert job is not None
    out = job.groups[0].output
    assert out.name == "run_stitched.ome.tiff"
    partial_tiff, fused = job.tmp_outputs
    partial_tiff.parent.mkdir(parents=True)
    partial_tiff.write_bytes(b"II*\0")
    fused.mkdir()

    registry = tmp_path / "registry"
    jd = JobDir(registry / job.job_id)
    jd.path.mkdir(parents=True)
    write_model(jd.job_file, job)
    write_model(
        jd.owner_file, OwnerV1(pid=subprocess_pid_that_exited(), create_time=1.0)
    )

    removed = cleanup_stale_jobs(registry)
    assert partial_tiff in removed
    assert fused in removed
    assert not partial_tiff.exists()
    assert not fused.exists()
