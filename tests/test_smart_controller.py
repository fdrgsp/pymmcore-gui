# pyright: reportArgumentType=false
# (useq models are built from plain values that pydantic coerces)
"""End-to-end smart runs on the demo core, in both execution modes."""

from __future__ import annotations

import json
import textwrap
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import useq

from pymmcore_gui._smart._controller import (
    SmartController,
    SmartRunConfig,
    SmartRunError,
)
from pymmcore_gui._smart._loader import inspect_script

if TYPE_CHECKING:
    import numpy as np
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.metadata import FrameMetaV1
    from pytestqt.qtbot import QtBot

TEMPLATES = Path(__file__).parent.parent / "src/pymmcore_gui/resources/smart_templates"
MODES = ["thread", "process"]


def _script(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "script.py"
    path.write_text(textwrap.dedent(source))
    return path


def _config(script: Path, **overrides: Any) -> SmartRunConfig:
    spec = inspect_script(script)
    params = spec.resolve_params(overrides.pop("params", None))
    fields: dict[str, Any] = {
        "spec": spec,
        "params": params,
        "execution": "thread",
        "sync": spec.sync,
        "filter": spec.filter,
    }
    fields.update(overrides)
    return SmartRunConfig(**fields)


def _run(
    qtbot: QtBot,
    controller: SmartController,
    config: SmartRunConfig,
    base: useq.MDASequence,
    run_dir: Path,
    timeout: int = 30_000,
) -> dict[str, Any]:
    controller.prepare(config, run_dir)
    with qtbot.waitSignal(controller.runFinished, timeout=timeout) as blocker:
        controller.start(base, output="memory")
    assert blocker.args is not None
    summary: dict[str, Any] = blocker.args[0]
    return summary


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("mode", MODES)
def test_adaptive_exposure_is_purely_reactive(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path, mode: str
) -> None:
    controller = SmartController(mmcore)
    config = _config(
        TEMPLATES / "adaptive_exposure.py",
        execution=mode,
        params={"n_frames": 4, "target_mean": 100.0},
    )
    summary = _run(
        qtbot, controller, config, useq.MDASequence(channels=["DAPI"]), tmp_path / "run"
    )

    assert summary["status"] == "stopped_by_script"
    assert summary["frames"] == 4
    frames = _lines(tmp_path / "run" / "frames.jsonl")
    assert [f["origin"] for f in frames] == ["base", "analysis", "analysis", "analysis"]
    assert [f["parent_frame_id"] for f in frames] == [None, 0, 1, 2]
    assert [f["t_index"] for f in frames] == [0, 1, 2, 3]
    exposures = [f["exposure_ms"] for f in frames]
    assert len(set(exposures)) > 1  # the script changed the exposure

    # the store holds every frame along one t axis
    view = mmcore.mda.get_view()
    assert view is not None and view.shape[0] == 4

    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["status"] == "stopped_by_script"
    assert run["execution"] == mode
    assert run["counts"]["frames"] == 4
    assert run["finished"] is not None
    script_copy = (tmp_path / "run" / "script.py").read_text()
    assert script_copy == config.spec.source

    analysis = _lines(tmp_path / "run" / "analysis.jsonl")
    assert [a["call"] for a in analysis][:1] == ["setup"]
    analyze_records = [a for a in analysis if a["call"] == "analyze"]
    assert len(analyze_records) == 4
    assert all("mean" in a["records"] for a in analyze_records)


@pytest.mark.parametrize("mode", MODES)
def test_detect_and_zstack_inserts_follow_up(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path, mode: str
) -> None:
    controller = SmartController(mmcore)
    config = _config(
        TEMPLATES / "detect_and_zstack.py",
        execution=mode,
        params={"threshold": 0.0, "max_stacks": 1, "z_range_um": 2.0, "z_step_um": 1.0},
    )
    base = useq.MDASequence(
        stage_positions=[(0, 0, 0), (100, 100, 0), (200, 200, 0)],
        channels=["DAPI"],
    )
    summary = _run(qtbot, controller, config, base, tmp_path / "run")

    assert summary["status"] == "completed"
    frames = _lines(tmp_path / "run" / "frames.jsonl")
    origins = [f["origin"] for f in frames]
    assert origins.count("base") == 3
    assert origins.count("analysis") == 3  # one 3-plane z-stack
    stack = [f for f in frames if f["origin"] == "analysis"]
    assert {f["parent_frame_id"] for f in stack} == {0}
    assert sorted(f["event"]["z_pos"] for f in stack) == [-1.0, 0.0, 1.0]


@pytest.mark.parametrize("mode", MODES)
def test_stop_when_ends_time_lapse_early(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path, mode: str
) -> None:
    controller = SmartController(mmcore)
    config = _config(
        TEMPLATES / "stop_when.py",
        execution=mode,
        params={"window": 2, "min_change": 65535.0},
    )
    base = useq.MDASequence(time_plan=useq.TIntervalLoops(interval=0, loops=50))
    summary = _run(qtbot, controller, config, base, tmp_path / "run")
    assert summary["status"] == "stopped_by_script"
    assert summary["frames"] == 2


def test_minimal_measures_every_frame(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    controller = SmartController(mmcore)
    received: list[dict[str, Any]] = []
    controller.analysisFinished.connect(received.append)
    base = useq.MDASequence(channels=["DAPI", "FITC"], z_plan={"range": 2, "step": 1})
    summary = _run(
        qtbot, controller, _config(TEMPLATES / "minimal.py"), base, tmp_path / "run"
    )
    assert summary["status"] == "completed"
    assert summary["frames"] == summary["analyses"] == 6
    assert sum(1 for r in received if r["call"] == "analyze") == 6


def test_frame_handler_runs_on_runner_thread(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    """In the GUI the MDA signals are Qt signals; a normal connection would
    queue the handler to the GUI thread (see the plan's G2)."""
    threads: list[threading.Thread] = []

    class Probe(SmartController):
        def _on_frame_ready(
            self, img: np.ndarray, event: useq.MDAEvent, meta: FrameMetaV1
        ) -> None:
            threads.append(threading.current_thread())
            super()._on_frame_ready(img, event, meta)

    controller = Probe(mmcore)
    _run(
        qtbot,
        controller,
        _config(TEMPLATES / "minimal.py"),
        useq.MDASequence(time_plan={"interval": 0, "loops": 2}),
        tmp_path / "run",
    )
    assert len(threads) == 2
    assert all(t is not threading.main_thread() for t in threads)


def test_returned_time_lapse_keeps_its_own_interval(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    script = _script(
        tmp_path,
        """
        import useq
        API_VERSION = 1
        def analyze(image, frame, ctx):
            if frame.frame_id == 0:
                return useq.MDASequence(
                    time_plan=useq.TIntervalLoops(interval=0.4, loops=2)
                )
        """,
    )
    controller = SmartController(mmcore)
    summary = _run(
        qtbot,
        controller,
        _config(script),
        useq.MDASequence(channels=["DAPI"]),
        tmp_path / "run",
    )
    assert summary["frames"] == 3
    times = [f["runner_time_ms"] for f in _lines(tmp_path / "run" / "frames.jsonl")]
    assert times[2] - times[1] >= 350  # interval honored...
    assert times[1] - times[0] < 350  # ...and the first one not delayed


@pytest.mark.parametrize(
    ("on_error", "status"), [("stop", "error"), ("skip", "completed")]
)
def test_script_error_policy(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path, on_error: str, status: str
) -> None:
    script = _script(
        tmp_path,
        """
        API_VERSION = 1
        def analyze(image, frame, ctx):
            if frame.frame_id == 1:
                raise ValueError("bad frame")
        """,
    )
    controller = SmartController(mmcore)
    errors: list[tuple[str, bool]] = []
    controller.analysisError.connect(lambda m, f: errors.append((m, f)))
    base = useq.MDASequence(time_plan={"interval": 0, "loops": 5})
    summary = _run(
        qtbot, controller, _config(script, on_error=on_error), base, tmp_path / "run"
    )
    assert summary["status"] == status
    assert summary["errors"] == 1
    if on_error == "stop":
        assert summary["frames"] < 5
        assert errors and "bad frame" in errors[0][0] and errors[0][1] is False
    else:
        assert summary["frames"] == 5
        assert not errors


def test_crashing_process_stops_run_cleanly(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    script = _script(
        tmp_path,
        """
        import os
        API_VERSION = 1
        def analyze(image, frame, ctx):
            os._exit(3)
        """,
    )
    controller = SmartController(mmcore)
    fatal: list[bool] = []
    controller.analysisError.connect(lambda _m, f: fatal.append(f))
    base = useq.MDASequence(time_plan={"interval": 0, "loops": 10})
    summary = _run(
        qtbot, controller, _config(script, execution="process"), base, tmp_path / "run"
    )
    assert summary["status"] == "error"
    assert fatal == [True]
    assert summary["frames"] < 10


def test_cancel_while_blocked_on_analysis(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    script = _script(
        tmp_path,
        """
        import time
        API_VERSION = 1
        SYNC = "blocking"
        def analyze(image, frame, ctx):
            time.sleep(1.5)
        """,
    )
    controller = SmartController(mmcore)
    controller.prepare(_config(script), tmp_path / "run")
    base = useq.MDASequence(time_plan={"interval": 0, "loops": 10})
    with qtbot.waitSignal(controller.analysisQueued, timeout=5000):
        controller.start(base, output="memory")

    cancelled_at = time.perf_counter()
    with qtbot.waitSignal(mmcore.mda.events.sequenceFinished, timeout=5000):
        controller.cancel()
    assert time.perf_counter() - cancelled_at < 1.0  # not waiting for analysis
    assert mmcore.mda.status.finish_reason == "canceled"  # pymmcore-plus fix

    with qtbot.waitSignal(controller.runFinished, timeout=15_000) as blocker:
        pass
    assert blocker.args is not None
    assert blocker.args[0]["status"] == "cancelled"
    assert blocker.args[0]["frames"] == 1


def test_cancel_from_outside_is_reported_as_cancelled(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    """E.g. the Acquire page's Cancel button calls mda.cancel() directly."""
    script = _script(
        tmp_path,
        """
        import time
        API_VERSION = 1
        def analyze(image, frame, ctx):
            time.sleep(0.5)
        """,
    )
    controller = SmartController(mmcore)
    controller.prepare(_config(script), tmp_path / "run")
    with qtbot.waitSignal(controller.analysisQueued, timeout=5000):
        controller.start(useq.MDASequence(time_plan={"interval": 0, "loops": 10}))
    with qtbot.waitSignal(controller.runFinished, timeout=15_000) as blocker:
        mmcore.mda.cancel()
    assert blocker.args is not None
    assert blocker.args[0]["status"] == "cancelled"


def test_prepare_reports_setup_failure(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    script = _script(
        tmp_path,
        """
        API_VERSION = 1
        def setup(ctx):
            raise RuntimeError("no GPU")
        def analyze(image, frame, ctx): ...
        """,
    )
    controller = SmartController(mmcore)
    with pytest.raises(SmartRunError, match="no GPU"):
        controller.prepare(_config(script), tmp_path / "run")
    assert not controller.is_active()


def test_stopped_run_drops_late_results(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    script = _script(
        tmp_path,
        """
        import time, useq
        API_VERSION = 1
        SYNC = "async"
        def analyze(image, frame, ctx):
            time.sleep(0.3)
            return useq.MDAEvent()
        """,
    )
    controller = SmartController(mmcore)
    controller.prepare(_config(script), tmp_path / "run")
    with qtbot.waitSignal(controller.analysisQueued, timeout=5000):
        controller.start(useq.MDASequence(time_plan={"interval": 0, "loops": 3}))
    controller.request_stop()
    with qtbot.waitSignal(controller.runFinished, timeout=15_000) as blocker:
        pass
    assert blocker.args is not None
    summary = blocker.args[0]
    assert summary["status"] == "stopped_by_user"
    assert summary["injected_events"] == 0
    assert summary["dropped_responses"] >= 1


def test_empty_base_sequence_is_refused(
    mmcore: CMMCorePlus, qtbot: QtBot, tmp_path: Path
) -> None:
    controller = SmartController(mmcore)
    controller.prepare(_config(TEMPLATES / "minimal.py"), tmp_path / "run")
    try:
        with pytest.raises(SmartRunError, match="no events"):
            controller.start(useq.MDASequence())
    finally:
        controller.abandon()
    assert not controller.is_active()
