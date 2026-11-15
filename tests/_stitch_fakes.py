"""A fake stitching worker, for tests that need the queue but not the engine.

Needs neither multiview-stitcher nor real data: it writes the protocol files
the real worker would, in one of the modes "ok", "fail", "crash" or "hang".
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from pymmcore_gui._stitching._controller import StitchingController

if TYPE_CHECKING:
    from pathlib import Path

FAKE_WORKER = r"""
import json, os, sys, time
from pathlib import Path

job_dir = Path(sys.argv[1])
mode = sys.argv[2]
job = json.loads((job_dir / "job.json").read_text())
log = open(job_dir / "worker.log", "a")
import psutil
me = psutil.Process()
(job_dir / "owner.json").write_text(json.dumps(
    {"schema_version": 1, "pid": me.pid, "create_time": me.create_time()}))

def write(name, obj):
    tmp = job_dir / (name + ".part")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, job_dir / name)

for g in job["groups"]:
    Path(g["output"] + ".tmp").mkdir(parents=True, exist_ok=True)
write("progress.json", {"schema_version": 1, "phase": "fusing", "done": 1,
                        "total": 4})
if mode == "ok":
    time.sleep(0.2)
    groups = []
    for g in job["groups"]:
        os.replace(g["output"] + ".tmp", g["output"])
        groups.append({"output": g["output"], "n_tiles": len(g["tiles"]),
                       "registered": True})
    write("result.json", {"schema_version": 1, "groups": groups,
                          "elapsed_s": 0.2})
    sys.exit(0)
if mode == "fail":
    write("error.json", {"schema_version": 1, "message": "no overlap at all"})
    sys.exit(1)
if mode == "crash":
    print("Segmentation fault (core dumped)", file=log, flush=True)
    os._exit(7)
if mode == "hang":
    time.sleep(600)
"""


def write_fake_worker(directory: Path) -> Path:
    script = directory / "fake_worker.py"
    script.write_text(FAKE_WORKER)
    return script


def fake_controller(
    tmp_path: Path, script: Path, modes: dict[str, str] | str
) -> StitchingController:
    """A controller whose jobs run `script` in `modes` (one, or per job id)."""

    def command(job_dir: Path) -> tuple[str, list[str]]:
        mode = modes if isinstance(modes, str) else modes[job_dir.name]
        return sys.executable, [str(script), str(job_dir), mode]

    return StitchingController(
        registry=tmp_path / "registry", command=command, cleanup_on_start=False
    )
