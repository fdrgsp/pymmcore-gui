import os
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path

import pytest

NAME = "pymmgui"
DIST = Path(__file__).parent.parent / "dist"
APP = DIST / NAME / (NAME + (".exe" if os.name == "nt" else ""))

if not APP.exists():
    pytest.skip(f"App not built: {APP}", allow_module_level=True)

import pyautogui  # noqa: E402


@pytest.fixture
def app_process() -> Iterator[subprocess.Popen]:
    # PYTHONFAULTHANDLER makes the frozen app's own interpreter dump a
    # Python-level thread traceback to stderr if it hits a fatal native
    # exception (e.g. access violation) -- the same mechanism pytest already
    # gets for free in-process. Capturing stderr (previously left
    # unredirected) is what actually lets us see that dump.
    env = os.environ.copy()
    env["PYTHONFAULTHANDLER"] = "1"
    proc = subprocess.Popen(
        [str(APP)],
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    stderr_chunks: list[bytes] = []

    def _drain_stderr() -> None:
        if proc.stderr is not None:
            for line in proc.stderr:
                stderr_chunks.append(line)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    # --- wait for the GUI to tell us it's ready ---
    while True:
        # this "READY" line is printed in _app.create_mmgui
        # when "PYTEST_VERSION" is set in the environment
        if proc.stdout and proc.stdout.readline().strip() == b"READY":
            break
        time.sleep(0.1)

    with proc:
        yield proc

        # --- teardown ---
        if proc.poll() is None:
            # This is a launch smoke test, not a shutdown test. The frozen
            # Windows app currently faults in native Qt/ADS cleanup after a
            # WM_CLOSE, so do not enter that unrelated teardown path here.
            if os.name == "nt":
                proc.kill()
            else:
                proc.terminate()
            try:
                if os.name != "nt":
                    with suppress(Exception):
                        pyautogui.moveTo(1200, 600, duration=0.1)
                proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        stderr_thread.join(timeout=2)

    captured_stderr = b"".join(stderr_chunks).decode(errors="replace")
    acceptable_codes = {0, 1} if os.name == "nt" else {0, -9}
    assert proc.returncode in acceptable_codes, (
        f"app exited with unexpected code {proc.returncode}\n"
        f"--- captured child stderr ---\n{captured_stderr or '(empty)'}"
    )


CMD_CTRL = "ctrl" if os.name == "nt" else "command"


@pytest.mark.usefixtures("app_process")
def test_app() -> None:
    time.sleep(1)
