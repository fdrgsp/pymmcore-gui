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
READY_TIMEOUT = 120
FAILURE_SCREENSHOT = DIST.parent / "bundle-failure.png"

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

    # Both pipes are drained by threads, so that nothing here can block on a
    # read: a stuck app must fail this test, not hang it.
    ready = threading.Event()
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []

    def _drain_stdout() -> None:
        if proc.stdout is not None:
            for line in proc.stdout:
                stdout_chunks.append(line)
                # this "READY" line is printed in _app.create_mmgui
                # when "PYTEST_VERSION" is set in the environment
                if line.strip() == b"READY":
                    ready.set()

    def _drain_stderr() -> None:
        if proc.stderr is not None:
            for line in proc.stderr:
                stderr_chunks.append(line)

    threads = [
        threading.Thread(target=_drain_stdout, daemon=True),
        threading.Thread(target=_drain_stderr, daemon=True),
    ]
    for thread in threads:
        thread.start()

    def _output() -> str:
        out = b"".join(stdout_chunks).decode(errors="replace")
        err = b"".join(stderr_chunks).decode(errors="replace")
        return (
            f"--- captured child stdout ---\n{out or '(empty)'}\n"
            f"--- captured child stderr ---\n{err or '(empty)'}"
        )

    def _release() -> None:
        """Wait for the app to exit and close its pipes."""
        proc.wait()
        for thread in threads:
            thread.join(timeout=2)
        # closing a pipe blocks if a thread is still reading it
        if not any(thread.is_alive() for thread in threads):
            for pipe in (proc.stdout, proc.stderr):
                if pipe is not None:
                    pipe.close()

    # --- wait for the GUI to tell us it's ready ---
    deadline = time.monotonic() + READY_TIMEOUT
    while not ready.wait(0.1):
        if proc.poll() is not None:
            _release()
            pytest.fail(
                f"app exited with code {proc.returncode} before it was ready\n"
                f"{_output()}"
            )
        if time.monotonic() > deadline:
            # Windowed builds report an unhandled startup exception in a modal
            # message box, which leaves the app alive and silent, so the screen
            # may be the only place the error is shown.
            with suppress(Exception):
                pyautogui.screenshot(str(FAILURE_SCREENSHOT))
            proc.kill()
            _release()
            pytest.fail(
                f"app was not ready after {READY_TIMEOUT} s, "
                f"see {FAILURE_SCREENSHOT.name}\n{_output()}"
            )

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

    _release()

    acceptable_codes = {0, 1} if os.name == "nt" else {0, -9}
    assert proc.returncode in acceptable_codes, (
        f"app exited with unexpected code {proc.returncode}\n{_output()}"
    )


CMD_CTRL = "ctrl" if os.name == "nt" else "command"


def test_app(app_process: subprocess.Popen) -> None:
    time.sleep(1)
    # The "READY" line the fixture waits for is printed before the window class
    # is even chosen, and Windows teardown below accepts exit code 1 (that's
    # what `proc.kill()` yields) -- so without this, a bundle that dies on
    # startup passes. That is exactly how a frozen build shipped that showed
    # the startup dialog and then silently exited.
    assert app_process.poll() is None, "app exited on its own shortly after startup"
