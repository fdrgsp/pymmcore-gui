import os
import subprocess
import sys
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


def _close_windows_app(pid: int) -> bool:
    """Post WM_CLOSE to each visible top-level window owned by *pid*."""
    if sys.platform != "win32":
        return False

    import ctypes
    from ctypes import wintypes

    found = False
    user32 = ctypes.windll.user32

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _close(hwnd: int, _lparam: int) -> bool:
        nonlocal found
        window_pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(window_pid))
        if window_pid.value == pid and user32.IsWindowVisible(hwnd):
            found = bool(user32.PostMessageW(hwnd, 0x0010, 0, 0)) or found
        return True

    user32.EnumWindows(_close, 0)
    return found


@pytest.fixture
def app_process() -> Iterator[subprocess.Popen]:
    proc = subprocess.Popen(
        [str(APP)],
        start_new_session=True,
        stdout=subprocess.PIPE,
    )

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
            closed_gracefully = _close_windows_app(proc.pid)
            if not closed_gracefully:
                proc.terminate()
            try:
                if not closed_gracefully and os.name != "nt":
                    with suppress(Exception):
                        pyautogui.moveTo(1200, 600, duration=0.1)
                proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    acceptable_codes = {0, 1} if os.name == "nt" else {0, -9}
    assert proc.returncode in acceptable_codes


CMD_CTRL = "ctrl" if os.name == "nt" else "command"


@pytest.mark.usefixtures("app_process")
def test_app() -> None:
    time.sleep(1)
