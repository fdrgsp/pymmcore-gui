import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pymmcore_plus import CMMCorePlus
from pytest import MonkeyPatch

from pymmcore_gui import __main__, _app
from pymmcore_gui._qt.QtWidgets import QApplication, QMessageBox
from pymmcore_gui._settings import Settings


@pytest.mark.order(0)
def test_main_app(monkeypatch: MonkeyPatch) -> None:
    with patch.object(
        _app.MMQApplication, "exec", lambda _: QApplication.processEvents()
    ):
        assert not QApplication.instance()
        monkeypatch.setattr(sys, "argv", ["mmgui", "run", "--demo-config"])
        with pytest.raises(SystemExit):
            __main__.main()

        assert QApplication.instance()
        assert isinstance(QApplication.instance(), _app.MMQApplication)
        assert hasattr(sys, "_original_excepthook_")
        for wdg in QApplication.topLevelWidgets():
            wdg.close()


def test_failed_startup_config_load_shows_a_dialog(
    settings: Settings, mmcore: CMMCorePlus
) -> None:
    """A failure loading the user's chosen/last-used config must be visible.

    Regression test: this used to only ``warnings.warn`` -- invisible in a
    windowed/frozen build with no console -- so clicking "Yes" on
    ``LoadConfigDialog`` (or relying on a stored auto-load preference) could
    silently fail with no feedback at all.
    """
    settings.last_config = Path("this-config-does-not-exist.cfg")
    settings.auto_load_last_config = True

    def _raise(*_a: object, **_k: object) -> None:
        raise RuntimeError("boom")

    with (
        patch.object(mmcore, "loadSystemConfiguration", _raise),
        patch.object(QMessageBox, "critical") as critical,
        pytest.warns(RuntimeWarning, match="boom"),
    ):
        _app.create_mmgui(
            mm_config=None,
            mmcore=mmcore,
            install_sys_excepthook=False,
            install_sentry=False,
            exec_app=False,
        )
        try:
            critical.assert_called_once()
            args = critical.call_args.args
            assert "this-config-does-not-exist.cfg" in args[2]
            assert "boom" in args[2]
        finally:
            for wdg in QApplication.topLevelWidgets():
                wdg.close()


def test_failed_explicit_config_load_is_not_fatal(mmcore: CMMCorePlus) -> None:
    """An explicitly requested config that won't load must not kill the app.

    Regression test: this branch (the startup dialog's choice, or ``-c``) had
    no error handling at all, so the exception escaped ``create_mmgui``. As
    the main window is only shown once the event loop starts, that killed the
    process with the window built but never shown -- and in the frozen build
    (windowed, no console) with no traceback either. The app simply appeared
    to do nothing after the startup dialog.

    The default first-launch choice, "Demo configuration", hits exactly this:
    it resolves against the Micro-Manager install ``pymmcore-plus`` finds, and
    the bundle ships no device adapters.
    """

    def _raise(*_a: object, **_k: object) -> None:
        raise FileNotFoundError("Path does not exist: MMConfig_demo.cfg")

    with (
        patch.object(mmcore, "loadSystemConfiguration", _raise),
        patch.object(QMessageBox, "critical") as critical,
        pytest.warns(RuntimeWarning, match="MMConfig_demo.cfg"),
    ):
        win = _app.create_mmgui(
            mm_config="MMConfig_demo.cfg",
            mmcore=mmcore,
            install_sys_excepthook=False,
            install_sentry=False,
            exec_app=False,
        )
        try:
            assert win is not None
            critical.assert_called_once()
            assert "MMConfig_demo.cfg" in critical.call_args.args[2]
        finally:
            for wdg in QApplication.topLevelWidgets():
                wdg.close()


def test_config_error_text_explains_a_missing_micromanager(
    monkeypatch: MonkeyPatch,
) -> None:
    """``Path does not exist: MMConfig_demo.cfg`` needs translating on a fresh box."""
    exc = FileNotFoundError("Path does not exist: MMConfig_demo.cfg")

    monkeypatch.setattr("pymmcore_plus.find_micromanager", lambda **_k: None)
    assert "No Micro-Manager installation was found" in _app._config_error_text(
        "MMConfig_demo.cfg", exc
    )

    monkeypatch.setattr("pymmcore_plus.find_micromanager", lambda **_k: "/some/mm")
    assert "No Micro-Manager installation was found" not in _app._config_error_text(
        "MMConfig_demo.cfg", exc
    )
