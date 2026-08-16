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
