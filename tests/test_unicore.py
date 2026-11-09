"""Running on UniMMCore: `--unicore`, auto-selection at startup, and load warnings."""

from __future__ import annotations

import weakref
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from pymmcore_plus.core import _mmcore_plus
from pymmcore_plus.experimental.unicore import UniMMCore

from pymmcore_gui import MicroManagerGUI, _app
from pymmcore_gui._qt.QtWidgets import QFileDialog, QMessageBox
from pymmcore_gui._theme import DARK_THEME, set_theme
from pymmcore_gui._unicore import config_needs_unicore
from pymmcore_gui.widgets._hardware import HardwareSetupPage
from pymmcore_gui.widgets._startup import StartupChoice

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot

TEST_CONFIG = Path(__file__).with_name("test_config.cfg")
PY_DEVICE = "PyShutter"
_DEVICE_MODULE = "pymmgui_test_pydevices"


@pytest.fixture
def py_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A .cfg declaring one Python shutter, whose module is importable."""
    (tmp_path / f"{_DEVICE_MODULE}.py").write_text(
        "from pymmcore_plus.experimental.unicore import ShutterDevice\n"
        "\n"
        "class MinimalShutter(ShutterDevice):\n"
        "    _open = False\n"
        "    def get_open(self) -> bool:\n"
        "        return self._open\n"
        "    def set_open(self, value: bool) -> None:\n"
        "        self._open = value\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    cfg = tmp_path / "py_devices.cfg"
    cfg.write_text(
        f"#py pyDevice,{PY_DEVICE},{_DEVICE_MODULE},MinimalShutter\n"
        f"#py Property,Core,Shutter,{PY_DEVICE}\n"
        "#py Property,Core,Initialize,1\n"
    )
    return cfg


def _launch(**kwargs: object) -> MicroManagerGUI:
    win = _app.create_mmgui(
        install_sys_excepthook=False,
        install_sentry=False,
        exec_app=False,
        **kwargs,  # type: ignore[arg-type]
    )
    assert isinstance(win, MicroManagerGUI)
    return win


def test_config_needs_unicore(tmp_path: Path, py_cfg: Path) -> None:
    assert config_needs_unicore(py_cfg)
    assert not config_needs_unicore(TEST_CONFIG)
    assert not config_needs_unicore(tmp_path / "missing.cfg")

    # same rule as UniMMCore's parser: `#py ` after stripping, nothing else
    indented = tmp_path / "indented.cfg"
    indented.write_text("Property,Core,Initialize,1\n   #py Property,Core,X,Y\n")
    assert config_needs_unicore(indented)
    look_alike = tmp_path / "look_alike.cfg"
    look_alike.write_text("#pyDevice is not a python line\n# py nor is this\n")
    assert not config_needs_unicore(look_alike)


def test_unicore_flag_runs_on_a_new_unicore(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    # the conftest `mmcore` is a standard global core; the explicit flag wins
    win = _launch(mm_config=False, unicore=True)
    qtbot.addWidget(win)
    assert isinstance(win.mmcore, UniMMCore)
    assert win.mmcore is not mmcore


def test_unicore_flag_reuses_a_global_unicore(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    uni = UniMMCore()
    monkeypatch.setattr(_mmcore_plus, "_instance", weakref.ref(uni))
    win = _launch(mm_config=False, unicore=True)
    qtbot.addWidget(win)
    assert win.mmcore is uni


def test_unicore_flag_rejects_a_standard_mmcore(mmcore: CMMCorePlus) -> None:
    with pytest.raises(ValueError, match="not a UniMMCore"):
        _app.create_mmgui(mmcore=mmcore, unicore=True, exec_app=False)


def test_standard_launch_keeps_the_standard_core(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    win = _launch(mm_config=str(TEST_CONFIG))
    qtbot.addWidget(win)
    assert win.mmcore is mmcore
    assert not isinstance(win.mmcore, UniMMCore)


@pytest.mark.parametrize("via", ["config_flag", "startup_dialog"])
def test_python_device_config_picks_unicore_at_startup(
    py_cfg: Path, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, via: str
) -> None:
    # a plain launch: nothing has created a core yet
    monkeypatch.setattr(_mmcore_plus, "_instance", None)

    if via == "config_flag":
        win = _launch(mm_config=str(py_cfg))
    else:
        choice = StartupChoice(layout="Default", config=str(py_cfg))
        with patch.object(
            MicroManagerGUI, "prompt_startup_choices", return_value=choice
        ):
            win = _launch(mm_config=None)
    qtbot.addWidget(win)

    assert isinstance(win.mmcore, UniMMCore)
    assert PY_DEVICE in win.mmcore.getLoadedDevices()
    assert win.mmcore.getShutterDevice() == PY_DEVICE


def test_python_device_config_keeps_an_existing_standard_core_but_warns(
    mmcore: CMMCorePlus, py_cfg: Path, qtbot: QtBot
) -> None:
    """A core the caller already set up is not replaced because of the file."""
    with pytest.warns(RuntimeWarning, match="declares Python devices"):
        win = _launch(mm_config=str(py_cfg))
    qtbot.addWidget(win)
    assert win.mmcore is mmcore
    assert PY_DEVICE not in mmcore.getLoadedDevices()


def _load_from_hardware_page(page: HardwareSetupPage, path: Path) -> None:
    with patch.object(QFileDialog, "getOpenFileName", return_value=(str(path), "")):
        page.load_config()


@pytest.mark.parametrize(
    "answer", [QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Yes]
)
def test_hardware_load_asks_before_dropping_python_devices(
    mmcore: CMMCorePlus,
    py_cfg: Path,
    qtbot: QtBot,
    answer: QMessageBox.StandardButton,
) -> None:
    set_theme(DARK_THEME)
    page = HardwareSetupPage(mmcore)
    qtbot.addWidget(page)

    with (
        patch.object(QMessageBox, "warning", return_value=answer) as warning,
        patch.object(mmcore, "loadSystemConfiguration") as load,
    ):
        _load_from_hardware_page(page, py_cfg)

    warning.assert_called_once()
    assert "pymm --unicore" in warning.call_args.args[2]
    if answer == QMessageBox.StandardButton.Yes:
        load.assert_called_once_with(str(py_cfg))
    else:
        load.assert_not_called()


def test_hardware_load_of_a_standard_config_does_not_ask(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    set_theme(DARK_THEME)
    page = HardwareSetupPage(mmcore)
    qtbot.addWidget(page)

    with patch.object(QMessageBox, "warning") as warning:
        _load_from_hardware_page(page, TEST_CONFIG)
    warning.assert_not_called()


def test_hardware_page_round_trips_python_devices_on_unicore(
    py_cfg: Path, tmp_path: Path, qtbot: QtBot
) -> None:
    set_theme(DARK_THEME)
    core = UniMMCore()
    page = HardwareSetupPage(core)
    qtbot.addWidget(page)

    with patch.object(QMessageBox, "warning") as warning:
        _load_from_hardware_page(page, py_cfg)

    warning.assert_not_called()
    assert PY_DEVICE in core.getLoadedDevices()
    assert page.model.config_file == str(py_cfg)

    # saving goes through the config model, which must keep the `#py` lines
    saved = tmp_path / "saved.cfg"
    assert page.save_to(str(saved))
    reloaded = UniMMCore()
    reloaded.loadSystemConfiguration(saved)
    assert reloaded.isPyDevice(PY_DEVICE)
    assert reloaded.getShutterDevice() == PY_DEVICE
