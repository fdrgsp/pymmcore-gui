"""Running on UniMMCore: `--unicore`, startup, warnings, Python device adapters."""

from __future__ import annotations

import sys
import weakref
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from pymmcore_plus import DeviceType
from pymmcore_plus.core import _mmcore_plus
from pymmcore_plus.experimental.unicore import UniMMCore

from pymmcore_gui import MicroManagerGUI, _app
from pymmcore_gui._qt.QtCore import Qt
from pymmcore_gui._qt.QtWidgets import QFileDialog, QMessageBox, QPushButton
from pymmcore_gui._theme import DARK_THEME, set_theme
from pymmcore_gui._unicore import (
    config_needs_unicore,
    py_devices_dir,
    scan_py_devices,
    startup_core,
)
from pymmcore_gui.widgets._hardware import HardwareSetupPage
from pymmcore_gui.widgets._hardware._peripherals import PeripheralsDialog
from pymmcore_gui.widgets._startup import StartupChoice

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.model import AvailableDevice
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


# ── Python device adapters folder ─────────────────────────────────

BENCH_ADAPTER = '''
from pymmcore_plus.experimental.unicore import ShutterDevice


class BenchShutter(ShutterDevice):
    """A shutter on the bench.

    Longer details that the device list doesn't show.
    """

    _open = False

    def get_open(self) -> bool:
        return self._open

    def set_open(self, value: bool) -> None:
        self._open = value
'''


def _py_page(qtbot: QtBot) -> tuple[HardwareSetupPage, UniMMCore]:
    set_theme(DARK_THEME)
    core = UniMMCore()
    page = HardwareSetupPage(core)
    qtbot.addWidget(page)
    return page, core


def _listed(page: HardwareSetupPage) -> dict[str, AvailableDevice]:
    return {d.adapter_name: d for d in page._available._devices if d.is_py_device}


def _select(page: HardwareSetupPage, adapter_name: str) -> None:
    table = page._available._table
    for row in range(table.rowCount()):
        item = table.item(row, 0)
        dev = item.data(Qt.ItemDataRole.UserRole) if item else None
        if dev is not None and dev.adapter_name == adapter_name:
            table.selectRow(row)
            return
    raise AssertionError(f"{adapter_name!r} not found in Available Devices")


def test_scan_lists_device_classes_without_running_them(
    py_devices_folder: Path,
) -> None:
    py_devices_folder.mkdir(exist_ok=True)
    (py_devices_folder / "lab_devices.py").write_text(
        'raise RuntimeError("scanning must not run this file")\n'
        "from pymmcore_plus.experimental import unicore\n"
        "from pymmcore_plus.experimental.unicore import ShutterDevice, StateDevice\n"
        "class FastShutter(ShutterDevice):\n"
        '    """A fast shutter.\n\n    Details.\n    """\n'
        "class BaseWheel(StateDevice): ...\n"
        "class FilterWheel(BaseWheel): ...\n"  # via another class in the file
        "class Stage(unicore.XYStageDevice): ...\n"  # attribute base
        "class _Hidden(ShutterDevice): ...\n"
        "class NotADevice: ...\n"
        "class LabHub(unicore.HubDevice): ...\n"
    )
    shutter = "from pymmcore_plus.experimental.unicore import ShutterDevice\n"
    shutter += "class Skipped(ShutterDevice): ...\n"
    for name in ("_helpers.py", "bad-name.py", "2cams.py", "class.py", "json.py"):
        (py_devices_folder / name).write_text(shutter)
    (py_devices_folder / "broken.py").write_text("x = 1\nclass (:\n")

    scan = scan_py_devices()
    found = {
        (d.library, d.adapter_name, d.device_type, d.description) for d in scan.devices
    }
    assert found == {
        ("lab_devices", "FastShutter", DeviceType.Shutter, "A fast shutter."),
        ("lab_devices", "BaseWheel", DeviceType.State, ""),
        ("lab_devices", "FilterWheel", DeviceType.State, ""),
        ("lab_devices", "Stage", DeviceType.XYStage, ""),
        ("lab_devices", "LabHub", DeviceType.Hub, ""),
    }
    assert all(d.is_py_device for d in scan.devices)

    # each unusable file says why (and `_`-prefixed helpers are left alone)
    problems = dict(scan.problems)
    assert set(problems) == {
        "bad-name.py",
        "2cams.py",
        "class.py",
        "json.py",
        "broken.py",
    }
    assert "e.g. bad_name.py" in problems["bad-name.py"]
    assert "e.g. devices_2cams.py" in problems["2cams.py"]
    assert "keyword" in problems["class.py"]
    assert "clashes with the module 'json'" in problems["json.py"]
    assert problems["broken.py"].startswith("line 2:")


def test_py_devices_dir_is_created_and_importable(py_devices_folder: Path) -> None:
    assert py_devices_dir() == py_devices_folder
    assert py_devices_folder.is_dir()
    # appended, so an adapter file can never shadow an installed module
    assert sys.path[-1] == str(py_devices_folder)


def test_hardware_page_lists_and_adds_python_devices(
    py_devices_folder: Path, tmp_path: Path, qtbot: QtBot
) -> None:
    py_devices_dir().joinpath("bench_devices.py").write_text(BENCH_ADAPTER)
    page, core = _py_page(qtbot)

    buttons = [b.text() for b in page.toolbar.findChildren(QPushButton)]
    assert "Python Devices…" in buttons
    listed = _listed(page)
    assert set(listed) == {"BenchShutter"}
    assert listed["BenchShutter"].library == "bench_devices"
    assert listed["BenchShutter"].description == "A shutter on the bench."

    _select(page, "BenchShutter")
    page._begin_add("Bench")
    assert core.isPyDevice("Bench")
    (dev,) = (d for d in page.model.devices if d.name == "Bench")
    assert dev.is_py_device

    saved = tmp_path / "saved.cfg"
    assert page.save_to(str(saved))
    assert "#py pyDevice,Bench,bench_devices,BenchShutter" in saved.read_text()


def test_standard_core_lists_no_python_devices(
    mmcore: CMMCorePlus, py_devices_folder: Path, qtbot: QtBot
) -> None:
    py_devices_folder.mkdir(exist_ok=True)
    (py_devices_folder / "bench_devices.py").write_text(BENCH_ADAPTER)
    set_theme(DARK_THEME)
    page = HardwareSetupPage(mmcore)
    qtbot.addWidget(page)

    assert not _listed(page)
    buttons = [b.text() for b in page.toolbar.findChildren(QPushButton)]
    assert "Python Devices…" not in buttons


def test_adapter_added_while_running_is_listed(
    py_devices_folder: Path, qtbot: QtBot
) -> None:
    page, _core = _py_page(qtbot)
    assert page._py_watcher is not None
    assert str(py_devices_folder) in page._py_watcher.directories()
    assert not _listed(page)

    (py_devices_folder / "bench_devices.py").write_text(BENCH_ADAPTER)
    page._on_py_devices_changed(str(py_devices_folder))

    assert set(_listed(page)) == {"BenchShutter"}
    # the file itself is now watched, so in-place edits are picked up too
    assert str(py_devices_folder / "bench_devices.py") in page._py_watcher.files()


def test_adapter_with_a_missing_package_names_it(
    py_devices_folder: Path, qtbot: QtBot
) -> None:
    py_devices_dir().joinpath("needs_hw.py").write_text(
        "import pymmgui_no_such_package\n" + BENCH_ADAPTER
    )
    page, core = _py_page(qtbot)

    _select(page, "BenchShutter")
    with patch.object(page, "_warn") as warn:
        page._begin_add("Bench")

    assert "No module named 'pymmgui_no_such_package'" in warn.call_args.args[0]
    assert "Bench" not in core.getLoadedDevices()


def test_startup_config_loads_adapters_from_the_devices_folder(
    py_devices_folder: Path,
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    py_devices_folder.mkdir(exist_ok=True)
    (py_devices_folder / "bench_devices.py").write_text(BENCH_ADAPTER)
    cfg = tmp_path / "bench.cfg"
    cfg.write_text(
        "#py pyDevice,Bench,bench_devices,BenchShutter\n"
        "#py Property,Core,Initialize,1\n"
        "#py Property,Core,Shutter,Bench\n"
    )
    monkeypatch.setattr(_mmcore_plus, "_instance", None)

    win = _launch(mm_config=str(cfg))
    qtbot.addWidget(win)

    assert win.mmcore.getShutterDevice() == "Bench"


def test_standalone_app_never_runs_on_unicore(
    mmcore: CMMCorePlus, py_cfg: Path, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(_mmcore_plus, "_instance", None)
    assert startup_core(True, None) is None
    assert startup_core(False, str(py_cfg)) is None

    # and the Hardware tab doesn't point at a flag the app refuses
    set_theme(DARK_THEME)
    page = HardwareSetupPage(mmcore)
    qtbot.addWidget(page)
    with patch.object(
        QMessageBox, "warning", return_value=QMessageBox.StandardButton.Cancel
    ) as warning:
        _load_from_hardware_page(page, py_cfg)
    assert "not in the standalone app" in warning.call_args.args[2]


def test_unreadable_adapter_files_are_flagged_above_the_list(
    py_devices_folder: Path, qtbot: QtBot
) -> None:
    broken = py_devices_dir() / "broken.py"
    broken.write_text("class (:\n")
    page, _core = _py_page(qtbot)

    problems = page._available._problems
    assert not problems.isHidden()
    assert "1 file in the Python devices folder couldn't be read" in problems.text()
    assert "broken.py: line 1" in problems.toolTip()
    with patch.object(QMessageBox, "warning") as warning:
        problems.linkActivated.emit("details")
    assert "broken.py: line 1" in warning.call_args.args[2]

    # fixing the file clears it, without restarting
    broken.write_text(BENCH_ADAPTER)
    page._on_py_devices_changed(str(broken))
    assert problems.isHidden()
    assert "BenchShutter" in _listed(page)


HUB_ADAPTER = '''
from pymmcore_plus.experimental.unicore import HubDevice, ShutterDevice


class LabHub(HubDevice):
    """A controller box."""

    def get_installed_peripherals(self):
        return [("PortShutter", "Shutter on port A")]


class PortShutter(ShutterDevice):
    _open = False

    def get_open(self) -> bool:
        return self._open

    def set_open(self, value: bool) -> None:
        self._open = value
'''


def test_python_hub_offers_its_peripherals(
    py_devices_folder: Path, tmp_path: Path, qtbot: QtBot
) -> None:
    py_devices_dir().joinpath("lab_hub.py").write_text(HUB_ADAPTER)
    page, core = _py_page(qtbot)

    # like a C++ hub library: the hub is listed, its peripheral hidden behind it
    listed = _listed(page)
    assert listed["LabHub"].device_type is DeviceType.Hub
    assert page._available._is_hub_child(listed["PortShutter"])

    def pick_all(dialog: PeripheralsDialog) -> int:
        offered = [dialog._table.item(r, 0) for r in range(dialog._table.rowCount())]
        assert [i.text() for i in offered if i] == ["PortShutter"]
        for item in offered:
            if item:
                item.setCheckState(Qt.CheckState.Checked)
        return 1

    _select(page, "LabHub")
    with patch.object(PeripheralsDialog, "exec", pick_all):
        page._begin_add("Hub")

    assert core.getParentLabel("PortShutter") == "Hub"
    assert {d.name for d in page.model.devices} >= {"Hub", "PortShutter"}
    # listed once, from the model (linked to its hub), not again from the scan
    assert [d.adapter_name for d in page._available._devices].count("PortShutter") == 1

    saved = tmp_path / "hub.cfg"
    assert page.save_to(str(saved))
    reloaded = UniMMCore()
    reloaded.loadSystemConfiguration(saved)
    assert reloaded.getParentLabel("PortShutter") == "Hub"
