"""Python devices on UniMMCore: startup core, device folder, loads and saves."""

from __future__ import annotations

import os
import sys
import weakref
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from pymmcore_plus.core import _mmcore_plus
from pymmcore_plus.experimental.unicore import UniMMCore

from pymmcore_gui import MicroManagerGUI, _app
from pymmcore_gui._light_sources import ChannelDeclaration, write_channel_comments
from pymmcore_gui._qt.QtWidgets import QFileDialog, QMessageBox
from pymmcore_gui._theme import DARK_THEME, set_theme
from pymmcore_gui._unicore import (
    DEVICES_PATH_KEY,
    PyDeviceFolder,
    config_needs_unicore,
    read_config,
    startup_core,
    write_devices_path,
)
from pymmcore_gui.widgets._hardware import HardwareSetupPage
from pymmcore_gui.widgets._startup import StartupChoice

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot

TEST_CONFIG = Path(__file__).with_name("test_config.cfg")

SHUTTERS = '''
from pymmcore_plus.experimental.unicore import ShutterDevice, pymm_property

from _helpers import CLOSED


class PyShutter(ShutterDevice):
    """A shutter for tests.

    More text.
    """

    _open = CLOSED
    _power = 0

    def get_open(self) -> bool:
        return self._open

    def set_open(self, value: bool) -> None:
        self._open = value

    @pymm_property
    def power(self) -> int:
        return self._power

    @power.setter
    def _set_power(self, value: int) -> None:
        self._power = value


class BrokenShutter(PyShutter):
    def initialize(self) -> None:
        raise RuntimeError("port busy")

    def busy(self) -> bool:
        raise RuntimeError("not initialized")


class _Private(PyShutter):
    pass
'''

PY_CFG = """\
Property,Core,Initialize,0
Device,Camera,DemoCamera,DCam
#py pyDevice,Shutter,shutters,PyShutter
Property,Core,Initialize,1
Property,Core,Camera,Camera
#py Property,Core,Shutter,Shutter
"""


@pytest.fixture(autouse=True)
def _restore_imports(tmp_path: Path) -> Iterator[None]:
    """Forget the device modules and `sys.path` entries a test added."""
    path = list(sys.path)
    yield
    sys.path[:] = path
    prefix = str(tmp_path.resolve())
    for name, module in list(sys.modules.items()):
        if (getattr(module, "__file__", None) or "").startswith(prefix):
            del sys.modules[name]


@pytest.fixture
def devices(tmp_path: Path) -> Path:
    """A folder with a device file and the helper it imports."""
    folder = tmp_path / "py_devices"
    folder.mkdir()
    (folder / "shutters.py").write_text(SHUTTERS)
    (folder / "_helpers.py").write_text("CLOSED = False\n")
    return folder


@pytest.fixture
def py_cfg(tmp_path: Path, devices: Path) -> Path:
    """A .cfg with a C++ camera and a Python shutter from `devices`."""
    cfg = tmp_path / "scope.cfg"
    cfg.write_text(f"{DEVICES_PATH_KEY},py_devices\n{PY_CFG}")
    return cfg


@pytest.fixture
def summaries() -> Iterator[list]:
    """Record load summaries instead of showing them."""
    shown: list = []
    with patch.object(
        HardwareSetupPage,
        "_show_load_summary",
        lambda self, info, report: shown.append(report),
    ):
        yield shown


def _page(qtbot: QtBot, core: CMMCorePlus) -> HardwareSetupPage:
    set_theme(DARK_THEME)
    page = HardwareSetupPage(core)
    qtbot.addWidget(page)
    return page


def _launch(**kwargs: object) -> MicroManagerGUI:
    win = _app.create_mmgui(
        install_sys_excepthook=False,
        install_sentry=False,
        exec_app=False,
        **kwargs,  # type: ignore[arg-type]
    )
    assert isinstance(win, MicroManagerGUI)
    return win


# ── the .cfg ──────────────────────────────────────────────────────


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


def test_read_config(py_cfg: Path, devices: Path) -> None:
    info = read_config(py_cfg)
    assert info.has_py_lines
    assert info.py_devices == ("Shutter",)
    # a relative folder resolves against the .cfg's own folder
    assert info.devices_path == devices.resolve()


def test_devices_path_line_survives_channel_comments(
    py_cfg: Path, devices: Path
) -> None:
    write_devices_path(py_cfg, devices)
    write_channel_comments(py_cfg, "Channel", {"DAPI": ChannelDeclaration(10)})
    assert read_config(py_cfg).devices_path == devices.resolve()
    assert py_cfg.read_text().count(DEVICES_PATH_KEY) == 1

    write_devices_path(py_cfg, None)
    assert DEVICES_PATH_KEY not in py_cfg.read_text()


# ── startup ───────────────────────────────────────────────────────


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


def test_a_config_with_groups_loads_into_a_unicore_window(qtbot: QtBot) -> None:
    """Widgets reacting to config events never query a half-loaded UniMMCore."""
    win = _launch(mm_config=str(TEST_CONFIG), unicore=True)
    qtbot.addWidget(win)
    assert isinstance(win.mmcore, UniMMCore)
    assert "Channel" in win.mmcore.getAvailableConfigGroups()


def test_standalone_app_never_starts_unicore(
    py_cfg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_mmcore_plus, "_instance", None)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert startup_core(True, None) is None
    assert startup_core(False, py_cfg) is None


@pytest.mark.parametrize("via", ["config_flag", "startup_dialog"])
def test_python_device_config_picks_unicore_at_startup(
    py_cfg: Path,
    devices: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    via: str,
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

    core = win.mmcore
    assert isinstance(core, UniMMCore)
    assert core.getShutterDevice() == "Shutter"
    # the folder came from the .cfg's path line
    assert win._py_folder.path == devices.resolve()
    assert "shutters" in win._py_folder.modules()
    assert "_helpers" in win._py_folder.modules()


@pytest.mark.parametrize("frozen", [False, True])
def test_python_device_config_on_the_standard_core_loads_the_rest(
    mmcore: CMMCorePlus,
    py_cfg: Path,
    qtbot: QtBot,
    summaries: list,
    monkeypatch: pytest.MonkeyPatch,
    frozen: bool,
) -> None:
    """A caller's standard core (or the standalone app) skips the `#py` lines."""
    if frozen:
        monkeypatch.setattr(sys, "frozen", True, raising=False)
    win = _launch(mm_config=str(py_cfg))
    qtbot.addWidget(win)

    assert win.mmcore is mmcore
    assert "Camera" in mmcore.getLoadedDevices()
    assert "Shutter" not in mmcore.getLoadedDevices()
    qtbot.waitUntil(lambda: bool(summaries))
    (note,) = summaries[0].notes
    assert ("pip install pymmcore-gui" in note) is frozen
    assert (f"pymm -c {py_cfg}" in note) is not frozen


# ── loads in the GUI ──────────────────────────────────────────────


def _load_from_hardware_page(page: HardwareSetupPage, path: Path) -> None:
    with patch.object(QFileDialog, "getOpenFileName", return_value=(str(path), "")):
        page.load_config()


@pytest.mark.parametrize("answer", [None, "Load without them"])
def test_hardware_load_asks_before_dropping_python_devices(
    mmcore: CMMCorePlus, py_cfg: Path, qtbot: QtBot, answer: str | None
) -> None:
    page = _page(qtbot, mmcore)
    with (
        patch.object(page, "_choose", return_value=answer) as choose,
        patch.object(mmcore, "loadSystemConfiguration") as load,
    ):
        _load_from_hardware_page(page, py_cfg)

    # the text gives the exact command to relaunch with
    assert f"pymm -c {py_cfg}" in choose.call_args.args[1]
    if answer:
        load.assert_called_once_with(str(py_cfg))
    else:
        load.assert_not_called()


def test_hardware_load_of_a_standard_config_does_not_ask(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    page = _page(qtbot, mmcore)
    with patch.object(page, "_choose") as choose:
        _load_from_hardware_page(page, TEST_CONFIG)
    choose.assert_not_called()


def test_relaunch_command_quotes_config_paths_with_spaces(
    mmcore: CMMCorePlus, py_cfg: Path, qtbot: QtBot
) -> None:
    path = py_cfg.rename(py_cfg.with_name("scope with spaces.cfg"))
    page = _page(qtbot, mmcore)

    with patch.object(page, "_choose", return_value=None) as choose:
        assert not page.load_configuration(path)

    quote = '"' if sys.platform == "win32" else "'"
    assert f"pymm -c {quote}{path}{quote}" in choose.call_args.args[1]


@pytest.mark.parametrize("check", ["missing", "busy", "unsaved", "startup_busy"])
def test_refused_loads_leave_the_session_alone(
    py_cfg: Path, tmp_path: Path, qtbot: QtBot, summaries: list, check: str
) -> None:
    core = UniMMCore()
    page = _page(qtbot, core)
    assert page.load_configuration(py_cfg)
    folder = page._py_folder.path
    sys_path = list(sys.path)

    path = py_cfg
    if check == "missing":
        path = tmp_path / "missing.cfg"
    elif check in ("busy", "startup_busy"):
        page.busy_reason = lambda: "an acquisition"
    else:
        page.confirm_other_edits = lambda: False

    with patch.object(page, "_warn"):
        if check == "missing":
            with pytest.raises(FileNotFoundError):
                page.load_configuration(path)
        else:
            assert not page.load_configuration(
                path, interactive=check != "startup_busy"
            )

    assert "Shutter" in core.getLoadedDevices()
    assert page._py_folder.path == folder
    assert sys.path == sys_path


@pytest.mark.parametrize("unicore", [False, True])
def test_live_mode_stops_before_a_load(
    mmcore: CMMCorePlus, qtbot: QtBot, unicore: bool
) -> None:
    core = UniMMCore() if unicore else mmcore
    page = _page(qtbot, core)
    calls: list[str] = []
    with (
        patch.object(core, "isSequenceRunning", return_value=True),
        patch.object(
            core,
            "stopSequenceAcquisition",
            side_effect=lambda: calls.append("stop"),
        ),
        patch.object(page, "_cancel_add", side_effect=lambda: calls.append("cancel")),
        patch.object(
            core,
            "loadSystemConfiguration",
            side_effect=lambda *a, **k: calls.append("load"),
        ),
        patch.object(
            core, "unloadAllDevices", side_effect=lambda: calls.append("unload")
        ),
    ):
        page.load_configuration(TEST_CONFIG)
    assert calls[:2] == ["stop", "cancel"]


def test_a_broken_python_device_is_skipped_and_summarized(
    py_cfg: Path, qtbot: QtBot, summaries: list
) -> None:
    text = py_cfg.read_text().replace(
        "Property,Core,Initialize,1",
        "#py pyDevice,Broken,shutters,BrokenShutter\n"
        "#py pyDevice,Missing,shutters,NoSuchClass\n"
        "#py pyDevice,Typo,typo,Cls\n"
        "Property,Core,Initialize,1",
    )
    py_cfg.write_text(text + "#py Property,Shutter,NoSuchProperty,1\n")
    (py_cfg.parent / "py_devices" / "typo.py").write_text("x = None\nx.oops\n")
    core = UniMMCore()
    page = _page(qtbot, core)

    assert page.load_configuration(py_cfg)

    assert {"Camera", "Shutter"} <= set(core.getLoadedDevices())
    qtbot.waitUntil(lambda: bool(summaries))
    messages = {(p.source, p.stage): p.message() for p in summaries[0].problems}
    assert messages[("Broken", "initialize")] == "failed to initialize: port busy"
    assert messages[("Missing", "import")] == "no class `NoSuchClass` in `shutters`"
    assert messages[("Typo", "import")].endswith("(line 2)")
    assert "the device was kept" in messages[("Shutter", "property")]


def test_a_failed_setting_alone_is_summarized(
    py_cfg: Path, qtbot: QtBot, summaries: list
) -> None:
    py_cfg.write_text(py_cfg.read_text() + "#py Property,Shutter,NoSuchProperty,1\n")
    page = _page(qtbot, UniMMCore())
    page.load_configuration(py_cfg)
    qtbot.waitUntil(lambda: bool(summaries))
    (problem,) = summaries[0].problems
    assert problem.device_kept
    assert problem.setting == "`NoSuchProperty = 1`"


def test_a_standard_config_loads_on_unicore(qtbot: QtBot, summaries: list) -> None:
    core = UniMMCore()
    page = _page(qtbot, core)
    assert page.load_configuration(TEST_CONFIG)
    assert "Camera" in core.getLoadedDevices()
    qtbot.wait(10)
    assert not summaries


def test_moved_folder_falls_back_to_the_active_one(
    py_cfg: Path, devices: Path, tmp_path: Path, qtbot: QtBot, summaries: list
) -> None:
    page = _page(qtbot, UniMMCore())
    page.load_configuration(py_cfg)
    py_cfg.write_text(py_cfg.read_text().replace("py_devices", "moved_away"))

    page.load_configuration(py_cfg, interactive=False)

    assert page._py_folder.path == devices.resolve()
    qtbot.waitUntil(lambda: bool(summaries))
    assert summaries[0].notes == [
        f"{tmp_path.resolve() / 'moved_away'} not found, used {devices.resolve()}"
    ]


def test_locate_folder_retries_the_load(
    py_cfg: Path, devices: Path, qtbot: QtBot, summaries: list
) -> None:
    py_cfg.write_text(py_cfg.read_text().replace("py_devices", "moved_away"))
    core = UniMMCore()
    page = _page(qtbot, core)

    page.load_configuration(py_cfg)
    qtbot.waitUntil(lambda: bool(summaries))
    assert summaries[0].locate_folder
    assert "Shutter" not in core.getLoadedDevices()

    with patch.object(QFileDialog, "getExistingDirectory", return_value=str(devices)):
        page._locate_py_folder(read_config(py_cfg))
    assert "Shutter" in core.getLoadedDevices()

    # the next save repairs the .cfg
    with patch.object(page, "_choose", return_value="Save anyway"):
        assert page.save_to(py_cfg)
    assert read_config(py_cfg).devices_path == devices.resolve()


def test_locate_folder_does_not_change_a_busy_session(
    py_cfg: Path, devices: Path, tmp_path: Path, qtbot: QtBot
) -> None:
    page = _page(qtbot, UniMMCore())
    page.load_configuration(py_cfg)
    module = sys.modules["shutters"]
    (other := tmp_path / "other").mkdir()
    page.busy_reason = lambda: "an acquisition"

    with (
        patch.object(QFileDialog, "getExistingDirectory", return_value=str(other)),
        patch.object(page, "_warn"),
    ):
        page._locate_py_folder(read_config(py_cfg))

    assert page._py_folder.path == devices.resolve()
    assert sys.modules["shutters"] is module
    assert "Shutter" in page._core.getLoadedDevices()


# ── the folder ────────────────────────────────────────────────────


def _problems(folder: PyDeviceFolder) -> dict[str, str]:
    return {p.source: p.message() for p in folder.scan(purge=True).problems}


def test_scan_lists_device_classes(devices: Path) -> None:
    folder = PyDeviceFolder()
    folder.activate(devices)
    (devices / "other.py").write_text("from shutters import PyShutter\n")

    result = folder.scan(purge=True)

    assert not result.problems
    names = {d.adapter_name: d for d in result.devices}
    # not `_Private`, nor `PyShutter` again as imported by `other.py`
    assert set(names) == {"PyShutter", "BrokenShutter"}
    shutter = names["PyShutter"]
    assert (shutter.library, shutter.device_type.name) == ("shutters", "ShutterDevice")
    assert shutter.description == "A shutter for tests."
    assert shutter.is_py_device


def test_scan_reports_broken_files(devices: Path) -> None:
    (devices / "syntax.py").write_text("x = 1\ndef broken(:\n")
    (devices / "needs_dep.py").write_text("import not_a_real_package.sub\n")
    (devices / "raises.py").write_text("x = 1\nraise ValueError('nope')\n")
    (devices / "json.py").write_text("")
    (devices / "my-cam.py").write_text("")
    folder = PyDeviceFolder()
    folder.activate(devices)

    problems = _problems(folder)

    assert problems["syntax.py"] == "line 2: invalid syntax"
    dep = problems["needs_dep.py"]
    assert "`not_a_real_package`" in dep
    assert sys.executable in dep
    assert problems["raises.py"] == (
        "error while importing: `ValueError: nope` (line 2)"
    )
    assert "already the installed module" in problems["json.py"]
    assert "rename it to e.g. `my_cam.py`" in problems["my-cam.py"]


def test_scan_reports_invalid_device_metadata(devices: Path) -> None:
    (devices / "metadata.py").write_text(
        "from shutters import PyShutter\n"
        "class BadMetadata(PyShutter):\n"
        "    @classmethod\n"
        "    def type(cls):\n"
        "        raise ValueError('invalid device type')\n"
    )
    folder = PyDeviceFolder()
    folder.activate(devices)

    result = folder.scan(purge=True)

    assert [p.source for p in result.problems] == ["metadata.py"]
    assert "invalid device type" in result.problems[0].message()
    assert any(d.adapter_name == "PyShutter" for d in result.devices)


def test_rescan_reads_edits_with_the_same_bytecode_timestamp(devices: Path) -> None:
    helper = devices / "_helpers.py"
    helper.write_text("CLOSED = 1\n")
    folder = PyDeviceFolder()
    folder.activate(devices)
    folder.scan(purge=True)
    stamp = helper.stat()
    helper.write_text("CLOSED = 2\n")
    os.utime(helper, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))

    folder.scan(purge=True)

    assert sys.modules["shutters"].CLOSED == 2


def test_folder_reimports_only_when_no_device_uses_it(
    devices: Path, qtbot: QtBot
) -> None:
    folder = PyDeviceFolder()
    folder.activate(devices)
    folder.scan(purge=True)
    first = sys.modules["shutters"]

    # nothing loaded from it: a scan reads the edited file
    (devices / "shutters.py").write_text(SHUTTERS + "\nEDITED = True\n")
    folder.scan(purge=True)
    assert sys.modules["shutters"] is not first
    assert sys.modules["shutters"].EDITED

    # a device uses it: the module is kept, and the edit is pending a reload
    core = UniMMCore()
    core.loadDevice("Shutter", "shutters", "PyShutter")
    assert folder.devices_in_use(core) == ["Shutter"]
    in_use = sys.modules["shutters"]
    (devices / "shutters.py").write_text(SHUTTERS)
    result = folder.scan(purge=False)
    assert sys.modules["shutters"] is in_use
    assert result.notes == [
        "Changes to shutters.py apply after reloading the configuration."
    ]


def test_switching_folders_purges_helpers(devices: Path, tmp_path: Path) -> None:
    folder = PyDeviceFolder()
    folder.activate(devices)
    folder.scan(purge=True)
    assert "_helpers" in sys.modules

    other = tmp_path / "other"
    other.mkdir()
    folder.activate(other)

    assert "_helpers" not in sys.modules
    assert str(devices.resolve()) not in sys.path
    assert sys.path[-1] == str(other.resolve())


def test_folder_owns_modules_imported_through_a_symlink(
    devices: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alias = tmp_path / "alias"
    alias.symlink_to(devices, target_is_directory=True)
    monkeypatch.syspath_prepend(str(alias))
    folder = PyDeviceFolder()
    folder.activate(devices)

    result = folder.scan(purge=True)

    assert not result.problems
    assert {"shutters", "_helpers"} <= set(folder.modules())
    folder.activate(None)
    assert "_helpers" not in sys.modules


def test_switching_folders_is_refused_while_their_devices_are_loaded(
    py_cfg: Path, tmp_path: Path, qtbot: QtBot
) -> None:
    page = _page(qtbot, UniMMCore())
    page.load_configuration(py_cfg)
    (other := tmp_path / "other").mkdir()

    with patch.object(page, "_warn") as warn:
        page._load_py_folder(str(other))

    assert "Remove Shutter, or start a New configuration" in warn.call_args.args[0]
    assert page._py_folder.path != other.resolve()


def test_row_load_lists_python_devices(devices: Path, qtbot: QtBot) -> None:
    page = _page(qtbot, UniMMCore())
    page._load_py_folder(str(devices))

    rows = page._available._table
    modules = {item.text() for r in range(rows.rowCount()) if (item := rows.item(r, 0))}
    assert "shutters (Py)" in modules


def test_row_is_disabled_off_unicore(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    page = _page(qtbot, mmcore)
    assert not page._py_row.isEnabled()
    assert "pymm -u" in page._py_row.toolTip()


# ── adding and saving ─────────────────────────────────────────────


def test_add_flow_unloads_a_device_that_failed_to_initialize(
    devices: Path, qtbot: QtBot
) -> None:
    core = UniMMCore()
    page = _page(qtbot, core)
    page._load_py_folder(str(devices))
    broken = next(d for d in page._py_available if d.adapter_name == "BrokenShutter")
    page._on_available_selected(broken)

    with (
        patch.object(page, "_warn") as warn,
        patch.object(core, "deviceBusy", side_effect=AssertionError("polled")),
    ):
        page._begin_add("Broken")

    assert "port busy" in warn.call_args.args[0]
    assert "Broken" not in core.getLoadedDevices()


def test_save_round_trips_python_devices_and_their_folder(
    py_cfg: Path, devices: Path, tmp_path: Path, qtbot: QtBot
) -> None:
    core = UniMMCore()
    page = _page(qtbot, core)
    page.load_configuration(py_cfg)

    saved = tmp_path / "saved.cfg"
    assert page.save_to(saved)

    info = read_config(saved)
    assert info.devices_path == devices.resolve()
    assert info.py_devices == ("Shutter",)
    reloaded = UniMMCore()
    reloaded.loadSystemConfiguration(saved)
    assert reloaded.getShutterDevice() == "Shutter"


def test_save_as_succeeds_after_the_loaded_file_was_deleted(
    py_cfg: Path, tmp_path: Path, qtbot: QtBot
) -> None:
    page = _page(qtbot, UniMMCore())
    page.load_configuration(py_cfg)
    py_cfg.unlink()
    target = tmp_path / "saved.cfg"
    target.touch()

    assert page.save_to(target)
    assert read_config(target).py_devices == ("Shutter",)


@pytest.mark.parametrize("lost", ["device", "setting", "both"])
def test_saving_over_a_file_warns_about_what_did_not_load(
    py_cfg: Path, qtbot: QtBot, summaries: list, lost: str
) -> None:
    text = py_cfg.read_text()
    if lost in ("device", "both"):
        text += "#py pyDevice,Missing,shutters,NoSuchClass\n"
    if lost in ("setting", "both"):
        text += "#py Property,Shutter,NoSuchProperty,1\n"
    py_cfg.write_text(text)
    page = _page(qtbot, UniMMCore())
    page.load_configuration(py_cfg)

    with patch.object(page, "_choose", return_value=None) as choose:
        assert not page.save_to(py_cfg)
    message = choose.call_args.args[1]
    assert ("removes a Python device that didn't load (Missing)" in message) is (
        lost != "setting"
    )
    assert ("`NoSuchProperty = 1` on Shutter" in message) is (lost != "device")

    # once saved over, the file no longer has them
    with patch.object(page, "_choose", return_value="Save anyway"):
        assert page.save_to(py_cfg)
    with patch.object(page, "_choose") as choose:
        assert page.save_to(py_cfg)
    choose.assert_not_called()


# ── the window ────────────────────────────────────────────────────


def test_window_guards_loads(
    py_cfg: Path, qtbot: QtBot, summaries: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_mmcore_plus, "_instance", None)
    win = _launch(mm_config=str(py_cfg))
    qtbot.addWidget(win)

    assert win._hardware_busy_reason() == ""
    win._on_pixel_calibration_running(True)
    assert win._hardware_busy_reason() == "pixel calibration"
    win._on_pixel_calibration_running(False)

    with (
        patch.object(win._configurations, "is_dirty", return_value=True),
        patch.object(win._configurations, "dirty_parts", return_value=["groups"]),
        patch(
            "pymmcore_gui._main_window.QMessageBox.question",
            return_value=0,  # neither Discard nor anything else
        ),
    ):
        assert not win._hardware.load_configuration(py_cfg)


def test_cancelled_load_preserves_configuration_edits(
    mmcore: CMMCorePlus, py_cfg: Path, qtbot: QtBot
) -> None:
    win = _launch(mmcore=mmcore, mm_config=False)
    qtbot.addWidget(win)
    with (
        patch.object(win._configurations, "is_dirty", return_value=True),
        patch.object(win._configurations, "dirty_parts", return_value=["groups"]),
        patch.object(win._configurations, "discard_changes") as discard,
        patch.object(
            QMessageBox, "question", return_value=QMessageBox.StandardButton.Discard
        ),
        patch.object(win._hardware, "_choose", return_value=None),
    ):
        assert not win._hardware.load_configuration(py_cfg)
        discard.assert_not_called()


def test_a_failed_startup_setting_says_so(
    py_cfg: Path, qtbot: QtBot, summaries: list
) -> None:
    py_cfg.write_text(
        py_cfg.read_text() + "#py ConfigGroup,System,Startup,Shutter,NoSuchProperty,1\n"
    )
    page = _page(qtbot, UniMMCore())
    page.load_configuration(py_cfg)
    qtbot.waitUntil(lambda: bool(summaries))
    (problem,) = summaries[0].problems
    assert problem.message().startswith(
        "System/Startup setting `NoSuchProperty = 1` failed"
    )
