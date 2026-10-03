# pyright: reportArgumentType=false
# (useq models are built from plain values that pydantic coerces)
from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
import useq

from pymmcore_gui import MicroManagerGUI
from pymmcore_gui._qt.QtWidgets import QApplication, QMessageBox
from pymmcore_gui._run_owner import RunOwner
from pymmcore_gui.widgets._smart import SmartMicroscopyPage
from pymmcore_gui.widgets._smart._page import TEMPLATES_DIR, list_templates

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot

    from pymmcore_gui._settings import Settings

BASE = useq.MDASequence(
    channels=[useq.Channel(config="DAPI", group="Channel", exposure=None)],
    time_plan=useq.TIntervalLoops(interval=0, loops=3),
)


@pytest.fixture
def page(qtbot: QtBot, mmcore: CMMCorePlus) -> Iterator[SmartMicroscopyPage]:
    page = SmartMicroscopyPage(mmcore)
    qtbot.addWidget(page)
    page.resize(1600, 900)
    page.show()
    yield page
    page.shutdown()


@pytest.fixture
def script(tmp_path: Path) -> Path:
    path = tmp_path / "minimal.py"
    shutil.copyfile(TEMPLATES_DIR / "minimal.py", path)
    return path


@pytest.fixture(autouse=True)
def _no_dialogs() -> Iterator[list[tuple[str, str]]]:
    """Record message boxes instead of blocking on them."""
    shown: list[tuple[str, str]] = []

    def _record(kind: str) -> Any:
        def _show(_parent: object, title: str, text: str, *_a: object) -> int:
            shown.append((kind, f"{title}: {text}"))
            return 0

        return _show

    with (
        patch.object(QMessageBox, "warning", _record("warning")),
        patch.object(QMessageBox, "critical", _record("critical")),
        patch.object(QMessageBox, "information", _record("information")),
        patch.object(
            QMessageBox, "exec", lambda self: shown.append(("exec", self.text()))
        ),
    ):
        yield shown


def _run(qtbot: QtBot, page: SmartMicroscopyPage) -> dict[str, Any]:
    page.mda.setValue(BASE)
    with qtbot.waitSignal(page.controller.runFinished, timeout=30_000) as blocker:
        page.mda.run_mda()
    assert blocker.args is not None
    summary: dict[str, Any] = blocker.args[0]
    return summary


def test_templates_menu_lists_every_template() -> None:
    names = [name for name, _ in list_templates()]
    assert len(names) == len(list(TEMPLATES_DIR.glob("*.py")))
    assert "Adaptive exposure" in names


def test_load_script_builds_parameter_form(
    page: SmartMicroscopyPage, tmp_path: Path, settings: Settings
) -> None:
    path = tmp_path / "adaptive.py"
    shutil.copyfile(TEMPLATES_DIR / "adaptive_exposure.py", path)
    assert page.load_script(path)
    form = page.script_panel.params
    assert set(form.values()) == {
        "target_mean",
        "min_exposure_ms",
        "max_exposure_ms",
        "n_frames",
    }
    form.editor("n_frames").setValue(7)  # type: ignore[attr-defined]
    assert page.script_panel.run_config().params["n_frames"] == 7
    # remembered per script, and restored when it is loaded again
    remembered = settings.smart_microscopy.settings_for(path)
    assert remembered["params"]["n_frames"] == 7
    assert settings.smart_microscopy.last_script == path.resolve()


def test_invalid_script_shows_error_and_refuses_to_run(
    page: SmartMicroscopyPage,
    qtbot: QtBot,
    tmp_path: Path,
    mmcore: CMMCorePlus,
    _no_dialogs: list[tuple[str, str]],
) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("API_VERSION = 1\ndef analyze(image):\n    pass\n")
    assert not page.load_script(bad)
    assert page.script_panel.spec is None
    assert "exactly 3" in page.script_panel._error.text()
    assert not page._test_btn.isEnabled()

    page.mda.setValue(BASE)
    page.mda.run_mda()
    QApplication.processEvents()
    assert not mmcore.mda.is_running()
    assert any("No valid script" in text for _, text in _no_dialogs)


def test_run_from_editor_opens_viewer_on_this_page(
    page: SmartMicroscopyPage, qtbot: QtBot, script: Path
) -> None:
    assert page.load_script(script)
    assert page._viewer_stack.currentWidget() is page._viewer_placeholder

    summary = _run(qtbot, page)

    assert summary["status"] == "completed"
    assert summary["frames"] == 3
    assert len(page.viewers._records) == 1
    assert page._viewer_stack.currentWidget() is page._viewer_dock_manager
    (record,) = page.viewers._records.values()
    viewer: Any = record.viewer
    assert viewer.source_title.startswith("Smart Minimal")
    # "Re-use MDA..." offers the run's *base* sequence
    assert [c.config for c in viewer.mda_sequence.channels] == ["DAPI"]
    assert viewer.mda_sequence.time_plan == BASE.time_plan
    assert page.monitor.model.rowCount() == 3
    assert page.monitor.run_dir is not None
    assert (page.monitor.run_dir / "frames.jsonl").exists()
    assert page.run_ownership.owner is None  # released after the run


def test_settings_locked_during_run(
    page: SmartMicroscopyPage, qtbot: QtBot, tmp_path: Path
) -> None:
    slow = tmp_path / "slow.py"
    slow.write_text(
        "import time\nAPI_VERSION = 1\n"
        "def analyze(image, frame, ctx):\n    time.sleep(0.2)\n"
    )
    assert page.load_script(slow)
    page.mda.setValue(BASE)
    with qtbot.waitSignal(page.mdaRunningChanged, timeout=10_000) as blocker:
        page.mda.run_mda()
    assert blocker.args == [True]
    assert not page._load_btn.isEnabled()
    assert not page.script_panel._settings_boxes[0].isEnabled()
    assert page._stop_btn.isEnabled()
    with qtbot.waitSignal(page.controller.runFinished, timeout=15_000):
        page._stop_btn.click()
    qtbot.waitUntil(lambda: page._load_btn.isEnabled(), timeout=5000)
    assert not page._stop_btn.isEnabled()


def test_script_modified_on_disk_is_flagged(
    page: SmartMicroscopyPage, qtbot: QtBot, script: Path
) -> None:
    assert page.load_script(script)
    script.write_text(script.read_text() + "\n# edited\n")
    qtbot.waitUntil(lambda: page.script_panel._modified, timeout=5000)
    assert "Modified" in page.script_panel._status_text.text()
    assert page.reload_script()
    assert not page.script_panel._modified


def test_test_on_last_image_acquires_nothing(
    page: SmartMicroscopyPage,
    qtbot: QtBot,
    tmp_path: Path,
    mmcore: CMMCorePlus,
    _no_dialogs: list[tuple[str, str]],
) -> None:
    path = tmp_path / "zstack.py"
    shutil.copyfile(TEMPLATES_DIR / "detect_and_zstack.py", path)
    assert page.load_script(path)
    page.script_panel.params.editor("threshold").setValue(0.0)  # type: ignore[attr-defined]

    page.test_on_last_image()
    assert any("No image yet" in text for _, text in _no_dialogs)

    mmcore.snapImage()
    with qtbot.waitSignal(page._tested, timeout=30_000):
        page.test_on_last_image()
    text = _no_dialogs[-1][1]
    assert "requested" in text and "event(s)" in text
    assert "hit=True" in text
    assert not mmcore.mda.is_running()


def test_last_script_restored_on_startup(
    qtbot: QtBot, mmcore: CMMCorePlus, script: Path, settings: Settings
) -> None:
    settings.smart_microscopy.remember_script(script, {"execution": "process"})
    page = SmartMicroscopyPage(mmcore)
    qtbot.addWidget(page)
    assert page.script_panel.spec is not None
    assert page.script_panel.path == script.resolve()
    assert page.script_panel.run_config().execution == "process"


# --------------------------------------------------------- main window


@pytest.fixture
def gui(qtbot: QtBot, mmcore: CMMCorePlus) -> Iterator[MicroManagerGUI]:
    gui = MicroManagerGUI(mmcore=mmcore)
    qtbot.addWidget(gui)
    yield gui


def _tab(gui: MicroManagerGUI, page: object) -> Any:
    return gui._mode_tabs._tabs[gui._stack.indexOf(page)]  # type: ignore[arg-type]


def test_window_has_smart_tab(gui: MicroManagerGUI) -> None:
    assert gui.TAB_LABELS[-1] == "Smart Microscopy"
    assert gui._stack.indexOf(gui.smart) == len(gui.TAB_LABELS) - 1


def test_smart_run_keeps_window_on_smart_page(
    gui: MicroManagerGUI, qtbot: QtBot, tmp_path: Path, script: Path
) -> None:
    slow = tmp_path / "slow.py"
    slow.write_text(
        "import time\nAPI_VERSION = 1\n"
        "def analyze(image, frame, ctx):\n    time.sleep(0.1)\n"
    )
    page = gui.smart
    assert page.load_script(slow)
    gui._mode_tabs._select(gui._stack.indexOf(page))
    page.mda.setValue(BASE)
    with qtbot.waitSignal(page.mdaRunningChanged, timeout=10_000):
        page.mda.run_mda()
    QApplication.processEvents()
    assert gui._run_ownership.owner is RunOwner.SMART
    assert gui._stack.currentWidget() is page
    assert _tab(gui, page).isEnabled()
    for other in (gui._installation, gui._hardware, gui._configurations, gui.acquire):
        assert not _tab(gui, other).isEnabled()

    with qtbot.waitSignal(page.controller.runFinished, timeout=15_000):
        pass
    qtbot.waitUntil(lambda: _tab(gui, gui.acquire).isEnabled(), timeout=5000)
    assert gui._stack.currentWidget() is page
    # the run's viewer opened on the Smart page, not on Acquire
    assert len(page.viewers._records) == 1
    assert not gui.acquire.viewers._records


def test_console_run_still_switches_to_acquire(
    gui: MicroManagerGUI, qtbot: QtBot
) -> None:
    gui._mode_tabs._select(gui._stack.indexOf(gui.smart))
    gui._stack.setCurrentWidget(gui.smart)
    slow = BASE.replace(time_plan=useq.TIntervalLoops(interval=0.5, loops=3))
    with qtbot.waitSignal(gui.acquire.mdaRunningChanged, timeout=10_000):
        gui._mmc.run_mda(slow, output="memory")
    QApplication.processEvents()
    assert gui._stack.currentWidget() is gui.acquire
    assert not _tab(gui, gui.smart).isEnabled()
    qtbot.waitUntil(lambda: not gui._mmc.mda.is_running(), timeout=10_000)
    qtbot.waitUntil(lambda: _tab(gui, gui.smart).isEnabled(), timeout=5000)
    assert len(gui.acquire.viewers._records) == 1
    assert not gui.smart.viewers._records
