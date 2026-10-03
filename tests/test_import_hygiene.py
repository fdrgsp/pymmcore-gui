"""Qt-free modules must stay cheap to import (see SMART_MICROSCOPY_PLAN.md, G8).

User analysis scripts import ``pymmcore_gui.smart`` inside a spawned worker
process; if that dragged in Qt and the whole window, every process-mode run
would pay for it at startup and carry the GUI's libraries for nothing.
Checked in a fresh interpreter, since this test session has Qt loaded already.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

HEAVY = ("PyQt6", "PySide6", "qtpy", "pymmcore_widgets", "ndv", "vispy")

QT_FREE_MODULES = [
    "pymmcore_gui",
    "pymmcore_gui.smart",
    "pymmcore_gui._smart._worker",
    "pymmcore_gui._smart._executors",
    "pymmcore_gui._smart._scheduler",
    "pymmcore_gui._smart._loader",
    "pymmcore_gui._smart._log",
]


@pytest.mark.parametrize("module", QT_FREE_MODULES)
def test_module_imports_without_qt(module: str) -> None:
    code = (
        "import sys, importlib\n"
        f"importlib.import_module({module!r})\n"
        f"heavy = sorted({{m.split('.')[0] for m in sys.modules}} & set({HEAVY!r}))\n"
        "print(','.join(heavy))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "", f"{module} imported: {result.stdout.strip()}"


def test_lazy_exports_still_resolve() -> None:
    import pymmcore_gui

    assert pymmcore_gui.MicroManagerGUI.__name__ == "MicroManagerGUI"
    assert callable(pymmcore_gui.create_mmgui)
    assert {"MicroManagerGUI", "create_mmgui", "CoreAction"} <= set(dir(pymmcore_gui))
    with pytest.raises(AttributeError):
        _ = pymmcore_gui.not_a_thing
