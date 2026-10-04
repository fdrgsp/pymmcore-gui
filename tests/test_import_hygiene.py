"""Qt-free modules must stay cheap to import.

Anything spawned from this application (e.g. a smart-microscopy analysis
process) may import ``pymmcore_gui`` submodules; if that dragged in Qt and the
whole window, every such process would pay for it at startup.
Checked in a fresh interpreter, since this test session has Qt loaded already.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

HEAVY = ("PyQt6", "PySide6", "qtpy", "pymmcore_widgets", "ndv", "vispy")

# The smart-microscopy engine (pymmcore_plus.smart) checks its own imports.
QT_FREE_MODULES = ["pymmcore_gui"]


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
    assert {"MicroManagerGUI", "create_mmgui"} <= set(dir(pymmcore_gui))
    with pytest.raises(AttributeError):
        _ = pymmcore_gui.not_a_thing
