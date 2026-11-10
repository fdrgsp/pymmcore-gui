"""Running on pymmcore-plus's `UniMMCore`, which can also load Python devices.

A `.cfg` declares Python devices on `#py ` lines, which the standard core reads
as comments, so it loads such a file without them. The core can't be swapped
under an open window (panels and viewers bind to it as they're built), so it is
chosen at startup: `pymm --unicore`, or a startup config with `#py` lines.

Python device adapters are `.py` files in pyMM's `devices` folder. It is put on
the import path, so `#py pyDevice,<label>,<file stem>,<class>` loads them.

Source runs only: the standalone app can't import packages it doesn't bundle,
which most hardware adapters need.
"""

from __future__ import annotations

import ast
import importlib.util
import keyword
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from pymmcore_plus import DeviceType

from pymmcore_gui._settings import USER_DATA_DIR

if TYPE_CHECKING:
    from typing import Literal

    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.model import AvailableDevice

# what UniMMCore's .cfg parser looks for (its own constant is private)
_PY_PREFIX = "#py "
_UNICORE_MODULE = "pymmcore_plus.experimental.unicore"
PY_DEVICES_DIR = USER_DATA_DIR / "devices"

# `pymmcore_plus.experimental.unicore` base classes -> the device type they make
_DEVICE_BASES: dict[str, DeviceType] = {
    "HubDevice": DeviceType.Hub,
    "CameraDevice": DeviceType.Camera,
    "SimpleCameraDevice": DeviceType.Camera,
    "ShutterDevice": DeviceType.Shutter,
    "StageDevice": DeviceType.Stage,
    "XYStageDevice": DeviceType.XYStage,
    "XYStepperStageDevice": DeviceType.XYStage,
    "StateDevice": DeviceType.State,
    "SLMDevice": DeviceType.SLM,
    "GenericDevice": DeviceType.Generic,
}


def unicore_supported() -> bool:
    """Return whether this build can run on UniMMCore (not the standalone app)."""
    return not getattr(sys, "frozen", False)


def config_needs_unicore(path: Path | str) -> bool:
    """Return whether the `.cfg` at `path` declares Python devices.

    A missing or unreadable file returns False: loading it reports the error.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return any(line.strip().startswith(_PY_PREFIX) for line in f)
    except OSError:
        return False


def is_unicore(core: CMMCorePlus | None) -> bool:
    """Return whether `core` is a `UniMMCore`."""
    # never imported means no UniMMCore exists, so don't import it to check
    unicore = sys.modules.get(_UNICORE_MODULE)
    return unicore is not None and isinstance(core, unicore.UniMMCore)


def startup_core(
    unicore: bool, mm_config: Path | str | Literal[False] | None
) -> CMMCorePlus | None:
    """Return a new `UniMMCore` if the launch needs one, else None.

    None leaves the window on `CMMCorePlus.instance()`. An existing global core
    is kept if it already is a UniMMCore, and also if only the config file asks
    for one: a core the caller set up isn't replaced behind their back.
    """
    if not unicore_supported():
        return None
    from pymmcore_plus.core import _mmcore_plus

    ref = _mmcore_plus._instance
    current = ref() if ref is not None else None
    needed = unicore or (
        current is None and mm_config and config_needs_unicore(mm_config)
    )
    if not needed or is_unicore(current):
        return None
    from pymmcore_plus.experimental.unicore import UniMMCore

    return UniMMCore()


def py_devices_dir() -> Path:
    """Return the Python device adapters folder, creating it and making it importable.

    `MMGUI_PY_DEVICES_DIR` overrides the location (read on every call, for tests).
    It is appended to `sys.path`, so an adapter file can't shadow a real module.
    """
    directory = Path(os.environ.get("MMGUI_PY_DEVICES_DIR") or PY_DEVICES_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    if str(directory) not in sys.path:
        sys.path.append(str(directory))
    return directory


class PyDeviceScan(NamedTuple):
    """What `scan_py_devices` found: devices, and files it couldn't use."""

    devices: list[AvailableDevice]
    problems: list[tuple[str, str]]  # (file name, what's wrong with it)


def scan_py_devices() -> PyDeviceScan:
    """Return the devices defined in the `.py` files of `py_devices_dir()`.

    Files are parsed, never run, so a broken or slow one can't hurt the GUI.
    A class counts if it subclasses a UniMMCore device base, directly or via
    another class in the same file. `_`-prefixed files and classes are skipped
    (helpers). A file that can't be parsed, or that a `.cfg` couldn't import
    under its own name, is reported in `problems` instead.
    """
    from pymmcore_plus.model import AvailableDevice

    importlib.invalidate_caches()  # see files added since the last import
    result = PyDeviceScan([], [])
    for path in sorted(py_devices_dir().glob("*.py")):
        module = path.stem
        if module.startswith("_"):
            continue
        if problem := _import_problem(module, path):
            result.problems.append((path.name, problem))
            continue
        try:
            tree = ast.parse(path.read_bytes(), filename=str(path))
        except SyntaxError as e:
            result.problems.append((path.name, f"line {e.lineno}: {e.msg}"))
            continue
        except (ValueError, OSError) as e:
            result.problems.append((path.name, str(e)))
            continue

        bases = dict(_DEVICE_BASES)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            kind = next(
                (bases[n] for b in node.bases if (n := _base_name(b)) in bases), None
            )
            if kind is None:
                continue
            bases[node.name] = kind  # subclasses of it count too
            if node.name.startswith("_"):
                continue
            doc = ast.get_docstring(node) or ""
            result.devices.append(
                AvailableDevice(
                    library=module,
                    adapter_name=node.name,
                    description=doc.strip().split("\n", 1)[0],
                    device_type=kind,
                    is_py_device=True,
                )
            )
    return result


def _base_name(node: ast.expr) -> str:
    """Return the class name a base expression ends in (`Foo` or `mod.Foo`)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _import_problem(module: str, path: Path) -> str:
    """Return why `import <module>` wouldn't load the file at `path`, or ''."""
    if not module.isidentifier():
        fixed = re.sub(r"\W+", "_", module)
        fixed = f"devices_{fixed}" if fixed[0].isdigit() else fixed
        return (
            "not a valid module name: use only letters, digits and _, not "
            f"starting with a digit (e.g. {fixed}.py)"
        )
    if keyword.iskeyword(module):
        return f"{module!r} is a Python keyword, so it can't be imported: rename it"
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError) as e:
        return f"can't be imported as {module!r}: {e}"
    origin = spec.origin if spec else None
    if origin is None:
        return f"can't be imported as {module!r}"
    if Path(origin).resolve() != path.resolve():
        return f"clashes with the module {module!r} at {origin}: rename it"
    return ""
