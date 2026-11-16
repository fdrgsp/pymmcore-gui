"""Startup core selection, configuration metadata, and Python device discovery."""

from __future__ import annotations

import functools
import importlib
import importlib.util
import inspect
import keyword
import os
import re
import sys
import traceback
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Literal

    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.experimental.unicore import PyDeviceLoadError, UniMMCore
    from pymmcore_plus.model import AvailableDevice

PY_PREFIX = "#py "
DEVICES_PATH_KEY = "#@PyDevicesPath"
_UNICORE_MODULE = "pymmcore_plus.experimental.unicore.core._unicore"
_SETTING_STAGES = ("property", "startup")


def unicore_supported() -> bool:
    """Return whether Python devices can be used: only when run from source."""
    return not getattr(sys, "frozen", False)


def is_unicore(core: CMMCorePlus | None) -> bool:
    """Return whether `core` is a `UniMMCore`, without importing it."""
    module = sys.modules.get(_UNICORE_MODULE)
    return module is not None and isinstance(core, module.UniMMCore)


@dataclass(frozen=True)
class ConfigInfo:
    """What a `.cfg` says about Python devices."""

    path: Path
    has_py_lines: bool = False
    py_devices: tuple[str, ...] = ()  # labels on `#py pyDevice` lines
    devices_path: Path | None = None  # its `#@PyDevicesPath`, made absolute


def read_config(path: Path | str) -> ConfigInfo:
    """Read the Python-device lines of the `.cfg` at `path` (raises `OSError`).

    A relative `#@PyDevicesPath` resolves against the `.cfg`'s folder, so a
    config and its device folder can move together.
    """
    path = Path(path)
    has_py, labels, folder = False, [], None
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith(PY_PREFIX):
                has_py = True
                cmd, _, rest = line[len(PY_PREFIX) :].lstrip().partition(",")
                if cmd == "pyDevice":
                    labels.append(rest.split(",", 1)[0])
            elif line.startswith(f"{DEVICES_PATH_KEY},"):
                if value := line.split(",", 1)[1].strip():
                    folder = (path.parent / Path(value).expanduser()).resolve()
    return ConfigInfo(path, has_py, tuple(labels), folder)


def config_needs_unicore(path: Path | str) -> bool:
    """Return whether the `.cfg` at `path` declares Python devices.

    A missing or unreadable file returns False: loading it reports the error.
    """
    try:
        return read_config(path).has_py_lines
    except OSError:
        return False


def write_devices_path(cfg: Path | str, folder: Path | None) -> None:
    """Record `folder` on the `.cfg`'s first line (or drop the line if None)."""
    cfg = Path(cfg)
    lines = [
        line
        for line in cfg.read_text().splitlines()
        if not line.strip().startswith(f"{DEVICES_PATH_KEY},")
    ]
    if folder is not None:
        lines.insert(0, f"{DEVICES_PATH_KEY},{folder.resolve()}")
    cfg.write_text("\n".join(lines) + "\n")


def startup_core(
    unicore: bool, mm_config: Path | str | Literal[False] | None
) -> CMMCorePlus | None:
    """Return a new `UniMMCore` if the launch needs one, else None.

    None leaves the window on `CMMCorePlus.instance()`. An existing global core
    is kept if it already is a UniMMCore, and also if only the config file asks
    for one: a core the caller set up isn't replaced behind their back. The
    standalone app never runs on UniMMCore.
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


@dataclass(frozen=True)
class PyDeviceProblem:
    """A device file, or a Python device in a `.cfg`, that couldn't be used.

    `stage` is "name" (bad file name) or "import" for a scanned file, else the
    stage of the loader's `PyDeviceLoadError`.
    """

    source: str  # the file name (scan) or the device label (load)
    stage: str
    exc: BaseException
    module: str = ""
    cls_name: str = ""
    line: str = ""  # the .cfg line, without its `#py ` prefix

    @classmethod
    def from_load_error(cls, err: PyDeviceLoadError) -> PyDeviceProblem:
        module = cls_name = ""
        cmd, *args = err.line.split(",")
        if cmd == "pyDevice" and len(args) == 3:
            _, module, cls_name = args
        return cls(err.label, err.stage, err.exc, module, cls_name, err.line)

    @property
    def device_kept(self) -> bool:
        """Whether only a setting failed, on a device that loaded."""
        return self.stage in _SETTING_STAGES

    def message(self) -> str:
        """Explain the problem in a sentence, without naming its `source`."""
        e, stage = self.exc, self.stage
        if stage == "name":
            return str(e)
        if stage == "import":
            return self._import_message()
        if stage == "construct":
            return f"could not create `{self.cls_name}`: {_text(e)}"
        if stage == "initialize":
            return f"failed to initialize: {_text(e)}"
        if stage == "wait":
            return f"stopped responding while loading: {_text(e)}"
        if stage == "unload":
            return (
                f"its shutdown also failed: {_text(e)}; hardware resources may "
                "still be held. If its port or camera stays busy, restart pyMM "
                "before loading again."
            )
        what = "System/Startup setting " if stage == "startup" else ""
        return f"{what}{self.setting} failed: {_text(e)}; the device was kept"

    @property
    def setting(self) -> str:
        """The failed setting, e.g. "`Power = 10`", for a kept device."""
        cmd, *args = self.line.split(",")
        if cmd == "Property" and len(args) == 3:
            return f"`{args[1]} = {args[2]}`"
        return f"`{self.line}`"

    def details(self) -> str:
        """The full traceback."""
        return "".join(traceback.format_exception(self.exc))

    def _import_message(self) -> str:
        e = self.exc
        if isinstance(e, SyntaxError):
            return f"line {e.lineno}: {e.msg}"
        if isinstance(e, ModuleNotFoundError) and e.name:
            if e.name == self.module:
                return f"no module `{e.name}`: set the Python devices folder"
            return missing_dependency_message(e.name)
        error = f"error while importing: `{type(e).__name__}: {e}`"
        if line := self._error_line():  # the module's own code raised
            return f"{error} (line {line})"
        if isinstance(e, AttributeError) and self.cls_name:
            return f"no class `{self.cls_name}` in `{self.module}`"
        if isinstance(e, TypeError) and self.cls_name:
            return f"`{self.cls_name}` is not a UniMMCore device"
        return error

    def _error_line(self) -> int | None:
        """The line of the module's own file where the import raised."""
        frames = traceback.extract_tb(self.exc.__traceback__)
        name = f"{self.module.rsplit('.', 1)[-1]}.py"
        return next(
            (f.lineno for f in reversed(frames) if Path(f.filename).name == name),
            None,
        )


def missing_dependency_message(module: str) -> str:
    """Say which package to install, and into which Python environment."""
    package = module.split(".")[0]
    return (
        f"needs the Python package `{package}`, which isn't installed in the "
        f"environment pyMM is running in ({sys.executable}). Install it there, "
        "then press Load again (or restart pyMM). The name to install can differ "
        "from the import name (e.g. `cv2` → `opencv-python`)."
    )


def _text(e: BaseException) -> str:
    return str(e) or type(e).__name__


@dataclass
class ScanResult:
    """The device classes found in a folder, and the files that failed."""

    devices: list[AvailableDevice] = field(default_factory=list)
    problems: list[PyDeviceProblem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class PyDeviceFolder:
    """The session's one folder of Python device files.

    The folder is appended to `sys.path` (so a file can't shadow an installed
    module), and its modules are those whose file lies inside it: device files,
    helpers they import, and anything imported later.
    """

    def __init__(self) -> None:
        self.path: Path | None = None
        self._sys_path_entry: str | None = None  # only if we added it
        self._mtimes: dict[str, float] = {}  # module -> file mtime at import

    def modules(self) -> list[str]:
        """Names of the imported modules whose file is inside the folder."""
        if self.path is None:
            return []
        prefix = f"{self.path}{os.sep}"
        return [
            name
            for name, module in list(sys.modules.items())
            if (origin := _origin(module)) and origin.startswith(prefix)
        ]

    def devices_in_use(self, core: CMMCorePlus) -> list[str]:
        """Labels of loaded devices whose class comes from the folder."""
        if self.path is None or not is_unicore(core):
            return []
        uni, modules = cast("UniMMCore", core), set(self.modules())
        return [
            label
            for label in uni.getLoadedDevices()
            if uni.isPyDevice(label) and uni.getDeviceLibrary(label) in modules
        ]

    def folder_for(self, info: ConfigInfo) -> tuple[Path | None, str]:
        """Return the folder to load `info` with, and a note on any fallback.

        The `.cfg`'s path wins if it exists; without one the active folder is
        kept; if it no longer exists, the active folder stands in for it.
        """
        saved = info.devices_path
        if saved is None:
            return self.path, ""
        if saved.is_dir():
            return saved, ""
        if self.path is not None:
            return self.path, f"{saved} not found, used {self.path}"
        return None, f"{saved} not found"

    def activate(self, path: Path | None) -> None:
        """Make `path` the folder (None: no folder), forgetting the old one.

        Only call it while no loaded device comes from the current folder.
        """
        self.purge()
        if (old := self._sys_path_entry) and old in sys.path:
            sys.path.remove(old)
        self._sys_path_entry = None
        self.path = path.resolve() if path is not None else None
        if self.path is not None and (entry := str(self.path)) not in sys.path:
            sys.path.append(entry)
            self._sys_path_entry = entry
        importlib.invalidate_caches()

    def purge(self) -> None:
        """Drop the folder's modules, so the next import reads the files again."""
        for name in self.modules():
            module = sys.modules.pop(name)
            # Timestamp-based .pyc files can hide same-size edits within a second.
            if cached := getattr(module, "__cached__", None):
                with suppress(OSError):
                    Path(cached).unlink(missing_ok=True)
        self._mtimes.clear()

    def scan(self, *, purge: bool) -> ScanResult:
        """Import each device file and list the device classes it defines.

        A device file is a top-level `*.py` not starting with `_`. Without
        `purge`, modules already imported are reused, and edits to them are
        reported as applying after a reload.
        """
        result = ScanResult()
        if self.path is None:
            return result
        if purge:
            self.purge()
        importlib.invalidate_caches()
        from pymmcore_plus.experimental.unicore import Device

        for file in sorted(self.path.glob("*.py")):
            if file.name.startswith("_"):
                continue
            if problem := self._name_problem(file):
                result.problems.append(problem)
                continue
            name, mtime = file.stem, file.stat().st_mtime
            if self._mtimes.setdefault(name, mtime) != mtime and name in sys.modules:
                result.notes.append(
                    f"Changes to {file.name} apply after reloading the configuration."
                )
            try:
                module = importlib.import_module(name)
                devices = _device_classes(module, Device)
            except Exception as e:
                self._mtimes.pop(name, None)
                result.problems.append(PyDeviceProblem(file.name, "import", e, name))
                continue
            result.devices.extend(devices)
        return result

    def _name_problem(self, file: Path) -> PyDeviceProblem | None:
        """Report a file name that can't be imported as its own module."""
        name = file.stem
        if not name.isidentifier() or keyword.iskeyword(name):
            fixed = re.sub(r"\W", "_", name).lower()
            if not fixed.isidentifier() or keyword.iskeyword(fixed):
                fixed = f"device_{fixed}"
            msg = f"not a valid module name: rename it to e.g. `{fixed}.py`"
            return PyDeviceProblem(file.name, "name", ValueError(msg), name)
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        origin = (spec.origin or "") if spec else ""
        if spec is not None and (
            not origin or Path(origin).resolve() != file.resolve()
        ):
            msg = f"`{name}` is already the installed module at {origin}: rename it"
            return PyDeviceProblem(file.name, "name", ValueError(msg), name)
        return None


def _device_classes(module: ModuleType, base: type) -> list[AvailableDevice]:
    """The public, concrete `base` subclasses defined in `module`."""
    from pymmcore_plus.model import AvailableDevice

    found = []
    for name, cls in vars(module).items():
        if (
            isinstance(cls, type)
            and issubclass(cls, base)
            and cls.__module__ == module.__name__
            and name == cls.__name__
            and not name.startswith("_")
            and not inspect.isabstract(cls)
        ):
            doc = (cls.__dict__.get("__doc__") or "").strip()
            found.append(
                AvailableDevice(
                    library=module.__name__,
                    adapter_name=name,
                    description=doc.splitlines()[0] if doc else "",
                    device_type=cls.type(),  # type: ignore[attr-defined]
                    is_py_device=True,
                )
            )
    return found


def _origin(module: ModuleType | None) -> str | None:
    """The module's file with its folder's links resolved, if it has one."""
    origin: str | None = getattr(getattr(module, "__spec__", None), "origin", None)
    if not origin or not os.path.isabs(origin):
        return None
    folder, name = os.path.split(origin)
    return os.path.join(_real_folder(folder), name)


@functools.cache
def _real_folder(folder: str) -> str:
    # modules share few folders: resolving each once keeps `modules()` cheap
    return os.path.realpath(folder)
