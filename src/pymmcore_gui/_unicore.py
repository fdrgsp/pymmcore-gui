"""Running on pymmcore-plus's `UniMMCore`, which can also load Python devices.

A `.cfg` declares Python devices on `#py ` lines, which the standard core reads
as comments, so it loads such a file without them. The core can't be swapped
under an open window (panels and viewers bind to it as they're built), so it is
chosen at startup: `pymm --unicore`, or a startup config with `#py` lines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

    from pymmcore_plus import CMMCorePlus

# what UniMMCore's .cfg parser looks for (its own constant is private)
_PY_PREFIX = "#py "


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
    if core is None:
        return False
    from pymmcore_plus.experimental.unicore import UniMMCore

    return isinstance(core, UniMMCore)


def startup_core(
    unicore: bool, mm_config: Path | str | Literal[False] | None
) -> CMMCorePlus | None:
    """Return a new `UniMMCore` if the launch needs one, else None.

    None leaves the window on `CMMCorePlus.instance()`. An existing global core
    is kept if it already is a UniMMCore, and also if only the config file asks
    for one: a core the caller set up isn't replaced behind their back.
    """
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
