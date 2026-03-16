from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

    from pymmcore_gui._qt.QtCore import QObject


def current_core(obj: QObject) -> CMMCorePlus | None:
    """Walk the Qt parent chain to find the nearest CMMCorePlus instance."""
    while obj is not None:
        if (core := getattr(obj, "mmcore", None)) is not None:
            return core
        obj = obj.parent()  # type: ignore[assignment]
    return None
