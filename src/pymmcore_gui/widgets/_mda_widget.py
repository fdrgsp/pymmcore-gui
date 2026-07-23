"""Shared MDA editor configured for the ome-writers sink."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_widgets import MDAWidget

if TYPE_CHECKING:
    from pathlib import Path

    from pymmcore_plus import CMMCorePlus

    from pymmcore_gui._qt.QtWidgets import QWidget


class MemoryMDAWidget(MDAWidget):
    """Christina-style MDA editor with a viewable memory-sink fallback."""

    def __init__(
        self, mmcore: CMMCorePlus, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent=parent, mmcore=mmcore)
        combo = self.save_info._writer_combo
        for idx in range(combo.count()):
            if combo.itemText(idx) == "tiff-sequence":
                combo.removeItem(idx)
                break
        combo.setCurrentText("ome-tiff")

    def prepare_mda(self) -> bool | str | Path | None:
        """Return a disk path or a scratch sink that supports live viewing."""
        output = super().prepare_mda()
        return "memory" if output is None else output
