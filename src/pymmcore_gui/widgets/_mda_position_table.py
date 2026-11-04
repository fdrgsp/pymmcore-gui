"""The MDA position table with feedback during HCS imports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_widgets.mda._core_tabs import _CoreConnectedPositionTable

from pymmcore_gui.widgets._busy import BusyOverlay, busy

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from useq import WellPlatePlan

    from pymmcore_gui._qt.QtWidgets import QWidget


class MDAPositionTable(_CoreConnectedPositionTable):
    def __init__(
        self,
        rows: int = 0,
        mmcore: CMMCorePlus | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(rows, mmcore, parent)
        self._import_overlay = BusyOverlay(self)

    def _update_table_positions(self, plan: WellPlatePlan) -> None:
        with busy(self._import_overlay, "Creating position list…"):
            super()._update_table_positions(plan)
