from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_widgets import StageExplorer
from pymmcore_widgets.control._rois.roi_manager import GRAY
from qtpy.QtGui import QAction
from superqt import QIconifyIcon

from pymmcore_gui._qt.QtCore import Signal

if TYPE_CHECKING:
    import useq
    from pymmcore_plus import CMMCorePlus
    from qtpy.QtWidgets import QWidget


class MMStageExplorer(StageExplorer):
    """StageExplorer that emits a signal when rois_to_useq_positions is called.

    Used to populate the MDAWidget with a list of positions with the GridFromPolygon
    as subsequence.
    """

    rois_to_positions = Signal(object)

    def __init__(
        self, parent: QWidget | None = None, mmcore: CMMCorePlus | None = None
    ) -> None:
        super().__init__(parent=parent, mmcore=mmcore)

        self.to_mda = QAction(
            QIconifyIcon("mdi:plus", color=GRAY), "Add ROIs to MDA", self
        )
        self.to_mda.triggered.connect(self.rois_to_useq_positions)
        self._toolbar.insertAction(self._toolbar.actions()[-2], self.to_mda)

    def rois_to_useq_positions(self) -> list[useq.AbsolutePosition] | None:
        """Convert ROIs to useq positions and emit the `rois_to_positions` signal."""
        positions = super().rois_to_useq_positions()
        self.rois_to_positions.emit(positions)
        return positions
