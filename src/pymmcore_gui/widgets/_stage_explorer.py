from __future__ import annotations

from signal import Sigmasks
from typing import TYPE_CHECKING

import useq
from pymmcore_widgets import StageExplorer
from pymmcore_widgets.control._rois.roi_manager import GRAY
from qtpy.QtGui import QAction
from superqt import QIconifyIcon

from pymmcore_gui._qt.QtCore import Signal

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from qtpy.QtWidgets import QWidget


class MMStageExplorer(StageExplorer):

    rois_to_positions = Signal(object)

    def __init__(
        self, parent: QWidget | None = None, mmcore: CMMCorePlus | None = None
    ) -> None:
        super().__init__(parent=parent, mmcore=mmcore)

        self.to_mda = QAction(
            QIconifyIcon("mdi:plus", color=GRAY), "Add ROIs to MDA", self
        )
        self.to_mda.triggered.connect(self.to_useq_positions)
        self._toolbar.insertAction(self._toolbar.actions()[-2], self.to_mda)

    def to_useq_positions(self) -> list[useq.AbsolutePosition] | None:
        if not (rois := self.roi_manager.all_rois()):
            return

        positions: list[useq.AbsolutePosition] = []
        for idx, roi in enumerate(rois):
            if plan := roi.create_grid_plan(*self._fov_w_h()):
                p: useq.AbsolutePosition = next(iter(plan.iter_grid_positions()))
                pos = useq.AbsolutePosition(
                    name=f"ROI_{idx}",
                    x=p.x,
                    y=p.y,
                    z=p.z,
                    sequence=useq.MDASequence(grid_plan=plan),
                )
                positions.append(pos)

        if not positions:
            return

        from rich import print
        print(positions)

        self.rois_to_positions.emit(positions)

        return positions


if __name__ == "__main__":
    import sys

    from pymmcore_plus import CMMCorePlus
    from qtpy.QtWidgets import QApplication

    app = QApplication(sys.argv)
    mmcore = CMMCorePlus.instance()
    mmcore.loadSystemConfiguration()
    window = MMStageExplorer(mmcore=mmcore)
    window.show()
    sys.exit(app.exec())
