from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_widgets import StageExplorer
from pymmcore_widgets.control._rois.roi_manager import GRAY
from qtpy.QtGui import QAction
from superqt import QIconifyIcon

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from qtpy.QtWidgets import QWidget


class MMStageExplorer(StageExplorer):

    def __init__(
        self, parent: QWidget | None = None, mmcore: CMMCorePlus | None = None
    ) -> None:
        super().__init__(parent=parent, mmcore=mmcore)

        self.to_mda = QAction(
            QIconifyIcon("mdi:plus", color=GRAY), "Add ROIs to MDA", self
        )
        self.to_mda.triggered.connect(self.add_to_mda)
        self._toolbar.insertAction(self._toolbar.actions()[-2], self.to_mda)

    def add_to_mda(self) -> None: ...


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
