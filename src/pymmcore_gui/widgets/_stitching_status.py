"""Background stitching status and per-job progress controls."""

from __future__ import annotations

from typing import TYPE_CHECKING

from superqt.iconify import QIconifyIcon

from pymmcore_gui._qt.QtCore import QEvent, QPoint, Qt
from pymmcore_gui._qt.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from pymmcore_gui._stitching._controller import JobState
from pymmcore_gui._theme import qcolor, theme

if TYPE_CHECKING:
    from pymmcore_gui._stitching._controller import StitchingController, StitchJob

PROGRESS_STEPS = 1000


class _JobRow(QWidget):
    """Name, phase, progress bar and Cancel for one job."""

    def __init__(self, handle: StitchJob, controller: StitchingController) -> None:
        super().__init__()
        self._handle = handle
        self._controller = controller
        self.title = QLabel(handle.title)
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)
        self.status = QLabel()
        self.bar = QProgressBar()
        self.bar.setRange(0, PROGRESS_STEPS)
        self.bar.setTextVisible(False)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setProperty("variant", "subtle")
        self.cancel_btn.clicked.connect(self._cancel)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.title, 1)
        top.addWidget(self.cancel_btn)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme().sp_xxs)
        layout.addLayout(top)
        layout.addWidget(self.bar)
        layout.addWidget(self.status)
        self.refresh()

    def refresh(self) -> None:
        handle = self._handle
        self.status.setText(handle.status_text)
        queued = handle.state == JobState.QUEUED
        self.cancel_btn.setText("Remove" if queued else "Cancel")
        self.cancel_btn.setEnabled(handle.state != JobState.CANCELLING)
        self.bar.setValue(
            0 if queued else round(handle.progress.fraction * PROGRESS_STEPS)
        )

    def _cancel(self) -> None:
        self._controller.cancel(self._handle)


class StitchingJobsPopup(QFrame):
    """One row per queued or running job."""

    def __init__(self, controller: StitchingController, parent: QWidget) -> None:
        super().__init__(parent, Qt.WindowType.Popup)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._controller = controller
        self._rows: dict[str, _JobRow] = {}
        self._layout = QVBoxLayout(self)
        sp = theme().sp_sm
        self._layout.setContentsMargins(sp, sp, sp, sp)
        self._layout.setSpacing(sp)
        self.setMinimumWidth(theme().scaled(320))
        controller.changed.connect(self.refresh)
        self.refresh()

    def rows(self) -> dict[str, _JobRow]:
        return dict(self._rows)

    def refresh(self) -> None:
        active = {j.id: j for j in self._controller.active_jobs}
        for job_id in list(self._rows):
            if job_id not in active:
                gone = self._rows.pop(job_id)
                self._layout.removeWidget(gone)
                gone.deleteLater()
        for index, (job_id, handle) in enumerate(active.items()):
            if (row := self._rows.get(job_id)) is not None:
                row.refresh()
                continue
            self._rows[job_id] = _JobRow(handle, self._controller)
            self._layout.insertWidget(index, self._rows[job_id])
        if not active:
            self.hide()
        elif self.isVisible():
            self.adjustSize()


class StitchingStatusButton(QPushButton):
    """`Stitching 1/2 · 42 %` while jobs are queued or running; hidden otherwise."""

    def __init__(
        self, controller: StitchingController, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self.setProperty("variant", "subtle")
        self.setFlat(True)
        self.setToolTip("Stitching in the background. Click for details.")
        self._popup = StitchingJobsPopup(controller, self)
        self.clicked.connect(self.show_popup)
        controller.changed.connect(self.refresh)
        self._apply_icon()
        self.refresh()

    @property
    def popup(self) -> StitchingJobsPopup:
        return self._popup

    def refresh(self) -> None:
        active = self._controller.active_jobs
        self.setVisible(bool(active))
        if not active:
            return
        running = self._controller.running_job
        percent = round(running.progress.fraction * 100) if running else 0
        count = f" 1/{len(active)}" if len(active) > 1 else ""
        self.setText(f"Stitching{count} · {percent} %")

    def show_popup(self) -> None:
        if not self._controller.active_jobs:
            return
        popup = self._popup
        popup.refresh()
        popup.adjustSize()
        # above the button, right-aligned with it: the status bar is at the bottom
        corner = self.mapToGlobal(QPoint(self.width(), 0))
        popup.move(corner - QPoint(popup.width(), popup.height()))
        popup.show()

    def _apply_icon(self) -> None:
        color = qcolor(theme().text_primary).name()
        self.setIcon(QIconifyIcon("mdi:vector-combine", color=color))

    def changeEvent(self, e: QEvent | None) -> None:
        super().changeEvent(e)
        if e is not None and e.type() == QEvent.Type.StyleChange:
            self._apply_icon()
