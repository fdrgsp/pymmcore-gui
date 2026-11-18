"""The Hardware tab's Python devices row: folder | Browse… | Load."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_gui._qt.QtCore import QEvent, Qt, Signal
from pymmcore_gui._qt.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from pymmcore_gui._theme import qcolor, theme

if TYPE_CHECKING:
    from pathlib import Path

HELP = (
    "Load runs the .py files in this folder. Install their dependencies in "
    "pyMM's Python environment first."
)


class PyDevicesRow(QWidget):
    """Pick the folder of Python device files and load it.

    A banner above the row counts the files that couldn't be used, until the
    next Load.
    """

    loadRequested = Signal(str)  # the folder, "" for none
    detailsRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._problem_count = 0

        self._banner = QLabel()
        self._banner.setTextFormat(Qt.TextFormat.RichText)
        self._banner.setWordWrap(True)
        self._banner.linkActivated.connect(self.detailsRequested)
        self._banner.hide()

        self._path = QLineEdit()
        self._path.setPlaceholderText("Folder of .py device files")
        self._path.returnPressed.connect(self._emit_load)
        self._browse = QPushButton("Browse…")
        self._browse.setProperty("variant", "subtle")
        self._browse.clicked.connect(self._browse_folder)
        self._load = QPushButton("Load")
        self._load.setProperty("variant", "primary")
        self._load.clicked.connect(self._emit_load)

        self._help = QLabel(HELP)
        self._help.setWordWrap(True)

        t = theme()
        row = QHBoxLayout()
        row.setSpacing(t.sp_xs)
        row.addWidget(QLabel("Python devices"))
        row.addWidget(self._path, 1)
        row.addWidget(self._browse)
        row.addWidget(self._load)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(t.sp_xs)
        layout.addWidget(self._banner)
        layout.addLayout(row)
        layout.addWidget(self._help)
        self._apply_colors()

    def path(self) -> str:
        """The folder typed or browsed to, stripped."""
        return self._path.text().strip()

    def set_path(self, path: Path | None) -> None:
        self._path.setText(str(path) if path else "")

    def set_unavailable(self, reason: str) -> None:
        """Disable the row, explaining why in its tooltip ("" enables it)."""
        self.setEnabled(not reason)
        self.setToolTip(reason)

    def set_problem_count(self, count: int) -> None:
        """Show (or, for 0, hide) the banner counting files that failed."""
        self._problem_count = count
        self._banner.setVisible(bool(count))
        self._apply_colors()

    def _browse_folder(self) -> None:
        if folder := QFileDialog.getExistingDirectory(
            self, "Python devices folder", self.path()
        ):
            self._path.setText(folder)

    def _emit_load(self) -> None:
        self.loadRequested.emit(self.path())

    def _apply_colors(self) -> None:
        t = theme()
        self._help.setStyleSheet(f"color: {qcolor(t.text_secondary).name()};")
        if n := self._problem_count:
            files = "1 file" if n == 1 else f"{n} files"
            amber = qcolor(t.status_amber).name()
            self._banner.setText(
                f"<span style='color:{amber}'>⚠ {files} in the Python devices "
                "folder couldn't be used.</span> <a href='details'>Details</a>"
            )

    def changeEvent(self, a0: QEvent | None) -> None:
        if a0 is not None and a0.type() == QEvent.Type.StyleChange:
            self._apply_colors()
        super().changeEvent(a0)
