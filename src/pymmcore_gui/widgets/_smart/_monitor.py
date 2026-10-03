"""Live view of a smart run: every frame, what analysis made of it, the script's log."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any, Final

from pymmcore_gui._qt.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QPersistentModelIndex,
    Qt,
    QUrl,
)
from pymmcore_gui._qt.QtGui import QDesktopServices, QFont
from pymmcore_gui._qt.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)
from pymmcore_gui._theme import qcolor, theme

if TYPE_CHECKING:
    from pathlib import Path

    from pymmcore_gui._smart._controller import SmartController
    from pymmcore_gui._smart._log import SmartRunLog

_ROOT: Final = QModelIndex()

MAX_ROWS: Final = 10_000
"""Rows kept in memory; the complete record is in the run folder."""

COLUMNS: Final = (
    "#",
    "Time (s)",
    "Origin",
    "Channel",
    "Position",
    "Analysis",
    "ms",
    "Action",
    "Results",
)


def _position(pos: object) -> str:
    if not isinstance(pos, dict):
        return ""
    parts = [
        f"{k}={pos[k]:.1f}"
        for k in ("x", "y", "z")
        if isinstance(pos.get(k), (int, float))
    ]
    return " ".join(parts)


def _action(record: dict[str, Any]) -> str:
    response = record.get("response") or {}
    parts = []
    if n := response.get("n_events"):
        if record.get("dropped"):
            parts.append(f"+{n} dropped")
        else:
            parts.append(f"+{n} → {response.get('priority', 'next')}")
    if response.get("drop_base"):
        parts.append("drop base")
    if response.get("stop"):
        parts.append("stop")
    return ", ".join(parts) or "—"


def _results(records: dict[str, Any]) -> str:
    def fmt(v: object) -> str:
        return f"{v:.4g}" if isinstance(v, float) else str(v)

    return "  ".join(f"{k}={fmt(v)}" for k, v in records.items())


class FramesModel(QAbstractTableModel):
    """One row per acquired frame, updated as its analysis completes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: list[list[str]] = []
        self._status: list[str] = []
        self._first_frame = 0  # frame_id of row 0 (older rows are dropped)

    def clear(self) -> None:
        self.beginResetModel()
        self._rows, self._status, self._first_frame = [], [], 0
        self.endResetModel()

    def add_frame(self, record: dict[str, Any]) -> None:
        if len(self._rows) >= MAX_ROWS:
            drop = MAX_ROWS // 10
            self.beginRemoveRows(QModelIndex(), 0, drop - 1)
            del self._rows[:drop]
            del self._status[:drop]
            self._first_frame += drop
            self.endRemoveRows()
        event = record.get("event") or {}
        channel = (event.get("channel") or {}).get("config", "")
        time_ms = record.get("runner_time_ms")
        row = [
            str(record["frame_id"]),
            "" if time_ms is None else f"{time_ms / 1000:.2f}",
            str(record.get("origin", "")),
            str(channel),
            _position(record.get("position")),
            "",
            "",
            "",
            "",
        ]
        n = len(self._rows)
        self.beginInsertRows(QModelIndex(), n, n)
        self._rows.append(row)
        self._status.append("")
        self.endInsertRows()

    def set_queued(self, frame_id: int) -> None:
        self._update(frame_id, {5: "queued"}, "queued")

    def set_result(self, record: dict[str, Any]) -> None:
        frame_id = record.get("frame_id")
        if frame_id is None:
            return
        if not record.get("ok"):
            status = "error"
        elif record.get("dropped"):
            status = "dropped"
        else:
            status = "done"
        self._update(
            frame_id,
            {
                5: status,
                6: f"{record.get('duration_ms', 0):.1f}",
                7: "error" if status == "error" else _action(record),
                8: _results(record.get("records") or {}),
            },
            status,
        )

    def _update(self, frame_id: int, cells: dict[int, str], status: str) -> None:
        row = frame_id - self._first_frame
        if not 0 <= row < len(self._rows):
            return
        for col, text in cells.items():
            self._rows[row][col] = text
        self._status[row] = status
        self.dataChanged.emit(self.index(row, 5), self.index(row, len(COLUMNS) - 1))

    # ----------------------------------------------------------- Qt model API

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = _ROOT) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex = _ROOT) -> int:
        return 0 if parent.isValid() else len(COLUMNS)

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            return self._rows[row][col]
        if role == Qt.ItemDataRole.ForegroundRole and col in (5, 7):
            status = self._status[row]
            t = theme()
            if status == "error":
                return qcolor(t.status_red)
            if status == "dropped":
                return qcolor(t.status_amber)
            if col == 7 and self._rows[row][7] not in ("", "—"):
                return qcolor(t.accent)
        if role == Qt.ItemDataRole.TextAlignmentRole and col in (0, 1, 6):
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        return None

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if (
            orientation == Qt.Orientation.Horizontal
            and role == Qt.ItemDataRole.DisplayRole
        ):
            return COLUMNS[section]
        return None


class SmartMonitor(QWidget):
    """Frames table, script log and running totals for the current run."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._run_dir: Path | None = None
        self._counts = {"frames": 0, "analyzed": 0, "requested": 0, "errors": 0}

        self.model = FramesModel(self)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)  # type: ignore[union-attr]
        header = self.table.horizontalHeader()
        assert header is not None
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(len(COLUMNS) - 1, QHeaderView.ResizeMode.Stretch)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setPlaceholderText("Messages from the script (ctx.log) appear here.")
        mono = QFont("Menlo")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.log.setFont(mono)

        self._summary = QLabel()
        self._folder = QLabel()
        self._folder.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._open_folder = QPushButton("Open run folder")
        self._open_folder.setProperty("variant", "subtle")
        self._open_folder.setEnabled(False)
        self._open_folder.clicked.connect(self._open_run_dir)
        footer = QHBoxLayout()
        footer.addWidget(self._summary)
        footer.addStretch()
        footer.addWidget(self._folder)
        footer.addWidget(self._open_folder)

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.table)
        split.addWidget(self.log)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(split, 1)
        layout.addLayout(footer)
        self._update_summary()

    @property
    def run_dir(self) -> Path | None:
        return self._run_dir

    def bind(self, controller: SmartController) -> None:
        controller.runStarted.connect(self._on_run_started)
        controller.frameAcquired.connect(self._on_frame)
        controller.analysisQueued.connect(self.model.set_queued)
        controller.analysisFinished.connect(self._on_analysis)
        controller.logMessage.connect(self.append_log)
        controller.analysisError.connect(
            lambda msg, _fatal: self.append_log("error", msg)
        )
        controller.runFinished.connect(self._on_run_finished)

    def append_log(self, level: str, message: str) -> None:
        t = theme()
        color = {
            "error": t.status_red,
            "warning": t.status_amber,
            "debug": t.text_secondary,
        }.get(level)
        text = html.escape(message).replace("\n", "<br>")
        if color is not None:
            text = f'<span style="color:{qcolor(color).name()}">{text}</span>'
        self.log.appendHtml(text)

    # ---------------------------------------------------------------- slots

    def _on_run_started(self, run_log: SmartRunLog) -> None:
        self.model.clear()
        self.log.clear()
        self._counts = dict.fromkeys(self._counts, 0)
        self._run_dir = run_log.run_dir
        self._folder.setText(str(run_log.run_dir))
        self._folder.setToolTip(str(run_log.run_dir))
        self._open_folder.setEnabled(True)
        self._update_summary("running")

    def _on_frame(self, record: dict[str, Any]) -> None:
        bar = self.table.verticalScrollBar()
        at_bottom = bar is None or bar.value() >= bar.maximum() - 2
        self.model.add_frame(record)
        self._counts["frames"] += 1
        self._update_summary("running")
        if at_bottom:
            self.table.scrollToBottom()

    def _on_analysis(self, record: dict[str, Any]) -> None:
        if record.get("call") != "analyze":
            if not record.get("ok"):
                self.append_log(
                    "error", f"{record.get('call')}() failed:\n{record.get('error')}"
                )
            return
        self.model.set_result(record)
        self._counts["analyzed"] += 1
        if not record.get("ok"):
            self._counts["errors"] += 1
        self._counts["requested"] += int(record.get("injected") or 0)
        self._update_summary("running")

    def _on_run_finished(self, summary: dict[str, Any]) -> None:
        status = str(summary.get("status", "")).replace("_", " ")
        self.append_log("info", f"Run finished: {status}.")
        self._update_summary(status)

    def _update_summary(self, status: str = "") -> None:
        c = self._counts
        text = (
            f"Frames {c['frames']} · Analyzed {c['analyzed']} · "
            f"Requested {c['requested']} · Errors {c['errors']}"
        )
        self._summary.setText(f"{text}   ({status})" if status else text)

    def _open_run_dir(self) -> None:
        if self._run_dir is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._run_dir)))
