"""Tabbed dialog showing the metadata of the acquisition a viewer displays.

Everything comes from the viewer's `AcquisitionRecord` (what its Save button
exports) and, for a reopened file, the raw documents on disk.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, cast

from pymmcore_plus.mda._sink import _serialize_summary_meta
from pymmcore_widgets.useq_widgets._column_fit import ColumnFitter

from pymmcore_gui._image_metadata import (
    FrameLookup,
    axis_labels,
    format_elapsed,
    frame_infos,
    frames_to_csv,
    mean_interval,
    previous_intervals,
    timepoints,
    value_label,
)
from pymmcore_gui._qt.QtCore import (
    QAbstractTableModel,
    QItemSelection,
    QItemSelectionModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    QTimer,
)
from pymmcore_gui._qt.QtGui import QFontDatabase, QKeySequence, QShortcut, QTextDocument
from pymmcore_gui._qt.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QTableView,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pymmcore_gui._settings import Settings
from pymmcore_gui._theme import qcolor, theme

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pymmcore_gui._array_viewer import MMArrayViewer
    from pymmcore_gui._image_metadata import FrameInfo, TimepointInfo
    from pymmcore_gui._qt.QtWidgets import QTableWidget

logger = logging.getLogger(__name__)

_FOLLOW_DELAY_MS = 50
_UPDATE_MS = 1000
_MAX_FRAME_ITEMS = 200
_PREVIEW_CHARS = 2_000_000
# rows measured when sizing table columns, so large tables size quickly
_SIZE_SAMPLE_ROWS = 200
_SORT_ROLE = Qt.ItemDataRole.UserRole


class _Snapshot:
    """What the viewer holds when the dialog opens or refreshes.

    Only the first `stop` metadata entries are used: frames a running
    acquisition appends later come with the next snapshot.
    """

    def __init__(self, viewer: MMArrayViewer) -> None:
        self.viewer = viewer
        self.origin = viewer.data_origin
        self.sequence = viewer.mda_sequence
        self.record = viewer._acquisition_record
        self.stop = len(self.record.frame_meta) if self.record else 0

    @property
    def has_frames(self) -> bool:
        return self.record is not None and any(
            meta is not None
            for meta in itertools.islice(self.record.frame_meta, self.stop)
        )

    @cached_property
    def frames(self) -> list[FrameInfo]:
        if self.record is None:
            return []
        return frame_infos(self.record.settings, self.record.frame_meta, self.stop)

    @cached_property
    def labels(self) -> dict[str, list[str]]:
        return axis_labels(self.record.settings) if self.record else {}

    @cached_property
    def lookup(self) -> FrameLookup:
        return FrameLookup(self.frames)

    def label(self, axis: str, value: int) -> str:
        return value_label(self.labels.get(axis, ()), value)


# ----------------------------- shared helpers -----------------------------


def _text_view() -> QPlainTextEdit:
    view = QPlainTextEdit()
    view.setReadOnly(True)
    view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
    view.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
    return view


def _button(text: str, slot: Callable[[], Any]) -> QPushButton:
    """An action button styled like the rest of the GUI's."""
    button = QPushButton(text)
    button.setProperty("variant", "primary")
    # a dialog would otherwise make each one an outlined default button
    button.setAutoDefault(False)
    button.clicked.connect(slot)
    return button


def _button_row(*buttons: QPushButton) -> QHBoxLayout:
    row = QHBoxLayout()
    row.addStretch()
    for button in buttons:
        row.addWidget(button)
    return row


def _copy(text: str) -> None:
    if clipboard := QApplication.clipboard():
        clipboard.setText(text)


def _save_text(parent: QWidget, caption: str, file_filter: str, text: str) -> None:
    path, _ = QFileDialog.getSaveFileName(parent, caption, "", file_filter)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


def _timestamp(value: Any) -> str:
    """An ISO date and time, to the second, without its time zone."""
    try:
        return datetime.fromisoformat(str(value)).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return str(value)


def _value_text(value: Any) -> str:
    if value is None or (isinstance(value, (Mapping, list, tuple)) and not value):
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


# keys that name an entry of a list (devices, properties, presets, ...)
_NAME_KEYS = ("label", "name", "camera_label")
# keys holding an entry's value, shown on the entry's own row
_VALUE_KEYS = ("value", "val")


def _list_entry(i: int, entry: Any) -> tuple[str, Any]:
    """Key a list entry by its own name (a preset setting by device-property)."""
    if isinstance(entry, Mapping):
        for key in _NAME_KEYS:
            if isinstance(name := entry.get(key), str):
                return name, {k: v for k, v in entry.items() if k != key}
        if {"dev", "prop"} <= entry.keys():
            rest = {k: v for k, v in entry.items() if k not in ("dev", "prop")}
            return f"{entry['dev']}-{entry['prop']}", rest
    return str(i), entry


def _fill_tree(parent: QTreeWidgetItem, value: Mapping | Sequence) -> None:
    """Add a (key, value) item per entry of `value`, nesting containers.

    List entries are keyed by their name, and an entry's own value shows on its
    row, so a device property reads `Exposure | 10`. Lists of plain values
    (allowed values, child names) show on one row.
    """
    if isinstance(value, Mapping):
        entries = list(value.items())
    else:
        entries = [_list_entry(i, entry) for i, entry in enumerate(value)]
    for key, child in entries:
        item = QTreeWidgetItem(parent, [str(key)])
        if isinstance(child, Mapping) and (
            shown := next((k for k in _VALUE_KEYS if k in child), None)
        ):
            item.setText(1, _value_text(child[shown]))
            child = {k: v for k, v in child.items() if k != shown}
        if isinstance(child, (list, tuple)) and not any(
            isinstance(c, (Mapping, list, tuple)) for c in child
        ):
            item.setText(1, ", ".join(_value_text(c) for c in child))
        elif isinstance(child, (Mapping, list, tuple)):
            if child:
                _fill_tree(item, child)
        else:
            item.setText(1, _value_text(child))


def _filter_tree(item: QTreeWidgetItem, text: str) -> bool:
    """Hide what does not match `text`; return whether `item` stays visible.

    An item matching by key or value keeps all its children visible, and an
    item with a matching descendant stays visible (and expanded) too.
    """
    matches = not text or text in item.text(0).lower() or text in item.text(1).lower()
    child_matches = False
    for i in range(item.childCount()):
        child = item.child(i)
        if child is not None:
            child_matches |= _filter_tree(child, "" if matches else text)
    if text and child_matches:
        item.setExpanded(True)
    item.setHidden(not (matches or child_matches))
    return matches or child_matches


def _tree_root(tree: QTreeWidget) -> QTreeWidgetItem:
    root = tree.invisibleRootItem()
    assert root is not None  # a tree always has one
    return root


def _tree() -> QTreeWidget:
    tree = QTreeWidget()
    tree.setHeaderLabels(["Key", "Value"])
    tree.setAlternatingRowColors(True)
    tree.setUniformRowHeights(True)
    # keys nested deeper only show once expanded: keep them readable. Expand
    # All emits once per item, so resize once after the whole batch.
    resize = QTimer(tree)
    resize.setSingleShot(True)
    resize.setInterval(0)
    resize.timeout.connect(lambda: tree.resizeColumnToContents(0))
    for signal in (tree.itemExpanded, tree.itemCollapsed):
        signal.connect(resize.start)
    return tree


# ------------------------------- table model ------------------------------


@dataclass(frozen=True)
class _Column:
    header: str
    value: Callable[[int], Any]
    """Row -> sortable value, or None when missing."""
    text: Callable[[Any], str] = str


class _TableModel(QAbstractTableModel):
    """Rows computed on demand, so large acquisitions cost no per-cell objects."""

    def __init__(
        self,
        columns: Sequence[_Column],
        n_rows: int,
        warn: Callable[[int], bool] | None = None,
    ) -> None:
        super().__init__()
        self.columns = list(columns)
        self._n_rows = n_rows
        self._warn = warn

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else self._n_rows

    def columnCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.columns)

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
            return self.columns[section].header
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        column = self.columns[index.column()]
        value = column.value(index.row())
        if role == Qt.ItemDataRole.DisplayRole:
            return "" if value is None else column.text(value)
        if role == _SORT_ROLE:
            return math.inf if value is None else value
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        if (
            role == Qt.ItemDataRole.ForegroundRole
            and self._warn is not None
            and self._warn(index.row())
        ):
            return qcolor(theme().status_amber)
        return None

    def empty_columns(self) -> list[int]:
        """Columns with no value in any row."""
        return [
            i
            for i, column in enumerate(self.columns)
            if all(column.value(row) is None for row in range(self._n_rows))
        ]


class _Table(QTableView):
    """A table view that `ColumnFitter` can size like a `QTableWidget`."""

    def rowCount(self) -> int:
        return model.rowCount() if (model := self.model()) else 0

    def columnCount(self) -> int:
        return model.columnCount() if (model := self.model()) else 0


class _PositionFilter(QSortFilterProxyModel):
    """Sorts by the numeric values and keeps only one position's rows."""

    def __init__(self) -> None:
        super().__init__()
        self.setSortRole(_SORT_ROLE)
        self.position_of: Callable[[int], int | None] = lambda row: None
        self.position: int | None = None

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        return self.position is None or self.position_of(source_row) == self.position

    def set_position(self, position: int | None) -> None:
        self.position = position
        self.invalidateFilter()


# ---------------------------------- pages ---------------------------------


class _Page(QWidget):
    """One tab. `load` fills it from a snapshot the first time it is shown."""

    title: ClassVar[str]

    def __init__(self, dialog: MetadataDialog) -> None:
        super().__init__()
        self.dialog = dialog
        self.body = QWidget()
        self._error = QLabel()
        self._error.setWordWrap(True)
        self._error.hide()
        layout = QVBoxLayout(self)
        layout.addWidget(self._error)
        layout.addWidget(self.body)

    def unavailable(self, snapshot: _Snapshot) -> str | None:
        """Why this tab has nothing to show, or None."""
        return None

    def load(self, snapshot: _Snapshot) -> None:
        raise NotImplementedError

    def display_changed(self) -> None:
        """The frame the viewer shows changed."""

    def show_error(self, message: str | None) -> None:
        self._error.setText(message or "")
        self._error.setVisible(bool(message))
        self.body.setVisible(not message)


class _SequencePage(_Page):
    title = "MDA Sequence"

    def __init__(self, dialog: MetadataDialog) -> None:
        super().__init__(dialog)
        self._text = _text_view()
        self._reuse = _button("Re-use MDA…", dialog.viewer.reuse_mda)
        layout = QVBoxLayout(self.body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._text)
        layout.addLayout(
            _button_row(
                _button("Copy", lambda: _copy(self._text.toPlainText())),
                _button("Save as YAML…", self._save),
                self._reuse,
            )
        )

    def unavailable(self, snapshot: _Snapshot) -> str | None:
        if snapshot.sequence is None:
            return "No MDA sequence was saved with this acquisition."
        return None

    def load(self, snapshot: _Snapshot) -> None:
        sequence = snapshot.sequence
        assert sequence is not None
        self._text.setPlainText(
            sequence.yaml(exclude_unset=True, exclude_defaults=True) or ""
        )
        self._reuse.setEnabled(self.dialog.viewer.can_reuse_mda())

    def _save(self) -> None:
        _save_text(
            self, "Save MDA Sequence", "YAML (*.yaml *.yml)", self._text.toPlainText()
        )


class _TimestampsPage(_Page):
    title = "Timestamps"

    def __init__(self, dialog: MetadataDialog) -> None:
        super().__init__(dialog)
        self._snapshot: _Snapshot | None = None
        self._timepoint_rows: list[TimepointInfo] = []
        self._summary = QLabel()
        self._summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self._frames_mode = QRadioButton("Frames")
        self._timepoints_mode = QRadioButton("Timepoints")
        self._frames_mode.setChecked(True)
        modes = QButtonGroup(self)
        modes.addButton(self._frames_mode)
        modes.addButton(self._timepoints_mode)
        self._frames_mode.toggled.connect(self._show_model)
        self._position = QComboBox()
        self._position.currentIndexChanged.connect(self._filter_position)
        self._follow = QCheckBox("Follow viewer")
        self._follow.setChecked(True)
        self._follow.toggled.connect(self.display_changed)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Show:"))
        controls.addWidget(self._frames_mode)
        controls.addWidget(self._timepoints_mode)
        controls.addSpacing(12)
        self._position_label = QLabel("Position:")
        controls.addWidget(self._position_label)
        controls.addWidget(self._position)
        controls.addStretch()
        controls.addWidget(self._follow)

        self._proxy = _PositionFilter()
        self._table = _Table()
        self._table.setModel(self._proxy)
        self._table.setSortingEnabled(True)
        self._table.sortByColumn(-1, Qt.SortOrder.AscendingOrder)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        if columns := self._table.horizontalHeader():
            columns.setResizeContentsPrecision(_SIZE_SAMPLE_ROWS)
        # resizable columns, fitted to their contents and spanning the table
        self._columns = ColumnFitter(cast("QTableWidget", self._table))
        if header := self._table.verticalHeader():
            header.hide()
        self._table.doubleClicked.connect(self._go_to_row)
        self._frame_model: _TableModel | None = None
        self._timepoint_model: _TableModel | None = None

        layout = QVBoxLayout(self.body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._summary)
        layout.addLayout(controls)
        layout.addWidget(self._table)
        layout.addLayout(
            _button_row(
                _button("Copy", self._copy_selection),
                _button("Export CSV…", self._export),
            )
        )

    def unavailable(self, snapshot: _Snapshot) -> str | None:
        if not snapshot.has_frames:
            return "No per-frame metadata was saved with this acquisition."
        return None

    def load(self, snapshot: _Snapshot) -> None:
        self._snapshot = snapshot
        frames = snapshot.frames
        self._timepoint_rows = timepoints(frames, snapshot.sequence)
        self._frame_model = self._frames_table(snapshot)
        self._timepoint_model = (
            self._timepoints_table(snapshot) if self._timepoint_rows else None
        )
        self._timepoints_mode.setEnabled(self._timepoint_model is not None)
        if self._timepoint_model is None:
            self._frames_mode.setChecked(True)

        shown = self._position.currentData()
        self._position.blockSignals(True)
        self._position.clear()
        self._position.addItem("All", None)
        positions = sorted({p for f in frames if (p := f.index.get("p")) is not None})
        for p in positions:
            self._position.addItem(snapshot.label("p", p), p)
        # a reload (e.g. of a running acquisition) keeps the chosen position
        self._position.setCurrentIndex(max(self._position.findData(shown), 0))
        self._position.blockSignals(False)
        self._position.setVisible(len(positions) > 1)
        self._position_label.setVisible(len(positions) > 1)
        self._proxy.set_position(self._position.currentData())

        self._summary.setText(self._summary_text(snapshot))
        self._show_model()

    def _summary_text(self, snapshot: _Snapshot) -> str:
        parts = []
        summary = snapshot.record.summary_meta if snapshot.record else None
        if summary and (started := summary.get("datetime")):
            parts.append(f"Started {_timestamp(started)}")
        if stamps := [f.delta_t for f in snapshot.frames if f.delta_t is not None]:
            parts.append(f"Lasted {format_elapsed(max(stamps))}")
        if (interval := mean_interval(self._timepoint_rows)) is not None:
            parts.append(f"Interval {interval:.3f} s")
        return " · ".join(parts)

    def _frames_table(self, snapshot: _Snapshot) -> _TableModel:
        frames = snapshot.frames
        intervals = previous_intervals(frames)
        axes = list(frames[0].index) if frames else []

        def scaled(value: float | None, factor: float) -> float | None:
            return None if value is None else value * factor

        def axis_column(axis: str) -> _Column:
            return _Column(
                axis,
                lambda row: frames[row].index[axis],
                lambda value: snapshot.label(axis, value),
            )

        def position_column(i: int, name: str) -> _Column:
            return _Column(
                f"{name} (µm)", lambda row: frames[row].position[i], "{:.2f}".format
            )

        columns = [axis_column(axis) for axis in axes]
        columns += [
            _Column("Elapsed", lambda row: frames[row].delta_t, format_elapsed),
            _Column("Δt (s)", lambda row: frames[row].delta_t, "{:.3f}".format),
            _Column(
                "Δprev (ms)", lambda row: scaled(intervals[row], 1000), "{:.1f}".format
            ),
            _Column(
                "Exposure (ms)",
                lambda row: scaled(frames[row].exposure, 1000),
                "{:g}".format,
            ),
        ]
        columns += [position_column(i, name) for i, name in enumerate("XYZ")]
        return _TableModel(columns, len(frames))

    def _timepoints_table(self, snapshot: _Snapshot) -> _TableModel:
        rows = self._timepoint_rows
        columns = []
        if rows[0].position is not None:
            columns.append(
                _Column(
                    "p",
                    lambda row: rows[row].position,
                    lambda value: snapshot.label("p", value),
                )
            )
        columns += [
            _Column("t", lambda row: rows[row].t),
            _Column("Start", lambda row: rows[row].start, format_elapsed),
            _Column("Interval (s)", lambda row: rows[row].interval, "{:.3f}".format),
            _Column("Requested (s)", lambda row: rows[row].requested, "{:.3f}".format),
            _Column(
                "Drift (%)",
                lambda row: None if (d := rows[row].drift) is None else d * 100,
                "{:+.1f}".format,
            ),
        ]

        def warn(row: int) -> bool:
            drift = rows[row].drift
            return drift is not None and abs(drift) > 0.1

        return _TableModel(columns, len(rows), warn)

    def _show_model(self) -> None:
        if self._snapshot is None:
            return
        frames_mode = self._frames_mode.isChecked()
        model = self._frame_model if frames_mode else self._timepoint_model
        if model is None:
            return
        if frames_mode:
            frames = self._snapshot.frames
            self._proxy.position_of = lambda row: frames[row].index.get("p")
        else:
            rows = self._timepoint_rows
            self._proxy.position_of = lambda row: rows[row].position
        self._proxy.setSourceModel(model)
        self._proxy.invalidateFilter()
        empty = set(model.empty_columns())
        for column in range(model.columnCount()):
            self._table.setColumnHidden(column, column in empty)
        self._columns.fit()
        self._follow.setEnabled(frames_mode)
        self.display_changed()

    def _filter_position(self) -> None:
        self._proxy.set_position(self._position.currentData())

    def display_changed(self) -> None:
        snapshot = self._snapshot
        if (
            snapshot is None
            or not self._follow.isChecked()
            or not self._frames_mode.isChecked()
        ):
            return
        rows = snapshot.lookup.rows(self.dialog.viewer.displayed_ranges())
        if (model := self._frame_model) is None:
            return
        selection = QItemSelection()
        last_column = model.columnCount() - 1
        for row in rows:
            selection.select(model.index(row, 0), model.index(row, last_column))
        selection = self._proxy.mapSelectionFromSource(selection)
        if selection_model := self._table.selectionModel():
            selection_model.select(
                selection,
                QItemSelectionModel.SelectionFlag.ClearAndSelect
                | QItemSelectionModel.SelectionFlag.Rows,
            )
        if not selection.isEmpty():
            self._table.scrollTo(selection.indexes()[0])

    def _go_to_row(self, proxy_index: QModelIndex) -> None:
        if self._snapshot is None:
            return
        row = self._proxy.mapToSource(proxy_index).row()
        if self._frames_mode.isChecked():
            index = self._snapshot.frames[row].index
        else:
            info = self._timepoint_rows[row]
            index = (
                {"t": info.t}
                if info.position is None
                else {"p": info.position, "t": info.t}
            )
        self.dialog.viewer.show_frame(index)

    def _copy_selection(self) -> None:
        if (selection_model := self._table.selectionModel()) is None:
            return
        columns = [
            c
            for c in range(self._proxy.columnCount())
            if not self._table.isColumnHidden(c)
        ]
        header = [
            str(self._proxy.headerData(c, Qt.Orientation.Horizontal)) for c in columns
        ]
        rows = sorted(index.row() for index in selection_model.selectedRows())
        lines = ["\t".join(header)]
        lines += [
            "\t".join(str(self._proxy.index(row, c).data()) for c in columns)
            for row in rows
        ]
        _copy("\n".join(lines))

    def _export(self) -> None:
        if self._snapshot is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Timestamps", "", "CSV (*.csv)"
        )
        if path:
            frames_to_csv(self._snapshot.frames, self._snapshot.labels, path)


class _FramePage(_Page):
    title = "Frame"

    def __init__(self, dialog: MetadataDialog) -> None:
        super().__init__(dialog)
        self._snapshot: _Snapshot | None = None
        self._tree = _tree()
        layout = QVBoxLayout(self.body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tree)

    def unavailable(self, snapshot: _Snapshot) -> str | None:
        if not snapshot.has_frames:
            return "No per-frame metadata was saved with this acquisition."
        return None

    def load(self, snapshot: _Snapshot) -> None:
        self._snapshot = snapshot

    def display_changed(self) -> None:
        snapshot = self._snapshot
        if snapshot is None:
            return
        rows = snapshot.lookup.rows(self.dialog.viewer.displayed_ranges())
        self._tree.clear()
        root = _tree_root(self._tree)
        if not rows:
            QTreeWidgetItem(root, ["No metadata for this frame"])
        for row in rows[:_MAX_FRAME_ITEMS]:
            frame = snapshot.frames[row]
            title = " · ".join(
                f"{axis} {snapshot.label(axis, value)}"
                for axis, value in frame.index.items()
            )
            item = QTreeWidgetItem(root, [title or "Frame"])
            _fill_tree(item, _frame_fields(frame))
        if len(rows) > _MAX_FRAME_ITEMS:
            QTreeWidgetItem(root, [f"… and {len(rows) - _MAX_FRAME_ITEMS} more"])
        self._tree.expandAll()


def _frame_fields(frame: FrameInfo) -> dict[str, Any]:
    """A frame's values for display, with `mda_event` decoded when it is JSON."""
    fields: dict[str, Any] = {
        "Elapsed": format_elapsed(frame.delta_t),
        "Δt (s)": frame.delta_t,
        "Exposure (ms)": None if frame.exposure is None else frame.exposure * 1000,
        "Position (µm)": {
            axis: value
            for axis, value in zip("xyz", frame.position, strict=True)
            if value is not None
        },
    }
    for key, value in frame.extra.items():
        if key == "mda_event" and isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        fields[key] = value
    return {key: value for key, value in fields.items() if value not in (None, "", {})}


class _AcquisitionPage(_Page):
    title = "Acquisition"

    def __init__(self, dialog: MetadataDialog) -> None:
        super().__init__(dialog)
        self._summary: dict[str, Any] = {}
        self._tree = _tree()
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter keys and values")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        layout = QVBoxLayout(self.body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._filter)
        layout.addWidget(self._tree)
        layout.addLayout(
            _button_row(
                _button("Expand All", self._tree.expandAll),
                _button("Collapse All", self._tree.collapseAll),
                _button("Copy as JSON", self._copy_json),
            )
        )

    def unavailable(self, snapshot: _Snapshot) -> str | None:
        if snapshot.record is None or snapshot.record.summary_meta is None:
            return "No acquisition summary was saved with this acquisition."
        return None

    def load(self, snapshot: _Snapshot) -> None:
        assert snapshot.record is not None and snapshot.record.summary_meta is not None
        self._summary = _serialize_summary_meta(snapshot.record.summary_meta)
        self._tree.clear()
        _fill_tree(_tree_root(self._tree), self._summary)
        self._tree.resizeColumnToContents(0)
        self._apply_filter(self._filter.text())

    def _apply_filter(self, text: str) -> None:
        root = _tree_root(self._tree)
        for i in range(root.childCount()):
            if item := root.child(i):
                _filter_tree(item, text.lower())

    def _copy_json(self) -> None:
        _copy(json.dumps(self._summary, indent=2, default=str))


class _RawPage(_Page):
    title = "Raw"

    def __init__(self, dialog: MetadataDialog) -> None:
        super().__init__(dialog)
        self._loaders: Mapping[str, Callable[[], str]] = {}
        self._texts: dict[str, str] = {}
        self._documents = QComboBox()
        self._documents.currentTextChanged.connect(self._show_document)
        self._note = QLabel()
        self._note.hide()
        self._text = _text_view()
        self._find = QLineEdit()
        self._find.setPlaceholderText("Find (Ctrl+F)")
        self._find.returnPressed.connect(lambda: self._find_next(backward=False))
        find_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Find), self)
        find_shortcut.activated.connect(self._find.setFocus)

        top = QHBoxLayout()
        self._documents_label = QLabel("Document:")
        top.addWidget(self._documents_label)
        top.addWidget(self._documents, 1)
        top.addWidget(self._find, 1)
        top.addWidget(_button("Previous", lambda: self._find_next(backward=True)))
        top.addWidget(_button("Next", lambda: self._find_next(backward=False)))
        layout = QVBoxLayout(self.body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addWidget(self._note)
        layout.addWidget(self._text)
        layout.addLayout(
            _button_row(_button("Copy", self._copy), _button("Save as…", self._save))
        )

    def unavailable(self, snapshot: _Snapshot) -> str | None:
        if snapshot.origin is None or snapshot.origin.raw_documents is None:
            return (
                "This viewer shows the acquisition as it was captured. Reopen the"
                " saved file to see the metadata stored on disk."
            )
        return None

    def load(self, snapshot: _Snapshot) -> None:
        assert snapshot.origin is not None and snapshot.origin.raw_documents is not None
        self._loaders = snapshot.origin.raw_documents()
        self._texts.clear()
        self._documents.blockSignals(True)
        self._documents.clear()
        self._documents.addItems(list(self._loaders))
        self._documents.blockSignals(False)
        self._documents.setVisible(len(self._loaders) > 1)
        self._documents_label.setVisible(len(self._loaders) > 1)
        self._show_document(self._documents.currentText())

    def _document(self, name: str) -> str:
        """The original text of document `name`, read on first use."""
        if name not in self._texts:
            self._texts[name] = self._loaders[name]()
        return self._texts[name]

    def _show_document(self, name: str) -> None:
        if name not in self._loaders:
            self._text.clear()
            return
        text = self._document(name)
        if name.endswith(".json"):
            try:
                text = json.dumps(json.loads(text), indent=2)
            except json.JSONDecodeError:
                pass
        truncated = len(text) > _PREVIEW_CHARS
        self._note.setText(
            f"Showing the first {_PREVIEW_CHARS:,} characters; Copy and Save keep"
            " the whole document."
        )
        self._note.setVisible(truncated)
        self._text.setPlainText(text[:_PREVIEW_CHARS])

    def _find_next(self, *, backward: bool) -> None:
        if not (query := self._find.text()):
            return

        def find() -> bool:
            if backward:
                return self._text.find(query, QTextDocument.FindFlag.FindBackward)
            return self._text.find(query)

        if not find():  # wrap around
            cursor = self._text.textCursor()
            cursor.movePosition(
                cursor.MoveOperation.End if backward else cursor.MoveOperation.Start
            )
            self._text.setTextCursor(cursor)
            find()

    def _copy(self) -> None:
        if name := self._documents.currentText():
            _copy(self._document(name))

    def _save(self) -> None:
        if name := self._documents.currentText():
            suffix = "JSON (*.json)" if name.endswith(".json") else "XML (*.xml)"
            _save_text(self, "Save Raw Metadata", suffix, self._document(name))


# --------------------------------- dialog ---------------------------------


class MetadataDialog(QDialog):
    """Metadata of `viewer`'s acquisition, in tabs built when first shown."""

    def __init__(self, viewer: MMArrayViewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.viewer = viewer
        self._snapshot = _Snapshot(viewer)
        self._stale: set[_Page] = set()

        self._header = QLabel()
        self._header.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._tabs = QTabWidget()
        self._pages: list[_Page] = [
            page(self)
            for page in (
                _SequencePage,
                _TimestampsPage,
                _FramePage,
                _AcquisitionPage,
                _RawPage,
            )
        ]
        for page in self._pages:
            self._tabs.addTab(page, page.title)
        self._tabs.currentChanged.connect(self._load_current)

        layout = QVBoxLayout(self)
        layout.addWidget(self._header)
        layout.addWidget(self._tabs)
        layout.addLayout(_button_row(_button("Close", self.close)))

        # picks up frames a running acquisition appends, and a renamed viewer
        self._update_timer = QTimer(self)
        self._update_timer.setInterval(_UPDATE_MS)
        self._update_timer.timeout.connect(self._update)
        self._update_timer.start()

        # batches slider drags into one update of the following tabs
        self._follow_timer = QTimer(self)
        self._follow_timer.setSingleShot(True)
        self._follow_timer.setInterval(_FOLLOW_DELAY_MS)
        self._follow_timer.timeout.connect(self._on_display_changed)
        model = viewer.display_model
        self._display_signals = (
            model.current_index.value_changed,
            model.events.visible_axes,
            model.events.channel_axis,
            model.events.channel_mode,
        )
        for signal in self._display_signals:
            signal.connect(self._schedule_follow)

        prefs = Settings.instance().modern_window
        if geometry := prefs.metadata_dialog_geometry:
            self.restoreGeometry(geometry)
        else:
            self.resize(900, 600)
        self.refresh()
        titles = [page.title for page in self._pages]
        if (tab := prefs.metadata_dialog_tab) in titles:
            index = titles.index(tab)
            if self._tabs.isTabEnabled(index):
                self._tabs.setCurrentIndex(index)

    def _update(self) -> None:
        record = self.viewer._acquisition_record
        grown = record is not None and len(record.frame_meta) != self._snapshot.stop
        if not grown and self.windowTitle() == self._title():
            return
        started = time.perf_counter()
        self.refresh()
        # reloading a long acquisition takes a while: check less often then
        took_ms = (time.perf_counter() - started) * 1000
        self._update_timer.setInterval(max(_UPDATE_MS, int(10 * took_ms)))

    def _title(self) -> str:
        return f"Metadata: {self.viewer.source_title}"

    def refresh(self) -> None:
        """Take a new snapshot of the viewer's metadata and rebuild the tabs."""
        self._snapshot = snapshot = _Snapshot(self.viewer)
        self._stale = set(self._pages)
        self.setWindowTitle(self._title())
        if origin := snapshot.origin:
            self._header.setText(" · ".join(filter(None, (origin.format, origin.path))))
        for i, page in enumerate(self._pages):
            reason = page.unavailable(snapshot)
            self._tabs.setTabEnabled(i, reason is None)
            self._tabs.setTabToolTip(i, reason or "")
        if not self._tabs.isTabEnabled(self._tabs.currentIndex()):
            enabled = (
                i for i in range(self._tabs.count()) if self._tabs.isTabEnabled(i)
            )
            self._tabs.setCurrentIndex(next(enabled, 0))
        self._load_current()

    def _load_current(self) -> None:
        page = self._tabs.currentWidget()
        if not isinstance(page, _Page) or page not in self._stale:
            return
        if not self._tabs.isTabEnabled(self._tabs.currentIndex()):
            return
        self._stale.discard(page)
        try:
            page.load(self._snapshot)
            page.display_changed()
        except Exception as e:
            logger.exception("Could not show the %s tab", page.title)
            page.show_error(f"Could not show this tab: {e}")
        else:
            page.show_error(None)

    def _schedule_follow(self) -> None:
        self._follow_timer.start()

    def _on_display_changed(self) -> None:
        page = self._tabs.currentWidget()
        if isinstance(page, _Page) and page not in self._stale:
            try:
                page.display_changed()
            except Exception:
                logger.exception(
                    "Could not follow the viewer in the %s tab", page.title
                )

    def done(self, a0: int) -> None:
        self._update_timer.stop()
        self._follow_timer.stop()
        for signal in self._display_signals:
            signal.disconnect(self._schedule_follow)
        settings = Settings.instance()
        prefs = settings.modern_window
        prefs.metadata_dialog_geometry = self.saveGeometry().data()
        prefs.metadata_dialog_tab = self._pages[self._tabs.currentIndex()].title
        settings.flush()
        super().done(a0)
