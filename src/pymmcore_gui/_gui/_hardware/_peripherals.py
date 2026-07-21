"""Peripheral selection for a newly added hub device."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pymmcore_gui._gui._theme import theme
from pymmcore_gui._qt.QtCore import Qt
from pymmcore_gui._qt.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pymmcore_plus.model import AvailableDevice, Device, Microscope


class PeripheralsDialog(QDialog):
    """Pick which of a hub's peripherals to add.

    Labels are editable; the returned devices are already parented to the hub.
    """

    HEADERS = ("Label", "Adapter", "Description")

    def __init__(
        self, hub: Device, model: Microscope, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._hub = hub
        self.setWindowTitle(f"Add {hub.name} peripherals")

        self._table = QTableWidget(0, len(self.HEADERS), self)
        self._table.setHorizontalHeaderLabels(list(self.HEADERS))
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._table.itemChanged.connect(self._sync_select_all)
        if vh := self._table.verticalHeader():
            vh.setVisible(False)
        if hh := self._table.horizontalHeader():
            hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)

        self._select_all = QCheckBox("Select all")
        self._select_all.setTristate(True)
        self._select_all.clicked.connect(self._on_select_all)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        bottom = QHBoxLayout()
        bottom.addWidget(self._select_all)
        bottom.addStretch()
        bottom.addWidget(buttons)

        t = theme()
        layout = QVBoxLayout(self)
        layout.setSpacing(t.sp_sm)
        layout.addWidget(QLabel(f"{hub.name} provides these peripherals:"))
        layout.addWidget(self._table, 1)
        layout.addLayout(bottom)

        self._populate(model)
        self.resize(t.scaled(560), t.scaled(360))

    def _populate(self, model: Microscope) -> None:
        peripherals = list(self._hub.available_peripherals(model))
        self._table.setRowCount(len(peripherals))
        taken = {d.name for d in model.devices}
        for row, child in enumerate(peripherals):
            label = QTableWidgetItem(_unique(child.adapter_name, taken))
            taken.add(label.text())
            label.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEditable
                | Qt.ItemFlag.ItemIsEnabled
            )
            label.setData(Qt.ItemDataRole.UserRole, child)
            label.setCheckState(Qt.CheckState.Unchecked)
            self._table.setItem(row, 0, label)

            for col, text in ((1, child.adapter_name), (2, child.description)):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemFlag.NoItemFlags)
                self._table.setItem(row, col, item)

    # ── results ───────────────────────────────────────────────────

    def selected_peripherals(self) -> Iterator[Device]:
        """Yield a `Device` for each checked row, parented to the hub."""
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item and item.checkState() == Qt.CheckState.Checked:
                child = cast("AvailableDevice", item.data(Qt.ItemDataRole.UserRole))
                yield child.replace(name=item.text(), parent_label=self._hub.name)

    def has_peripherals(self) -> bool:
        """Whether the hub offered anything at all."""
        return self._table.rowCount() > 0

    # ── select-all plumbing ───────────────────────────────────────

    def _on_select_all(self) -> None:
        state = (
            Qt.CheckState.Checked
            if self._select_all.checkState() != Qt.CheckState.Unchecked
            else Qt.CheckState.Unchecked
        )
        self._select_all.setCheckState(state)
        with _blocked(self._table):
            for row in range(self._table.rowCount()):
                if item := self._table.item(row, 0):
                    item.setCheckState(state)

    def _sync_select_all(self) -> None:
        checked = sum(
            1
            for row in range(self._table.rowCount())
            if (item := self._table.item(row, 0))
            and item.checkState() == Qt.CheckState.Checked
        )
        if checked == 0:
            state = Qt.CheckState.Unchecked
        elif checked == self._table.rowCount():
            state = Qt.CheckState.Checked
        else:
            state = Qt.CheckState.PartiallyChecked
        self._select_all.setCheckState(state)


def _unique(base: str, taken: set[str]) -> str:
    """Return `base`, suffixed if needed, so it isn't already in `taken`."""
    if base not in taken:
        return base
    i = 2
    while f"{base}-{i}" in taken:
        i += 1
    return f"{base}-{i}"


class _blocked:
    """Context manager blocking a widget's signals."""

    def __init__(self, widget: QWidget) -> None:
        self._widget = widget

    def __enter__(self) -> None:
        self._widget.blockSignals(True)

    def __exit__(self, *args: object) -> None:
        self._widget.blockSignals(False)
