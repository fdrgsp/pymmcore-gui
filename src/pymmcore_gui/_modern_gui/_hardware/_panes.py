"""Device list panes (available / installed) for the Hardware Setup tab."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pymmcore_plus import DeviceType

from pymmcore_gui._modern_gui._theme import theme
from pymmcore_gui._qt.QtCore import Qt, pyqtSignal
from pymmcore_gui._qt.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymmcore_plus.model import AvailableDevice, Device


def pane_title(text: str) -> QLabel:
    """Return the section heading used at the top of each pane."""
    return QLabel(text)


class _DeviceTable(QTableWidget):
    """Read-only table with single full-row selection."""

    def __init__(self, headers: Sequence[str], parent: QWidget | None = None) -> None:
        super().__init__(0, len(headers), parent)
        self.setHorizontalHeaderLabels(list(headers))
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(True)
        self.setWordWrap(False)
        if vh := self.verticalHeader():
            vh.setVisible(False)
        if hh := self.horizontalHeader():
            hh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            hh.setHighlightSections(False)

    def selected_object(self) -> Any | None:
        """Return the object attached to the selected row, if any."""
        for item in self.selectedItems():
            if first := self.item(item.row(), 0):
                return first.data(Qt.ItemDataRole.UserRole)
        return None

    def add_row(self, values: Sequence[str], obj: Any, tooltip: str = "") -> None:
        """Append a row, attaching `obj` to its first cell."""
        row = self.rowCount()
        self.insertRow(row)
        for col, text in enumerate(values):
            item = QTableWidgetItem(text)
            if tooltip:
                item.setToolTip(tooltip)
            if col == 0:
                item.setData(Qt.ItemDataRole.UserRole, obj)
            self.setItem(row, col, item)


class AvailableDevicesPane(QWidget):
    """Left pane: every device offered by the installed device adapters."""

    deviceSelected = pyqtSignal(object)  # AvailableDevice | None

    HEADERS = ("Module", "Adapter", "Type")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._devices: list[AvailableDevice] = []
        self._hub_libraries: set[str] = set()

        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter devices…")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)

        self._type = QComboBox()
        self._type.currentIndexChanged.connect(self._apply_filter)

        self._hub_children = QCheckBox("Show hub children")
        self._hub_children.setToolTip(
            "Uncheck to collapse each hub library down to just its hub device."
        )
        # Collapsed by default (as the Java wizard does): a hub library is
        # normally added via its hub, then its peripherals.
        self._hub_children.setChecked(False)
        self._hub_children.toggled.connect(self._apply_filter)

        self._table = _DeviceTable(self.HEADERS)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)

        t = theme()
        type_row = QHBoxLayout()
        type_row.setSpacing(t.sp_xs)
        type_row.addWidget(self._type, 1)
        type_row.addWidget(self._hub_children)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(t.sp_xs)
        layout.addWidget(pane_title("Available Devices"))
        layout.addWidget(self._filter)
        layout.addLayout(type_row)
        layout.addWidget(self._table, 1)

    # ── data ──────────────────────────────────────────────────────

    def set_devices(self, devices: Sequence[AvailableDevice]) -> None:
        """Populate the pane with the model's available devices.

        Devices are grouped by library with the hub first, so the intended
        "add the hub, then pick its peripherals" flow is the obvious one.
        """
        self._devices = sorted(
            devices,
            key=lambda d: (
                d.library.lower(),
                d.device_type is not DeviceType.Hub,
                d.adapter_name.lower(),
            ),
        )
        # Mirror the Java wizard's "compact" list (MicroscopeModel.
        # getAvailableDevicesCompact): a library that provides a hub offers its
        # other devices as peripherals of that hub, not as standalone entries.
        self._hub_libraries = {
            d.library for d in self._devices if d.device_type is DeviceType.Hub
        }
        self._rebuild_types()
        self._rebuild_table()
        self._apply_filter()

    def _is_hub_child(self, dev: AvailableDevice) -> bool:
        """Whether `dev` is reachable as a peripheral rather than on its own."""
        if dev.library_hub is not None:
            # discovered by querying an already-loaded hub
            return True
        # NOTE: pymmcore-plus never sets `library_hub` for devices reported by
        # getAvailableDevices (it keys the hub lookup by adapter_name as well as
        # library, which cannot match), so fall back to matching on library.
        return (
            dev.device_type is not DeviceType.Hub and dev.library in self._hub_libraries
        )

    def _rebuild_types(self) -> None:
        current = self._type.currentData()
        self._type.blockSignals(True)
        self._type.clear()
        self._type.addItem("All Types", None)
        for dt in sorted({d.device_type for d in self._devices}, key=lambda d: d.name):
            self._type.addItem(dt.name, dt)
        if (idx := self._type.findData(current)) >= 0:
            self._type.setCurrentIndex(idx)
        self._type.blockSignals(False)

    def _rebuild_table(self) -> None:
        self._table.setRowCount(0)
        for dev in self._devices:
            adapter = dev.adapter_name
            if dev.library_hub is not None:
                adapter = f"[{dev.library_hub.adapter_name}] {adapter}"
            self._table.add_row(
                (dev.library, adapter, dev.device_type.name), dev, dev.description
            )

    # ── filtering ─────────────────────────────────────────────────

    def _apply_filter(self) -> None:
        terms = self._filter.text().lower().split()
        dev_type = self._type.currentData()
        show_children = self._hub_children.isChecked()

        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            dev: AvailableDevice | None = (
                item.data(Qt.ItemDataRole.UserRole) if item else None
            )
            if dev is None:  # pragma: no cover
                continue
            hidden = (
                (dev_type is not None and dev.device_type is not dev_type)
                or (self._is_hub_child(dev) and not show_children)
                or not self._matches(row, terms)
            )
            self._table.setRowHidden(row, hidden)

    def _matches(self, row: int, terms: Sequence[str]) -> bool:
        if not terms:
            return True
        haystack = " ".join(
            item.text().lower()
            for col in range(self._table.columnCount())
            if (item := self._table.item(row, col))
        )
        return all(term in haystack for term in terms)

    # ── selection ─────────────────────────────────────────────────

    def clear_selection(self) -> None:
        """Deselect whatever row is currently selected, if any."""
        self._table.clearSelection()

    def _on_selection_changed(self) -> None:
        self.deviceSelected.emit(self._table.selected_object())


class InstalledDevicesPane(QWidget):
    """Middle pane: devices currently part of the configuration."""

    deviceSelected = pyqtSignal(object)  # Device | None
    removeRequested = pyqtSignal(object)  # Device

    HEADERS = ("Label", "Adapter", "Type")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._table = _DeviceTable(self.HEADERS)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)

        self._remove_btn = QPushButton("Remove")
        self._remove_btn.setProperty("variant", "danger")
        self._remove_btn.setEnabled(False)
        self._remove_btn.clicked.connect(self._emit_remove)

        t = theme()
        bottom = QHBoxLayout()
        bottom.setSpacing(t.sp_xs)
        bottom.addStretch()
        bottom.addWidget(self._remove_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(t.sp_xs)
        layout.addWidget(pane_title("Installed Devices"))
        layout.addWidget(self._table, 1)
        layout.addLayout(bottom)

    def set_devices(self, devices: Sequence[Device]) -> None:
        """Populate the pane with the model's installed devices."""
        selected = getattr(self._table.selected_object(), "name", None)
        self._table.setRowCount(0)
        for dev in devices:
            self._table.add_row(
                (dev.name, dev.adapter_name, dev.device_type.name),
                dev,
                dev.description,
            )
        self._reselect(selected)

    def _reselect(self, name: str | None) -> None:
        """Restore selection by device label after a rebuild."""
        if name is None:
            return
        for row in range(self._table.rowCount()):
            if (item := self._table.item(row, 0)) and item.text() == name:
                self._table.selectRow(row)
                return

    def clear_selection(self) -> None:
        """Deselect whatever row is currently selected, if any."""
        self._table.clearSelection()

    def _on_selection_changed(self) -> None:
        dev = self._table.selected_object()
        self._remove_btn.setEnabled(dev is not None)
        self.deviceSelected.emit(dev)

    def _emit_remove(self) -> None:
        if (dev := self._table.selected_object()) is not None:
            self.removeRequested.emit(dev)
