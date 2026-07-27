"""Right-hand pane of the Hardware Setup tab.

Doubles as the *setup* pane for a device about to be added (label + pre-init
properties) and the *property* pane for a device already installed.
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from pymmcore_plus import DeviceType, Keyword, PropertyType

from pymmcore_gui._modern_gui._theme import theme
from pymmcore_gui._qt.QtCore import Qt, pyqtSignal
from pymmcore_gui._qt.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ._panes import pane_title

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pymmcore_plus.model import AvailableDevice, Device, Property


def _editor_for(prop: Property, on_change: Callable[[str], None]) -> QWidget:
    """Build the most appropriate editor widget for `prop`."""
    if prop.is_read_only:
        ro = QLineEdit(str(prop.value))
        ro.setReadOnly(True)
        ro.setEnabled(False)
        return ro

    if prop.allowed_values:
        combo = QComboBox()
        combo.addItems([str(v) for v in prop.allowed_values])
        combo.setCurrentText(str(prop.value))
        combo.currentTextChanged.connect(on_change)
        return combo

    if prop.has_limits and prop.property_type is PropertyType.Integer:
        spin = QSpinBox()
        spin.setRange(int(prop.lower_limit), int(prop.upper_limit))
        with suppress(TypeError, ValueError):
            spin.setValue(int(float(prop.value)))
        spin.valueChanged.connect(lambda v: on_change(str(v)))
        return spin

    if prop.has_limits and prop.property_type is PropertyType.Float:
        dspin = QDoubleSpinBox()
        dspin.setRange(prop.lower_limit, prop.upper_limit)
        dspin.setDecimals(3)
        with suppress(TypeError, ValueError):
            dspin.setValue(float(prop.value))
        dspin.valueChanged.connect(lambda v: on_change(str(v)))
        return dspin

    line = QLineEdit(str(prop.value))
    line.editingFinished.connect(lambda: on_change(line.text()))
    return line


class DeviceSetupPane(QWidget):
    """Setup / property pane for the selected device."""

    addRequested = pyqtSignal(str)  # label typed by the user
    addConfirmed = pyqtSignal()  # pre-init configured, finish adding
    addCancelled = pyqtSignal()
    propertyChanged = pyqtSignal(object, str)  # (Property, new value)
    delayChanged = pyqtSignal(object, float)  # (Device, delay in ms)
    renameRequested = pyqtSignal(object, str)  # (Device, new label)
    stateLabelChanged = pyqtSignal(object, int, str)  # (Device, state, new label)
    portSelected = pyqtSignal(str, str)  # (serial adapter name, library)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._title = pane_title("Device Setup")

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(theme().sp_sm)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(self._body)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme().sp_xs)
        layout.addWidget(self._title)
        layout.addWidget(scroll, 1)

        self.show_empty()

    # ── states ────────────────────────────────────────────────────

    def show_empty(self) -> None:
        """Nothing selected."""
        self._reset("Device Setup")
        hint = QLabel("Select a device to view its settings.")
        hint.setWordWrap(True)
        hint.setEnabled(False)
        self._body_layout.addWidget(hint)
        self._body_layout.addStretch()

    def show_available(self, dev: AvailableDevice, suggested_label: str) -> None:
        """An available device is selected — offer to add it."""
        self._reset(dev.adapter_name)
        self._add_info(dev.library, dev.adapter_name, dev.device_type.name)
        self._add_description(dev.description)

        label_edit = QLineEdit(suggested_label)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Label:", label_edit)
        self._body_layout.addLayout(form)

        def _emit_add() -> None:
            self.addRequested.emit(label_edit.text())

        add_btn = QPushButton("Add Device")
        add_btn.setProperty("variant", "primary")
        add_btn.clicked.connect(_emit_add)
        label_edit.returnPressed.connect(_emit_add)

        self._body_layout.addStretch()
        self._body_layout.addWidget(add_btn)

    def show_pending(
        self,
        dev: Device,
        serial_devices: Sequence[Device] = (),
        port_device: Device | None = None,
    ) -> None:
        """Device loaded but not yet initialized — configure pre-init props.

        If the device declares a "Port" property, `serial_devices` populates the
        port chooser and `port_device` (once loaded) contributes its own serial
        settings.
        """
        self._reset(dev.name)
        self._add_info(
            dev.library, dev.adapter_name, dev.device_type.name, dev.parent_label
        )

        pre_init = [p for p in dev.properties if p.is_pre_init]
        if pre_init:
            self._add_section("Setup properties")
            self._add_properties(pre_init, serial_devices)
        else:  # pragma: no cover - page adds directly in this case
            self._body_layout.addWidget(QLabel("No pre-initialization settings."))

        if port_device is not None:
            self._add_section(f"Serial port — {port_device.name}")
            self._add_properties(
                [p for p in port_device.properties if not p.is_read_only]
            )

        row = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.addCancelled.emit)
        confirm = QPushButton("Finish")
        confirm.setProperty("variant", "primary")
        confirm.clicked.connect(self.addConfirmed.emit)
        row.addStretch()
        row.addWidget(cancel)
        row.addWidget(confirm)

        self._body_layout.addStretch()
        self._body_layout.addLayout(row)

    def show_installed(self, dev: Device, port_device: Device | None = None) -> None:
        """An installed device is selected — show its *setup* settings.

        These are the values needed to configure the device (pre-init
        properties such as Port, plus the serial settings and delay) — not the
        runtime properties shown by a property browser.
        """
        self._reset(dev.name)
        self._add_info(
            dev.library, dev.adapter_name, dev.device_type.name, dev.parent_label
        )
        self._add_description(dev.description)

        # editable label — renaming reloads the device in the core
        label_edit = QLineEdit(dev.name)

        def _commit_rename() -> None:
            if (new := label_edit.text().strip()) and new != dev.name:
                self.renameRequested.emit(dev, new)

        label_edit.editingFinished.connect(_commit_rename)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Label:", label_edit)
        self._body_layout.addLayout(form)

        if setup_props := [p for p in dev.properties if p.is_pre_init]:
            self._add_section("Setup properties")
            self._add_properties(setup_props)
        else:
            none_lbl = QLabel("No setup properties required.")
            none_lbl.setEnabled(False)
            self._body_layout.addWidget(none_lbl)

        # State devices (filter wheels, objective turrets, ...) have discrete
        # positions that can be given friendly names — e.g. an objective
        # turret's positions named "10X"/"20X"/"40X" instead of "State-0" etc.
        if dev.device_type == DeviceType.StateDevice and dev.labels:
            self._add_section("State Labels")
            self._add_state_labels(dev)

        # a device with a "Port" is configured through that serial device
        if port_device is not None:
            self._add_section(f"Serial port — {port_device.name}")
            self._add_properties(
                [p for p in port_device.properties if not p.is_read_only]
            )

        if dev.uses_delay:
            self._add_section("Timing")
            delay = QDoubleSpinBox()
            delay.setRange(0.0, 100_000.0)
            delay.setDecimals(1)
            delay.setSuffix(" ms")
            delay.setValue(dev.delay_ms)
            delay.valueChanged.connect(lambda v: self.delayChanged.emit(dev, v))
            form = QFormLayout()
            form.setContentsMargins(0, 0, 0, 0)
            form.addRow("Delay:", delay)
            self._body_layout.addLayout(form)

        self._body_layout.addStretch()

    # ── building blocks ───────────────────────────────────────────

    def _reset(self, title: str) -> None:
        """Clear the body and set the pane title."""
        self._title.setText(title)
        _clear_layout(self._body_layout)

    def _add_section(self, title: str) -> None:
        """Add a sub-heading separating groups of settings."""
        self._body_layout.addWidget(QLabel(title))

    def _add_info(
        self, module: str, adapter: str, dev_type: str, parent: str = ""
    ) -> None:
        rows = [("Module:", module), ("Adapter:", adapter), ("Type:", dev_type)]
        if parent:
            rows.append(("Hub:", parent))
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(theme().sp_xxs)
        for name, value in rows:
            val = QLabel(value)
            val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow(name, val)
        self._body_layout.addLayout(form)

    def _add_description(self, description: str) -> None:
        if not description:
            return
        desc = QLabel(description)
        desc.setWordWrap(True)
        desc.setEnabled(False)
        self._body_layout.addWidget(desc)

    def _add_state_labels(self, dev: Device) -> None:
        """Editable State/Label table for a state device."""
        table = QTableWidget(len(dev.labels), 2)
        table.setHorizontalHeaderLabels(["State", "Label"])
        if (hdr := table.horizontalHeader()) is not None:
            hdr.setStretchLastSection(True)
        if (vh := table.verticalHeader()) is not None:
            vh.setVisible(False)
        table.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked
            | QTableWidget.EditTrigger.EditKeyPressed
        )

        # populate without emitting itemChanged for the initial (unedited) values
        table.blockSignals(True)
        try:
            for state, label in enumerate(dev.labels):
                state_item = QTableWidgetItem(str(state))
                state_item.setFlags(state_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(state, 0, state_item)
                table.setItem(state, 1, QTableWidgetItem(label))
        finally:
            table.blockSignals(False)

        def _on_item_changed(item: QTableWidgetItem) -> None:
            if item.column() == 1:
                self.stateLabelChanged.emit(dev, item.row(), item.text())

        table.itemChanged.connect(_on_item_changed)
        # Give it priority over show_installed()'s trailing addStretch() (an
        # unstretched widget leaves all leftover vertical space to that
        # stretch instead, which reads fine when a "Timing" section follows
        # but looks like a bug -- a small table over a big dead area -- when
        # this is the last section, as it is for most state devices).
        self._body_layout.addWidget(table, 1)

    def _add_properties(
        self, props: Sequence[Property], serial_devices: Sequence[Device] = ()
    ) -> None:
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(theme().sp_xxs)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for prop in props:
            editor: QWidget
            if prop.name == Keyword.Port and serial_devices:
                editor = self._port_chooser(prop, serial_devices)
            else:

                def _on_change(value: str, p: Property = prop) -> None:
                    self.propertyChanged.emit(p, value)

                editor = _editor_for(prop, _on_change)
            form.addRow(f"{prop.name}:", editor)
        self._body_layout.addLayout(form)

    def _port_chooser(
        self, prop: Property, serial_devices: Sequence[Device]
    ) -> QComboBox:
        """Combo of available serial devices.

        MMCore only offers *loaded* serial devices as allowed values for "Port",
        so the choices come from the model's available devices instead.
        """
        combo = QComboBox()
        combo.addItem("", "")
        for dev in serial_devices:
            combo.addItem(dev.adapter_name, dev.library)
        if prop.value:
            combo.setCurrentText(str(prop.value))
        # connect after populating so seeding the value doesn't emit
        combo.currentIndexChanged.connect(
            lambda _i: self.portSelected.emit(
                combo.currentText(), combo.currentData() or ""
            )
        )
        return combo


def _clear_layout(layout: QLayout) -> None:
    """Remove and delete every item in `layout`, recursing into sub-layouts.

    Note: iterate on `count()` rather than until `takeAt` returns None —
    QFormLayout logs a warning when asked for an index it no longer has.
    """
    while layout.count():
        if (item := layout.takeAt(0)) is None:  # pragma: no cover
            break
        if w := item.widget():
            w.deleteLater()
        elif child := item.layout():
            _clear_layout(child)
            child.deleteLater()
