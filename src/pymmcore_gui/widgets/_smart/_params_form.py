"""A form generated from a Smart Microscopy script's ``PARAMETERS``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pymmcore_gui._qt.QtCore import QSignalBlocker, Signal
from pymmcore_gui._qt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QWidget,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymmcore_plus.smart import ParamDef

_INT_LIMIT = 2**31 - 1
_FLOAT_LIMIT = 1e12


class ParamsForm(QWidget):
    """One editor per parameter; `values` returns them, ready for the script."""

    valuesChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QFormLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self._params: tuple[ParamDef, ...] = ()
        self._editors: dict[str, QWidget] = {}
        self._empty = QLabel("This script has no parameters.")
        self._empty.setEnabled(False)
        self._layout.addRow(self._empty)

    def set_params(
        self, params: Sequence[ParamDef], values: dict[str, Any] | None = None
    ) -> None:
        """Rebuild the form for *params*, showing *values* (defaults otherwise)."""
        while self._layout.rowCount():
            self._layout.removeRow(0)  # deletes the row's widgets
        self._params = tuple(params)
        self._editors = {}
        values = values or {}
        if not self._params:
            self._empty = QLabel("This script has no parameters.")
            self._empty.setEnabled(False)
            self._layout.addRow(self._empty)
            return
        editors: dict[str, QWidget] = {}
        for param in self._params:
            editor = self._make_editor(param)
            editors[param.name] = editor
            # Initial values are not user edits: no valuesChanged for them.
            with QSignalBlocker(editor):
                self._set_editor_value(
                    param, editor, values.get(param.name, param.default)
                )
            label = QLabel(param.label or param.name)
            if param.tooltip:
                label.setToolTip(param.tooltip)
                editor.setToolTip(param.tooltip)
            self._layout.addRow(label, editor)
        self._editors = editors

    def values(self) -> dict[str, Any]:
        """Current values, keyed by parameter name."""
        return {
            p.name: self._editor_value(p, self._editors[p.name]) for p in self._params
        }

    def reset(self) -> None:
        """Put every editor back to its parameter's default."""
        for param in self._params:
            self._set_editor_value(param, self._editors[param.name], param.default)

    def editor(self, name: str) -> QWidget:
        return self._editors[name]

    # ---------------------------------------------------------------- editors

    def _make_editor(self, param: ParamDef) -> QWidget:
        editor: QWidget
        if param.kind == "choice":
            combo = QComboBox()
            for choice in param.choices or ():
                combo.addItem(str(choice), choice)
            combo.currentIndexChanged.connect(self.valuesChanged)
            editor = combo
        elif param.kind == "bool":
            check = QCheckBox()
            check.toggled.connect(self.valuesChanged)
            editor = check
        elif param.kind == "int":
            spin = QSpinBox()
            spin.setRange(
                int(param.min) if param.min is not None else -_INT_LIMIT,
                int(param.max) if param.max is not None else _INT_LIMIT,
            )
            if param.step is not None:
                spin.setSingleStep(max(1, int(param.step)))
            spin.valueChanged.connect(self.valuesChanged)
            editor = spin
        elif param.kind == "float":
            dspin = QDoubleSpinBox()
            dspin.setDecimals(_decimals(param))
            dspin.setRange(
                param.min if param.min is not None else -_FLOAT_LIMIT,
                param.max if param.max is not None else _FLOAT_LIMIT,
            )
            if param.step is not None:
                dspin.setSingleStep(param.step)
            dspin.valueChanged.connect(self.valuesChanged)
            editor = dspin
        else:
            line = QLineEdit()
            line.textChanged.connect(self.valuesChanged)
            editor = line
        editor.setObjectName(f"param_{param.name}")
        # A spin box sizes itself to its range's longest value; an unbounded
        # one (±1e12) would force the whole settings column wide.
        editor.setMinimumWidth(80)
        return editor

    @staticmethod
    def _set_editor_value(param: ParamDef, editor: QWidget, value: Any) -> None:
        if isinstance(editor, QComboBox):
            index = editor.findData(value)
            editor.setCurrentIndex(max(index, 0))
        elif isinstance(editor, QCheckBox):
            editor.setChecked(bool(value))
        elif isinstance(editor, QSpinBox):
            editor.setValue(int(value))
        elif isinstance(editor, QDoubleSpinBox):
            editor.setValue(float(value))
        elif isinstance(editor, QLineEdit):
            editor.setText(str(value))

    @staticmethod
    def _editor_value(param: ParamDef, editor: QWidget) -> Any:
        if isinstance(editor, QComboBox):
            return editor.currentData()
        if isinstance(editor, QCheckBox):
            return editor.isChecked()
        if isinstance(editor, (QSpinBox, QDoubleSpinBox)):
            return editor.value()
        if isinstance(editor, QLineEdit):
            return editor.text()
        return param.default  # pragma: no cover


def _decimals(param: ParamDef) -> int:
    """Enough decimals to show the step and the default exactly (2..6)."""
    decimals = 2
    for value in (param.step, param.default, param.min, param.max):
        if isinstance(value, float) and value != int(value):
            text = f"{value:.6f}".rstrip("0")
            decimals = max(decimals, len(text.split(".")[1]))
    return min(decimals, 6)
