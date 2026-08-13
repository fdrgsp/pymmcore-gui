from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pymmcore_widgets import PixelConfigurationWidget as _UpstreamPixelConfiguration
from superqt.utils import signals_blocked

from pymmcore_gui._pixel_calibration import PixelCalibrationResult
from pymmcore_gui._qt.QtCore import QTimer, Signal
from pymmcore_gui.widgets._pixel_calibration_panel import (
    CalibrationTarget,
    PixelCalibrationPanel,
)

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.model import PixelSizePreset

    from pymmcore_gui._qt.QtGui import QCloseEvent
    from pymmcore_gui._qt.QtWidgets import (
        QDoubleSpinBox,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )


_CALIBRATION_DECIMALS = 8


class PixelConfigurationWidget(_UpstreamPixelConfiguration):
    """Pixel-configuration editor with automatic camera/stage calibration."""

    calibrationRunningChanged = Signal(bool)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        mmcore: CMMCorePlus | None = None,
    ) -> None:
        super().__init__(parent, mmcore=mmcore)
        self._calibration_panel = PixelCalibrationPanel(self._mmc, self)

        layout = cast("QVBoxLayout", self.layout())
        layout_item = layout.itemAt(0)
        splitter = cast(
            "QSplitter | None",
            layout_item.widget() if layout_item is not None else None,
        )
        if (
            splitter is None
        ):  # pragma: no cover - protects against upstream layout drift
            raise RuntimeError("PixelConfigurationWidget has no content splitter")
        splitter.insertWidget(1, self._calibration_panel)
        self._content_splitter = splitter
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 3)
        splitter.setSizes([300, 700, 430])

        self._calibration_panel.resultReady.connect(self._apply_calibration_result)
        self._calibration_panel.calibrationRunningChanged.connect(
            self._on_calibration_running_changed
        )
        self._px_table._table.itemSelectionChanged.connect(
            self._sync_calibration_target
        )
        self._px_table._table.itemChanged.connect(self._sync_calibration_target)
        self._px_table.valueChanged.connect(self._sync_calibration_target)
        self._props_selector.valueChanged.connect(self._sync_calibration_target)
        self._mmc.events.systemConfigurationLoaded.connect(
            self._sync_calibration_target
        )
        model = self._px_table.table().model()
        if model is not None:
            model.rowsInserted.connect(self._schedule_precision_update)

        self._set_numeric_precision()
        self._sync_calibration_target()

    def _schedule_precision_update(self, *_: object) -> None:
        QTimer.singleShot(0, self._set_numeric_precision)

    def _set_numeric_precision(self) -> None:
        for row in range(self._px_table.table().rowCount()):
            if spin := cast(
                "QDoubleSpinBox | None", self._px_table.table().cellWidget(row, 1)
            ):
                spin.setDecimals(_CALIBRATION_DECIMALS)
                spin.setSingleStep(0.0001)
        for row in range(2):
            for column in range(3):
                if spin := cast(
                    "QDoubleSpinBox | None",
                    self._affine_table.cellWidget(row, column),
                ):
                    spin.setDecimals(_CALIBRATION_DECIMALS)
                    spin.setSingleStep(0.0001)

    @staticmethod
    def _preset_settings(
        preset: PixelSizePreset,
    ) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            sorted(
                (
                    str(setting.device_name),
                    str(setting.property_name),
                    str(setting.property_value),
                )
                for setting in preset.settings
            )
        )

    def _binding_is_saved(self, resolution_id: str) -> bool:
        """Whether ``resolution_id`` is a real, already-saved core pixel-size config.

        Deliberately does *not* also require the property selector's current
        device/property list to match what's saved to core: that list only
        matters when it's actually used (to apply this resolution's optical
        state before a Snap/Test-frame/Start-calibration run), where a
        mismatch already surfaces as its own clear error. Gating calibration
        itself on it too just blocked runs whenever the property selector's
        UI state was unsaved or empty -- e.g. a resolution with no
        identifying property at all, such as a manually-swapped objective
        with no state device tracking it.
        """
        try:
            return resolution_id in self._mmc.getAvailablePixelSizeConfigs()
        except Exception:
            return False

    def _selected_preset(self) -> tuple[int, PixelSizePreset] | None:
        items = self._px_table._table.selectedItems()
        if len(items) != 1:
            return None
        row = items[0].row()
        preset = self._resID_map.get(row)
        return (row, preset) if preset is not None else None

    def _sync_calibration_target(self, *_: object) -> None:
        selected = self._selected_preset()
        if selected is None:
            self._calibration_panel.setTarget(None)
            return
        row, preset = selected
        del row
        settings = self._preset_settings(preset)
        self._calibration_panel.setTarget(
            CalibrationTarget(
                resolution_id=str(preset.name),
                settings=settings,
                binding_is_saved=self._binding_is_saved(str(preset.name)),
            )
        )

    def _apply_calibration_result(self, payload: object, resolution_id: str) -> None:
        """Populate the selected editor row and make the widget dirty."""
        if not isinstance(payload, PixelCalibrationResult):  # pragma: no cover
            return
        selected = self._selected_preset()
        if selected is None:
            return
        row, preset = selected
        if str(preset.name) != resolution_id:
            return
        if self._preset_settings(preset) != payload.fingerprint.config_settings:
            return

        matrix = payload.raw_matrix
        affine = (
            float(matrix[0, 0]),
            float(matrix[0, 1]),
            0.0,
            float(matrix[1, 0]),
            float(matrix[1, 1]),
            0.0,
        )
        pixel_size = float(payload.raw_pixel_size_um)
        preset.pixel_size_um = pixel_size
        preset.affine = affine

        spin = cast("QDoubleSpinBox", self._px_table.table().cellWidget(row, 1))
        with signals_blocked(spin):
            spin.setValue(pixel_size)
        with signals_blocked(self._affine_table):
            self._affine_table.setValue(affine)
        self._update_clean_state()

    def _on_calibration_running_changed(self, running: bool) -> None:
        self._px_table.setEnabled(not running)
        self._props_selector.setEnabled(
            not running and self._selected_preset() is not None
        )
        self._affine_table.setEnabled(not running)
        self._apply_btn.setEnabled(not running and not self.isClean())
        self.calibrationRunningChanged.emit(running)

    def shutdownCalibration(self) -> None:
        """Cancel calibration and wait until its hardware state is restored."""
        self._calibration_panel.shutdownCalibration()

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        self.shutdownCalibration()
        super().closeEvent(a0)


__all__ = ["PixelConfigurationWidget"]
