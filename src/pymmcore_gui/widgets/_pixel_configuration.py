from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pymmcore_widgets import PixelConfigurationWidget as _UpstreamPixelConfiguration
from pymmcore_widgets._icons import StandardIcon
from superqt.utils import signals_blocked

from pymmcore_gui._modern_gui._theme import qcolor, theme
from pymmcore_gui._pixel_calibration import PixelCalibrationResult
from pymmcore_gui._qt.QtCore import QEvent, QTimer, Signal
from pymmcore_gui._qt.QtWidgets import QSizePolicy
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
        # Keep the two configuration editors together on the left and give
        # calibration the wider right half of the page.
        splitter.insertWidget(2, self._calibration_panel)
        self._content_splitter = splitter
        # Let the splitter shrink both editor panes below their size hints so
        # those hints cannot override the intended 1:1:2 proportions.
        for pane in (splitter.widget(0), splitter.widget(1)):
            assert pane is not None
            policy = pane.sizePolicy()
            policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
            pane.setSizePolicy(policy)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 2)
        # Seed above every child's minimum size hint.  QSplitter preserves its
        # clamped initial sizes during later resizes, so smaller seed values
        # would lose the intended 1:1:2 proportions before the page is shown.
        splitter.setSizes([700, 700, 1400])

        self._calibration_panel.resultReady.connect(self._apply_calibration_result)
        self._calibration_panel.calibrationRunningChanged.connect(
            self._on_calibration_running_changed
        )
        self._px_table._table.itemSelectionChanged.connect(
            self._sync_calibration_target
        )
        self._px_table._table.itemChanged.connect(self._sync_calibration_target)
        self._px_table.valueChanged.connect(self._sync_calibration_target)
        # Property values now live in the selected resolution preset itself.
        # Keep the calibration target current after value edits and after the
        # shared property set is changed from either toolbar action.
        self._value_table.valueEdited.connect(self._sync_calibration_target)
        self._value_table.act_edit_props.triggered.connect(
            self._sync_calibration_target
        )
        self._value_table.act_remove_props.triggered.connect(
            self._sync_calibration_target
        )
        self._mmc.events.systemConfigurationLoaded.connect(
            self._sync_calibration_target
        )
        self._precision_timer = QTimer(self)
        self._precision_timer.setSingleShot(True)
        self._precision_timer.timeout.connect(self._set_numeric_precision)
        model = self._px_table.table().model()
        if model is not None:
            model.rowsInserted.connect(self._schedule_precision_update)

        self._set_numeric_precision()
        self._apply_themed_action_icons()
        self._sync_calibration_target()

    def _apply_themed_action_icons(self) -> None:
        """Apply semantic theme colors to the pixel-configuration actions."""
        green = qcolor(theme().status_green).name()
        red = qcolor(theme().status_red).name()
        self._px_table.act_remove_row.setIcon(StandardIcon.DELETE.icon(red))
        self._px_table.act_clear.setIcon(StandardIcon.DELETE_ALL.icon(red))
        self._value_table.act_edit_props.setIcon(StandardIcon.PROPERTY_ADD.icon(green))
        self._value_table.act_remove_props.setIcon(StandardIcon.DELETE.icon(red))

    def _schedule_precision_update(self, *_: object) -> None:
        # A child-owned timer is cancelled when this widget is destroyed.  A
        # static singleShot can otherwise retain this Python wrapper past C++
        # teardown and call into deleted table widgets under PySide6.
        self._precision_timer.start(0)

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

        Deliberately does *not* also require the editor's current property set
        to match what's saved to core: those values only matter when they are
        actually applied before a Snap/Test-frame/Start-calibration run, where
        a mismatch already surfaces as its own clear error. Gating calibration
        itself on it too would block unsaved or property-free resolutions, such
        as a manually swapped objective with no state device tracking it.
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
        self._value_table.setEnabled(not running)
        self._affine_table.setEnabled(not running)
        self._apply_btn.setEnabled(not running and not self.isClean())
        self.calibrationRunningChanged.emit(running)

    def shutdownCalibration(self) -> None:
        """Cancel calibration and wait until its hardware state is restored."""
        self._calibration_panel.shutdownCalibration()

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        self.shutdownCalibration()
        super().closeEvent(a0)

    def changeEvent(self, a0: QEvent | None) -> None:
        super().changeEvent(a0)
        if (
            a0 is not None
            and a0.type() == QEvent.Type.StyleChange
            and hasattr(self, "_value_table")
        ):
            self._apply_themed_action_icons()


__all__ = ["PixelConfigurationWidget"]
