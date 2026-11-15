"""Stitching settings dialog."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from pymmcore_gui._qt.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from pymmcore_gui._stitching._models import (
    Alignment,
    Blending,
    OutputFormat,
    StitchSettingsV1,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

AUTO = "Auto"
FIRST_CHANNEL = "First channel"

BLENDING_LABELS: dict[Blending, str] = {
    Blending.SMOOTH: "Smooth (weighted average)",
    Blending.MAX: "Maximum",
    Blending.AVERAGE: "Average",
}

FORMAT_LABELS: dict[OutputFormat, str] = {
    OutputFormat.SAME: "Same as the acquisition",
    OutputFormat.OME_ZARR: "OME-Zarr",
    OutputFormat.OME_TIFF: "OME-TIFF",
}

Binning = Literal[1, 2, 4, 8]
BINNINGS: tuple[Binning, ...] = (1, 2, 4, 8)


class StitchingSettingsDialog(QDialog):
    """Edit a `StitchSettingsV1`: alignment, blending, output and performance.

    `channels` are the sequence's channel names, offered for registration;
    a saved channel the sequence no longer has is kept and shown as missing.
    """

    def __init__(
        self,
        settings: StitchSettingsV1,
        channels: Sequence[str] = (),
        *,
        max_threads: int = 1,
        title: str = "Stitching Settings",
        accept_text: str = "OK",
        note: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self._channels = list(dict.fromkeys(channels))

        self.register_radio = QRadioButton("Refine with registration")
        self.register_radio.setToolTip(
            "Start from the stage positions, then correct each tile's position "
            "from the image content where neighbouring tiles overlap."
        )
        self.stage_radio = QRadioButton("Use stage positions only")
        self.stage_radio.setToolTip(
            "Place every tile exactly where the stage says it was. Fast, and the "
            "only choice when the tiles do not overlap."
        )
        self._alignment_group = QButtonGroup(self)
        self._alignment_group.addButton(self.register_radio)
        self._alignment_group.addButton(self.stage_radio)

        self.channel_combo = QComboBox()
        self.binning_combo = QComboBox()
        self.binning_combo.addItem(AUTO, None)
        for b in BINNINGS:
            self.binning_combo.addItem(str(b), b)
        self.binning_combo.setToolTip(
            "Registration works on binned copies of the tiles: faster, and "
            "usually as accurate. Auto bins to about 1024 pixels per side."
        )
        self.max_projection_check = QCheckBox(
            "Register z-stacks on their max projection"
        )
        self.max_projection_check.setToolTip(
            "Much faster than registering in 3D. Corrects X and Y only; Z stays "
            "where the stage put it."
        )
        self.timepoint_combo = QComboBox()
        self.timepoint_combo.addItem("First timepoint only", False)
        self.timepoint_combo.addItem("Every timepoint", True)
        self.timepoint_combo.setToolTip(
            "Every timepoint also corrects drift between tiles over time, at the "
            "cost of one registration per timepoint."
        )
        self.reject_check = QCheckBox("Reject poor matches below")
        self.reject_check.setToolTip(
            "Overlaps whose match quality is below the threshold are ignored. A "
            "tile left without any match stays at its stage position."
        )
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setDecimals(2)
        reject_row = QHBoxLayout()
        reject_row.setContentsMargins(0, 0, 0, 0)
        reject_row.addWidget(self.reject_check)
        reject_row.addWidget(self.threshold_spin)
        reject_row.addStretch()

        self._registration_widgets: tuple[QWidget, ...] = (
            self.channel_combo,
            self.binning_combo,
            self.max_projection_check,
            self.timepoint_combo,
            self.reject_check,
            self.threshold_spin,
        )

        align_form = QFormLayout()
        align_form.addRow(self.register_radio)
        align_form.addRow(self.stage_radio)
        align_form.addRow("Registration channel:", self.channel_combo)
        align_form.addRow("Registration binning:", self.binning_combo)
        align_form.addRow("Time-lapse:", self.timepoint_combo)
        align_form.addRow(self.max_projection_check)
        align_form.addRow(reject_row)
        align_box = QGroupBox("Alignment")
        align_box.setLayout(align_form)

        self.blending_combo = QComboBox()
        for blending, label in BLENDING_LABELS.items():
            self.blending_combo.addItem(label, blending)
        self.blend_width_spin = QDoubleSpinBox()
        self.blend_width_spin.setRange(0.0, 10_000.0)
        self.blend_width_spin.setDecimals(1)
        self.blend_width_spin.setSuffix(" µm")
        # the minimum reads "Auto": multiview-stitcher's own default width
        self.blend_width_spin.setSpecialValueText(AUTO)
        self.blend_width_spin.setToolTip(
            "Width of the band at tile edges where neighbours are blended."
        )
        blend_form = QFormLayout()
        blend_form.addRow("Blending:", self.blending_combo)
        blend_form.addRow("Blend width:", self.blend_width_spin)
        blend_box = QGroupBox("Blending")
        blend_box.setLayout(blend_form)

        self.format_combo = QComboBox()
        for fmt, label in FORMAT_LABELS.items():
            self.format_combo.addItem(label, fmt)
        self.format_combo.setToolTip(
            "OME-Zarr is written directly. OME-TIFF is written as OME-Zarr first "
            "and then copied into one tiled, pyramidal OME-TIFF file, which "
            "takes a little longer and briefly needs twice the disk space."
        )
        self.open_check = QCheckBox("Open the mosaic in a viewer when done")
        output_note = QLabel(
            "Mosaics are saved next to the acquisition, as "
            "<i>&lt;name&gt;_stitched</i>. Existing files are never overwritten."
        )
        output_note.setWordWrap(True)
        format_form = QFormLayout()
        format_form.setContentsMargins(0, 0, 0, 0)
        format_form.addRow("Format:", self.format_combo)
        output_layout = QVBoxLayout()
        output_layout.addLayout(format_form)
        output_layout.addWidget(self.open_check)
        output_layout.addWidget(output_note)
        output_box = QGroupBox("Output")
        output_box.setLayout(output_layout)

        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(0, max(max_threads, 1))
        self.threads_spin.setSpecialValueText(AUTO)
        self.threads_spin.setToolTip(
            "Threads the stitching process may use for everything. Auto uses "
            "half the CPU cores, leaving the rest to the acquisition."
        )
        self.low_priority_check = QCheckBox("Run at low CPU priority")
        perf_form = QFormLayout()
        perf_form.addRow("Worker threads:", self.threads_spin)
        perf_form.addRow(self.low_priority_check)
        perf_box = QGroupBox("Performance")
        perf_box.setLayout(perf_form)

        self.defaults_btn = QPushButton("Restore Defaults")
        self.cancel_btn = QPushButton("Cancel")
        self.ok_btn = QPushButton(accept_text)
        self.ok_btn.setDefault(True)
        self.ok_btn.setProperty("variant", "primary")
        buttons = QHBoxLayout()
        buttons.addWidget(self.defaults_btn)
        buttons.addStretch()
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.ok_btn)

        layout = QVBoxLayout(self)
        self.note_label = QLabel(note)
        self.note_label.setWordWrap(True)
        self.note_label.setVisible(bool(note))
        layout.addWidget(self.note_label)
        layout.addWidget(align_box)
        layout.addWidget(blend_box)
        layout.addWidget(output_box)
        layout.addWidget(perf_box)
        layout.addLayout(buttons)

        self.register_radio.toggled.connect(self._sync_enabled)
        self.reject_check.toggled.connect(self._sync_enabled)
        self.defaults_btn.clicked.connect(self._restore_defaults)
        self.cancel_btn.clicked.connect(self.reject)
        self.ok_btn.clicked.connect(self.accept)

        self.setValue(settings)

    def setValue(self, settings: StitchSettingsV1) -> None:
        """Show `settings` in the editors."""
        self._settings = settings
        register = settings.alignment == Alignment.REGISTER
        self.register_radio.setChecked(register)
        self.stage_radio.setChecked(not register)

        self.channel_combo.clear()
        self.channel_combo.addItem(FIRST_CHANNEL, None)
        for name in self._channels:
            self.channel_combo.addItem(name, name)
        if (reg := settings.reg_channel) is not None:
            if reg not in self._channels:
                self.channel_combo.addItem(f"{reg} (not in this sequence)", reg)
            self.channel_combo.setCurrentIndex(self.channel_combo.findData(reg))

        self.binning_combo.setCurrentIndex(
            max(self.binning_combo.findData(settings.reg_binning), 0)
        )
        self.max_projection_check.setChecked(settings.z_max_projection)
        self.timepoint_combo.setCurrentIndex(
            self.timepoint_combo.findData(settings.register_every_timepoint)
        )
        self.reject_check.setChecked(settings.reject_poor_matches)
        self.threshold_spin.setValue(settings.quality_threshold)

        self.blending_combo.setCurrentIndex(
            self.blending_combo.findData(settings.blending)
        )
        self.blend_width_spin.setValue(settings.blend_width_um or 0.0)
        self.open_check.setChecked(settings.open_when_done)
        self.format_combo.setCurrentIndex(
            self.format_combo.findData(settings.output_format)
        )
        self.threads_spin.setValue(settings.worker_threads or 0)
        self.low_priority_check.setChecked(settings.low_priority)
        self._sync_enabled()

    def value(self) -> StitchSettingsV1:
        """The settings as edited."""
        binning = self.binning_combo.currentData()
        width = self.blend_width_spin.value()
        threads = self.threads_spin.value()
        return self._settings.model_copy(
            update={
                "alignment": (
                    Alignment.REGISTER
                    if self.register_radio.isChecked()
                    else Alignment.STAGE
                ),
                "reg_channel": self.channel_combo.currentData(),
                "reg_binning": cast("Binning | None", binning),
                "z_max_projection": self.max_projection_check.isChecked(),
                "register_every_timepoint": bool(self.timepoint_combo.currentData()),
                "reject_poor_matches": self.reject_check.isChecked(),
                "quality_threshold": self.threshold_spin.value(),
                "blending": Blending(self.blending_combo.currentData()),
                "blend_width_um": width if width > 0 else None,
                "open_when_done": self.open_check.isChecked(),
                "output_format": OutputFormat(self.format_combo.currentData()),
                "worker_threads": threads if threads > 0 else None,
                "low_priority": self.low_priority_check.isChecked(),
            }
        )

    def _restore_defaults(self) -> None:
        self.setValue(StitchSettingsV1())

    def _sync_enabled(self, *_: object) -> None:
        register = self.register_radio.isChecked()
        for widget in self._registration_widgets:
            widget.setEnabled(register)
        self.threshold_spin.setEnabled(register and self.reject_check.isChecked())
