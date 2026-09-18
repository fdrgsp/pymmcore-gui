"""Application-wide preferences, independent of any hardware configuration.

Reachable from the gear button in the main window's top toolbar. Kept
deliberately separate from the Configurations page: that page edits the
hardware ``.cfg`` file (with its own dirty-tracking and Save button), while
these settings apply across every configuration and are persisted straight to
the user's ``pmm_settings.json`` (see ``pymmcore_gui._settings``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from superqt.iconify import QIconifyIcon

from pymmcore_gui._modern_gui._theme import qcolor, theme
from pymmcore_gui._qt.QtCore import QEvent, QSize, Qt
from pymmcore_gui._qt.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from pymmcore_gui._settings import Settings

_MIN_WIDTH: Final = 420
_MAX_MEMORY_GB_RANGE: Final = (0.1, 256.0)


class PreferencesButton(QPushButton):
    """Gear button that opens the :class:`PreferencesDialog`.

    Same "chrome, rebuild the icon on a theme change" treatment as the other
    small icon buttons in the top toolbar (``NotificationBellButton``,
    ``LayoutMenuButton``).
    """

    _ICON = "mdi:cog"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFlat(True)
        self.setProperty("variant", "subtle")
        self.setFixedSize(32, 32)
        self.setToolTip("Preferences")
        self.clicked.connect(self._open)
        self._apply_icon()

    def _apply_icon(self) -> None:
        color = qcolor(theme().text_secondary).name()
        self.setIcon(QIconifyIcon(self._ICON, color=color))
        size = theme().scaled(18)
        self.setIconSize(QSize(size, size))

    def changeEvent(self, e: QEvent | None) -> None:
        if e is not None and e.type() == QEvent.Type.StyleChange:
            self._apply_icon()
        super().changeEvent(e)

    def _open(self) -> None:
        dlg = PreferencesDialog(self.window())
        dlg.exec()


class PreferencesDialog(QDialog):
    """Data & Memory preferences today; more sections can join it later.

    Owns no application state beyond what's in the form: values are read from
    ``Settings.instance()`` at construction and written back (then flushed to
    disk) only when "Save" is clicked.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setModal(True)

        t = theme()
        prefs = Settings.instance().scratch

        group = QGroupBox("Data && Memory")
        group.setToolTip(
            "Controls where an acquisition's data lives when the Saving "
            "section is unchecked."
        )
        form = QFormLayout()
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(t.sp_sm)
        form.setVerticalSpacing(t.sp_xs)
        group.setLayout(form)

        self._max_memory = QDoubleSpinBox()
        self._max_memory.setRange(*_MAX_MEMORY_GB_RANGE)
        self._max_memory.setDecimals(1)
        self._max_memory.setSingleStep(0.5)
        self._max_memory.setSuffix(" GB")
        self._max_memory.setValue(prefs.max_memory_gb)
        self._max_memory.setToolTip(
            "The largest run held entirely in RAM before spilling to disk\n"
            "(or refusing to run, depending on the option below)."
        )
        form.addRow("Max in-memory size:", self._max_memory)

        self._spill_to_disk = QCheckBox("Spill to disk when exceeded")
        self._spill_to_disk.setChecked(prefs.spill_to_disk)
        self._spill_to_disk.setToolTip(
            "Checked: continue the run on disk once the limit above is hit.\n"
            "Unchecked: refuse to start a run that would exceed it."
        )
        self._spill_to_disk.toggled.connect(self._update_scratch_dir_enabled)
        form.addRow("", self._spill_to_disk)

        self._scratch_dir = QLineEdit()
        self._scratch_dir.setPlaceholderText("System temp folder")
        if prefs.scratch_dir is not None:
            self._scratch_dir.setText(str(prefs.scratch_dir))
        self._scratch_dir.setToolTip(
            "Parent folder for spilled data. Leave empty to use the system\n"
            "temp folder -- set this to a drive with enough free space if\n"
            "runs are spilling to a small system disk."
        )
        self._browse_btn = QPushButton("...")
        self._browse_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._browse_btn.setFixedWidth(t.scaled(32))
        self._browse_btn.clicked.connect(self._browse_scratch_dir)
        dir_row = QHBoxLayout()
        dir_row.setContentsMargins(0, 0, 0, 0)
        dir_row.addWidget(self._scratch_dir)
        dir_row.addWidget(self._browse_btn)
        form.addRow("Scratch folder:", dir_row)
        self._update_scratch_dir_enabled(self._spill_to_disk.isChecked())

        note = QLabel(
            "Applies to acquisitions run without saving to disk (the "
            "Saving section unchecked)."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {qcolor(t.text_secondary).name()};")

        save_btn = QPushButton("Save")
        save_btn.setProperty("variant", "primary")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setProperty("variant", "subtle")
        cancel_btn.setAutoDefault(False)
        cancel_btn.clicked.connect(self.reject)

        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(cancel_btn)
        buttons.addWidget(save_btn)

        self.setMinimumWidth(t.scaled(_MIN_WIDTH))
        outer = QVBoxLayout(self)
        outer.setContentsMargins(t.sp_lg, t.sp_lg, t.sp_lg, t.sp_lg)
        outer.setSpacing(t.sp_lg)
        outer.addWidget(group)
        outer.addWidget(note)
        outer.addLayout(buttons)

    def _update_scratch_dir_enabled(self, spill_to_disk: bool) -> None:
        self._scratch_dir.setEnabled(spill_to_disk)
        self._browse_btn.setEnabled(spill_to_disk)

    def _browse_scratch_dir(self) -> None:
        start = self._scratch_dir.text() or str(Path.home())
        if directory := QFileDialog.getExistingDirectory(
            self, "Select Scratch Folder", start
        ):
            self._scratch_dir.setText(directory)

    def _save(self) -> None:
        prefs = Settings.instance().scratch
        prefs.max_memory_gb = self._max_memory.value()
        prefs.spill_to_disk = self._spill_to_disk.isChecked()
        text = self._scratch_dir.text().strip()
        prefs.scratch_dir = Path(text) if text else None
        Settings.instance().flush()
        self.accept()
