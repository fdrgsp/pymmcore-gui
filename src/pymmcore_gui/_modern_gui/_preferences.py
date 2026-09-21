"""Application-wide preferences, independent of any hardware configuration.

Reachable from the gear button in the Acquire tab's toolbar, right after the
panel buttons -- today's preferences (which widgets show, the active layout,
Data & Memory) all affect acquisitions specifically, so that's where the
entry point lives, even though Data & Memory is persisted globally like other
app settings. Kept deliberately separate from the Configurations page: that
page edits the hardware ``.cfg`` file (with its own dirty-tracking and Save
button), while Data & Memory applies across every configuration and is
persisted straight to the user's ``pmm_settings.json`` (see
``pymmcore_gui._settings``).
"""

from __future__ import annotations

import tempfile
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final

from superqt.iconify import QIconifyIcon

from pymmcore_gui._layouts import RESERVED_LAYOUT_NAMES, available_layouts
from pymmcore_gui._modern_gui._theme import qcolor, theme
from pymmcore_gui._qt.QtCore import QEvent, QSize, Qt
from pymmcore_gui._qt.QtGui import QCursor
from pymmcore_gui._qt.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from pymmcore_gui._settings import Settings
from pymmcore_gui._utils import system_memory_gb

from ._panels import PANELS

if TYPE_CHECKING:
    from ._acquire import AcquirePage

_MIN_WIDTH: Final = 420


def _resync_hover_state(widget: QWidget) -> None:
    """Resync hover after a nested event loop (``PreferencesDialog.exec()``).

    Qt tracks hover through Enter/Leave events generated from real cursor
    motion. A blocking ``exec()`` called from inside a ``clicked`` handler
    runs its own nested loop; once it returns, this widget's hover flag can
    be left stale, so it keeps painting as hovered even after the cursor has
    moved off it. Recomputing it from the live cursor position and
    repainting fixes that.
    """
    under_mouse = widget.rect().contains(widget.mapFromGlobal(QCursor.pos()))
    widget.setAttribute(Qt.WidgetAttribute.WA_UnderMouse, under_mouse)
    widget.update()


def _max_memory_gb_range() -> tuple[float, float]:
    """(min, max) allowed for the "Max in-memory size" spinbox.

    The upper bound is the machine's total physical RAM.
    """
    total_gb, _ = system_memory_gb()
    return (0.1, round(total_gb, 1))


class PreferencesButton(QPushButton):
    """Gear button that opens the :class:`PreferencesDialog`.

    Lives in the Acquire tab's toolbar, grouped directly after the panel
    buttons (see ``AcquirePage._place_panel_bar``) -- no separator between
    them, styled the same as the other icon buttons on that row
    (``SnapButton``, the panel toggles, ...): "subtle" variant, unfixed size
    (the style computes it from the icon), same icon size.

    Takes the owning :class:`AcquirePage` itself (also its Qt parent) rather
    than a generic ``QWidget``: the dialog it opens now reaches into that
    page's panel-visibility and layout state, not just app-wide settings.
    """

    _ICON = "mdi:cog"

    def __init__(self, acquire_page: AcquirePage) -> None:
        super().__init__(acquire_page)
        self._acquire_page = acquire_page
        self.setProperty("variant", "subtle")
        self.setToolTip("Preferences")
        self.clicked.connect(self._open)
        self._apply_icon()

    def _apply_icon(self) -> None:
        color = qcolor(theme().text_secondary).name()
        self.setIcon(QIconifyIcon(self._ICON, color=color))
        # Same size as the other Acquire toolbar icon buttons
        # (``_acquire_toolbar._icon_size``); kept in sync manually since
        # importing that private helper across modules isn't worth it here.
        size = theme().scaled(20)
        self.setIconSize(QSize(size, size))

    def changeEvent(self, e: QEvent | None) -> None:
        if e is not None and e.type() == QEvent.Type.StyleChange:
            self._apply_icon()
        super().changeEvent(e)

    def _open(self) -> None:
        dlg = PreferencesDialog(self._acquire_page, self.window())
        dlg.exec()
        _resync_hover_state(self)


class PreferencesDialog(QDialog):
    """Data & Memory, Layout, and Show Widgets -- Acquire-page preferences.

    Two different apply models live side by side here:

    * **Show Widgets** and **Layout** act on the live :class:`AcquirePage`
      immediately -- checking a box shows/hides its toolbar button right
      now, clicking a layout switches to it right now. There's nothing to
      "save": closing the dialog (Save *or* Cancel) never undoes them,
      exactly like the toolbar buttons they replaced never needed an undo.
    * **Data & Memory** is the odd one out: values are read from
      ``Settings.instance()`` at construction and written back (then flushed
      to disk) only when "Save" is clicked; "Cancel" discards the edit.
    """

    def __init__(
        self, acquire_page: AcquirePage, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setModal(True)
        self._acquire_page = acquire_page

        t = theme()

        widgets_group = self._build_widgets_group(acquire_page)
        memory_group = self._build_memory_group()
        layout_group = self._build_layout_group()

        self.setMinimumWidth(t.scaled(_MIN_WIDTH))
        outer = QVBoxLayout(self)
        outer.setContentsMargins(t.sp_lg, t.sp_lg, t.sp_lg, t.sp_lg)
        outer.setSpacing(t.sp_lg)
        outer.addWidget(memory_group)
        outer.addWidget(layout_group)
        outer.addWidget(widgets_group)

    # ── Show Widgets ─────────────────────────────────────────────

    def _build_widgets_group(self, acquire_page: AcquirePage) -> QGroupBox:
        """One checkbox per hideable panel; toggling applies immediately.

        Replaces the old ⋯ customize menu -- same panels (every registry
        entry except the always-visible MDA one), same effect
        (:meth:`AcquirePage.set_panel_visible`).
        """
        group = QGroupBox("Show Widgets")
        group.setToolTip(
            "Choose which of the Acquire toolbar's tool buttons are shown.\n"
            "Takes effect immediately."
        )
        # 2 columns rather than 1 -- six checkboxes in a single column reads
        # as a long, sparse list; a grid keeps the group compact.
        columns = 2
        layout = QGridLayout()
        layout.setHorizontalSpacing(theme().sp_lg)
        layout.setVerticalSpacing(theme().sp_xs)
        hidden = acquire_page.hidden_panels()
        self._widget_checkboxes: dict[str, QCheckBox] = {}
        hideable = [info for info in PANELS if not info.always_visible]
        for index, info in enumerate(hideable):
            checkbox = QCheckBox(info.title)
            checkbox.setChecked(info.key not in hidden)
            checkbox.setToolTip(info.tooltip)
            checkbox.toggled.connect(partial(acquire_page.set_panel_visible, info.key))
            layout.addWidget(checkbox, index // columns, index % columns)
            self._widget_checkboxes[info.key] = checkbox
        group.setLayout(layout)
        return group

    # ── Data & Memory ────────────────────────────────────────────

    def _build_memory_group(self) -> QGroupBox:
        t = theme()
        prefs = Settings.instance().scratch

        group = QGroupBox("Data && Memory")
        group.setToolTip(
            "Controls where an acquisition's data lives when the Saving "
            "section is unchecked."
        )
        grid = QGridLayout()
        grid.setHorizontalSpacing(t.sp_sm)
        grid.setVerticalSpacing(t.sp_xs)
        grid.setColumnStretch(1, 1)
        group.setLayout(grid)

        # Left-align every row label and size them all to the widest one, so
        # the fields that follow start at a common x position.
        row_labels = ("Max in-memory size:", "Spill folder:")
        label_width = max(
            self.fontMetrics().horizontalAdvance(text) for text in row_labels
        )

        def row_label(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setFixedWidth(label_width)
            lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            return lbl

        self._max_memory = QDoubleSpinBox()
        self._max_memory.setRange(*_max_memory_gb_range())
        self._max_memory.setDecimals(1)
        self._max_memory.setSingleStep(0.5)
        self._max_memory.setSuffix(" GB")
        self._max_memory.setValue(prefs.max_memory_gb)
        self._max_memory.setToolTip(
            "The largest run held entirely in RAM before spilling to disk\n"
            "(or refusing to run, depending on the option below). Defaults\n"
            "to 80% of the RAM free on this machine."
        )
        grid.addWidget(row_label("Max in-memory size:"), 0, 0)
        grid.addWidget(self._max_memory, 0, 1)

        self._spill_to_disk = QCheckBox("Spill to disk when exceeded")
        self._spill_to_disk.setChecked(prefs.spill_to_disk)
        self._spill_to_disk.setToolTip(
            "Checked: continue the run on disk once the limit above is hit.\n"
            "Unchecked: refuse to start a run that would exceed it."
        )
        self._spill_to_disk.toggled.connect(self._update_scratch_dir_enabled)
        grid.addWidget(self._spill_to_disk, 1, 1)

        self._scratch_dir = QLineEdit()
        self._scratch_dir.setPlaceholderText("System temp folder")
        self._scratch_dir.setText(
            str(prefs.scratch_dir) if prefs.scratch_dir is not None else _system_tmp()
        )
        self._scratch_dir.setToolTip(
            "Parent folder for spilled data, shared by all runs that spill.\n"
            "Defaults to the system temp folder -- point this at a drive\n"
            "with more free space if runs are spilling to a small system\n"
            "disk."
        )
        self._browse_btn = QPushButton("...")
        self._browse_btn.setProperty("variant", "subtle")
        self._browse_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._browse_btn.setFixedWidth(t.scaled(32))
        self._browse_btn.clicked.connect(self._browse_scratch_dir)
        dir_row = QHBoxLayout()
        dir_row.setContentsMargins(0, 0, 0, 0)
        dir_row.setSpacing(5)
        dir_row.addWidget(self._scratch_dir)
        dir_row.addWidget(self._browse_btn)
        grid.addWidget(row_label("Spill folder:"), 2, 0)
        grid.addLayout(dir_row, 2, 1)
        self._update_scratch_dir_enabled(self._spill_to_disk.isChecked())

        # Save/Cancel live here, not as a dialog-wide footer: they only ever
        # apply to this group -- Show Widgets and Layout take effect the
        # moment you click them (see the class docstring).
        save_btn = QPushButton("Save")
        save_btn.setProperty("variant", "primary")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setProperty("variant", "subtle")
        cancel_btn.setAutoDefault(False)
        cancel_btn.clicked.connect(self.reject)

        buttons = QHBoxLayout()
        buttons.setSpacing(5)
        buttons.setContentsMargins(0, t.sp_sm, 0, 0)
        buttons.addStretch()
        buttons.addWidget(cancel_btn)
        buttons.addWidget(save_btn)
        grid.addLayout(buttons, 3, 0, 1, 2)

        return group

    def _update_scratch_dir_enabled(self, spill_to_disk: bool) -> None:
        self._scratch_dir.setEnabled(spill_to_disk)
        self._browse_btn.setEnabled(spill_to_disk)

    def _browse_scratch_dir(self) -> None:
        start = self._scratch_dir.text() or str(Path.home())
        if directory := QFileDialog.getExistingDirectory(
            self, "Select Spill Folder", start
        ):
            self._scratch_dir.setText(directory)

    def _save(self) -> None:
        prefs = Settings.instance().scratch
        prefs.max_memory_gb = self._max_memory.value()
        prefs.spill_to_disk = self._spill_to_disk.isChecked()
        text = self._scratch_dir.text().strip()
        # Treat "still the system temp folder" as "no override", so it keeps
        # tracking the OS temp dir rather than pinning today's resolved path.
        prefs.scratch_dir = None if (not text or text == _system_tmp()) else Path(text)
        Settings.instance().flush()
        self.accept()

    # ── Layout ───────────────────────────────────────────────────

    def _build_layout_group(self) -> QGroupBox:
        """List every layout (Last session, Default, saved ones); pick = apply now.

        Replaces the old Layout toolbar button's drop-down: reuses
        ``AcquirePage.select_layout``/``prompt_save_layout``/
        ``prompt_delete_layout`` verbatim, just triggered from this list
        instead of a menu.
        """
        group = QGroupBox("Layout")
        group.setToolTip(
            "Switch, save, or delete Acquire panel arrangements.\n"
            "Selecting a layout applies it immediately."
        )
        layout = QVBoxLayout()
        layout.setSpacing(theme().sp_xs)

        self._layout_list = QListWidget()
        self._layout_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._layout_list.setMaximumHeight(theme().scaled(120))
        self._layout_list.itemClicked.connect(self._on_layout_clicked)
        self._layout_list.currentItemChanged.connect(
            lambda *_: self._update_delete_enabled()
        )
        layout.addWidget(self._layout_list)

        row = QHBoxLayout()
        self._add_layout_btn = QPushButton("Add Current…")
        self._add_layout_btn.setProperty("variant", "subtle")
        self._add_layout_btn.setToolTip("Save the current arrangement under a name")
        self._add_layout_btn.clicked.connect(self._add_layout)
        self._delete_layout_btn = QPushButton("Delete")
        self._delete_layout_btn.setProperty("variant", "subtle")
        self._delete_layout_btn.setToolTip("Delete the selected saved layout")
        self._delete_layout_btn.clicked.connect(self._delete_selected_layout)
        row.addStretch()
        row.addWidget(self._add_layout_btn)
        row.addWidget(self._delete_layout_btn)
        layout.addLayout(row)

        group.setLayout(layout)
        self._refresh_layout_list()
        return group

    def _refresh_layout_list(self) -> None:
        self._layout_list.blockSignals(True)
        self._layout_list.clear()
        current = self._acquire_page.layout_name
        current_item: QListWidgetItem | None = None
        for name in available_layouts():
            item = QListWidgetItem(name)
            self._layout_list.addItem(item)
            if name == current:
                current_item = item
        if current_item is not None:
            self._layout_list.setCurrentItem(current_item)
        self._layout_list.blockSignals(False)
        self._update_delete_enabled()

    def _update_delete_enabled(self) -> None:
        item = self._layout_list.currentItem()
        self._delete_layout_btn.setEnabled(
            item is not None and item.text() not in RESERVED_LAYOUT_NAMES
        )

    def _on_layout_clicked(self, item: QListWidgetItem) -> None:
        self._acquire_page.select_layout(item.text())
        self._refresh_layout_list()

    def _add_layout(self) -> None:
        self._acquire_page.prompt_save_layout()
        self._refresh_layout_list()

    def _delete_selected_layout(self) -> None:
        item = self._layout_list.currentItem()
        if item is None or item.text() in RESERVED_LAYOUT_NAMES:
            return
        self._acquire_page.prompt_delete_layout(item.text())
        self._refresh_layout_list()


def _system_tmp() -> str:
    return str(Path(tempfile.gettempdir()))
