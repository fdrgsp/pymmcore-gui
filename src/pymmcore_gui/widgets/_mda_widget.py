"""Shared MDA editor configured for the ome-writers sink."""

from __future__ import annotations

from collections import defaultdict
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, cast

from pymmcore_widgets import MDAWidgetCollapsible
from pymmcore_widgets._icons import StandardIcon
from pymmcore_widgets.mda import (
    ChannelProperty,
    CollapsibleCoreMDATabs,
    SectionMetrics,
)
from pymmcore_widgets.mda._core_channels import NO_LIGHT_SOURCE
from pymmcore_widgets.useq_widgets._positions import MDAButton
from superqt.iconify import QIconifyIcon

from pymmcore_gui._array_viewer import (
    ensure_visible_icon,
    set_source_icon,
    unstyle_widgets,
)
from pymmcore_gui._light_sources import (
    parse_light_source_comments,
    write_light_source_comments,
)
from pymmcore_gui._modern_gui._busy import BusyOverlay
from pymmcore_gui._modern_gui._theme import qcolor, theme
from pymmcore_gui._qt.QtCore import (
    QEvent,
    QModelIndex,
    QObject,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from pymmcore_gui._qt.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QMessageBox,
    QPushButton,
    QWidget,
)

from ._active_channel_table import (
    CURRENT_CHANNEL_COLUMN,
    ActiveChannelCollapsibleCoreMDATabs,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    import useq
    from pymmcore_plus import CMMCorePlus
    from pymmcore_widgets.mda._xy_bounds import CoreXYBoundsControl

    from pymmcore_gui._qt.QtGui import QResizeEvent

    from ._active_channel_table import ActiveChannelTable


def _align_bounds_grid(bounds: CoreXYBoundsControl) -> None:
    """Line up the mark/visit icon grid with the Left/Top/Right/Bottom fields.

    CoreXYBoundsControl (pymmcore_widgets/mda/_xy_bounds.py) places its icon
    button grid (Fixed size policy) and its Left/Top/Right/Bottom fields (a
    QFormLayout-based _BoundsWidget) side by side in a plain QHBoxLayout
    with no alignment flags. That's fine at the control's own sizeHint, but
    it lives inside GridPlanWidget's QStackedWidget, which stretches every
    page to fill the full stack area -- Qt then vertically centers the
    Fixed-policy icon grid while the QFormLayout stays top-anchored.
    AlignTop fixes that block-level offset, but the two remain independent
    layouts with independently-computed row heights/spacing (~30px icon
    buttons vs. whatever height our theme gives a spinbox+label row), so
    each row still drifts further out of sync than the last one. Measure
    the *actual* rendered row geometry of both and adjust the icon grid's
    spacing/margins to match, rather than hardcoding pixel constants that
    would only happen to be correct at one specific zoom level.
    """
    top_layout = bounds.layout()
    if top_layout is None:
        return
    grid = None
    for i in range(top_layout.count()):
        item = top_layout.itemAt(i)
        w = item.widget() if item is not None else None
        if w is not None:
            top_layout.setAlignment(w, Qt.AlignmentFlag.AlignTop)
            if w is not bounds._bounds_wdg:
                grid = w
    grid_layout = grid.layout() if grid is not None else None
    if not isinstance(grid_layout, QGridLayout):
        return

    def row_center(w: QWidget) -> float:
        top = w.mapTo(bounds, w.rect().topLeft()).y()
        return top + w.height() / 2

    field_step = row_center(bounds.top) - row_center(bounds.left)
    icon_step = row_center(bounds.btn_left) - row_center(bounds.btn_top)
    new_spacing = max(0, grid_layout.verticalSpacing() + (field_step - icon_step))
    grid_layout.setVerticalSpacing(round(new_spacing))

    # Re-measure after the spacing change: it only affects rows below the
    # first, so btn_top (row 0) hasn't moved, but re-deriving from current
    # geometry rather than assuming that keeps this correct either way.
    top_shift = row_center(bounds.left) - row_center(bounds.btn_top)
    left, top, right, bottom = grid_layout.getContentsMargins()
    grid_layout.setContentsMargins(
        left or 0, round((top or 0) + top_shift), right or 0, bottom or 0
    )


class MemoryMDAWidget(MDAWidgetCollapsible):
    """Collapsible-sections MDA editor with a viewable memory-sink fallback.

    The sectioned presentation now lives upstream in ``MDAWidgetCollapsible``;
    this subclass only adds the app theme, semantic icons, the channel-selection
    core bridge, and the memory-sink output fallback.
    """

    _storeCreationFinished = Signal()

    def _create_tab_widget(self) -> CollapsibleCoreMDATabs:
        return ActiveChannelCollapsibleCoreMDATabs(None, self._mmc)

    # ----------- Override type hints in superclass -----------
    # _create_tab_widget above always installs an ActiveChannelTable, which adds
    # activeRow/setActiveRow to the upstream CoreConnectedChannelTable
    # interface; narrow the property's declared return type to match.

    @property
    def channels(self) -> ActiveChannelTable:
        return cast("ActiveChannelTable", super().channels)

    def __init__(self, mmcore: CMMCorePlus, parent: QWidget | None = None) -> None:
        self._restoring_sequence = False
        self._applying_channel_config = False
        # {channel preset: [(device, property, intensity), ...]}, parsed from the
        # loaded cfg's comment block. Held in memory rather than re-read per use, so
        # it stays right after saving to a *different* file than the loaded one.
        self._light_source_declarations: dict[str, list[tuple[str, str, float]]] = {}
        super().__init__(parent=parent, mmcore=mmcore)
        self.camera_roi.setRoiInfoVisible(False)
        self._update_time_estimate()
        self._store_overlay = BusyOverlay(self)
        self._storeCreationFinished.connect(self._store_overlay.stop)
        self._mmc.mda.events.sequenceStarted.connect(self._on_store_creation_finished)
        self._mmc.mda.events.sequenceFinished.connect(self._on_store_creation_finished)
        self._apply_theme_metrics()
        combo = self.save_info._writer_combo
        for idx in range(combo.count()):
            if combo.itemText(idx) == "tiff-sequence":
                combo.removeItem(idx)
                break
        combo.setCurrentText("ome-tiff")

        # Run/Pause/Cancel/Save/Load default to the "ghost" variant (no
        # visible box until hovered) — give them a persistently visible one,
        # matching the rest of the app's action buttons.
        for btn in (
            self.control_btns.run_btn,
            self.control_btns.pause_btn,
            self.control_btns.cancel_btn,
            self._save_button,
            self._load_button,
        ):
            btn.setProperty("variant", "subtle")

        # Goes after the stretch that follows the upstream "Show Light Source" /
        # "Advanced" checkboxes, so it right-aligns in that row. Created before the
        # unstyle_widgets sweep below so it gets themed with everything else.
        self._save_light_sources_btn = QPushButton("Save Light Sources to cfg")
        self._save_light_sources_btn.setToolTip(
            "Write each channel's light source into a Micro-Manager configuration "
            "file, so it is restored the next time that file is loaded."
        )
        self._save_light_sources_btn.setProperty("variant", "subtle")
        self._save_light_sources_btn.clicked.connect(self._save_light_sources_to_config)
        self.channels._btn_row.insertWidget(3, self._save_light_sources_btn)

        # All sub-tabs (Channels/Positions/Z/Time/Grid) are constructed
        # eagerly by CoreMDATabs.create_subwidgets() during super().__init__,
        # so one recursive sweep here reaches every descendant -- strips
        # hardcoded stylesheets (e.g. useq_widgets' border-less spinboxes,
        # gray range labels) and normalizes buttons the same way as
        # everywhere else in the app.
        unstyle_widgets(self)
        # unstyle_widgets clears every stylesheet, including the one the upstream
        # Saving section installs to hide the embedded QGroupBox's native header
        # and indicator. Re-apply it here (and after each row insert below).
        self._collapsible_tabs().apply_save_body_style()
        self._apply_table_toolbar_icon_size()

        # Channels/Positions/Time are tables: each row is a fresh cell widget
        # built on demand (e.g. the Positions row's black "mdi:axis" Sub-
        # Sequence button, or the row's spinboxes), created only when a row
        # is actually added -- long after the sweep above already ran. Without
        # this, a row added interactively is never covered by any sweep at
        # all until the next light/dark toggle (which only revisits icons,
        # not stylesheets).
        for table_widget in (self.channels, self.stage_positions, self.time_plan):
            model = table_widget.table().model()
            if model is not None:
                model.rowsInserted.connect(self._on_table_rows_inserted)

        # Upstream replaces Pause/Resume icons whenever acquisition state
        # changes. Re-apply our semantic colors after those runtime swaps.
        self._mmc.mda.events.sequencePauseToggled.connect(self._apply_themed_icons)
        self._mmc.mda.events.sequenceFinished.connect(self._apply_themed_icons)
        self._connect_position_icon_updates()
        self._apply_themed_icons()
        self._connect_channel_selection()

        # Connected here rather than in the table itself so it runs *after*
        # CoreConnectedChannelTable's own systemConfigurationLoaded handler, which
        # rebuilds the light source combo we are about to pick an entry from.
        self._mmc.events.systemConfigurationLoaded.connect(
            self._reload_light_source_declarations
        )
        # a configuration is usually already loaded by the time this is constructed
        self._reload_light_source_declarations()

        # CoreXYBoundsControl swaps each edge button's icon whenever its
        # Mark/Move action changes.  That replacement installs the upstream
        # raw (black) icon after unstyle_widgets() has already made the initial
        # icon theme-aware, so it becomes nearly invisible in dark mode.
        # Connect after the upstream swap handler and refresh the new glyph.
        self.grid_plan._core_xy_bounds.go_middle.toggled.connect(
            self._refresh_grid_bounds_icons
        )

        _align_bounds_grid(self.grid_plan._core_xy_bounds)

    def _update_time_estimate(self) -> None:
        """Keep the upstream acquisition-duration display hidden in this GUI."""
        self._time_warning.hide()
        self._duration_label.hide()

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        """Keep the data-store startup overlay fitted to the MDA editor."""
        super().resizeEvent(a0)
        if hasattr(self, "_store_overlay"):
            self._store_overlay.setGeometry(self.rect())

    def run_mda(self) -> None:
        """Start an MDA, showing progress while a disk store is initialized."""
        output = self.prepare_mda()
        if isinstance(output, bool):
            return

        show_store_progress = self.save_info.isChecked() and self.isVisible()
        if show_store_progress:
            self._store_overlay.start("Creating data store…")
        try:
            self.execute_mda(output)
        except Exception:
            # A synchronous launch failure emits neither sequenceStarted nor
            # sequenceFinished, so clear the overlay here.
            if show_store_progress:
                self._store_overlay.stop()
            raise

    def _on_store_creation_finished(self, *_: object) -> None:
        """Relay worker-thread MDA signals safely back to the Qt GUI thread."""
        self._storeCreationFinished.emit()

    def _disconnect(self) -> None:
        """Disconnect this subclass's MDA callbacks, then the upstream ones."""
        with suppress(Exception):
            self._mmc.mda.events.sequenceStarted.disconnect(
                self._on_store_creation_finished
            )
            self._mmc.mda.events.sequenceFinished.disconnect(
                self._on_store_creation_finished
            )
            self._mmc.events.systemConfigurationLoaded.disconnect(
                self._reload_light_source_declarations
            )
        super()._disconnect()

    def _on_table_rows_inserted(self, *_: object) -> None:
        """Theme cell widgets that are constructed only when a row is added."""
        unstyle_widgets(self)
        self._collapsible_tabs().apply_save_body_style()
        self._install_channel_editor_filters()
        self._connect_position_icon_updates()
        self._apply_themed_icons()
        self._apply_light_source_declarations()

    def setValue(self, value: useq.MDASequence) -> None:
        """Restore an MDA sequence without applying a selected row to the core."""
        selected = self._selected_channel_identity()
        selected_row = self.channels.activeRow()
        self._restoring_sequence = True
        try:
            super().setValue(value)
            self._restore_channel_selection(selected, selected_row)
        finally:
            self._restoring_sequence = False
        self._collapsible_tabs().refresh_summaries()

    def refresh_channel_table(self) -> None:
        """Refresh core-backed channel choices without changing microscope state."""
        channels = self.channels
        table = channels.table()
        selected = self._selected_channel_identity()
        selected_row = channels.activeRow()
        values = channels.value(exclude_unchecked=False)
        properties = channels.channelProperties(exclude_unchecked=False)
        light_sources_visible = channels.lightSourceVisible()

        selector_col = table._get_selector_col()
        selector = table.columnInfo(selector_col)
        checked = (
            [
                selector.isChecked(table, row, selector_col)
                for row in range(table.rowCount())
            ]
            if selector is not None
            else []
        )

        self._restoring_sequence = True
        try:
            channels.refresh()
            channels.setValue(values)
            channels.setLightSourceVisible(light_sources_visible)
            channels.setChannelProperties(properties)

            selector_col = table._get_selector_col()
            if (selector := table.columnInfo(selector_col)) is not None:
                for row, is_checked in enumerate(checked):
                    check_state = (
                        Qt.CheckState.Checked if is_checked else Qt.CheckState.Unchecked
                    )
                    selector.setCheckState(
                        table,
                        row,
                        selector_col,
                        cast("bool", check_state),
                    )
            self._restore_channel_selection(selected, selected_row)
        finally:
            self._restoring_sequence = False
        # After the guard is cleared, so the declarations actually apply: the
        # snapshot restored above predates whatever configuration change prompted
        # this refresh, and the loaded cfg is the more authoritative source.
        self._apply_light_source_declarations()
        self._collapsible_tabs().refresh_summaries()

    # ------------- light source declarations (see LIGHT_SOURCE_COMMENT) ---------

    def _reload_light_source_declarations(self) -> None:
        """Re-read the declaration block from the newly loaded cfg, and apply it."""
        self._light_source_declarations = parse_light_source_comments(
            self._mmc.systemConfigurationFile() or "", self._mmc.getChannelGroup()
        )
        self._apply_light_source_declarations()

    def _apply_light_source_declarations(
        self, rows: Iterable[int] | None = None, clear_missing: bool = False
    ) -> None:
        """Select each row's declared light source and default intensity.

        Only rows whose channel preset is declared in the cfg are touched; there is
        deliberately no fallback that infers a light source from a channel preset's
        own properties, so an undeclared channel simply has none. Pass
        ``clear_missing`` to also reset rows whose preset is undeclared (used when
        the user picks a different preset for a row).
        """
        if self._restoring_sequence:
            return
        channels = self.channels.value(exclude_unchecked=False)
        indices = range(len(channels)) if rows is None else rows

        # A declaration listing several properties is one multi-slider light source
        # (e.g. a Lumencor LIDA): map the whole pair set back to the single-preset
        # config group entry whose one spin box drives all of them at once.
        by_pairs = {
            frozenset(pairs): label
            for label, pairs in self.channels.lightSources().items()
        }

        props: list[ChannelProperty] = []
        for row in indices:
            if not 0 <= row < len(channels):  # pragma: no cover
                continue
            preset = str(channels[row].config or "")
            if entries := self._light_source_declarations.get(preset):
                pairs = [(device, prop) for device, prop, _ in entries]
                # An unmatched pair set leaves the label empty, and
                # setChannelProperties then resolves each entry by (device,
                # property) instead -- which is what a single-property declaration
                # wants anyway, and which drops a declaration naming a property
                # this configuration does not offer.
                label = by_pairs.get(frozenset(pairs), "")
                props.extend(
                    ChannelProperty(
                        channel_index=row,
                        config=preset,
                        group=label,
                        device=device,
                        property=prop,
                        value=entries[0][2],
                    )
                    for device, prop in pairs
                )
            elif clear_missing:
                self._clear_light_source(row)
        if props:
            self.channels.setChannelProperties(props)

    def _clear_light_source(self, row: int) -> None:
        """Reset one row's light source and intensity to "none"."""
        channels = self.channels
        table = channels.table()
        ls_col = table.indexOf(channels._light_source_column)
        int_col = table.indexOf(channels.INTENSITY)
        if ls_col < 0 or int_col < 0:  # pragma: no cover
            return
        # Left unblocked on purpose: the resulting valueChanged drives upstream's
        # _sync_intensity_widgets, which re-ranges and disables the spin box.
        channels._light_source_column.set_cell_data(table, row, ls_col, NO_LIGHT_SOURCE)
        channels.INTENSITY.set_cell_data(table, row, int_col, 0.0)

    def _save_light_sources_to_config(self) -> None:
        """Save the whole cfg to a file, with the light source declarations appended.

        The block is rewritten from scratch so a light source the user cleared
        actually disappears, and two rows sharing one channel preset collapse to a
        single declaration (the later row wins). The configuration itself is dumped
        from the core, so uncommitted edits in the Hardware Setup page are not
        included.
        """
        if not (channel_group := self._mmc.getChannelGroup()):
            QMessageBox.warning(
                self,
                "Save Light Sources",
                "No channel group is set on the current configuration, so there is "
                "nothing to attach the light sources to.",
            )
            return
        if not self.channels.lightSourceVisible():
            QMessageBox.warning(
                self,
                "Save Light Sources",
                "Light sources are turned off. Check 'Show Light Source' and pick "
                "the light source for each channel before saving.",
            )
            return

        current = self._mmc.systemConfigurationFile() or ""
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save Micro-Manager configuration file", current, "cfg(*.cfg)"
        )
        if not path_str:
            return
        path = Path(path_str)
        if path.suffix.lower() != ".cfg":
            path = path.with_suffix(".cfg")

        by_row: dict[int, list[ChannelProperty]] = defaultdict(list)
        for entry in self.channels.channelProperties(exclude_unchecked=False):
            by_row[entry["channel_index"]].append(entry)
        declarations: dict[str, list[tuple[str, str, float]]] = {}
        for row, channel in enumerate(self.channels.value(exclude_unchecked=False)):
            if (preset := str(channel.config or "")) and (entries := by_row.get(row)):
                declarations[preset] = [
                    (entry["device"], entry["property"], float(entry["value"]))
                    for entry in entries
                ]

        try:
            # dumps the configuration itself; comments do not survive this, which is
            # why the declaration block is appended afterwards rather than merged
            self._mmc.saveSystemConfiguration(str(path))
            write_light_source_comments(path, channel_group, declarations)
        except Exception as exc:
            QMessageBox.critical(
                self, "Save Light Sources", f"Could not save {path}:\n{exc}"
            )
            return
        # The core is untouched by all of this, so the table needs no refresh -- but
        # what was just written is now what a reload of this file would restore.
        self._light_source_declarations = declarations

    def _connect_channel_selection(self) -> None:
        table = self.channels.table()
        selection_model = table.selectionModel()
        if selection_model is None:  # pragma: no cover
            return
        # Hardware fires only from a click on the ● column here; the Config
        # combo's own activation (connected in _install_channel_editor_filters)
        # is the other trigger.
        table.clicked.connect(self._on_channel_cell_clicked)
        # QAbstractItemView syncs currentIndex to whichever cell widget last
        # took focus (e.g. clicking into the Exposure/Intensity spin box) --
        # entirely internal to Qt, so it can't be stopped by filtering mouse or
        # focus events. Snap it back to the active row whenever that happens.
        selection_model.currentRowChanged.connect(self._on_table_current_row_changed)
        model = table.model()
        if model is not None:
            model.columnsInserted.connect(self._schedule_channel_editor_filters)
        self._install_channel_editor_filters()

        # Reverse sync: when the active preset changes on the core from
        # anywhere else (setConfig, the Groups & Presets table, etc.), update
        # the ● indicator and select the matching row to reflect the new state.
        self._mmc.events.configSet.connect(self._on_core_config_set)

    def _schedule_channel_editor_filters(self, *_: object) -> None:
        # columnsInserted fires from insertColumn(), before DataTable.addColumn()
        # has populated that column's cell widgets. Install after the insertion
        # event has completed so newly rebuilt config/property columns are
        # included.
        QTimer.singleShot(0, self._install_channel_editor_filters)

    def _install_channel_editor_filters(self) -> None:
        """Wire up the two ways to move hardware, and the live-apply editors.

        A row's Config combo `activated` signal (user-only; a programmatic
        `currentTextChanged` during a sequence restore/refresh must not move
        hardware) applies that channel, same as clicking the ● column. A row's
        Exposure/Intensity spin box `editingFinished` pushes a live edit of the
        *active* row to hardware (see _on_channel_value_committed) but never
        moves the row highlight -- see _on_table_current_row_changed, which
        undoes Qt's own focus-driven selection change for every other column.
        """
        table = self.channels.table()
        try:
            config_col = table.indexOf(self.channels._config_column)
            exposure_col = table.indexOf(self.channels.EXPOSURE)
            intensity_col = table.indexOf(self.channels.INTENSITY)
        except RuntimeError:
            # A zero-delay install queued during table construction may outlive
            # the widget when a window is closed immediately afterward. PySide6
            # raises for that deleted C++ wrapper; there is nothing left to wire.
            return
        for row in range(table.rowCount()):
            for col in range(table.columnCount()):
                cell = table.cellWidget(row, col)
                if cell is None:
                    continue

                if col == config_col:
                    combos = ((cell,) if isinstance(cell, QComboBox) else ()) + tuple(
                        cell.findChildren(QComboBox)
                    )
                    for combo in combos:
                        if combo.property("_pymmcore_gui_channel_combo_connected"):
                            continue
                        combo.setProperty("_pymmcore_gui_channel_combo_connected", True)
                        # activated is user-only.  currentTextChanged would also
                        # fire while loading/refreshing an MDA and could move
                        # hardware.
                        #
                        # The declaration must be applied first: it updates this
                        # row's Light Source/Intensity, which the apply-to-core
                        # below then pushes to hardware.
                        combo.activated.connect(self._on_channel_config_activated)
                        combo.activated.connect(self._on_channel_combo_activated)
                elif col in (exposure_col, intensity_col) and isinstance(
                    cell, QDoubleSpinBox
                ):
                    if cell.property("_pymmcore_gui_channel_value_connected"):
                        continue
                    cell.setProperty("_pymmcore_gui_channel_value_connected", True)
                    # editingFinished is user-only (Enter or focus-out) and, unlike
                    # valueChanged, never fires from a programmatic setValue() --
                    # e.g. while restoring a sequence or re-ranging the intensity
                    # spin box for a new light source.
                    cell.editingFinished.connect(self._on_channel_value_committed)

    def _channel_index_for_editor(self, editor: QObject | None) -> QModelIndex:
        if not isinstance(editor, QWidget):
            return QModelIndex()
        table = self.channels.table()
        model = table.model()
        if model is None:  # pragma: no cover
            return QModelIndex()
        for row in range(table.rowCount()):
            for col in range(table.columnCount()):
                cell = table.cellWidget(row, col)
                if cell is editor or (cell is not None and cell.isAncestorOf(editor)):
                    return model.index(row, col)
        return QModelIndex()

    def _on_channel_combo_activated(self, *_: object) -> None:
        """Apply a channel to the microscope, same as clicking the ● column.

        Only for the *active* row, though: picking a new value in a row that
        ISN'T already active just updates that row's stored channel for later
        use -- it doesn't move hardware or change which row is active.
        Otherwise, switching which channel is loaded on the microscope while
        merely reconfiguring some other row (e.g. typing a new value into a
        row below the active one) would be a surprising side effect.
        """
        index = self._channel_index_for_editor(self.sender())
        if not index.isValid() or index.row() != self.channels.activeRow():
            return
        self._on_channel_row_selected(index)

    def _on_channel_config_activated(self, *_: object) -> None:
        """Follow the new preset's light source declaration for the edited row.

        Unlike _on_channel_combo_activated this applies to *any* row, not just the
        active one: the row now holds a different channel, so its light source
        should describe that channel whether or not hardware moves.
        """
        index = self._channel_index_for_editor(self.sender())
        if index.isValid():
            self._apply_light_source_declarations([index.row()], clear_missing=True)

    def _on_channel_cell_clicked(self, index: QModelIndex) -> None:
        """Apply a channel to the microscope only when the ● column is clicked."""
        if not index.isValid() or self._restoring_sequence:
            return
        current_col = self.channels.table().indexOf(CURRENT_CHANNEL_COLUMN)
        if index.column() != current_col:
            return
        self._on_channel_row_selected(index)

    def _on_channel_value_committed(self, *_: object) -> None:
        """Push a live-edited exposure/intensity value to the microscope.

        Fires when the user finishes editing the active row's Exposure or
        Intensity spin box (Enter or focus-out). Only while a sequence (live) is
        running: idle edits still follow the just-in-time rule and only take
        effect at the next capture. Edits to a row other than the active one
        never move hardware -- they just update the table for later use.
        """
        if self._restoring_sequence or self._applying_channel_config:
            return
        if not self._mmc.isSequenceRunning():
            return
        index = self._channel_index_for_editor(self.sender())
        if not index.isValid() or index.row() != self.channels.activeRow():
            return
        self._apply_channel_row_to_core(index.row(), include_capture_settings=True)

    def _on_table_current_row_changed(
        self, current: QModelIndex, previous: QModelIndex
    ) -> None:
        """Snap the row highlight back to the active row if anything else moved it.

        QAbstractItemView makes a cell widget's row "current" as soon as that
        widget gains keyboard focus -- e.g. simply clicking into the Exposure or
        Intensity spin box -- entirely internal to Qt, so it can't be stopped by
        filtering mouse or focus events. Undo that drift immediately (before any
        repaint, so there's no visible flicker) so the highlighted row always
        matches setActiveRow and nothing else. A legitimate change (the ● column
        or the Config combo, both via setActiveRow) already left the active row
        equal to ``current``, so this is a no-op for those -- and re-asserting
        via setActiveRow (rather than duplicating its table calls here) means
        this converges after one bounce instead of looping.
        """
        active = self.channels.activeRow()
        if current.row() != active:
            self.channels.setActiveRow(active)

    def _on_channel_row_selected(self, current: QModelIndex, *_: object) -> None:
        """Activate a channel on the microscope and update the ● indicator.

        Called when the user clicks the ● indicator column, and exposed for
        programmatic use (e.g. during tests and sequence-running channel switches).
        """
        if (
            self._restoring_sequence
            or self._applying_channel_config
            or not current.isValid()
        ):
            return
        row = current.row()
        self.channels.setActiveRow(row)
        self._apply_channel_row_to_core(
            row,
            include_capture_settings=self._mmc.isSequenceRunning(),
        )

    def _on_core_config_set(self, group: str, preset: str) -> None:
        """Select the matching row when the active preset changes on the core.

        Fires from ``configSet`` whenever any config changes (e.g. from the
        Groups & Presets table, not just this widget). Only the channel group
        is mirrored — other config groups (objective, etc.) are ignored.
        Guarded against the table→core path so we don't loop. setActiveRow
        handles both the ● indicator and the row highlight.
        """
        if self._applying_channel_config or self._restoring_sequence:
            return
        if group != self._mmc.getChannelGroup():
            return
        self.channels.setActiveRow(self._find_channel_row(preset))

    def _find_channel_row(self, preset: str) -> int:
        """Row whose channel config == ``preset``, preferring a checked one."""
        table = self.channels.table()
        channels = self.channels.value(exclude_unchecked=False)
        selector_col = table._get_selector_col()
        selector = table.columnInfo(selector_col)
        match = -1
        for row, channel in enumerate(channels):
            if str(channel.config or "") != preset:
                continue
            if match < 0:
                match = row
            if selector is not None and selector.isChecked(table, row, selector_col):
                return row
        return match

    def apply_active_channel_for_capture(self) -> bool:
        """Apply the active row's channel, exposure, and property before imaging.

        The acquisition checkbox is intentionally ignored: it controls inclusion in
        an MDA, while the table's current row is the independent live/snap selection.
        """
        return self._apply_channel_row_to_core(
            self.channels.activeRow(),
            include_capture_settings=True,
        )

    def _apply_channel_row_to_core(
        self, row: int, *, include_capture_settings: bool = False
    ) -> bool:
        if self._restoring_sequence or self._applying_channel_config:
            return False

        channels = self.channels.value(exclude_unchecked=False)
        if not 0 <= row < len(channels):
            return False

        channel = channels[row]
        config = str(channel.config or "")
        if not config:
            return False

        available_groups = set(self._mmc.getAvailableConfigGroups())
        group = str(channel.group or "")
        current_channel_group = self._mmc.getChannelGroup()
        if not group or (group == "Channel" and group not in available_groups):
            group = current_channel_group
        if not group or group not in available_groups:
            return False
        if config not in self._mmc.getAvailableConfigs(group):
            return False

        self._applying_channel_config = True
        try:
            if self._mmc.getCurrentConfig(group) != config:
                self._mmc.setConfig(group, config)

            if include_capture_settings:
                # Ensure any device delays from the selected optical preset have
                # completed before exposure/property settings and imaging.
                self._mmc.waitForConfig(group, config)

                if channel.exposure is not None:
                    self._mmc.setExposure(float(channel.exposure))
                    if camera := self._mmc.getCameraDevice():
                        self._mmc.waitForDevice(camera)

                # Although today's table exposes one ranged property per
                # channel, iterate all matching entries so this remains correct if
                # the upstream model later permits multiple ranged properties.
                for entry in self.channels.channelProperties(exclude_unchecked=False):
                    if entry["channel_index"] != row:
                        continue
                    device = entry["device"]
                    prop = entry["property"]
                    self._mmc.setProperty(device, prop, entry["value"])
                    self._mmc.waitForDevice(device)
        finally:
            self._applying_channel_config = False
        return True

    def _selected_channel_identity(self) -> tuple[str, str] | None:
        row = self.channels.activeRow()
        channels = self.channels.value(exclude_unchecked=False)
        if not 0 <= row < len(channels):
            return None
        channel = channels[row]
        return str(channel.group or ""), str(channel.config or "")

    def _restore_channel_selection(
        self,
        identity: tuple[str, str] | None,
        preferred_row: int = -1,
    ) -> None:
        """Restore the ● row (and its highlight, via setActiveRow) after a rebuild."""
        if identity is None:
            return
        channels = self.channels.value(exclude_unchecked=False)
        if 0 <= preferred_row < len(channels):
            channel = channels[preferred_row]
            candidate = str(channel.group or ""), str(channel.config or "")
            if candidate == identity:
                self.channels.setActiveRow(preferred_row)
                return
        for row, channel in enumerate(channels):
            candidate = str(channel.group or ""), str(channel.config or "")
            if candidate == identity:
                self.channels.setActiveRow(row)
                return

    def _collapsible_tabs(self) -> CollapsibleCoreMDATabs:
        return self.tabs

    def _apply_table_toolbar_icon_size(self) -> None:
        """Shrink the axis-table toolbars to the app's compact action-icon size.

        Upstream ``DataTableWidget`` hardcodes a 22px toolbar icon size, and the
        app's zoom pass makes every ``QToolBar`` even larger
        (``PM_ToolBarIconSize``) — both bigger than the toolbar actions the user
        sees elsewhere in Acquire. Match those instead (≈20px at the default
        zoom), scaled with the theme. Re-applied on theme/zoom changes from
        ``changeEvent`` because the app-wide pass would otherwise reset it.
        """
        icon = theme().scaled(16)
        size = QSize(icon, icon)
        for table in (self.channels, self.stage_positions, self.time_plan):
            table.toolBar().setIconSize(size)

    def _apply_theme_metrics(self) -> None:
        """Feed the app's zoom-scaled spacing into the upstream sections."""
        t = theme()
        self.set_section_metrics(
            SectionMetrics(
                header_height=t.row_height,
                disclosure_width=t.scaled(24),
                header_spacing=t.sp_xxs,
                body_margin_h=t.sp_sm,
                body_margin_top=t.sp_xs,
                body_margin_bottom=t.sp_sm,
                body_spacing=t.sp_sm,
                content_spacing=t.sp_xxs,
                footer_margin_h=t.sp_sm,
                footer_margin_top=t.sp_xs,
                footer_margin_bottom=t.sp_sm,
            )
        )

    def _connect_position_icon_updates(self) -> None:
        """Keep per-position sub-sequence icons themed after value changes."""
        for btn in self.findChildren(MDAButton):
            if btn.property("_pymmcore_gui_icon_colors_connected"):
                continue
            btn.setProperty("_pymmcore_gui_icon_colors_connected", True)
            btn.valueChanged.connect(self._apply_themed_icons)

    def _refresh_grid_bounds_icons(self, *_: object) -> None:
        """Theme the edge-button glyphs replaced by a Mark/Move action change."""
        bounds = self.grid_plan._core_xy_bounds
        for btn in bounds.findChildren(QPushButton):
            # The action toggle has just installed a different raw glyph, so
            # replace ensure_visible_icon's previously stashed Mark/Move source
            # before deriving the themed version from it.
            set_source_icon(btn, btn.icon())
            ensure_visible_icon(btn)

    def _apply_themed_icons(self, *_: object) -> None:
        """Apply the app's semantic green/red to every MDA action icon."""
        green = qcolor(theme().status_green).name()
        red = qcolor(theme().status_red).name()

        controls = self.control_btns
        set_source_icon(
            controls.run_btn,
            QIconifyIcon("mdi:play-circle-outline", color=green),
        )
        pause_glyph = (
            "mdi:play-circle-outline"
            if controls.pause_btn.text() == "Resume"
            else "mdi:pause-circle-outline"
        )
        set_source_icon(
            controls.pause_btn,
            QIconifyIcon(pause_glyph, color=green),
        )
        set_source_icon(
            controls.cancel_btn,
            QIconifyIcon("mdi:stop-circle-outline", color=red),
        )

        for table in (self.channels, self.stage_positions, self.time_plan):
            table.act_add_row.setIcon(QIconifyIcon("mdi:plus-thick", color=green))
            table.act_remove_row.setIcon(StandardIcon.DELETE.icon(red))
            table.act_clear.setIcon(StandardIcon.DELETE_ALL.icon(red))

        for btn in self.findChildren(MDAButton):
            configured = not btn.clear_btn.isHidden()
            seq_icon = (
                QIconifyIcon("mdi:axis-arrow", color=green)
                if configured
                else QIconifyIcon("mdi:axis")
            )
            set_source_icon(btn.seq_btn, seq_icon)
            ensure_visible_icon(btn.seq_btn)
            set_source_icon(
                btn.clear_btn,
                QIconifyIcon("mdi:close-circle", color=red),
            )

    def changeEvent(self, a0: QEvent | None) -> None:
        """Re-theme icons and rescale section spacing after a style/zoom change."""
        super().changeEvent(a0)
        if (
            a0 is not None
            and a0.type() == QEvent.Type.StyleChange
            and hasattr(self, "control_btns")
        ):
            self._apply_themed_icons()
            self._apply_theme_metrics()
            self._apply_table_toolbar_icon_size()

    def prepare_mda(self) -> bool | str | Path | None:
        """Return a disk path or a scratch sink that supports live viewing."""
        output = super().prepare_mda()
        return "memory" if output is None else output
