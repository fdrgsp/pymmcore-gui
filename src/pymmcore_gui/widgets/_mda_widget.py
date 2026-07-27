"""Shared MDA editor configured for the ome-writers sink."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pymmcore_widgets import MDAWidgetCollapsible
from pymmcore_widgets.mda import CollapsibleCoreMDATabs, SectionMetrics
from pymmcore_widgets.useq_widgets._positions import MDAButton
from superqt.iconify import QIconifyIcon

from pymmcore_gui._array_viewer import (
    ensure_visible_icon,
    set_source_icon,
    unstyle_widgets,
)
from pymmcore_gui._gui._theme import qcolor, theme
from pymmcore_gui._qt.QtCore import QEvent, QModelIndex, QObject, QSize, Qt, QTimer
from pymmcore_gui._qt.QtWidgets import QComboBox, QGridLayout, QWidget

from ._ranged_property_channels import RangedPropertyCollapsibleCoreMDATabs

if TYPE_CHECKING:
    from pathlib import Path

    import useq
    from pymmcore_plus import CMMCorePlus
    from pymmcore_widgets.mda._xy_bounds import CoreXYBoundsControl


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

    def _create_tab_widget(self) -> CollapsibleCoreMDATabs:
        return RangedPropertyCollapsibleCoreMDATabs(None, self._mmc)

    def __init__(self, mmcore: CMMCorePlus, parent: QWidget | None = None) -> None:
        self._restoring_sequence = False
        self._applying_channel_config = False
        super().__init__(parent=parent, mmcore=mmcore)
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

        _align_bounds_grid(self.grid_plan._core_xy_bounds)

    def _on_table_rows_inserted(self, *_: object) -> None:
        """Theme cell widgets that are constructed only when a row is added."""
        unstyle_widgets(self)
        self._collapsible_tabs().apply_save_body_style()
        self._install_channel_editor_filters()
        self._connect_position_icon_updates()
        self._apply_themed_icons()

    def setValue(self, value: useq.MDASequence) -> None:
        """Restore an MDA sequence without applying a selected row to the core."""
        selected = self._selected_channel_identity()
        selected_row = self.channels.table().currentRow()
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
        selected_row = table.currentRow()
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
        self._collapsible_tabs().refresh_summaries()

    def _connect_channel_selection(self) -> None:
        table = self.channels.table()
        selection_model = table.selectionModel()
        if selection_model is None:  # pragma: no cover
            return
        selection_model.currentRowChanged.connect(self._on_channel_row_selected)
        table.clicked.connect(self._on_channel_row_selected)
        model = table.model()
        if model is not None:
            model.columnsInserted.connect(self._schedule_channel_editor_filters)
        self._install_channel_editor_filters()

    def _schedule_channel_editor_filters(self, *_: object) -> None:
        # columnsInserted fires from insertColumn(), before DataTable.addColumn()
        # has populated that column's cell widgets. Install after the insertion
        # event has completed so newly rebuilt config/property columns are
        # included.
        QTimer.singleShot(0, self._install_channel_editor_filters)

    def _install_channel_editor_filters(self) -> None:
        """Make interaction with any channel-cell editor activate its row.

        QTableWidget does not receive mouse events handled by cell widgets such as
        the channel combo and exposure spin box. Installing the filter on every
        editor and descendant lets those normal editing interactions also establish
        the table's single active/current row without stealing focus from the editor.
        """
        table = self.channels.table()
        config_col = table.indexOf(self.channels._config_column)
        for row in range(table.rowCount()):
            for col in range(table.columnCount()):
                cell = table.cellWidget(row, col)
                if cell is None:
                    continue
                editors = (cell, *cell.findChildren(QWidget))
                for editor in editors:
                    if not editor.property("_pymmcore_gui_channel_row_filter"):
                        editor.installEventFilter(self)
                        editor.setProperty("_pymmcore_gui_channel_row_filter", True)

                if col != config_col:
                    continue
                combos = ((cell,) if isinstance(cell, QComboBox) else ()) + tuple(
                    cell.findChildren(QComboBox)
                )
                for combo in combos:
                    if combo.property("_pymmcore_gui_channel_combo_connected"):
                        continue
                    combo.setProperty("_pymmcore_gui_channel_combo_connected", True)
                    # activated is user-only.  currentTextChanged would also fire
                    # while loading/refreshing an MDA and could move hardware.
                    combo.activated.connect(self._on_channel_combo_activated)

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

    def _activate_channel_index(self, index: QModelIndex) -> None:
        """Select/highlight ``index``'s row and apply its current config."""
        if not index.isValid() or self._restoring_sequence:
            return
        table = self.channels.table()
        previous_row = table.currentRow()
        table.setCurrentCell(index.row(), index.column())
        table.selectRow(index.row())
        # A row change is handled synchronously by currentRowChanged. Re-apply
        # explicitly for another cell in the same row, most importantly after a
        # user changes that row's channel combo.
        if previous_row == index.row():
            self._on_channel_row_selected(index)

    def _on_channel_combo_activated(self, *_: object) -> None:
        self._activate_channel_index(self._channel_index_for_editor(self.sender()))

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if a1 is not None and a1.type() in (
            QEvent.Type.MouseButtonPress,
            QEvent.Type.FocusIn,
        ):
            index = self._channel_index_for_editor(a0)
            if index.isValid():
                self._activate_channel_index(index)
        return super().eventFilter(a0, a1)

    def _on_channel_row_selected(self, current: QModelIndex, *_: object) -> None:
        if (
            self._restoring_sequence
            or self._applying_channel_config
            or not current.isValid()
        ):
            return

        self._apply_channel_row_to_core(
            current.row(),
            include_capture_settings=self._mmc.isSequenceRunning(),
        )

    def apply_active_channel_for_capture(self) -> bool:
        """Apply the active row's channel, exposure, and property before imaging.

        The acquisition checkbox is intentionally ignored: it controls inclusion in
        an MDA, while the table's current row is the independent live/snap selection.
        """
        return self._apply_channel_row_to_core(
            self.channels.table().currentRow(),
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
        row = self.channels.table().currentRow()
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
        if identity is None:
            return
        table = self.channels.table()
        config_col = table.indexOf(self.channels._config_column)
        channels = self.channels.value(exclude_unchecked=False)
        if 0 <= preferred_row < len(channels):
            channel = channels[preferred_row]
            candidate = str(channel.group or ""), str(channel.config or "")
            if candidate == identity:
                table.setCurrentCell(preferred_row, max(0, config_col))
                return
        for row, channel in enumerate(channels):
            candidate = str(channel.group or ""), str(channel.config or "")
            if candidate == identity:
                table.setCurrentCell(row, max(0, config_col))
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
            table.act_remove_row.setIcon(
                QIconifyIcon("mdi:close-box-outline", color=red)
            )
            table.act_clear.setIcon(
                QIconifyIcon("mdi:close-box-multiple-outline", color=red)
            )

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
