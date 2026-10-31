"""Shared MDA editor configured for the ome-writers sink."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from ome_writers import AcquisitionSettings, OmeTiffFormat, ScratchFormat
from ome_writers._schema import _ome_stem_suffix
from pymmcore_widgets import (
    MDAWidget,
    MDAWidgetCollapsible,
    MDAWidgetTopbar,
)
from pymmcore_widgets._icons import StandardIcon
from pymmcore_widgets.mda import (
    CollapsibleCoreMDATabs,
    SectionMetrics,
    TopbarMDATabs,
)
from pymmcore_widgets.mda._collapsible_mda import _ClickableLabel
from pymmcore_widgets.useq_widgets._positions import MDAButton
from superqt.iconify import QIconifyIcon

from pymmcore_gui._array_viewer import (
    _recolor_icon,
    ensure_visible_icon,
    set_source_icon,
    unstyle_widgets,
)
from pymmcore_gui._channel_settings_store import channel_settings_store
from pymmcore_gui._qt.QtCore import (
    QEvent,
    QModelIndex,
    QObject,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from pymmcore_gui._qt.QtGui import QPainter, QPixmap
from pymmcore_gui._qt.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QToolButton,
    QWidget,
)
from pymmcore_gui._settings import Settings
from pymmcore_gui._theme import qcolor, theme
from pymmcore_gui.widgets._busy import BusyOverlay

from ._mda_tabs import (
    GuiCollapsibleCoreMDATabs,
    GuiTopbarMDATabs,
    install_third_party_window_theming,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    import useq
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.mda import SingleOutput
    from pymmcore_widgets.mda._xy_bounds import CoreXYBoundsControl

    from pymmcore_gui._qt.QtGui import QResizeEvent

    from ._mda_channel_table import MDAChannelTable

# Qt's "no maximum", needed to undo a setFixedHeight before recomputing it.
QWIDGETSIZE_MAX = (1 << 24) - 1

# The subset of `OmeTiffFormat.multi_file_metadata` this GUI offers. "companion-file"
# is deliberately left out: it splits metadata into a sidecar that is easy to lose,
# with no advantage over "master-tiff".
TiffLayout = Literal["self-contained", "master-tiff", "redundant"]

# (label, multi_file_metadata, tooltip) for the OME-TIFF per-position layouts,
# in the order they are offered. The first is the default.
TIFF_LAYOUTS: tuple[tuple[str, TiffLayout, str], ...] = (
    (
        "Independent file per position",
        "self-contained",
        "<b>Each file stands alone.</b><br><br>A file describes only its own "
        "position and refers to no other, so any file can be opened, moved, "
        "renamed or deleted on its own, and the data store is created in constant "
        "time per position.<br><br>Readers show separate single-position images "
        "rather than one grouped acquisition. Each file still records its own "
        "stage position, and for a plate its own well, field index and the full "
        "plate layout — only the grouping is lost.",
    ),
    (
        "Linked dataset, metadata in the first file",
        "master-tiff",
        "<b>One grouped acquisition, described once.</b><br><br>The first file "
        "carries the metadata for every position; the rest hold pixels and point "
        "back to it. Readers open all positions as a single acquisition, and the "
        "data store is created as quickly as with independent files.<br><br>The "
        "set must be kept together: a file on its own carries no metadata at all, "
        "and losing the first file leaves the others undescribed.",
    ),
    (
        "Linked dataset, metadata in every file",
        "redundant",
        "<b>One grouped acquisition, described in every file.</b><br><br>Every "
        "file repeats the metadata for all positions, so any of them can serve as "
        "the entry point and no single file is critical.<br><br>That copying makes "
        "store creation slow down sharply as positions are added — noticeable past "
        "~100, and tens of seconds for several hundred. The files still reference "
        "each other, so one taken out of the folder is not usable on its own.",
    ),
)

TIFF_LAYOUT_TOOLTIP = (
    "How the per-position OME-TIFF files relate to each other. "
    "Hover an entry for what it means."
)


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


def _section_metrics() -> SectionMetrics:
    """The app's zoom-scaled spacing for the collapsible presentation."""
    t = theme()
    return SectionMetrics(
        header_height=t.row_height,
        disclosure_width=t.scaled(24),
        header_spacing=t.sp_xxs,
        body_margin_h=t.sp_sm,
        body_margin_top=t.sp_xs,
        body_margin_bottom=t.sp_sm,
        body_spacing=t.sp_sm,
        content_spacing=t.sp_xxs,
        content_margin_h=t.sp_sm,
        footer_margin_h=t.sp_sm,
        footer_margin_top=t.sp_xs,
        footer_margin_bottom=t.sp_sm,
    )


def _memory_output_settings() -> AcquisitionSettings:
    """Build the "no saving" scratch output from the user's Data & Memory prefs.

    Used whenever the Saving section is unchecked -- see ``MemoryMDAWidget.
    prepare_mda``. The resulting `AcquisitionSettings` still gives the run a
    live, viewable array (via `ome_writers`' scratch backend); nothing is
    written to disk unless the in-memory limit below is exceeded.
    """
    prefs = Settings.instance().scratch
    fmt = ScratchFormat(
        max_memory_bytes=round(prefs.max_memory_gb * 1024**3),
        spill_to_disk=prefs.spill_to_disk,
        spill_dir=str(prefs.scratch_dir) if prefs.scratch_dir else None,
    )
    return AcquisitionSettings(format=fmt)


if TYPE_CHECKING:
    # Type-checkers see the real base, so everything the mixin touches
    # (save_info, channels, control_btns, ...) resolves normally.
    _MixinBase = MDAWidget
else:
    # At runtime it must NOT be a QObject: PyQt cannot combine two
    # QObject-derived classes, and inheriting MDAWidget here as well as through
    # the concrete flavour corrupts the generated meta-object -- signals then
    # resolve to the wrong slot (roiSelectionRequested came back as the bool
    # mdaLockChanged). A plain object mixin sidesteps that entirely.
    _MixinBase = object


class MemoryMDAWidgetBase(_MixinBase):
    """Everything this app adds to an MDA editor, independent of its presentation.

    Both upstream presentations (`MDAWidgetCollapsible`'s sections,
    `MDAWidgetTopbar`'s tab bar) derive from `MDAWidget` and expose the same
    editors, footer and saving widgets, so the app theme, semantic icons, the
    channel-selection core bridge, the acquisition lock and the memory-sink
    output fallback all live here and are shared verbatim. Both flavours must
    behave identically; only their layout differs.

    What genuinely differs is confined to `tabs`: the collapsible one has
    `sections`/`saving_section`/`refresh_summaries`, while the top bar is a
    real `QTabWidget` addressed through its native tab API
    (`setChecked`/`setCurrentWidget`/`indexOf`) instead. Reached through the
    small hooks below (`_flavor_*`) rather than by branching through the rest
    of the class.
    """

    if TYPE_CHECKING:
        # Supplied by whichever concrete flavour this is mixed into. Their tab
        # widgets share every member used here but have no common subclass that
        # declares them, so they are typed loosely rather than narrowed to one.
        #
        # Both flavours expose `tabs` as a property, so it is declared as one
        # here too -- a plain annotation would read as an incompatible override.
        @property
        def tabs(self) -> Any: ...

        # Declared for real on each concrete flavour -- a signal has to be
        # defined on a QObject subclass, which this mixin deliberately is not.
        _sequenceStartedInGui: Any
        _sequenceFinishedInGui: Any
        mdaLockChanged: Any

        # The settings card `_install_layout` builds on each concrete flavour
        # (axis order / keep shutter open / autofocus axis) -- not part of
        # upstream `MDAWidget`, so `_MixinBase` doesn't carry it either.
        _settings_box: Any

    # ----------- Override type hints in superclass -----------
    # _create_tab_widget above always installs an MDAChannelTable, which adds
    # the settings store link to the upstream CoreConnectedChannelTable
    # interface; narrow the property's declared return type to match.

    @property
    def channels(self) -> MDAChannelTable:
        return cast("MDAChannelTable", super().channels)

    def __init__(self, mmcore: CMMCorePlus, parent: QWidget | None = None) -> None:
        self._restoring_sequence = False
        self._cancel_requested = False
        self._mda_locked = False
        super().__init__(parent=parent, mmcore=mmcore)
        install_third_party_window_theming()
        self._update_time_estimate()
        self._progress_overlay = BusyOverlay(self)
        self._sequenceStartedInGui.connect(self._on_sequence_started_in_gui)
        self._sequenceFinishedInGui.connect(self._on_sequence_finished_in_gui)
        self._mmc.mda.events.sequenceStarted.connect(self._relay_sequence_started)
        self._mmc.mda.events.sequenceFinished.connect(self._relay_sequence_finished)
        self._mda_state_timer = QTimer(self)
        self._mda_state_timer.setInterval(100)
        self._mda_state_timer.timeout.connect(self._sync_mda_state)
        self.control_btns.cancel_btn.released.connect(self._on_cancel_requested)
        self._apply_theme_metrics()
        combo = self.save_info._writer_combo
        for idx in range(combo.count()):
            if combo.itemText(idx) == "tiff-sequence":
                combo.removeItem(idx)
                break
        combo.setCurrentText("ome-tiff")
        self._install_ome_tiff_options()

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
        # built on demand (e.g. the Positions row's black "mdi:grid" per-
        # position grid button, or the row's spinboxes), created only when a
        # row is actually added -- long after the sweep above already ran.
        # Without this, a row added interactively is never covered by any
        # sweep at all until the next light/dark toggle (which only revisits
        # icons, not stylesheets).
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

        # Link the channel rows to the per-preset settings the Channels panel
        # edits: new rows start from them, and user edits are stored back.
        self.channels.setSettingsStore(channel_settings_store(mmcore))
        self.channels.applyStoredSettings(range(self.channels.table().rowCount()))
        if (model := self.channels.table().model()) is not None:
            model.columnsInserted.connect(self._schedule_channel_editor_filters)
        self._install_channel_editor_filters()

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
        """Keep the acquisition progress overlay fitted to the MDA editor."""
        super().resizeEvent(a0)
        if hasattr(self, "_progress_overlay"):
            self._progress_overlay.setGeometry(self.rect())

    def run_mda(self) -> None:
        """Start an MDA, showing progress while a disk store is initialized."""
        output = self.prepare_mda()
        if isinstance(output, bool):
            return

        show_store_progress = self.save_info.isChecked() and self.isVisible()
        if show_store_progress:
            self._progress_overlay.start("Creating data store…")
        try:
            self.execute_mda(output)
        except Exception:
            # A synchronous launch failure emits neither sequenceStarted nor
            # sequenceFinished, so clear the overlay here.
            if show_store_progress:
                self._progress_overlay.stop()
            raise

    def _relay_sequence_started(self, *_: object) -> None:
        """Relay the worker-thread start event safely to the GUI thread."""
        self._sequenceStartedInGui.emit()

    def _relay_sequence_finished(self, *_: object) -> None:
        """Relay the worker-thread finish event safely to the GUI thread."""
        self._sequenceFinishedInGui.emit()

    def _on_sequence_started_in_gui(self) -> None:
        self._cancel_requested = False
        self._progress_overlay.stop()
        self._mda_state_timer.start()
        self._set_mda_lock(True)

    def _on_sequence_finished_in_gui(self) -> None:
        self._sync_mda_state()

    def cancel_acquisition(self) -> None:
        """Cancel a run from outside this widget, with the same feedback as its button.

        Exactly what ``control_btns.cancel_btn`` does (it is wired to both
        ``_on_cancel_requested`` here and ``mda.cancel`` upstream), so a
        cancellation requested by e.g. the window closing still gets the
        "Cancelling acquisition…" overlay. A no-op when nothing is running.
        """
        if self._mmc.mda.status.phase.value == "idle":
            return
        self._on_cancel_requested()
        self._mmc.mda.cancel()
        # Normally already running (sequenceStarted starts it). Starting it here
        # too means a caller waiting on the idle transition still gets it even
        # if that signal was missed and the timer was never started.
        self._mda_state_timer.start()

    def _set_mda_lock(self, locked: bool) -> None:
        """Announce a change in whether an acquisition owns the hardware."""
        if locked != self._mda_locked:
            self._mda_locked = locked
            self.mdaLockChanged.emit(locked)

    def _on_cancel_requested(self) -> None:
        """Show cancellation progress until the runner has completely stopped."""
        if self._mmc.mda.status.phase.value == "idle":
            return
        self._cancel_requested = True
        if self.isVisible():
            self._progress_overlay.start("Cancelling acquisition…")

    def _sync_mda_state(self) -> None:
        """Show finalization progress and recover controls from missed signals."""
        phase = self._mmc.mda.status.phase.value
        if self._cancel_requested and phase != "idle":
            if self.isVisible() and (
                self._progress_overlay.isHidden()
                or self._progress_overlay._message != "Cancelling acquisition…"
            ):
                self._progress_overlay.start("Cancelling acquisition…")
            return

        if phase == "finishing":
            if (
                self.save_info.isChecked()
                and self.isVisible()
                and (
                    self._progress_overlay.isHidden()
                    or self._progress_overlay._message != "Finalizing data store…"
                )
            ):
                self._progress_overlay.start("Finalizing data store…")
            return

        if phase != "idle":
            return

        self._cancel_requested = False
        self._mda_state_timer.stop()
        self._progress_overlay.stop()
        # sequenceFinished normally restores these through pymmcore-widgets.
        # Polling the runner provides a fallback if that GUI notification is
        # delayed or missed after cancellation/writer teardown.
        self._enable_widgets(True)
        controls = self.control_btns
        controls.run_btn.show()
        controls.pause_btn.hide()
        controls.cancel_btn.hide()
        controls._on_mda_paused(False)
        self._apply_themed_icons()
        self._set_mda_lock(False)

    def _disconnect(self) -> None:
        """Disconnect this subclass's MDA callbacks, then the upstream ones."""
        with suppress(Exception):
            self._mmc.mda.events.sequenceStarted.disconnect(
                self._relay_sequence_started
            )
            self._mmc.mda.events.sequenceFinished.disconnect(
                self._relay_sequence_finished
            )
            self.control_btns.cancel_btn.released.disconnect(self._on_cancel_requested)
        super()._disconnect()

    def _on_table_rows_inserted(
        self, _parent: QModelIndex, first: int, last: int
    ) -> None:
        """Theme new cells and give new channel rows their presets' settings."""
        unstyle_widgets(self)
        self._collapsible_tabs().apply_save_body_style()
        self._install_channel_editor_filters()
        self._connect_position_icon_updates()
        self._apply_themed_icons()
        if (
            self.sender() is self.channels.table().model()
            and not self._restoring_sequence
        ):
            self.channels.applyStoredSettings(range(first, last + 1))

    def setValue(self, value: useq.MDASequence) -> None:
        """Restore an MDA sequence, keeping its own channel settings."""
        self._restoring_sequence = True
        try:
            super().setValue(value)
        finally:
            self._restoring_sequence = False
        self._flavor_refresh_summaries()

    def refresh_channel_table(self) -> None:
        """Refresh core-backed channel choices, keeping the rows."""
        channels = self.channels
        table = channels.table()
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
        finally:
            self._restoring_sequence = False
        self._flavor_refresh_summaries()

    def _schedule_channel_editor_filters(self, *_: object) -> None:
        # columnsInserted fires from insertColumn(), before DataTable.addColumn()
        # has populated that column's cell widgets. Install after the insertion
        # event has completed so newly rebuilt config/property columns are
        # included.
        QTimer.singleShot(0, self._install_channel_editor_filters)

    def _install_channel_editor_filters(self) -> None:
        """Connect the editors whose user edits change a channel's settings.

        A row's Config combo switching to another preset gives the row that
        preset's settings; an edit of its exposure, light source, intensity or
        camera properties is stored for its preset. Only user-driven signals are
        used -- ``activated`` and ``editingFinished`` -- so that restoring a
        sequence or rebuilding a column never overwrites a preset's settings.
        """
        channels = self.channels
        table = channels.table()
        try:
            config_col = table.indexOf(channels._config_column)
            settings_cols = {
                table.indexOf(channels.EXPOSURE),
                table.indexOf(channels.INTENSITY),
                table.indexOf(channels._light_source_column),
                *(table.indexOf(info) for info in channels._camera_columns.values()),
            }
        except RuntimeError:
            # A zero-delay install queued during table construction may outlive
            # the widget when a window is closed immediately afterward. PySide6
            # raises for that deleted C++ wrapper; there is nothing left to wire.
            return
        for row in range(table.rowCount()):
            for col in range(table.columnCount()):
                if (cell := table.cellWidget(row, col)) is None:
                    continue
                if col in settings_cols:
                    self._connect_channel_editor(cell, self._on_channel_settings_edited)
                elif col == config_col:
                    for combo in ((cell,) if isinstance(cell, QComboBox) else ()) + (
                        tuple(cell.findChildren(QComboBox))
                    ):
                        self._connect_channel_editor(
                            combo, self._on_channel_config_activated
                        )

    def _connect_channel_editor(self, cell: QWidget, slot: Any) -> None:
        if cell.property("_pymmcore_gui_channel_editor_connected"):
            return
        cell.setProperty("_pymmcore_gui_channel_editor_connected", True)
        if isinstance(cell, QComboBox):
            cell.activated.connect(slot)
        elif isinstance(cell, QDoubleSpinBox):
            cell.editingFinished.connect(slot)

    def _on_channel_settings_edited(self, *_: object) -> None:
        if self._restoring_sequence:
            return
        index = self._channel_index_for_editor(self.sender())
        if index.isValid():
            self.channels.storeRowSettings(index.row())

    def _on_channel_config_activated(self, *_: object) -> None:
        """Give a row switched to another preset that preset's settings."""
        index = self._channel_index_for_editor(self.sender())
        if index.isValid():
            self.channels.restoreChannelSettings(index.row())

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

    def _collapsible_tabs(self) -> Any:
        """The tab widget, whichever presentation this flavour installed."""
        return self.tabs

    # ------------------------- per-flavour hooks -------------------------

    def _flavor_apply_theme_metrics(self) -> None:
        """Feed zoom-scaled spacing to a presentation that has such knobs."""

    def _flavor_refresh_summaries(self) -> None:
        """Re-render whatever per-axis summary the presentation shows."""

    def _flavor_icon_labels(self) -> Iterable[QLabel]:
        """Header labels whose pixmaps need tinting to the app's icon colour."""
        return ()

    def _flavor_apply_tab_icons(self) -> None:
        """Tint native `QTabBar` tab icons, for a presentation built on one."""

    def _flavor_reveal_axis(self, axis: str, editor: QWidget) -> None:
        """Bring `axis` into view in whatever way this presentation allows."""

    def axis_editor(self, axis: str) -> QWidget:
        """The editor widget for an axis key, present in every presentation."""
        editors: dict[str, QWidget] = {
            "c": self.channels,
            "p": self.stage_positions,
            "g": self.grid_plan,
            "z": self.z_plan,
            "t": self.time_plan,
        }
        if (editor := editors.get(axis)) is None:  # pragma: no cover
            raise ValueError(f"Unknown MDA axis: {axis!r}")
        return editor

    def revealAxis(self, axis: str) -> None:
        """Turn an axis on and bring its editor into view.

        Flavour-agnostic so callers (e.g. Stage Explorer's "send to MDA") do not
        have to know whether the axes are collapsible sections or top-bar tabs.
        """
        self._flavor_reveal_axis(axis, self.axis_editor(axis))

    def _apply_table_toolbar_icon_size(self) -> None:
        """Shrink the axis-table toolbars to the app's compact action-icon size.

        Upstream ``DataTableWidget`` hardcodes a 22px toolbar icon size, and the
        app's zoom pass makes every ``QToolBar`` even larger
        (``PM_ToolBarIconSize``) — both bigger than the toolbar actions the user
        sees elsewhere in Acquire. Match those instead (≈20px at the default
        zoom), scaled with the theme. Re-applied on theme/zoom changes from
        ``changeEvent`` because the app-wide pass would otherwise reset it.

        Also turns off each button's auto-raise, matching every other action
        button in Acquire (see Stage Explorer's own toolbar for the same
        fix): a QToolBar's buttons default to auto-raise ("ghost" style,
        fill-only, borderless hover), which reads as barely-there next to the
        rest of the app's persistently boxed "subtle" buttons.
        """
        icon = theme().scaled(16)
        size = QSize(icon, icon)
        for table in (self.channels, self.stage_positions, self.time_plan):
            toolbar = table.toolBar()
            toolbar.setIconSize(size)
            for action in toolbar.actions():
                button = toolbar.widgetForAction(action)
                if isinstance(button, QToolButton):
                    button.setAutoRaise(False)
                    if not button.property("variant"):
                        button.setProperty("variant", "subtle")
                    ensure_visible_icon(button)

    def _apply_theme_metrics(self) -> None:
        """Feed the app's zoom-scaled spacing into the upstream presentation."""
        self._flavor_apply_theme_metrics()

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

    def _apply_themed_section_icons(self) -> None:
        """Tint the collapsible section header icons to the app's icon color.

        Upstream renders them as uncolored `QIconifyIcon` pixmaps on labels
        (`_collapsible_mda.py`), so they come out black, and the app's icon
        sweeps never reach them -- those recolor `QAbstractButton` icons, not
        label pixmaps. Only `_ClickableLabel` is considered, which is the
        header's own class: a plain `QLabel` in a section body may hold a
        deliberately colored pixmap that must not be tinted.

        Tinting is idempotent (`SourceIn` replaces RGB and keeps alpha), so
        this can re-run on every theme change without drifting.
        """
        color = qcolor(theme().text_secondary)
        for label in self._flavor_icon_labels():
            pixmap = label.pixmap()
            if pixmap is None or pixmap.isNull():
                continue  # the header's title/summary labels carry text, not icons
            tinted = QPixmap(pixmap)
            painter = QPainter(tinted)
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceIn
            )
            painter.fillRect(tinted.rect(), color)
            painter.end()
            label.setPixmap(tinted)

    def _apply_themed_icons(self, *_: object) -> None:
        """Apply the app's semantic green/red to every MDA action icon."""
        self._apply_themed_section_icons()
        self._flavor_apply_tab_icons()
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
                QIconifyIcon("mdi:grid", color=green)
                if configured
                else QIconifyIcon("mdi:grid")
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

    # ------------------------- OME-TIFF layout option -------------------------

    def _install_ome_tiff_options(self) -> None:
        """Add the per-position file layout choice to the Saving section."""
        self._tiff_layout_label = QLabel("Tiff Mode:")
        self._tiff_layout_combo = QComboBox()
        self._tiff_layout_combo.setToolTip(TIFF_LAYOUT_TOOLTIP)
        for idx, (label, mode, tip) in enumerate(TIFF_LAYOUTS):
            self._tiff_layout_combo.addItem(label, userData=mode)
            self._tiff_layout_combo.setItemData(idx, tip, Qt.ItemDataRole.ToolTipRole)
        # TIFF_LAYOUTS is ordered, and its first entry is the default.
        self._tiff_layout_combo.setCurrentIndex(0)

        layout = cast("QGridLayout", self.save_info.layout())
        layout.addWidget(self._tiff_layout_label, 2, 0)
        layout.addWidget(self._tiff_layout_combo, 2, 1, 1, 3)
        self.save_info._writer_combo.currentTextChanged.connect(
            self._sync_ome_tiff_options
        )
        self._sync_ome_tiff_options()

    def _sync_ome_tiff_options(self, *_: object) -> None:
        """Show the layout choice only for the writer it applies to."""
        visible = self._is_ome_tiff_selected()
        self._tiff_layout_label.setVisible(visible)
        self._tiff_layout_combo.setVisible(visible)
        # SaveGroupBox pins its own height to avoid jiggling when toggled, so the
        # pin has to be recomputed whenever a row appears or disappears.
        self.save_info.setMinimumHeight(0)
        self.save_info.setMaximumHeight(QWIDGETSIZE_MAX)
        self.save_info.setFixedHeight(self.save_info.minimumSizeHint().height())

    def _is_ome_tiff_selected(self) -> bool:
        return self.save_info._writer_combo.currentText() == "ome-tiff"

    def tiffLayout(self) -> TiffLayout:
        """The selected `OmeTiffFormat.multi_file_metadata` mode."""
        # every item's data came from TIFF_LAYOUTS, so it is one of the literals
        return cast("TiffLayout", self._tiff_layout_combo.currentData())

    def setTiffLayout(self, mode: TiffLayout) -> None:
        """Select the layout whose `multi_file_metadata` mode is `mode`."""
        idx = self._tiff_layout_combo.findData(mode)
        if idx < 0:  # pragma: no cover
            raise ValueError(f"Unknown OME-TIFF layout {mode!r}")
        self._tiff_layout_combo.setCurrentIndex(idx)

    def _sequence_uses_grid(self) -> bool:
        """Whether the sequence to run has a grid plan, global or per position."""
        sequence = self.value()
        return sequence.grid_plan is not None or any(
            pos.sequence is not None and pos.sequence.grid_plan is not None
            for pos in sequence.stage_positions
        )

    def _confirm_missing_pixel_size(self) -> bool:
        """Ask before running a grid plan that has no pixel size to scale it."""
        response = QMessageBox.warning(
            self,
            "No Pixel Size",
            "The current objective/resolution preset has no pixel size, but this "
            "sequence uses a grid plan. A grid's spacing and overlap come from the "
            "camera's field of view, which needs the pixel size, so the tiles would "
            "not be placed correctly. Set a pixel size in the Pixel Configuration "
            "Editor (Configurations tab) first."
            "\n\nRun anyway?",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return bool(response == QMessageBox.StandardButton.Ok)

    def prepare_mda(self) -> bool | SingleOutput | None:
        """Return a disk path or a scratch sink that supports live viewing."""
        if (
            self._mmc.getPixelSizeUm() <= 0
            and self._sequence_uses_grid()
            and not self._confirm_missing_pixel_size()
        ):
            return False
        output = super().prepare_mda()
        if isinstance(output, bool):
            return output
        if output is None:
            return _memory_output_settings()
        if isinstance(output, (str, Path)) and self._is_ome_tiff_selected():
            # Always state the layout rather than letting any of them fall through
            # to ome-writers' default: what the user picked here should not change
            # meaning if that default is ever changed.
            # Stating the format means stating its suffix too: OmeTiffFormat
            # rebuilds the output path from `suffix`, whose default (.ome.tiff)
            # would quietly rewrite a destination the user named .ome.tif.
            suffix = _ome_stem_suffix(str(output))[1]
            tiff_format = (
                OmeTiffFormat(multi_file_metadata=self.tiffLayout(), suffix=suffix)
                if suffix
                else OmeTiffFormat(multi_file_metadata=self.tiffLayout())
            )
            return AcquisitionSettings(root_path=str(output), format=tiff_format)
        return output


class MemoryMDAWidget(MemoryMDAWidgetBase, MDAWidgetCollapsible):
    """The app's MDA editor, presented as upstream's collapsible sections."""

    _sequenceStartedInGui = Signal()
    _sequenceFinishedInGui = Signal()

    mdaLockChanged = Signal(bool)
    """Emitted True when a run takes the hardware, False once the runner is idle.

    The app-wide acquisition lock hangs off this (see
    ``AcquirePage.set_mda_lock``). The False edge is emitted from
    ``_sync_mda_state``, which polls the runner for as long as a run is active,
    so the same safety net that recovers this widget's own controls from a
    missed ``sequenceFinished`` also releases the rest of the GUI -- the lock
    can never outlive the acquisition.
    """

    def _create_tab_widget(self) -> CollapsibleCoreMDATabs:
        return GuiCollapsibleCoreMDATabs(None, self._mmc)

    def _flavor_apply_theme_metrics(self) -> None:
        self.set_section_metrics(_section_metrics())

    def _flavor_refresh_summaries(self) -> None:
        self.tabs.refresh_summaries()

    def _flavor_icon_labels(self) -> Iterable[QLabel]:
        return list(self.tabs.findChildren(_ClickableLabel))

    def _flavor_reveal_axis(self, axis: str, editor: QWidget) -> None:
        section = self.tabs.section(axis)
        section.set_checked(True)
        section.set_expanded(True)
        self.tabs.refresh_summaries()


class TopbarMemoryMDAWidget(MemoryMDAWidgetBase, MDAWidgetTopbar):
    """The same editor, presented as upstream's native icon-labelled top tabs.

    Identical in behaviour to `MemoryMDAWidget` -- everything the app adds lives
    in `MemoryMDAWidgetBase`. This flavour is structurally the closest to plain
    `MDAWidget`: `tabs` is a real `QTabWidget`, so the hooks below use its
    native tab API (`setChecked`/`setCurrentWidget`) directly rather than a
    presentation-specific row/section object.
    """

    _sequenceStartedInGui = Signal()
    _sequenceFinishedInGui = Signal()

    mdaLockChanged = Signal(bool)
    """Emitted True when a run takes the hardware, False once the runner is idle.

    The app-wide acquisition lock hangs off this (see
    ``AcquirePage.set_mda_lock``). The False edge is emitted from
    ``_sync_mda_state``, which polls the runner for as long as a run is active,
    so the same safety net that recovers this widget's own controls from a
    missed ``sequenceFinished`` also releases the rest of the GUI -- the lock
    can never outlive the acquisition.
    """

    def _create_tab_widget(self) -> TopbarMDATabs:
        return GuiTopbarMDATabs(None, self._mmc)

    def _enable_widgets(self, enable: bool) -> None:
        """Disable editors while keeping the topbar footer controls clickable.

        Upstream's generic implementation disables every direct child except
        ``control_btns``.  In the topbar presentation those controls are not a
        direct child: they live inside ``mdaExecutionFooter`` with Save/Load.
        Disabling that frame also disables Pause and Cancel through their
        ancestor, despite the buttons themselves remaining enabled.  Keep the
        footer alive and lock only its editing actions explicitly.
        """
        self.tabs._enable_tabs(enable)
        self._settings_box.setEnabled(enable)
        self._save_button.setEnabled(enable)
        self._load_button.setEnabled(enable)

    def _flavor_apply_tab_icons(self) -> None:
        """Tint every tab's icon to the app's icon colour.

        Upstream sets each tab's icon from a plain, uncolored `QIconifyIcon`
        (`TopbarMDATabs._add_tab`), which is also cached per index
        (`tab_icon`) precisely so this can re-derive a themed icon from that
        stable source on every call instead of compounding tints onto an
        already-recolored one. Applied through `set_tab_icon` (the button
        widget's icon), not `QTabBar.setTabIcon` (the native tab's, which
        this presentation deliberately leaves empty -- see `set_tab_icon`).
        """
        color = qcolor(theme().text_secondary)
        for i in range(self.tabs.count()):
            if (source := self.tabs.tab_icon(i)) is not None:
                self.tabs.set_tab_icon(i, _recolor_icon(source, color))

    def _flavor_reveal_axis(self, axis: str, editor: QWidget) -> None:
        self.tabs.setChecked(editor, True)
        self.tabs.setCurrentWidget(editor)
