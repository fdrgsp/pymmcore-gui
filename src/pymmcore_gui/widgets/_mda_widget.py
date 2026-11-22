"""Shared MDA editor configured for the ome-writers sink."""

from __future__ import annotations

import os
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from ome_writers import AcquisitionSettings, OmeTiffFormat, ScratchFormat
from ome_writers._schema import _ome_stem_suffix
from pymmcore_widgets import MDAWidget, MDAWidgetCollapsible
from pymmcore_widgets._busy import BusyOverlay
from pymmcore_widgets._util import fov_pixel_size_warning
from pymmcore_widgets.mda import (
    CollapsibleCoreMDATabs,
    CoreMDATabs,
    SectionMetrics,
)
from pymmcore_widgets.mda._collapsible_mda import _ClickableLabel
from pymmcore_widgets.useq_widgets import MDASequenceWidget
from superqt.iconify import QIconifyIcon

from pymmcore_gui._array_viewer import (
    _recolor_icon,
    set_icon_tint,
    set_source_icon,
    unstyle_widgets,
)
from pymmcore_gui._channel_settings_store import channel_settings_store
from pymmcore_gui._qt.QtCore import (
    QEvent,
    QModelIndex,
    QObject,
    Qt,
    QTimer,
    Signal,
)
from pymmcore_gui._qt.QtGui import QPainter, QPixmap
from pymmcore_gui._qt.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QToolButton,
    QWidget,
)
from pymmcore_gui._sequence_support import drop_unsupported
from pymmcore_gui._settings import Settings
from pymmcore_gui._stitching import has_mosaic
from pymmcore_gui._stitching import is_available as stitching_available
from pymmcore_gui._stitching._models import (
    METADATA_NAMESPACE,
    SequenceStitchingV1,
    stored_request,
    with_request,
)
from pymmcore_gui._theme import qcolor, theme
from pymmcore_gui._utils import NO_PIXEL_SIZE

from ._mda_tabs import (
    GuiCollapsibleCoreMDATabs,
    GuiCoreMDATabs,
    install_third_party_window_theming,
    style_table_toolbar,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import useq
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.mda import SingleOutput

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

STITCHING_TOOLTIP = (
    "After the acquisition completes, fuse each grid into one mosaic, saved as "
    "<i>&lt;name&gt;_stitched</i> next to the acquisition, in its format unless "
    "the settings say otherwise. It runs in the background; its progress is "
    "shown in the status bar."
)
STITCHING_NO_OVERLAP_TOOLTIP = (
    "<br><br><b>The grid has no overlap</b>, so tiles are placed by stage "
    "position only: registration needs neighbouring tiles to overlap."
)


def _grids(sequence: useq.MDASequence) -> list[Any]:
    """The sequence's grid plans: the global one and those of its positions."""
    grids = [sequence.grid_plan] if sequence.grid_plan is not None else []
    grids += [
        pos.sequence.grid_plan
        for pos in sequence.stage_positions
        if pos.sequence is not None and pos.sequence.grid_plan is not None
    ]
    return grids


def _grids_overlap(sequence: useq.MDASequence) -> bool:
    """Whether any grid of `sequence` asks for overlapping tiles."""
    for grid in _grids(sequence):
        overlap = getattr(grid, "overlap", (0.0, 0.0))
        if max(overlap) > 0:
            return True
    return False


def _section_metrics() -> SectionMetrics:
    """The app's zoom-scaled spacing for the collapsible presentation."""
    t = theme()
    return SectionMetrics(
        header_height=t.row_height - t.scaled(4),
        disclosure_width=t.scaled(24),
        header_spacing=t.sp_xxs,
        body_margin_h=t.sp_sm,
        body_margin_top=t.sp_xxs,
        body_margin_bottom=t.sp_xs,
        body_spacing=t.sp_sm,
        content_spacing=t.sp_xxs,
        footer_margin_h=t.sp_sm,
        footer_margin_top=t.sp_xxs,
        footer_margin_bottom=t.sp_xs,
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
    """Shared application behavior for both MDA editor presentations.

    Own theming, channel settings, acquisition locking and scratch output here.
    Presentation-specific work belongs in the _flavor_* hooks.
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
        settingsIgnored: Any

    # ----------- Override type hints in superclass -----------
    # _create_tab_widget above always installs an MDAChannelTable, which adds
    # the settings store link to the upstream CoreConnectedChannelTable
    # interface; narrow the property's declared return type to match.

    @property
    def channels(self) -> MDAChannelTable:
        return cast("MDAChannelTable", super().channels)

    def __init__(self, mmcore: CMMCorePlus, parent: QWidget | None = None) -> None:
        self._restoring_sequence = False
        self._drop_unsupported = False
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
        self._install_stitching_option()

        # Match the application's persistently visible action buttons.
        for btn in (
            self.control_btns.run_btn,
            self.control_btns.pause_btn,
            self.control_btns.cancel_btn,
            self._save_button,
            self._load_button,
        ):
            btn.setProperty("variant", "subtle")

        # Normalize the eagerly constructed upstream editors to the app theme.
        unstyle_widgets(self)
        # unstyle_widgets clears the Saving section's stylesheet too, which
        # hides the embedded QGroupBox's native header and indicator.
        self._collapsible_tabs().apply_save_body_style()
        self._apply_table_toolbar_icon_size()
        # the position table paints its buttons as this hidden one, never shown
        # to become primary as the other action buttons do
        self.stage_positions.table().buttonTemplate().setProperty("variant", "primary")

        # Theme cell widgets created by later row insertions too.
        for table_widget in (self.channels, self.time_plan):
            model = table_widget.table().model()
            if model is not None:
                model.rowsInserted.connect(self._on_table_rows_inserted)

        # Upstream replaces Pause/Resume icons whenever acquisition state
        # changes. Re-apply our semantic colors after those runtime swaps.
        self._mmc.mda.events.sequencePauseToggled.connect(self._apply_themed_icons)
        self._mmc.mda.events.sequenceFinished.connect(self._apply_themed_icons)
        self._apply_themed_icons()

        # Link the channel rows to the per-preset settings the Channels panel
        # edits: new rows start from them, and user edits are stored back.
        self.channels.setSettingsStore(channel_settings_store(mmcore))
        self.channels.applyStoredSettings(range(self.channels.table().rowCount()))
        if (model := self.channels.table().model()) is not None:
            model.columnsInserted.connect(self._schedule_channel_editor_filters)
        self._install_channel_editor_filters()

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
        sender = self.sender()
        for editor in (self.channels, self.time_plan):
            table = editor.table()
            if sender is not table.model():
                continue
            for row in range(first, last + 1):
                for col in range(table.columnCount()):
                    if (cell := table.cellWidget(row, col)) is not None:
                        unstyle_widgets(cell)
            if editor is self.channels:
                self._install_channel_editor_filters()
                if not self._restoring_sequence:
                    self.channels.applyStoredSettings(range(first, last + 1))
            break

    def setValue(self, value: useq.MDASequence) -> None:
        """Restore an MDA sequence, keeping its own channel settings."""
        dropped: list[str] = []
        if self._drop_unsupported:
            value, dropped = drop_unsupported(value, self._mmc)
        self._restoring_sequence = True
        try:
            super().setValue(value)
        finally:
            self._restoring_sequence = False
        self._restore_stitching_request(value)
        self._flavor_refresh_summaries()
        if dropped:
            self.settingsIgnored.emit(
                "Cannot re-use the following MDA settings with the current "
                "Micro-Manager configuration:\n\n"
                + "\n".join(f"• {item}" for item in dropped)
            )

    @contextmanager
    def _dropping_unsupported(self) -> Iterator[None]:
        previous = self._drop_unsupported
        self._drop_unsupported = True
        try:
            yield
        finally:
            self._drop_unsupported = previous

    def load(self, file: str | Path | None = None) -> None:
        """Load an MDA file, leaving out what this configuration can't run."""
        with self._dropping_unsupported():
            super().load(file)

    def loadSequence(self, sequence: useq.MDASequence) -> None:
        """Load a sequence, omitting unsupported settings and reporting them."""
        with self._dropping_unsupported():
            self.setValue(sequence)

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

        Picking another preset in a row's Config combo gives the row that preset's
        settings; an edit of its exposure, light source or intensity is stored for
        its preset. Only user-driven signals are used -- ``activated`` and
        ``editingFinished`` -- so that restoring a sequence or rebuilding a column
        never overwrites a preset's settings.
        """
        channels = self.channels
        table = channels.table()
        try:
            config_col = table.indexOf(channels._config_column)
            settings_cols = {
                table.indexOf(channels.EXPOSURE),
                table.indexOf(channels.INTENSITY),
                table.indexOf(channels._light_source_column),
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
            # unlike a spin box, a combo does not take the focus that selects its row
            cell.activated.connect(self._select_channel_editor_row)
            cell.activated.connect(slot)
        elif isinstance(cell, QDoubleSpinBox):
            cell.editingFinished.connect(slot)

    def _select_channel_editor_row(self, *_: object) -> None:
        index = self._channel_index_for_editor(self.sender())
        if index.isValid():
            self.channels.table().selectRow(index.row())

    def _on_channel_settings_edited(self, *_: object) -> None:
        if self._restoring_sequence:
            return
        index = self._channel_index_for_editor(self.sender())
        if index.isValid():
            self.channels.storeRowSettings(index.row())

    def _on_channel_config_activated(self, *_: object) -> None:
        index = self._channel_index_for_editor(self.sender())
        if index.isValid():
            self.channels.presetPicked(index.row())

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

    def apply_active_channel_for_capture(self) -> bool:
        """Apply the selected channel's settings before a snap or live.

        Those of the row marked in the channel table, whose values may differ from
        its preset's (several rows of one preset, a restored sequence); otherwise, or
        with the table's channel control off, of the preset set on the core. The
        row's acquisition checkbox is ignored: it controls inclusion in an MDA, not
        the live/snap selection.

        An editor still being typed in goes first, here or in the Channels panel: on
        macOS, clicking Snap or Live leaves it the focus, so it has not taken its value.
        """
        focused = QApplication.focusWidget()
        if isinstance(focused, (QAbstractSpinBox, QLineEdit)):
            focused.clearFocus()
        channels = self.channels
        if channels.channelControlEnabled() and (row := channels.activeRow()) >= 0:
            return channels.applyRow(row, wait=True)
        return channel_settings_store(self._mmc).apply_current_channel()

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
        """Restyle the axis-table toolbars, and Surface Focus's once opened."""
        tables = [self.channels, self.stage_positions, self.time_plan]
        if (surface := self.stage_positions._surface_dialog) is not None:
            tables.append(surface.points)
        for table in tables:
            style_table_toolbar(table)

    def _apply_theme_metrics(self) -> None:
        """Feed the app's zoom-scaled spacing into the upstream presentation."""
        self._flavor_apply_theme_metrics()

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

    def _apply_surface_focus_icons(self, green: str, red: str) -> None:
        """Theme the gears the Surface Focus button swaps between.

        Upstream re-picks its icon from `_gears` on every setting change, and
        its neutral gear is uncolored (black), so the dict itself is themed.
        """
        positions = self.stage_positions
        neutral = qcolor(theme().text_primary).name()
        positions._gears.update(
            {
                "": QIconifyIcon("tabler:settings", color=neutral),
                "valid": QIconifyIcon("tabler:settings-check", color=green),
                "error": QIconifyIcon("tabler:settings-x", color=red),
            }
        )
        positions._update_surface_focus()

    def _apply_themed_icons(self, *_: object) -> None:
        """Apply the app's semantic green/red to every MDA action icon."""
        self._apply_themed_section_icons()
        if hasattr(self, "_stitching_settings_btn"):
            set_source_icon(
                self._stitching_settings_btn,
                QIconifyIcon(
                    "mdi:cog-outline", color=qcolor(theme().text_primary).name()
                ),
            )
        self._flavor_apply_tab_icons()
        set_icon_tint(self.af_axis.settings_button, qcolor(theme().text_primary))
        green = qcolor(theme().status_green).name()
        red = qcolor(theme().status_red).name()
        self._apply_surface_focus_icons(green, red)

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

        self.stage_positions.delegate().setIcons(
            QIconifyIcon("mdi:grid", color=qcolor(theme().text_primary).name()),
            QIconifyIcon("mdi:grid", color=green),
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
        self._repin_save_height()

    def _repin_save_height(self) -> None:
        """Refit the Saving section after one of its rows appeared or disappeared."""
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

    def _install_stitching_option(self) -> None:
        """Add the stitching checkbox and settings button to the Saving section."""
        self._stitching_available = stitching_available()
        self._stitching_request = SequenceStitchingV1(
            settings=Settings.instance().stitching
        )
        self._stitching_label = QLabel("Stitching:")
        self._stitching_check = QCheckBox("Also save a stitched mosaic")
        self._stitching_settings_btn = QToolButton()
        self._stitching_settings_btn.setToolTip("Stitching settings…")
        self._stitching_settings_btn.setProperty("variant", "subtle")
        self._stitching_row = QWidget()
        row = QHBoxLayout(self._stitching_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._stitching_check)
        row.addWidget(self._stitching_settings_btn)
        row.addStretch()

        layout = cast("QGridLayout", self.save_info.layout())
        layout.addWidget(self._stitching_label, 3, 0)
        layout.addWidget(self._stitching_row, 3, 1, 1, 3)
        self._stitching_check.toggled.connect(self._on_stitching_toggled)
        self._stitching_settings_btn.clicked.connect(self.edit_stitching_settings)
        self.valueChanged.connect(self._sync_stitching_option)
        self.stage_positions.valueChanged.connect(self._sync_stitching_option)
        self._sync_stitching_option()

    def _sync_stitching_option(self, *_: object) -> None:
        """Show the row only when there is a grid to stitch."""
        editor_sequence = MDASequenceWidget.value(self)
        visible = self._stitching_available and has_mosaic(editor_sequence)
        tip = STITCHING_TOOLTIP
        if visible and not _grids_overlap(editor_sequence):
            tip += STITCHING_NO_OVERLAP_TOOLTIP
        self._stitching_check.setToolTip(tip)
        if visible != self._stitching_row.isVisibleTo(self.save_info):
            self._stitching_label.setVisible(visible)
            self._stitching_row.setVisible(visible)
            self._repin_save_height()

    def _on_stitching_toggled(self, checked: bool) -> None:
        self.setStitchingRequest(
            self._stitching_request.model_copy(update={"enabled": checked})
        )

    def stitchingRequest(self) -> SequenceStitchingV1:
        """Whether to stitch, and how; stored in `value().metadata`."""
        return self._stitching_request

    def setStitchingRequest(self, request: SequenceStitchingV1) -> None:
        """Set whether to stitch and with which settings."""
        changed = request != self._stitching_request
        self._stitching_request = request
        if self._stitching_check.isChecked() != request.enabled:
            self._stitching_check.setChecked(request.enabled)
        if changed:
            self.valueChanged.emit()

    def edit_stitching_settings(self) -> bool:
        """Open the ⚙ dialog; return whether the settings were accepted.

        Accepted settings also become the defaults of new sequences.
        """
        from pymmcore_gui.widgets._stitching_dialog import StitchingSettingsDialog

        channels = [c.config for c in MDASequenceWidget.value(self).channels]
        dialog = StitchingSettingsDialog(
            self._stitching_request.settings,
            channels,
            max_threads=os.cpu_count() or 1,
            parent=cast("QWidget", self),
        )
        try:
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return False
            settings = dialog.value()
        finally:
            dialog.deleteLater()
        self.setStitchingRequest(
            self._stitching_request.model_copy(update={"settings": settings})
        )
        prefs = Settings.instance()
        prefs.stitching = settings
        prefs.flush()
        return True

    def _restore_stitching_request(self, sequence: useq.MDASequence) -> None:
        """Take the request a restored sequence carries, or start unchecked."""
        if not hasattr(self, "_stitching_request"):
            return  # upstream's __init__, before the row exists
        request = stored_request(sequence.metadata) or SequenceStitchingV1(
            settings=Settings.instance().stitching
        )
        self.setStitchingRequest(request)

    def value(self) -> useq.MDASequence:
        """The sequence to run, carrying the stitching request in its metadata."""
        val = super().value()
        if getattr(self, "_stitching_available", False):
            meta = with_request(val.metadata, self._stitching_request)
            val.metadata[METADATA_NAMESPACE] = meta[METADATA_NAMESPACE]
        return val

    def prepare_mda(self) -> bool | SingleOutput | None:
        """Return a disk path or a scratch sink that supports live viewing."""
        self._commit_editors()
        sequence = self.value()
        if dropped := drop_unsupported(sequence, self._mmc)[1]:
            QMessageBox.warning(
                self,
                "Not Available in This Configuration",
                "The loaded configuration cannot run these MDA settings:\n\n"
                + "\n".join(f"• {item}" for item in dropped)
                + "\n\nChange or remove them, then run again.",
            )
            return False
        if _grids(sequence) and (
            problem := fov_pixel_size_warning(self._mmc, NO_PIXEL_SIZE)
        ):
            QMessageBox.warning(
                self,
                "Pixel Size Needed for the Grid",
                f"{problem}\n\nThis sequence uses a grid plan, whose spacing and "
                "overlap come from the camera's field of view, so its tiles would "
                "not be placed correctly.",
            )
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

    settingsIgnored = Signal(str)
    """Emitted with a message when `load`/`loadSequence` left settings out."""

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


class TopbarMemoryMDAWidget(MemoryMDAWidgetBase, MDAWidget):
    """The same editor, presented as upstream's native icon-labelled top tabs.

    Identical in behaviour to `MemoryMDAWidget` -- everything the app adds lives
    in `MemoryMDAWidgetBase`. This flavour is upstream's plain `MDAWidget`:
    `tabs` is a real `QTabWidget`, so the hooks below use its native tab API
    (`setChecked`/`setCurrentWidget`) directly rather than a
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

    settingsIgnored = Signal(str)
    """Emitted with a message when `load`/`loadSequence` left settings out."""

    def _create_tab_widget(self) -> CoreMDATabs:
        return GuiCoreMDATabs(None, self._mmc)

    def _flavor_apply_tab_icons(self) -> None:
        """Tint tab-button icons from their cached originals on each theme change."""
        color = qcolor(theme().text_secondary)
        for i in range(self.tabs.count()):
            if (source := self.tabs.tab_icon(i)) is not None:
                self.tabs.set_tab_icon(i, _recolor_icon(source, color))

    def _flavor_reveal_axis(self, axis: str, editor: QWidget) -> None:
        self.tabs.setChecked(editor, True)
        self.tabs.setCurrentWidget(editor)
