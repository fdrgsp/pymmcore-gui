"""Channel table that marks which channel is live on the microscope."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pymmcore_widgets.mda import (
    CollapsibleCoreMDATabs,
    CoreConnectedChannelTable,
    SectionMetrics,
)
from pymmcore_widgets.useq_widgets._column_info import ColumnInfo
from superqt.utils import signals_blocked

from pymmcore_gui._array_viewer import (
    ensure_visible_icon,
    set_source_icon,
    unstyle_widgets,
)
from pymmcore_gui._modern_gui._theme import theme
from pymmcore_gui._qt.QtCore import QEvent, QSize, Qt, QTimer
from pymmcore_gui._qt.QtWidgets import (
    QAbstractButton,
    QHeaderView,
    QPushButton,
    QTableWidgetItem,
)

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from qtpy.QtCore import SignalInstance  # type: ignore[attr-defined]
    from qtpy.QtWidgets import QTableWidget

    from pymmcore_gui._qt.QtWidgets import QWidget


_CURRENT_ACTIVE = "●"
_CURRENT_INACTIVE = "○"
_CURRENT_COL_WIDTH = 28


@dataclass(frozen=True)
class _CurrentChannelColumn(ColumnInfo):
    """Narrow indicator showing which channel is currently active on the microscope.

    Displays ``●`` in the active row and ``○`` in all others. Clicking this
    column activates the corresponding channel on the microscope.
    """

    key: str = "_current_channel"
    data_type: type = str  # unused; cells are plain QTableWidgetItems
    # Left blank rather than "Current": the column is too narrow for that text
    # to render without being clipped. The header tooltip carries the label
    # instead (see ActiveChannelTable.__init__).
    header: str | None = ""

    def init_cell(
        self,
        table: QTableWidget,
        row: int,
        col: int,
        change_signal: SignalInstance,
    ) -> None:
        """Populate the cell with an inactive indicator."""
        item = QTableWidgetItem(_CURRENT_INACTIVE)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        item.setToolTip("Click to activate this channel on the microscope")
        table.setItem(row, col, item)

    def get_cell_data(self, table: QTableWidget, row: int, col: int) -> dict[str, Any]:
        """Return an empty dict — this column carries no MDA sequence data."""
        return {}

    def set_cell_data(
        self, table: QTableWidget, row: int, col: int, value: Any
    ) -> None:
        """Set the cell to ``●`` when *value* is truthy, ``○`` otherwise."""
        if item := table.item(row, col):
            item.setText(_CURRENT_ACTIVE if value else _CURRENT_INACTIVE)


CURRENT_CHANNEL_COLUMN = _CurrentChannelColumn()


class ActiveChannelTable(CoreConnectedChannelTable):
    """Core channel table that tracks which channel is live on the microscope.

    A narrow ``Current`` column is prepended that shows ``●`` in the row whose
    channel is presently active on the microscope and ``○`` in all others.
    Clicking that column, or picking a value in a row's Config combo, activates
    the channel; no other column does so (see ``MemoryMDAWidget`` for the wiring).
    """

    def __init__(
        self,
        rows: int = 0,
        mmcore: CMMCorePlus | None = None,
        parent: QWidget | None = None,
    ) -> None:
        self._active_row: int = -1
        super().__init__(rows, mmcore, parent)
        # Prepend the active-channel indicator at the leftmost position.
        table = self.table()
        table.addColumn(CURRENT_CHANNEL_COLUMN, 0)
        table.setColumnWidth(0, _CURRENT_COL_WIDTH)
        if (header := table.horizontalHeader()) is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        if header_item := table.horizontalHeaderItem(0):
            header_item.setToolTip("Channel currently active on the microscope")

    def setActiveRow(self, row: int) -> None:
        """Mark *row* as the channel currently active on the microscope.

        Updates the ``●``/``○`` indicator for every row in the ``Current``
        column, and moves the table's own row highlight to match -- the
        highlighted row always mirrors the ``●`` row and never changes for any
        other reason (clicking into an Exposure or Intensity editor, focusing a
        cell, etc. never moves it). Pass ``-1`` to clear both the indicator and
        the highlight without marking any row active.
        """
        self._active_row = row
        table = self.table()
        col = table.indexOf(CURRENT_CHANNEL_COLUMN)
        if col < 0:  # pragma: no cover
            return
        with signals_blocked(table):
            for r in range(table.rowCount()):
                CURRENT_CHANNEL_COLUMN.set_cell_data(table, r, col, r == row)
        if row < 0:
            table.clearSelection()
            if (selection_model := table.selectionModel()) is not None:
                selection_model.clearCurrentIndex()
            return
        table.setCurrentCell(row, col)
        table.selectRow(row)

    def activeRow(self) -> int:
        """Return the row currently active on the microscope, or ``-1`` if none."""
        return self._active_row


class ActiveChannelCollapsibleCoreMDATabs(CollapsibleCoreMDATabs):
    """Collapsible MDA tabs using :class:`ActiveChannelTable`."""

    def __init__(
        self,
        parent: QWidget | None = None,
        core: CMMCorePlus | None = None,
    ) -> None:
        # Position-table sub-sequences are hosted by pymmcore-widgets' private
        # _MDAPopup.  Remember that context before the superclass builds the
        # complete editor tree; the same tab class is also used by the main MDA
        # widget, where Channels intentionally starts expanded.
        self._is_subsequence_editor = (
            parent is not None and type(parent).__name__ == "_MDAPopup"
        )
        super().__init__(parent, core)
        if self._is_subsequence_editor:
            self._configure_subsequence_editor()

    def create_subwidgets(self) -> None:
        super().create_subwidgets()
        inherited_channels = self.channels
        self.channels = ActiveChannelTable(1, self._mmc)
        inherited_channels.deleteLater()

    def _apply_editor_min_heights(self) -> None:
        """Ignore a queued upstream resize after this tab widget was deleted."""
        with suppress(RuntimeError):
            super()._apply_editor_min_heights()

    def _configure_subsequence_editor(self) -> None:
        """Make a position sub-sequence popup match the app's MDA styling."""
        for section in self.sections:
            section.set_expanded(False)

        self._apply_subsequence_theme()
        self.grid_plan.valueChanged.connect(self._apply_subsequence_theme)
        bounds = cast("Any", self.grid_plan)._core_xy_bounds
        bounds.go_middle.toggled.connect(
            self._refresh_subsequence_bounds_icons
        )

        # _MDAPopup creates its OK/Cancel button box after constructing us.
        # Re-run once its constructor has completed so the dialog chrome, not
        # just this child editor, receives the same normalization.
        QTimer.singleShot(0, self._apply_subsequence_theme)

    def _apply_subsequence_theme(self, *_: object) -> None:
        popup = self.parentWidget()
        unstyle_widgets(popup if popup is not None else self)

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

        icon_size = QSize(t.scaled(16), t.scaled(16))
        for table in (self.channels, self.stage_positions, self.time_plan):
            table.toolBar().setIconSize(icon_size)
        self._refresh_subsequence_bounds_icons()

        root = popup if popup is not None else self
        for button in root.findChildren(QAbstractButton):
            ensure_visible_icon(button)

    def _refresh_subsequence_bounds_icons(self, *_: object) -> None:
        """Re-theme the raw Mark/Move glyphs installed by the bounds editor."""
        bounds = cast("Any", self.grid_plan)._core_xy_bounds
        for button in bounds.findChildren(QPushButton):
            set_source_icon(button, button.icon())
            ensure_visible_icon(button)

    def changeEvent(self, event: QEvent | None) -> None:
        super().changeEvent(event)
        if (
            event is not None
            and event.type() == QEvent.Type.StyleChange
            and getattr(self, "_is_subsequence_editor", False)
        ):
            self._apply_subsequence_theme()


__all__ = [
    "CURRENT_CHANNEL_COLUMN",
    "ActiveChannelCollapsibleCoreMDATabs",
    "ActiveChannelTable",
]
