"""Channel table that marks which channel is live on the microscope."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pymmcore_widgets.mda import CollapsibleCoreMDATabs, CoreConnectedChannelTable
from pymmcore_widgets.useq_widgets._column_info import ColumnInfo
from superqt.utils import signals_blocked

from pymmcore_gui._qt.QtCore import Qt
from pymmcore_gui._qt.QtWidgets import QHeaderView, QTableWidgetItem

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

    def create_subwidgets(self) -> None:
        super().create_subwidgets()
        inherited_channels = self.channels
        self.channels = ActiveChannelTable(1, self._mmc)
        inherited_channels.deleteLater()


__all__ = [
    "CURRENT_CHANNEL_COLUMN",
    "ActiveChannelCollapsibleCoreMDATabs",
    "ActiveChannelTable",
]
