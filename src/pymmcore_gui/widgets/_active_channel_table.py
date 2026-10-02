"""Channel table that marks which channel is live on the microscope."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pymmcore_widgets import HCSWizard
from pymmcore_widgets.mda import (
    CollapsibleCoreMDATabs,
    CoreConnectedChannelTable,
    TopbarMDATabs,
)
from pymmcore_widgets.useq_widgets import CustomPlateWidget
from pymmcore_widgets.useq_widgets._column_info import ColumnInfo
from superqt.utils import signals_blocked

from pymmcore_gui._array_viewer import (
    ensure_visible_icon,
    set_source_icon,
    unstyle_widgets,
)
from pymmcore_gui._qt.QtCore import QEvent, QObject, QPointF, Qt
from pymmcore_gui._qt.QtGui import QBrush, QPainter, QPen
from pymmcore_gui._qt.QtWidgets import (
    QApplication,
    QHeaderView,
    QPushButton,
    QStyledItemDelegate,
    QTableWidgetItem,
)
from pymmcore_gui._theme import qcolor, theme

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from qtpy.QtCore import QModelIndex, SignalInstance  # type: ignore[attr-defined]
    from qtpy.QtWidgets import QStyleOptionViewItem, QTableWidget

    from pymmcore_gui._qt.QtWidgets import QWidget


_ACTIVE_ROLE = Qt.ItemDataRole.UserRole
_CURRENT_COL_WIDTH = 46
# Fraction of the cell's shorter side the dot/ring's diameter occupies.
_CURRENT_DOT_FRACTION = 0.4


class _CurrentChannelDelegate(QStyledItemDelegate):
    """Paints the ``Current`` column's dot/ring indicator.

    Drawn with ``QPainter`` rather than a ``"●"``/``"○"`` text glyph: those two
    Unicode characters are unrelated glyphs, and at least one common system
    font (Windows' Segoe UI) renders the filled "black circle" glyph visibly
    smaller than the outlined "white circle" one at the same point size --
    drawing our own ellipse keeps both states pixel-identical in size on
    every platform.
    """

    def paint(
        self,
        painter: QPainter | None,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> None:
        super().paint(painter, option, index)
        if painter is None:
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = option.rect
        d = min(rect.width(), rect.height()) * _CURRENT_DOT_FRACTION
        radius = d / 2
        center = QPointF(rect.center())
        if index.data(_ACTIVE_ROLE):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(qcolor(theme().status_yellow)))
        else:
            painter.setPen(QPen(qcolor(theme().text_secondary), 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(center, radius, radius)
        painter.restore()


@dataclass(frozen=True)
class _CurrentChannelColumn(ColumnInfo):
    """Narrow indicator showing which channel is currently active on the microscope.

    Draws a filled dot in the active row and a hollow ring in all others (see
    ``_CurrentChannelDelegate``). Clicking this column activates the
    corresponding channel on the microscope.
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
        item = QTableWidgetItem("")
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        item.setToolTip("Click to activate this channel on the microscope")
        item.setData(_ACTIVE_ROLE, False)
        table.setItem(row, col, item)

    def get_cell_data(self, table: QTableWidget, row: int, col: int) -> dict[str, Any]:
        """Return an empty dict — this column carries no MDA sequence data."""
        return {}

    def set_cell_data(
        self, table: QTableWidget, row: int, col: int, value: Any
    ) -> None:
        """Mark the cell active (filled dot) or inactive (hollow ring)."""
        if item := table.item(row, col):
            item.setData(_ACTIVE_ROLE, bool(value))


CURRENT_CHANNEL_COLUMN = _CurrentChannelColumn()


class ActiveChannelTable(CoreConnectedChannelTable):
    """Core channel table that tracks which channel is live on the microscope.

    A narrow ``Current`` column is prepended that shows a filled dot in the
    row whose channel is presently active on the microscope and a hollow ring
    in all others. Clicking that column, or picking a value in a row's Config
    combo, activates the channel; no other column does so (see
    ``MemoryMDAWidget`` for the wiring).
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
        table.setItemDelegateForColumn(0, _CurrentChannelDelegate(table))
        if (header := table.horizontalHeader()) is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        if header_item := table.horizontalHeaderItem(0):
            header_item.setToolTip("Channel currently active on the microscope")
        self.apply_theme_metrics()

    def apply_theme_metrics(self) -> None:
        """Zoom-scale the Current column's width.

        Re-called on theme/zoom changes (see `MemoryMDAWidgetBase`); the
        indicator itself is painted relative to the cell's rect (see
        `_CurrentChannelDelegate`), so it scales automatically with the row
        height and needs no per-row update here.
        """
        table = self.table()
        table.setColumnWidth(0, theme().scaled(_CURRENT_COL_WIDTH))

    def setActiveRow(self, row: int) -> None:
        """Mark *row* as the channel currently active on the microscope.

        Updates the dot/ring indicator for every row in the ``Current``
        column, and moves the table's own row highlight to match -- the
        highlighted row always mirrors the active-dot row and never changes
        for any other reason (clicking into an Exposure or Intensity editor,
        focusing a cell, etc. never moves it). Pass ``-1`` to clear both the
        indicator and the highlight without marking any row active.
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

    def _apply_editor_min_heights(self) -> None:
        """Ignore a queued upstream resize after this tab widget was deleted."""
        with suppress(RuntimeError):
            super()._apply_editor_min_heights()


class ActiveChannelTopbarMDATabs(TopbarMDATabs):
    """Top-bar MDA tabs using :class:`ActiveChannelTable`.

    The top-bar twin of :class:`ActiveChannelCollapsibleCoreMDATabs`: both
    presentations must offer the same channel table, so the swap between them
    changes only the layout.
    """

    def create_subwidgets(self) -> None:
        super().create_subwidgets()
        inherited_channels = self.channels
        self.channels = ActiveChannelTable(1, self._mmc)
        inherited_channels.deleteLater()


def _theme_subsequence_popup(popup: QWidget) -> None:
    """Match a position sub-sequence popup's styling to the rest of the app."""
    unstyle_widgets(popup)

    # The grid's Mark/Move bounds buttons swap their raw icon at runtime
    # (mode toggle, go_middle checkbox), so re-theme them whenever that
    # happens rather than relying on the one-off sweep above.
    grid_plan = getattr(getattr(popup, "mda_tabs", None), "grid_plan", None)
    bounds = getattr(grid_plan, "_core_xy_bounds", None)
    if grid_plan is None or bounds is None:
        return

    def _refresh_bounds_icons(*_: object) -> None:
        for button in cast("Any", bounds).findChildren(QPushButton):
            set_source_icon(button, button.icon())
            ensure_visible_icon(button)

    _refresh_bounds_icons()
    bounds.go_middle.toggled.connect(_refresh_bounds_icons)
    grid_plan.valueChanged.connect(_refresh_bounds_icons)


class _ThirdPartyWindowThemer(QObject):
    """Applies the app's styling to windows pymmcore-widgets opens itself.

    The position sub-sequence popup, the HCS wizard, and the wizard's "New
    Plate" dialog are all constructed on demand, deep inside
    pymmcore-widgets, with no app-side subclass to hook construction-time
    theming into -- the popup is private (``_MDAPopup``, matched by class
    name for want of a public hook), the wizard is created lazily by the
    position table's "Well Plate..." button, and the custom-plate dialog is
    created lazily by the wizard's "+" button (so it postdates the wizard's
    own ``unstyle_widgets()`` sweep and needs its own). Watch every Show
    event application-wide instead and theme each the moment it appears.
    """

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if (
            a1 is None
            or a1.type() != QEvent.Type.Show
            or a0 is None
            or a0.property("_pymmcore_gui_themed")
        ):
            return False
        if type(a0).__name__ == "_MDAPopup":
            a0.setProperty("_pymmcore_gui_themed", True)
            _theme_subsequence_popup(cast("QWidget", a0))
        elif isinstance(a0, (HCSWizard, CustomPlateWidget)):
            a0.setProperty("_pymmcore_gui_themed", True)
            unstyle_widgets(a0)
        return False


_window_themer: _ThirdPartyWindowThemer | None = None


def install_third_party_window_theming() -> None:
    """Install the app-wide filter that themes pymmcore-widgets' own windows.

    Idempotent -- safe to call from every ``MemoryMDAWidget`` instance.
    """
    global _window_themer
    app = QApplication.instance()
    if app is None or _window_themer is not None:  # pragma: no cover
        return
    _window_themer = _ThirdPartyWindowThemer(app)
    app.installEventFilter(_window_themer)


__all__ = [
    "CURRENT_CHANNEL_COLUMN",
    "ActiveChannelCollapsibleCoreMDATabs",
    "ActiveChannelTable",
    "ActiveChannelTopbarMDATabs",
    "install_third_party_window_theming",
]
