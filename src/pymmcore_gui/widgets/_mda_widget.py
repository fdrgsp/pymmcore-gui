"""Shared MDA editor configured for the ome-writers sink."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_widgets import MDAWidget
from pymmcore_widgets.useq_widgets._positions import MDAButton
from superqt.iconify import QIconifyIcon

from pymmcore_gui._array_viewer import (
    ensure_visible_icon,
    set_source_icon,
    unstyle_widgets,
)
from pymmcore_gui._gui._theme import qcolor, theme
from pymmcore_gui._qt.QtCore import QEvent, Qt
from pymmcore_gui._qt.QtWidgets import QGridLayout

if TYPE_CHECKING:
    from pathlib import Path

    from pymmcore_plus import CMMCorePlus
    from pymmcore_widgets.mda._xy_bounds import CoreXYBoundsControl

    from pymmcore_gui._qt.QtWidgets import QWidget


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


class MemoryMDAWidget(MDAWidget):
    """Christina-style MDA editor with a viewable memory-sink fallback."""

    def __init__(self, mmcore: CMMCorePlus, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent, mmcore=mmcore)
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

        _align_bounds_grid(self.grid_plan._core_xy_bounds)

    def _on_table_rows_inserted(self, *_: object) -> None:
        """Theme cell widgets that are constructed only when a row is added."""
        unstyle_widgets(self)
        self._connect_position_icon_updates()
        self._apply_themed_icons()

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
        """Recreate semantic icons when switching between light and dark themes."""
        super().changeEvent(a0)
        if (
            a0 is not None
            and a0.type() == QEvent.Type.StyleChange
            and hasattr(self, "control_btns")
        ):
            self._apply_themed_icons()

    def prepare_mda(self) -> bool | str | Path | None:
        """Return a disk path or a scratch sink that supports live viewing."""
        output = super().prepare_mda()
        return "memory" if output is None else output
