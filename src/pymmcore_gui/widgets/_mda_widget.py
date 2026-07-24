"""Shared MDA editor configured for the ome-writers sink."""

from __future__ import annotations

from typing import TYPE_CHECKING

import useq
from pymmcore_widgets import MDAWidget
from pymmcore_widgets.mda._core_mda import _MDAControlButtons
from pymmcore_widgets.useq_widgets._positions import NULL_SEQUENCE, MDAButton
from superqt.iconify import QIconifyIcon

from pymmcore_gui._array_viewer import unstyle_widgets
from pymmcore_gui._gui._theme import qcolor, theme
from pymmcore_gui._qt.QtCore import QEvent, Qt
from pymmcore_gui._qt.QtWidgets import QGridLayout

if TYPE_CHECKING:
    from pathlib import Path

    from pymmcore_plus import CMMCorePlus
    from pymmcore_widgets.mda._xy_bounds import CoreXYBoundsControl

    from pymmcore_gui._qt.QtWidgets import QWidget


# ═══════════════════════════════════════════════════════════════
# Upstream hardcodes its own literal color at nearly every run/pause/
# cancel/add/remove/clear icon site across MDAWidget and its Channels/
# Positions/Time sub-tables -- and no two of them actually agree with each
# other ("green", "lime", "#3A3", "#C33", "red", ...), let alone with this
# app's theme. Each one is patched at its exact source (the method that
# decides the icon, not a color-guessing sweep over existing pixmaps) so
# every "this is a go/success action" icon becomes theme().status_green and
# every "this is a stop/danger action" icon becomes theme().status_red.
# ═══════════════════════════════════════════════════════════════


def _themed_set_value(self: MDAButton, value: useq.MDASequence | dict | None) -> None:
    """Replaces MDAButton.setValue -- same logic, theme-derived colors.

    Position rows in the MDA widget's Positions table each get one of these
    for an optional per-position sub-sequence. Patched here (not swept
    per-row) so it also stays correct for any *future* edit made through
    the popup dialog, not just the icon's state at row-creation time.
    """
    if isinstance(value, dict):
        value = useq.MDASequence(**value)
    elif value and not isinstance(value, useq.MDASequence):
        raise TypeError(f"Expected useq.MDASequence, got {type(value)}")
    old_val, self._value = getattr(self, "_value", None), value

    # Unconditional (not just on change) so this also corrects clear_btn's
    # hardcoded-red icon the first time setValue(None) runs from __init__.
    red = qcolor(theme().status_red).name()
    self.clear_btn.setIcon(QIconifyIcon("mdi:close-circle", color=red))

    if old_val != value:
        if value and value != NULL_SEQUENCE:
            green = qcolor(theme().status_green).name()
            self.seq_btn.setIcon(QIconifyIcon("mdi:axis-arrow", color=green))
            self.clear_btn.show()
        else:
            self.seq_btn.setIcon(QIconifyIcon("mdi:axis"))
            self.clear_btn.hide()
        self.valueChanged.emit()


MDAButton.setValue = _themed_set_value  # type: ignore[method-assign]


def _themed_on_mda_paused(self: _MDAControlButtons, paused: bool) -> None:
    """Replaces _MDAControlButtons._on_mda_paused -- theme-derived colors.

    This is the method upstream uses to swap pause_btn's icon between
    "paused" (play/resume) and "running" (pause) -- patched here so every
    pause/resume toggle during a real run re-derives from the *current*
    theme, not just the icon set once at construction.
    """
    color = qcolor(theme().status_green).name()
    if paused:
        self.pause_btn.setIcon(QIconifyIcon("mdi:play-circle-outline", color=color))
        self.pause_btn.setText("Resume")
    else:
        self.pause_btn.setIcon(QIconifyIcon("mdi:pause-circle-outline", color=color))
        self.pause_btn.setText("Pause")


_MDAControlButtons._on_mda_paused = (  # type: ignore[method-assign]
    _themed_on_mda_paused
)


def _apply_mda_theme_colors(widget: MemoryMDAWidget) -> None:
    """(Re)apply status colors to every icon patched/overridden above.

    Covers what the two whole-method patches above don't reach on their
    own: the run/cancel buttons (their icon is only ever set once, at
    construction, with no method controlling it later) and each table's
    toolbar actions (same: set once, never revisited). Called once at
    construction and again on every theme change (see changeEvent below).
    """
    green = qcolor(theme().status_green).name()
    red = qcolor(theme().status_red).name()

    control = widget.control_btns
    control.run_btn.setIcon(QIconifyIcon("mdi:play-circle-outline", color=green))
    control.cancel_btn.setIcon(QIconifyIcon("mdi:stop-circle-outline", color=red))
    # pause_btn toggles between two icons/labels at runtime (see
    # _themed_on_mda_paused) -- reapply whichever one is currently shown.
    _themed_on_mda_paused(control, control.pause_btn.text() == "Resume")

    for table_widget in (widget.channels, widget.stage_positions, widget.time_plan):
        table_widget.act_add_row.setIcon(QIconifyIcon("mdi:plus-thick", color=green))
        table_widget.act_remove_row.setIcon(
            QIconifyIcon("mdi:close-box-outline", color=red)
        )
        table_widget.act_clear.setIcon(
            QIconifyIcon("mdi:close-box-multiple-outline", color=red)
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


class MemoryMDAWidget(MDAWidget):
    """Christina-style MDA editor with a viewable memory-sink fallback."""

    def __init__(
        self, mmcore: CMMCorePlus, parent: QWidget | None = None
    ) -> None:
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
                model.rowsInserted.connect(lambda *_: unstyle_widgets(self))

        _align_bounds_grid(self.grid_plan._core_xy_bounds)
        _apply_mda_theme_colors(self)

    def prepare_mda(self) -> bool | str | Path | None:
        """Return a disk path or a scratch sink that supports live viewing."""
        output = super().prepare_mda()
        return "memory" if output is None else output

    def changeEvent(self, a0: QEvent | None) -> None:
        # status_green/status_red differ between light/dark themes -- the
        # icons _apply_mda_theme_colors sets are otherwise only ever
        # (re)computed at construction or on a real state change (run/pause/
        # stop), so a mere theme toggle would leave them showing stale
        # colors from whichever theme was active when this widget was built.
        if a0 is not None and a0.type() == QEvent.Type.StyleChange:
            _apply_mda_theme_colors(self)
        super().changeEvent(a0)
