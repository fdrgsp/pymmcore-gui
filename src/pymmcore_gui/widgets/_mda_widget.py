"""Shared MDA editor configured for the ome-writers sink."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_widgets import MDAWidget

from pymmcore_gui._array_viewer import unstyle_widgets
from pymmcore_gui._qt.QtCore import Qt

if TYPE_CHECKING:
    from pathlib import Path

    from pymmcore_plus import CMMCorePlus

    from pymmcore_gui._qt.QtWidgets import QWidget


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

        # The Grid tab's "Absolute Bounds" page (CoreXYBoundsControl) packs a
        # Fixed-size-policy icon-button grid next to the Left/Top/Right/Bottom
        # QFormLayout fields in a plain QHBoxLayout with no alignment flags.
        # That's fine at the control's own sizeHint, but it lives inside
        # GridPlanWidget's QStackedWidget, which stretches every page to fill
        # the full stack area -- Qt then vertically centers the Fixed-policy
        # icon grid while the QFormLayout naturally top-anchors, so the two
        # drift apart whenever the stack is taller than this page needs.
        bounds = self.grid_plan._core_xy_bounds
        if (bounds_layout := bounds.layout()) is not None:
            for i in range(bounds_layout.count()):
                item = bounds_layout.itemAt(i)
                if item is not None and (w := item.widget()) is not None:
                    bounds_layout.setAlignment(w, Qt.AlignmentFlag.AlignTop)

    def prepare_mda(self) -> bool | str | Path | None:
        """Return a disk path or a scratch sink that supports live viewing."""
        output = super().prepare_mda()
        return "memory" if output is None else output
