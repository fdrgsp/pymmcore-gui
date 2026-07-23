"""Compact acquisition-only group/preset browser (no editing controls)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_widgets import GroupPresetTableWidget

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

    from pymmcore_gui._qt.QtWidgets import QWidget


class AcquisitionPresetSelector(GroupPresetTableWidget):
    """The upstream group/preset table, with editing controls hidden.

    Adding/removing/editing groups and presets already happens on the
    Configurations tab, and saving/loading a whole .cfg already happens on
    the Hardware tab — this sidebar widget is only for quickly picking a
    preset during acquisition, so everything but the table itself is hidden.
    """

    def __init__(
        self, *, mmcore: CMMCorePlus | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(mmcore=mmcore, parent=parent)

        # hide each button directly (not just its wrapper row), so isHidden()
        # is unambiguous regardless of whether this widget has been shown
        for btn in (
            self.groups_add_btn,
            self.groups_remove_btn,
            self.groups_edit_btn,
            self.presets_add_btn,
            self.presets_remove_btn,
            self.presets_edit_btn,
            self.save_btn,
            self.load_btn,
        ):
            btn.hide()

        # also hide the two rows themselves, so their labels ("Group:",
        # "Preset:") disappear along with the buttons rather than dangling
        # with nothing next to them
        if (layout := self.layout()) is not None:
            for i in range(layout.count()):
                item = layout.itemAt(i)
                widget = item.widget() if item is not None else None
                if widget is not None and widget is not self.table_wdg:
                    widget.hide()

        # upstream hardcodes a 200px floor on the table so it reads well as a
        # standalone panel; here it lives in a collapsible sidebar panel that
        # needs to shrink to just its header when collapsed
        self.table_wdg.setMinimumHeight(0)

    def refresh(self) -> None:
        """Re-scan the core for config groups/presets.

        GroupPresetTableWidget already keeps itself in sync via core events;
        this is a defensive re-scan for edits made on another tab that don't
        happen to fire one of those events.
        """
        self._populate_table()
