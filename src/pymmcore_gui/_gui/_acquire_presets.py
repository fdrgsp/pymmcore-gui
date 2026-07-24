"""Compact acquisition-only group/preset browser (no editing controls)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_widgets import GroupPresetTableWidget
from pymmcore_widgets.control._presets_widget import PresetsWidget

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

    from pymmcore_gui._qt.QtWidgets import QWidget


def _no_presets_to_strip(self: PresetsWidget, preset: str) -> list[tuple[str, str]]:
    return []


# PresetsWidget (constructed internally, one per multi-preset config group, by
# GroupPresetTableWidget's table -- including our own AcquisitionPresetSelector
# below) treats any (device, property) pair that isn't present in *every*
# other preset of the same group as a mistake, and silently deletes it -- both
# once at construction time and again on every live configDefined event. That
# is wrong for a legitimate, intentional per-preset override (e.g. a different
# camera for just one channel), a real, supported Micro-Manager config
# pattern. Confirmed this silently strips such an override just from loading
# a config and mounting this widget -- no editing involved. Patched on the
# class itself (not just our subclass) since the bug lives in PresetsWidget
# wherever it's constructed, not in anything we do with it.
PresetsWidget._find_dev_prop_to_remove = _no_presets_to_strip  # type: ignore[method-assign]


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
