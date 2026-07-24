"""Compact acquisition-only group/preset browser (no editing controls)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_widgets import GroupPresetTableWidget
from pymmcore_widgets.control._presets_widget import PresetsWidget

from pymmcore_gui._array_viewer import unstyle_widgets

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

    from pymmcore_gui._qt.QtWidgets import QWidget


def _refresh_only_new_group_preset(
    self: PresetsWidget,
    group: str,
    preset: str,
    device: str,
    property: str,
    value: str,
) -> None:
    """Refresh the combo without validating/"fixing" the group's data.

    PresetsWidget's own `_on_new_group_preset` (connected to the core's
    `configDefined` signal, and also called directly -- for every *other*
    preset in the group -- from `_delete_presets_with_different_properties()`
    at construction time) treats the group's *first* preset as the one true
    baseline: any preset with a different property *set* gets its extras
    silently deleted (`deleteConfig` + redefine without them), and any preset
    with a different property *count* gets a `UserWarning`, in both
    directions (missing or extra).

    Both reactions are wrong for a legitimate, intentional per-preset
    override (e.g. a different camera for just one channel) -- a real,
    supported Micro-Manager config pattern, not a mistake to "fix". Worse,
    `configDefined` fires once per (device, property, value) triple, and
    MMCore's own C++ config-file loader defines a preset's properties one at
    a time with no way for us to suppress that -- so reloading a config from
    the Hardware tab was reacting to every single *incomplete, mid-load*
    snapshot as if it were the final preset, both corrupting data (before
    `_find_dev_prop_to_remove` was neutered below) and flooding a cascade of
    "missing properties" warnings (one per property, shrinking as the reload
    caught up) even after.

    None of this validation is needed: the widget's only real job here is
    keeping its combo box in sync with the core, which `_refresh()` alone
    already does correctly by reading the group's *current* state directly,
    regardless of how many properties any preset happens to have.
    """
    if group == self._group:
        self._refresh()


# Patched on the class itself (not just wrapped where we construct it),
# since PresetsWidget is built internally by GroupPresetTableWidget's table
# -- including our own AcquisitionPresetSelector below -- not by us directly.
PresetsWidget._on_new_group_preset = (  # type: ignore[method-assign]
    _refresh_only_new_group_preset
)


def _no_destructive_group_resync(
    self: GroupPresetTableWidget,
    group: str,
    preset: str,
    device: str,
    property: str,
    value: str,
) -> None:
    """Refresh the table without ever deleting the group's data.

    A second, independent instance of the same destructive-mutation pattern:
    GroupPresetTableWidget's own `_on_new_group_preset` (also connected to
    the core's `configDefined` signal) calls `deleteConfigGroup(group)`
    whenever a configDefined event arrives for a group whose table row is
    *currently* showing a single-property PropertyWidget cell -- meaning it
    intends to "upgrade" that cell to a multi-preset PresetsWidget once a
    second preset is defined. But it then redefines only the *one* preset
    named in this specific event (from a fresh getConfigData() snapshot),
    silently destroying every *other* preset the group may already have.
    This doesn't need any UI action at all: `_populate_table()` alone
    already rebuilds every row's cell widget by inspecting the group's
    *current* preset count, so it displays correctly whether that count is
    1 or many -- no group deletion was ever necessary to get there.
    """
    self._populate_table()


# Patched the same way as PresetsWidget above, and for the same reason: the
# bug lives on this class, wherever it's constructed, not in anything our
# subclass below does with it.
GroupPresetTableWidget._on_new_group_preset = (  # type: ignore[method-assign]
    _no_destructive_group_resync
)


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
        unstyle_widgets(self)

    def refresh(self) -> None:
        """Re-scan the core for config groups/presets.

        GroupPresetTableWidget already keeps itself in sync via core events;
        this is a defensive re-scan for edits made on another tab that don't
        happen to fire one of those events.
        """
        self._populate_table()
        # _populate_table() rebuilds every row's cell widget from scratch
        # (fresh PresetsWidget/PropertyWidget instances) -- re-sweep so new
        # cells get themed too, not just the ones present at construction.
        unstyle_widgets(self)
