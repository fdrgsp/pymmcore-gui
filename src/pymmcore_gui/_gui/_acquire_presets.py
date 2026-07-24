"""Compact acquisition-only group/preset browser (no editing controls)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from pymmcore_widgets import GroupPresetTableWidget
from pymmcore_widgets.control._presets_widget import PresetsWidget

from pymmcore_gui._array_viewer import unstyle_widgets

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

# Same root cause, a separate (non-destructive) symptom: _on_new_group_preset
# also warns whenever a preset's property *count* differs from the group's
# first preset, then reports "missing" properties as the *set difference*
# (baseline - preset). For a preset that has one *extra* property (the same
# legitimate per-preset override as above), that difference is empty, so the
# message is always the literally-nonsensical "missing the following
# properties: []." No data is touched in this branch -- only the message is
# wrong -- so this is a plain warnings filter, not a behavior patch.
warnings.filterwarnings(
    "ignore",
    message=r".*preset is missing the following properties: \[\]\.$",
    category=UserWarning,
    module=r"pymmcore_widgets\.control\._presets_widget",
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
