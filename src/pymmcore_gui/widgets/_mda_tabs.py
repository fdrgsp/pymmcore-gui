"""The MDA editor's tabs, and the styling of windows pymmcore-widgets opens."""

from __future__ import annotations

from pymmcore_widgets import HCSWizard
from pymmcore_widgets.mda import CollapsibleCoreMDATabs, CoreMDATabs
from pymmcore_widgets.useq_widgets import CustomPlateWidget
from pymmcore_widgets.useq_widgets._z_surface_dialog import ZSurfaceDialog

from pymmcore_gui._array_viewer import (
    unstyle_widgets,
)
from pymmcore_gui._qt.QtCore import QEvent, Qt
from pymmcore_gui._qt.QtWidgets import QWidget
from pymmcore_gui._theme._events import add_event_hook

from ._mda_channel_table import MDAChannelTable

_THEMED = "_pymmcore_gui_themed"


def _install_gui_tables(tabs: CoreMDATabs) -> None:
    """Swap in this app's channel table for the upstream one."""
    inherited_channels = tabs.channels
    tabs.channels = MDAChannelTable(0, tabs._mmc)
    inherited_channels.deleteLater()


class GuiCollapsibleCoreMDATabs(CollapsibleCoreMDATabs):
    """Collapsible MDA tabs using :class:`MDAChannelTable`."""

    def create_subwidgets(self) -> None:
        super().create_subwidgets()
        _install_gui_tables(self)


class GuiCoreMDATabs(CoreMDATabs):
    """Upstream's top-bar MDA tabs using :class:`MDAChannelTable`.

    The top-bar twin of :class:`GuiCollapsibleCoreMDATabs`: both presentations
    must offer the same channel table, so the swap between them changes only the
    layout.
    """

    def create_subwidgets(self) -> None:
        super().create_subwidgets()
        _install_gui_tables(self)


def _theme(widget: QWidget) -> None:
    """Unstyle ``widget``'s subtree and mark all of it as themed."""
    unstyle_widgets(widget)
    for w in (widget, *widget.findChildren(QWidget)):
        w.setProperty(_THEMED, True)


def _theme_subsequence_popup(popup: QWidget) -> None:
    """Match a position sub-sequence popup's styling to the rest of the app."""
    _theme(popup)

    # Left to itself, the dialog hands initial focus to the first focusable
    # widget in it -- the Grid/Tiles tab's own checkbox -- which then shows the
    # focus ring as a blue square inside the box, as if it were half-checked.
    # Focusing the dialog itself instead shows no ring anywhere; Esc/Enter
    # still work, and Tab still moves into the controls.
    popup.setFocus(Qt.FocusReason.OtherFocusReason)


def _theme_third_party_window(widget: QWidget, _event: QEvent) -> bool:
    """Apply the app's styling to windows pymmcore-widgets opens itself.

    The position sub-sequence popup, the HCS wizard, the wizard's "New
    Plate" dialog and the Surface Focus dialog are all constructed on demand,
    deep inside pymmcore-widgets, with no app-side subclass to hook
    construction-time theming into -- the popup is private (``_MDAPopup``,
    matched by class name for want of a public hook). Watch every Show event
    application-wide instead and theme each the moment it appears.

    These windows also build parts of themselves later (e.g. the wizard's
    per-well Manual calibration panels, created when a well is first
    selected), after the window's own sweep: any unthemed widget shown inside
    an already-themed window is themed on its first Show too.
    """
    if widget.property(_THEMED):
        return False
    if type(widget).__name__ == "_MDAPopup":
        _theme_subsequence_popup(widget)
    elif isinstance(widget, (HCSWizard, CustomPlateWidget, ZSurfaceDialog)) or (
        (window := widget.window()) is not None and window.property(_THEMED)
    ):
        _theme(widget)
    return False


def install_third_party_window_theming() -> None:
    """Install the app-wide hook that themes pymmcore-widgets' own windows.

    Idempotent -- safe to call from every ``MemoryMDAWidget`` instance.
    """
    add_event_hook(_theme_third_party_window, QEvent.Type.Show)


__all__ = [
    "GuiCollapsibleCoreMDATabs",
    "GuiCoreMDATabs",
    "install_third_party_window_theming",
]
