"""The MDA editor's tabs, and the styling of windows pymmcore-widgets opens."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pymmcore_widgets import HCSWizard
from pymmcore_widgets.mda import CollapsibleCoreMDATabs, TopbarMDATabs
from pymmcore_widgets.useq_widgets import CustomPlateWidget

from pymmcore_gui._array_viewer import (
    ensure_visible_icon,
    set_source_icon,
    unstyle_widgets,
)
from pymmcore_gui._qt.QtCore import QEvent, QObject, Qt
from pymmcore_gui._qt.QtWidgets import QApplication, QPushButton

from ._mda_channel_table import MDAChannelTable

if TYPE_CHECKING:
    from pymmcore_gui._qt.QtWidgets import QWidget


class GuiCollapsibleCoreMDATabs(CollapsibleCoreMDATabs):
    """Collapsible MDA tabs using :class:`MDAChannelTable`."""

    def create_subwidgets(self) -> None:
        super().create_subwidgets()
        inherited_channels = self.channels
        self.channels = MDAChannelTable(0, self._mmc)
        inherited_channels.deleteLater()


class GuiTopbarMDATabs(TopbarMDATabs):
    """Top-bar MDA tabs using :class:`MDAChannelTable`.

    The top-bar twin of :class:`GuiCollapsibleCoreMDATabs`: both presentations
    must offer the same channel table, so the swap between them changes only the
    layout.
    """

    def create_subwidgets(self) -> None:
        super().create_subwidgets()
        inherited_channels = self.channels
        self.channels = MDAChannelTable(0, self._mmc)
        inherited_channels.deleteLater()


def _theme_subsequence_popup(popup: QWidget) -> None:
    """Match a position sub-sequence popup's styling to the rest of the app."""
    unstyle_widgets(popup)

    # Left to itself, the dialog hands initial focus to the first focusable
    # widget in it -- the Grid/Tiles tab's own checkbox -- which then shows the
    # focus ring as a blue square inside the box, as if it were half-checked.
    # Focusing the dialog itself instead shows no ring anywhere; Esc/Enter
    # still work, and Tab still moves into the controls.
    popup.setFocus(Qt.FocusReason.OtherFocusReason)

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
    "GuiCollapsibleCoreMDATabs",
    "GuiTopbarMDATabs",
    "install_third_party_window_theming",
]
