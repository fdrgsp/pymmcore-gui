"""Acquire tab.

Currently hosts the device Property Browser in a sub-tab; acquisition-related
widgets will be added here as the GUI grows.
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from pymmcore_plus import CMMCorePlus
from pymmcore_widgets import PropertyBrowser

from pymmcore_gui._qt.QtCore import QTimer
from pymmcore_gui._qt.QtWidgets import QTabWidget, QWidget

from ._tab_page import TabPage

if TYPE_CHECKING:
    from pymmcore_gui._qt.QtGui import QShowEvent


class AcquirePage(TabPage):
    """Acquisition tab with sub-tabbed tools."""

    def __init__(
        self, mmcore: CMMCorePlus | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._core = mmcore or CMMCorePlus.instance()

        # NOTE: PropertyBrowser is a QDialog upstream; parenting it into a
        # layout makes it behave as a plain child widget (isWindow() is False).
        self._property_browser = PropertyBrowser(mmcore=self._core)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._property_browser, "Property Browser")

        self.add_content_widget(self._tabs)
        # the tools fill the page; the docks would only crowd them
        self.left.hide()
        self.right.hide()
        self.bottom.hide()
        self.toolbar.hide()

    def showEvent(self, event: QShowEvent | None) -> None:
        # Devices added on the Hardware tab don't fire systemConfigurationLoaded,
        # so refresh the property table when shown. Deferred to the next
        # event-loop turn so it sizes to the settled (not first-show) geometry.
        super().showEvent(event)
        QTimer.singleShot(0, self._refresh_property_browser)

    def _refresh_property_browser(self) -> None:
        # PropertyBrowser exposes no public refresh; rebuild its table directly
        # (guarded, in case the internals change). The widget itself already
        # handles systemConfigurationLoaded.
        with suppress(RuntimeError):
            fn = getattr(self._property_browser._prop_table, "_rebuild_table", None)
            if callable(fn):
                with suppress(Exception):
                    fn()
