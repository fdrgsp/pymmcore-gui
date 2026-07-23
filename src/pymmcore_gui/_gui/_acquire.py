"""Acquire tab.

Toolbar of snap/live/optical-config/shutter controls over a (currently empty)
central area, with the device Property Browser in a right panel toggled from
the toolbar.
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from pymmcore_plus import CMMCorePlus
from pymmcore_widgets import LiveButton, PropertyBrowser, SnapButton

from pymmcore_gui._qt.QtCore import QTimer
from pymmcore_gui._qt.QtWidgets import QPushButton, QWidget

from ._acquire_toolbar import ChannelPresetsBar, ShuttersBar, toolbar_separator
from ._tab_page import TabPage

if TYPE_CHECKING:
    from pymmcore_gui._qt.QtGui import QShowEvent


class AcquirePage(TabPage):
    """Acquisition tab: control toolbar + toggleable property browser."""

    def __init__(
        self, mmcore: CMMCorePlus | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._core = mmcore or CMMCorePlus.instance()

        # NOTE: PropertyBrowser is a QDialog upstream; parenting it into a
        # layout makes it behave as a plain child widget (isWindow() is False).
        self._property_browser = PropertyBrowser(mmcore=self._core)
        self.right.add_widget(self._property_browser, 1)

        # central area is empty for now (the image viewport will live here)
        self.add_content_widget(QWidget())
        self.left.hide()
        self.right.hide()  # revealed by the toolbar's "Properties" button
        self.bottom.hide()

        # toolbar: snap | live ‖ optical configs ‖ shutters … [ Properties ]
        self._channels = ChannelPresetsBar(self._core)
        self._shutters = ShuttersBar(self._core)
        self.toolbar.add_widget(SnapButton(mmcore=self._core))
        self.toolbar.add_widget(LiveButton(mmcore=self._core))
        self.toolbar.add_widget(toolbar_separator())
        self.toolbar.add_widget(self._channels)
        self.toolbar.add_widget(toolbar_separator())
        self.toolbar.add_widget(self._shutters)
        self.toolbar.add_stretch()

        self._props_btn = QPushButton("Properties")
        self._props_btn.setCheckable(True)
        self._props_btn.setToolTip("Show the device property browser")
        self._props_btn.toggled.connect(self._toggle_properties)
        self.toolbar.add_widget(self._props_btn)

    def _toggle_properties(self, checked: bool) -> None:
        self.right.setVisible(checked)
        if checked:
            # give the panel a usable width, then refresh once laid out
            self._h_split.setSizes([0, 700, 460])
            QTimer.singleShot(0, self._refresh_property_browser)

    def showEvent(self, event: QShowEvent | None) -> None:
        # Devices added on the Hardware tab load into the core but don't fire
        # systemConfigurationLoaded, so the toolbar bars (and property table)
        # would be stale. Re-scan the core whenever this tab is shown.
        super().showEvent(event)
        self._channels.refresh()
        self._shutters.refresh()
        if self._props_btn.isChecked():
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
