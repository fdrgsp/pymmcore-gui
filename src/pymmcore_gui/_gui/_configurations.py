"""Configurations tab: property browser, group editor and pixel configuration.

These are the upstream ``pymmcore_widgets`` editors; this module only arranges
them as sub-tabs and keeps them in step with the core.
"""

from __future__ import annotations

from contextlib import suppress

from pymmcore_plus import CMMCorePlus
from pymmcore_widgets import (
    ConfigGroupsEditor,
    PixelConfigurationWidget,
    PropertyBrowser,
)

from pymmcore_gui._qt.QtWidgets import QTabWidget, QWidget

from ._tab_page import TabPage


class ConfigurationsPage(TabPage):
    """Sub-tabbed editors for device properties, config groups and pixel sizes."""

    def __init__(
        self, mmcore: CMMCorePlus | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._core = mmcore or CMMCorePlus.instance()

        # NOTE: PropertyBrowser is a QDialog upstream; parenting it into a
        # layout makes it behave as a plain child widget (isWindow() is False).
        self._property_browser = PropertyBrowser(mmcore=self._core)
        self._group_editor = ConfigGroupsEditor.create_from_core(self._core)
        self._pixel_config = PixelConfigurationWidget(mmcore=self._core)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._property_browser, "Property Browser")
        self._tabs.addTab(self._group_editor, "Group Editor")
        self._tabs.addTab(self._pixel_config, "Pixel Configuration")

        self.add_content_widget(self._tabs)
        # these editors fill the page; the docks would only crowd them
        self.left.hide()
        self.right.hide()
        self.bottom.hide()
        self.toolbar.hide()

        # a configuration may be loaded after this page is built
        self._core.events.systemConfigurationLoaded.connect(
            self._on_system_config_loaded
        )

    def _on_system_config_loaded(self) -> None:
        """Refresh the group editor when a new configuration is loaded."""
        # the widget may already be torn down on the C++ side
        with suppress(RuntimeError):
            self.refresh()

    def refresh(self) -> None:
        """Re-read the editors from the current state of the core."""
        with suppress(Exception):
            self._group_editor.update_from_core(self._core, update_configs=True)
