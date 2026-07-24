"""Acquire tab with camera controls, live preview, and MDA viewers."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from pymmcore_plus import CMMCorePlus
from pymmcore_widgets import PropertyBrowser

from pymmcore_gui._qt.QtCore import Qt, QTimer
from pymmcore_gui._qt.QtWidgets import (
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QWidget,
)
from pymmcore_gui.widgets._mda_widget import MemoryMDAWidget

from ._acquire_presets import AcquisitionPresetSelector
from ._acquire_toolbar import (
    ChannelPresetsBar,
    LiveButton,
    ShuttersBar,
    SnapButton,
    toolbar_separator,
)
from ._acquire_viewers import AcquireViewers
from ._collapsible_panel import CollapsiblePanel
from ._stage_controls import StageControls
from ._tab_page import TabPage

if TYPE_CHECKING:
    from pymmcore_gui._qt.QtGui import QShowEvent


class AcquirePage(TabPage):
    """Acquisition controls with tabbed preview and MDA display."""

    def __init__(
        self, mmcore: CMMCorePlus | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._core = mmcore or CMMCorePlus.instance()

        self._presets = AcquisitionPresetSelector(mmcore=self._core)
        presets_panel = CollapsiblePanel("Groups & Presets", expanded=True)
        presets_panel.body_layout.addWidget(self._presets)

        self._stages = StageControls(self._core)
        stages_panel = CollapsiblePanel("Stages", expanded=True)
        stages_panel.body_layout.addWidget(self._stages)

        # a splitter (not a plain stack) so the user can drag to trade space
        # between the two panels — e.g. shrink the mostly-empty group/preset
        # table to give the stage controls more room
        left_split = QSplitter(Qt.Orientation.Vertical)
        left_split.addWidget(presets_panel)
        left_split.addWidget(stages_panel)
        left_split.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
        self.left.add_widget(left_split)

        self._right_tabs = QTabWidget()
        self._right_tabs.setDocumentMode(True)
        self._right_tabs.setTabsClosable(True)
        self._right_tabs.tabCloseRequested.connect(self._close_right_tab)

        self._mda = MemoryMDAWidget(mmcore=self._core)
        self._right_tabs.addTab(self._mda, "MDA")

        self._property_browser: PropertyBrowser | None = None
        self.right.add_widget(self._right_tabs, 1)

        self._viewers = AcquireViewers(self._core)
        self.add_content_widget(self._viewers)
        self.bottom.hide()
        self._h_split.setSizes([420, 800, 520])

        # toolbar: snap | live ‖ optical configs ‖ shutters … [ MDA | Properties ]
        self._channels = ChannelPresetsBar(self._core)
        self._shutters = ShuttersBar(self._core)
        self.toolbar.add_widget(SnapButton(mmcore=self._core))
        self.toolbar.add_widget(LiveButton(mmcore=self._core))
        self.toolbar.add_widget(toolbar_separator())
        self.toolbar.add_widget(self._channels)
        self.toolbar.add_widget(toolbar_separator())
        self.toolbar.add_widget(self._shutters)
        self.toolbar.add_stretch()

        self._mda_btn = QPushButton("MDA")
        self._mda_btn.setCheckable(True)
        self._mda_btn.setChecked(True)
        self._mda_btn.setToolTip("Show the multi-dimensional acquisition tab")
        self._mda_btn.toggled.connect(self._toggle_mda)
        self.toolbar.add_widget(self._mda_btn)

        self._props_btn = QPushButton("Properties")
        self._props_btn.setCheckable(True)
        self._props_btn.setToolTip("Open the device property browser tab")
        self._props_btn.toggled.connect(self._toggle_properties)
        self.toolbar.add_widget(self._props_btn)

    def _toggle_mda(self, checked: bool) -> None:
        idx = self._right_tabs.indexOf(self._mda)
        if checked:
            if idx < 0:
                idx = self._right_tabs.insertTab(0, self._mda, "MDA")
            self._right_tabs.setCurrentIndex(idx)
        elif idx >= 0:
            self._right_tabs.removeTab(idx)
        self._update_right_sidebar()

    def _toggle_properties(self, checked: bool) -> None:
        browser = self._property_browser
        if checked:
            if browser is None:
                # PropertyBrowser is a QDialog upstream. Adding it to QTabWidget
                # reparents it as a regular embedded page.
                browser = self._property_browser = PropertyBrowser(mmcore=self._core)
            idx = self._right_tabs.indexOf(browser)
            if idx < 0:
                idx = self._right_tabs.addTab(browser, "Properties")
            self._right_tabs.setCurrentIndex(idx)
            QTimer.singleShot(0, self._refresh_property_browser)
        elif browser is not None and (idx := self._right_tabs.indexOf(browser)) >= 0:
            self._right_tabs.removeTab(idx)
        self._update_right_sidebar()

    def _close_right_tab(self, index: int) -> None:
        widget = self._right_tabs.widget(index)
        if widget is self._mda:
            self._mda_btn.setChecked(False)
        elif widget is self._property_browser:
            self._props_btn.setChecked(False)

    def _update_right_sidebar(self) -> None:
        visible = self._right_tabs.count() > 0
        self.right.setVisible(visible)
        if visible:
            self._h_split.setSizes([360, 800, 520])

    def showEvent(self, event: QShowEvent | None) -> None:
        # Devices added on the Hardware tab load into the core but don't fire
        # systemConfigurationLoaded, so the toolbar bars (and property table)
        # would be stale. Re-scan the core whenever this tab is shown.
        super().showEvent(event)
        self._channels.refresh()
        self._shutters.refresh()
        self._presets.refresh()
        self._stages.refresh_devices()
        if (
            self._property_browser is not None
            and self._right_tabs.indexOf(self._property_browser) >= 0
        ):
            QTimer.singleShot(0, self._refresh_property_browser)

    def _refresh_property_browser(self) -> None:
        # PropertyBrowser exposes no public refresh; rebuild its table directly
        # (guarded, in case the internals change). The widget itself already
        # handles systemConfigurationLoaded.
        with suppress(RuntimeError):
            browser = self._property_browser
            if browser is None:
                return
            fn = getattr(browser._prop_table, "_rebuild_table", None)
            if callable(fn):
                with suppress(Exception):
                    fn()
