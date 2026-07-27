"""Acquire tab with camera controls, live preview, and MDA viewers."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from pymmcore_plus import CMMCorePlus
from pymmcore_widgets import PropertyBrowser

from pymmcore_gui._array_viewer import unstyle_widgets
from pymmcore_gui._qt.QtCore import QTimer
from pymmcore_gui._qt.QtWidgets import QPushButton, QTabWidget, QWidget
from pymmcore_gui.widgets._mda_widget import MemoryMDAWidget
from pymmcore_gui.widgets._stage_explorer import ThemedStageExplorer

from ._acquire_presets import AcquisitionPresetSelector
from ._acquire_toolbar import (
    LiveButton,
    ShuttersBar,
    SnapButton,
    toolbar_separator,
)
from ._acquire_viewers import AcquireViewers
from ._tab_bar import ThemedTabBar
from ._tab_page import TabPage

if TYPE_CHECKING:
    from useq import Position

    from pymmcore_gui._qt.QtGui import QShowEvent
    from pymmcore_gui.widgets._mm_console import MMConsole

_PRESETS_LABEL = "Groups and Presets"
_MDA_PANEL_MIN_WIDTH = 100
_RIGHT_PANEL_MIN_WIDTH = 100
# Wide enough that the panels don't open clipped -- the user can still drag
# them down to the minimums above.
_MDA_PANEL_INITIAL_WIDTH = 700
_RIGHT_PANEL_INITIAL_WIDTH = 420


class AcquirePage(TabPage):
    """Acquisition controls with tabbed preview and MDA display."""

    def __init__(
        self, mmcore: CMMCorePlus | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._core = mmcore or CMMCorePlus.instance()

        self._presets = AcquisitionPresetSelector(mmcore=self._core)
        self._mda = MemoryMDAWidget(mmcore=self._core)
        self.left.add_widget(self._mda, 1)
        self.left.setMinimumWidth(_MDA_PANEL_MIN_WIDTH)

        self._right_tabs = QTabWidget()
        self._right_tabs.setTabBar(ThemedTabBar(self._right_tabs))
        self._right_tabs.setDocumentMode(True)
        self._right_tabs.setTabsClosable(True)
        self._right_tabs.tabCloseRequested.connect(self._close_right_tab)

        self._right_tabs.addTab(self._presets, _PRESETS_LABEL)
        self._right_tabs.setCurrentWidget(self._presets)

        self._property_browser: PropertyBrowser | None = None
        self._console: MMConsole | None = None
        self.right.add_widget(self._right_tabs, 1)
        self.right.setMinimumWidth(_RIGHT_PANEL_MIN_WIDTH)

        # Center content is itself split into two tabs: "Viewer" (a lazy snap
        # preview + one viewer per MDA run) and "Explorer" (a stage-explorer
        # map). The Viewer tab keeps its own inner Preview/MDA tab bar.
        self._content_tabs = QTabWidget()
        self._content_tabs.setTabBar(ThemedTabBar(self._content_tabs))
        self._content_tabs.setDocumentMode(True)

        self._viewers = AcquireViewers(self._core)
        self._content_tabs.addTab(self._viewers, "Viewer")

        self._explorer = ThemedStageExplorer(mmcore=self._core)
        self._explorer.sendToMDARequested.connect(self._on_explorer_positions)
        self._content_tabs.addTab(self._explorer, "Explorer")

        self.add_content_widget(self._content_tabs)
        self.bottom.hide()
        self._h_split.setSizes(
            [_MDA_PANEL_INITIAL_WIDTH, 900, _RIGHT_PANEL_INITIAL_WIDTH]
        )

        # toolbar: snap|live ‖ shutters … [Presets|Properties]
        self._shutters = ShuttersBar(self._core)
        self._snap_btn = SnapButton(mmcore=self._core)
        self._snap_btn.snapRequested.connect(self._mda.apply_active_channel_for_capture)
        self._snap_btn.snapRequested.connect(self._viewers.ensure_preview)
        self.toolbar.add_widget(self._snap_btn)
        self._live_btn = LiveButton(mmcore=self._core)
        self._live_btn.liveStartedRequested.connect(
            self._mda.apply_active_channel_for_capture
        )
        self._live_btn.liveStartedRequested.connect(self._viewers.ensure_preview)
        self.toolbar.add_widget(self._live_btn)
        self.toolbar.add_widget(toolbar_separator())
        self.toolbar.add_widget(self._shutters)
        self.toolbar.add_stretch()

        # "subtle" = a persistently visible box (rather than the default
        # "ghost", which is borderless until hovered), matching Snap/Live/
        # Shutters and the channel-preset buttons.
        self._presets_btn = QPushButton(_PRESETS_LABEL)
        self._presets_btn.setProperty("variant", "subtle")
        self._presets_btn.setCheckable(True)
        self._presets_btn.setChecked(True)
        self._presets_btn.setToolTip("Show the group/preset selection tab")
        self._presets_btn.toggled.connect(self._toggle_presets)
        self.toolbar.add_widget(self._presets_btn)

        self._props_btn = QPushButton("Properties")
        self._props_btn.setProperty("variant", "subtle")
        self._props_btn.setCheckable(True)
        self._props_btn.setToolTip("Open the device property browser tab")
        self._props_btn.toggled.connect(self._toggle_properties)
        self.toolbar.add_widget(self._props_btn)

        self._console_btn = QPushButton("Console")
        self._console_btn.setProperty("variant", "subtle")
        self._console_btn.setCheckable(True)
        self._console_btn.setToolTip("Open an IPython console tab")
        self._console_btn.toggled.connect(self._toggle_console)
        self.toolbar.add_widget(self._console_btn)

    def _on_explorer_positions(self, positions: list[Position], replace: bool) -> None:
        """Transfer Stage Explorer regions into the MDA position table."""
        existing = [] if replace else list(self._mda.stage_positions.value())
        self._mda.stage_positions.setValue([*existing, *positions])
        self._mda._collapsible_tabs().section("p").set_expanded(True)

    def _toggle_presets(self, checked: bool) -> None:
        idx = self._right_tabs.indexOf(self._presets)
        if checked:
            if idx < 0:
                idx = self._right_tabs.insertTab(0, self._presets, _PRESETS_LABEL)
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
                unstyle_widgets(browser)
            idx = self._right_tabs.indexOf(browser)
            if idx < 0:
                idx = self._right_tabs.addTab(browser, "Properties")
            self._right_tabs.setCurrentIndex(idx)
            QTimer.singleShot(0, self._refresh_property_browser)
        elif browser is not None and (idx := self._right_tabs.indexOf(browser)) >= 0:
            self._right_tabs.removeTab(idx)
        self._update_right_sidebar()

    def _toggle_console(self, checked: bool) -> None:
        console = self._console
        if checked:
            if console is None:
                from pymmcore_gui.widgets._mm_console import MMConsole

                console = self._console = MMConsole(mmcore=self._core)
            idx = self._right_tabs.indexOf(console)
            if idx < 0:
                idx = self._right_tabs.addTab(console, "Console")
            self._right_tabs.setCurrentIndex(idx)
        elif console is not None and (idx := self._right_tabs.indexOf(console)) >= 0:
            self._right_tabs.removeTab(idx)
        self._update_right_sidebar()

    def _close_right_tab(self, index: int) -> None:
        widget = self._right_tabs.widget(index)
        if widget is self._presets:
            self._presets_btn.setChecked(False)
        elif widget is self._property_browser:
            self._props_btn.setChecked(False)
        elif widget is self._console:
            self._console_btn.setChecked(False)

    def _update_right_sidebar(self) -> None:
        visible = self._right_tabs.count() > 0
        self.right.setVisible(visible)
        # Only touch the right panel's own size, and only when it's actually
        # collapsing/reappearing -- leave whatever the user already dragged
        # the left (MDA) and right panels to alone otherwise (e.g. switching
        # between an already-open Presets/Properties tab shouldn't resize
        # anything).
        sizes = self._h_split.sizes()
        if visible:
            if not sizes[2]:
                sizes[2] = _RIGHT_PANEL_INITIAL_WIDTH
        else:
            sizes[2] = 0
        self._h_split.setSizes(sizes)

    def showEvent(self, a0: QShowEvent | None) -> None:
        # Devices added on the Hardware tab load into the core but don't fire
        # systemConfigurationLoaded, so the toolbar bars (and property table)
        # would be stale. Re-scan the core whenever this tab is shown.
        #
        # The MDA channel table needs this too: config groups edited on the
        # Configurations tab are written inside a block_core() block that emits
        # no signals (see ConfigurationsPage.save), so its channel-group and
        # ranged-property columns can't learn about those edits any other way.
        super().showEvent(a0)
        self._shutters.refresh()
        self._presets.refresh()
        self._mda.refresh_channel_table()
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
