"""Acquire tab: camera controls, live preview, and a dockable MDA/tools layout."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from pymmcore_plus import CMMCorePlus
from pymmcore_widgets import PropertyBrowser

from pymmcore_gui._array_viewer import (
    ensure_visible_icon,
    set_icon_tint,
    unstyle_widgets,
)
from pymmcore_gui._qt.QtAds import CDockManager, CDockWidget, DockWidgetArea
from pymmcore_gui._qt.QtCore import QEvent, QTimer
from pymmcore_gui._qt.QtGui import QFont
from pymmcore_gui._qt.QtWidgets import (
    QAbstractButton,
    QAbstractSlider,
    QPushButton,
    QWidget,
)
from pymmcore_gui.widgets._mda_widget import MemoryMDAWidget

from ._acquire_presets import AcquisitionPresetSelector
from ._acquire_toolbar import (
    LiveButton,
    ShuttersBar,
    SnapButton,
    toolbar_separator,
)
from ._acquire_viewers import AcquireViewersManager
from ._tab_page import TabPage
from ._theme import qcolor, theme

if TYPE_CHECKING:
    from pymmcore_gui._qt.QtAds import CDockAreaWidget
    from pymmcore_gui._qt.QtGui import QShowEvent
    from pymmcore_gui.widgets._mm_console import MMConsole

_MDA_LABEL = "MDA"
_PRESETS_LABEL = "Groups and Presets"
_PROPS_LABEL = "Properties"
_CONSOLE_LABEL = "Console"
_DOCK_MIN_WIDTH = 0
_MDA_DOCK_WIDTH = 700
_RIGHT_DOCK_MAX_WIDTH = 500
_ADS_NEUTRAL_ICON_BUTTONS = frozenset(
    {
        "tabsMenuButton",
        "detachGroupButton",
        "dockAreaAutoHideButton",
        "dockAreaMinimizeButton",
        "dockAreaCloseButton",
    }
)
_ADS_TAB_CLOSE_BUTTON = "tabCloseButton"

_ads_configured = False


def _configure_ads() -> None:
    """Apply the process-wide ADS flags, once.

    Mirrors ``_main_window.py``'s classic-GUI setup so both GUIs' dock
    managers behave identically. These setters are static -- they affect every
    ``CDockManager`` in the process, and only the *first* call before any
    manager is constructed has an effect. ``DockAreaHasAutoHideButton`` is the
    one addition beyond what the classic GUI sets: without it, pinning a dock
    to a side bar is drag-only and undiscoverable.

    Note: emptying a dock area (moving its last widget elsewhere, whether to
    another regular area or to auto-hide) reproducibly segfaults under
    ``QT_QPA_PLATFORM=offscreen`` + pytest-qt, independent of any app code,
    on both PyQt6Ads 4.4.0.post2 and 5.0.0 -- confirmed test-harness-only,
    since interactive drag-and-drop on a real display does not reproduce it.
    Automated tests therefore stick to what doesn't empty an area (open/close
    toggling, tabbing a dock into an existing one); actual rearranging is a
    manual smoke-test item (see the PR description).
    """
    global _ads_configured
    if _ads_configured:
        return
    _ads_configured = True
    CDockManager.setConfigFlag(CDockManager.eConfigFlag.DockAreaHasCloseButton, False)
    CDockManager.setConfigFlag(CDockManager.eConfigFlag.OpaqueSplitterResize, True)
    CDockManager.setAutoHideConfigFlag(
        CDockManager.eAutoHideFlag.AutoHideFeatureEnabled, True
    )
    CDockManager.setAutoHideConfigFlag(
        CDockManager.eAutoHideFlag.DockAreaHasAutoHideButton, True
    )


class AcquirePage(TabPage):
    """Acquisition controls with a dockable MDA/tools layout and image viewers.

    Every panel below the toolbar -- MDA, Groups and Presets, the Property
    Browser, and the console -- is a QtAds dock widget, so the user can
    rearrange them, pin any of them to a side bar, or move them to the bottom.
    Only ``toolbar`` is inherited from ``TabPage``; the dock manager fills the
    whole content area, replacing the fixed left/right sidebar split the other
    pages still use.
    """

    def __init__(
        self, mmcore: CMMCorePlus | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._core = mmcore or CMMCorePlus.instance()
        # AcquirePage doesn't use TabPage's left sidebar -- the dock manager
        # below supplies its own left/right/bottom docking.
        self.left.hide()
        self._content_layout.setContentsMargins(0, 0, 0, 0)

        _configure_ads()
        self._dock_manager = CDockManager(self.content)
        self._dock_manager.dockWidgetAdded.connect(self._queue_dock_icon_refresh)
        self.add_content_widget(self._dock_manager)
        self._base_dock_style = self._dock_manager.styleSheet()
        self._apply_dock_style()

        # ── central: blank placeholder; Preview / MDA-run viewers are each a
        # real CDockWidget tabbed into its dock area (see AcquireViewersManager) ──
        self._central = CDockWidget(self._dock_manager, "Viewers", self)
        self._central.setObjectName("acquire_viewers")
        self._central.setFeature(CDockWidget.DockWidgetFeature.NoTab, True)
        self._central.setFeature(
            CDockWidget.DockWidgetFeature.DockWidgetClosable, False
        )
        blank = QWidget()
        blank.setObjectName("blank")
        self._central.setWidget(blank)
        central_dock_area = self._dock_manager.setCentralWidget(self._central)
        assert central_dock_area is not None
        self._central_dock_area = central_dock_area

        self._viewers = AcquireViewersManager(
            self._dock_manager, self._central_dock_area, self._core, parent=self
        )

        # ── eager docks: MDA is shown by default ─────────────────────────
        self._mda = MemoryMDAWidget(mmcore=self._core)
        self._mda_dock = self._add_dock(
            "acquire_mda", _MDA_LABEL, self._mda, DockWidgetArea.LeftDockWidgetArea
        )

        # ── lazy docks: built on first open ───────────────────────────────
        self._presets: AcquisitionPresetSelector | None = None
        self._presets_dock: CDockWidget | None = None
        self._property_browser: PropertyBrowser | None = None
        self._props_dock: CDockWidget | None = None
        self._console: MMConsole | None = None
        self._console_dock: CDockWidget | None = None
        self._right_dock_area: CDockAreaWidget | None = None

        area = self._mda_dock.dockAreaWidget()
        if area is None:
            return
        central = self._dock_manager.width() - _MDA_DOCK_WIDTH
        self._dock_manager.setSplitterSizes(area, [_MDA_DOCK_WIDTH, central])

        # toolbar: snap|live ‖ shutters … [MDA][Groups and Presets][Properties][Console]
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

        self._mda_btn = self._add_panel_button(_MDA_LABEL, "Show or hide the MDA panel")
        self._mda_btn.setChecked(True)
        self._mda_btn.toggled.connect(self._mda_dock.toggleView)
        self._sync_button(self._mda_btn, self._mda_dock)

        self._presets_btn = self._add_panel_button(
            _PRESETS_LABEL, "Show or hide the group/preset selection panel"
        )
        self._presets_btn.toggled.connect(self._toggle_presets)

        self._props_btn = self._add_panel_button(
            _PROPS_LABEL, "Open the device property browser panel"
        )
        self._props_btn.toggled.connect(self._toggle_properties)

        self._console_btn = self._add_panel_button(
            _CONSOLE_LABEL, "Open an IPython console panel"
        )
        self._console_btn.toggled.connect(self._toggle_console)
        self._refresh_dock_icons()

    # ------------------------------------------------------------------ dock helpers

    def _add_dock(
        self,
        name: str,
        title: str,
        widget: QWidget,
        area: DockWidgetArea,
        into: CDockAreaWidget | None = None,
    ) -> CDockWidget:
        """Create a dock wrapping *widget* and add it to the manager at *area*."""
        dock = CDockWidget(self._dock_manager, title, self)
        dock.setObjectName(name)
        dock.setWidget(widget, CDockWidget.eInsertMode.ForceNoScrollArea)
        # AcquirePage lives inside MainWindow's QStackedWidget; an ADS floating
        # container is a top-level window that would linger after switching
        # to another mode tab, so none of these docks may float.
        dock.setFeature(CDockWidget.DockWidgetFeature.DockWidgetFloatable, False)
        self._dock_manager.addDockWidget(area, dock, into)
        return dock

    def _add_side_dock(self, name: str, title: str, widget: QWidget) -> CDockWidget:
        """Add a dock to the right area, tabbing into it if it already exists.

        The first call creates the right area and caps its initial width at
        ``_RIGHT_DOCK_MAX_WIDTH``; subsequent calls tab into that same area.
        """
        if self._right_dock_area is not None:
            return self._add_dock(
                name,
                title,
                widget,
                DockWidgetArea.CenterDockWidgetArea,
                self._right_dock_area,
            )
        dock = self._add_dock(name, title, widget, DockWidgetArea.RightDockWidgetArea)
        self._right_dock_area = dock.dockAreaWidget()
        # Cap the right column width.  setSplitterSizes expects sizes for all
        # columns in the splitter; use the MDA area as the reference since it
        # is the leftmost area of the main horizontal splitter.
        if (mda_area := self._mda_dock.dockAreaWidget()) is not None:
            total = self._dock_manager.width()
            right = min(total // 4, _RIGHT_DOCK_MAX_WIDTH)
            central = max(total - _MDA_DOCK_WIDTH - right, _DOCK_MIN_WIDTH)
            self._dock_manager.setSplitterSizes(
                mda_area, [_MDA_DOCK_WIDTH, central, right]
            )
        return dock

    def _add_panel_button(self, label: str, tooltip: str) -> QPushButton:
        """A checkable toolbar toggle for a dock, styled like the other actions."""
        btn = QPushButton(label)
        btn.setProperty("variant", "subtle")
        btn.setCheckable(True)
        btn.setToolTip(tooltip)
        self.toolbar.add_widget(btn)
        return btn

    def _sync_button(self, btn: QPushButton, dock: CDockWidget) -> None:
        """Keep *btn* checked in sync with whether *dock* is open.

        Connects ``viewToggled`` rather than ``dock.toggleViewAction().toggled``:
        pinning a dock to an auto-hide side bar transiently unchecks and
        re-checks that action without ever emitting ``viewToggled``, so
        binding the button to the action would close the dock the moment the
        user pinned it. Both directions are idempotent (``setChecked`` and
        ``toggleView`` no-op when already in the target state), so this never
        loops.
        """
        dock.viewToggled.connect(btn.setChecked)

    # ------------------------------------------------------------------ theming

    def _apply_dock_style(self) -> None:
        """Append theme-derived overrides to ADS's built-in stylesheet.

        ADS ships its own ~9KB sheet written entirely against ``palette(...)``
        roles, which mostly tracks this app's light/dark themes for free. The
        exception is the dock tab labels: inactive ones are ``palette(dark)``,
        a *shadow* role, which under the dark theme renders near-black on a
        near-black tab and is effectively invisible. Re-point both states at
        the theme's own text colors instead. Appended (not replaced) so the
        rest of ADS's chrome -- including the title-bar icons, which come from
        ``qproperty-icon`` rules in that sheet -- is left intact.
        """
        t = theme()
        self._dock_manager.setStyleSheet(
            self._base_dock_style
            + f"""
            ads--CDockWidgetTab QLabel {{
                color: {qcolor(t.text_secondary).name()};
            }}
            ads--CDockWidgetTab[activeTab="true"] QLabel {{
                color: {qcolor(t.text_primary).name()};
            }}
            ads--CAutoHideTab {{
                color: {qcolor(t.text_secondary).name()};
            }}
            ads--CAutoHideTab[activeTab="true"] {{
                color: {qcolor(t.text_primary).name()};
            }}
            QWidget#blank {{
                background-color: {qcolor(t.bg_deepest).name()};
            }}
            """
        )

    def _queue_dock_icon_refresh(self, _dock: CDockWidget) -> None:
        """Refresh after ADS has finished constructing a newly added dock's chrome."""
        QTimer.singleShot(0, self._refresh_dock_icons)

    def _refresh_dock_icons(self) -> None:
        """Make ADS chrome visible and keep tab-close glyphs theme red.

        ADS assigns fixed black pixmaps to its title-bar buttons. Docks are
        created after the application's initial dark-theme sweep, so those
        icons otherwise remain black until the first light/dark toggle.
        Reusing the shared contrast correction fixes their initial state.

        The close button beside each dock tab is semantic rather than neutral:
        force it to the active theme's status-red. ``set_icon_tint`` records
        that intent so the application-wide contrast sweep cannot turn it
        white again on the next theme change.
        """
        red = qcolor(theme().status_red)
        for btn in self._dock_manager.findChildren(QAbstractButton):
            name = btn.objectName()
            if name == _ADS_TAB_CLOSE_BUTTON:
                set_icon_tint(btn, red)
            elif name in _ADS_NEUTRAL_ICON_BUTTONS:
                ensure_visible_icon(btn)

    def _refresh_dock_fonts(self) -> None:
        """Let the dock subtree follow app-font (zoom) changes again.

        ``CDockManager`` applies a stylesheet in its constructor, and applying
        *any* stylesheet makes Qt resolve and freeze the fonts of that widget's
        whole subtree -- so every widget inside a dock stops following
        ``QApplication.setFont()``, which is exactly how Cmd+Shift+± zooming
        works (see ``set_zoom``). Without this the MDA table, its cell
        editors, the presets table etc. stay pinned at whatever size was
        current when the manager was built, while everything outside the dock
        area rescales normally.

        Resetting each descendant to a default-constructed (unresolved) QFont
        makes it re-inherit the now-current app font. Same fix, same reason, as
        ``_GroupEditorTab.changeEvent`` in ``_configurations.py``.
        """
        dm = self._dock_manager
        for w in (dm, *dm.findChildren(QWidget)):
            # QAbstractSlider is excluded for the same reason as in
            # _configurations.py: its groove/handle metrics are derived from
            # the font, and resetting it mid-StyleChange fights the style.
            if not isinstance(w, QAbstractSlider):
                w.setFont(QFont())

    def changeEvent(self, a0: QEvent | None) -> None:
        """Re-apply dock theming and un-freeze dock fonts after a theme/zoom change."""
        super().changeEvent(a0)
        if a0 is not None and a0.type() == QEvent.Type.StyleChange:
            if hasattr(self, "_dock_manager"):
                self._apply_dock_style()
                self._refresh_dock_icons()
                self._refresh_dock_fonts()

    def _toggle_presets(self, checked: bool) -> None:
        dock = self._presets_dock
        if dock is None:
            if not checked:
                return
            presets = self._presets = AcquisitionPresetSelector(mmcore=self._core)
            dock = self._presets_dock = self._add_side_dock(
                "acquire_presets", _PRESETS_LABEL, presets
            )
            self._sync_button(self._presets_btn, dock)
        dock.toggleView(checked)
        if checked:
            dock.setAsCurrentTab()

    def _toggle_properties(self, checked: bool) -> None:
        dock = self._props_dock
        if dock is None:
            if not checked:
                return
            # PropertyBrowser is a QDialog upstream; docking reparents it as a
            # regular embedded page.
            browser = self._property_browser = PropertyBrowser(mmcore=self._core)
            unstyle_widgets(browser)
            dock = self._props_dock = self._add_side_dock(
                "acquire_properties", _PROPS_LABEL, browser
            )
            self._sync_button(self._props_btn, dock)
        dock.toggleView(checked)
        if checked:
            dock.setAsCurrentTab()
            QTimer.singleShot(0, self._refresh_property_browser)

    def _toggle_console(self, checked: bool) -> None:
        dock = self._console_dock
        if dock is None:
            if not checked:
                return
            from pymmcore_gui.widgets._mm_console import MMConsole

            console = self._console = MMConsole(mmcore=self._core)
            dock = self._console_dock = self._add_side_dock(
                "acquire_console", _CONSOLE_LABEL, console
            )
            self._sync_button(self._console_btn, dock)
        dock.toggleView(checked)
        if checked:
            dock.setAsCurrentTab()

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
        if self._presets is not None:
            self._presets.refresh()
        self._mda.refresh_channel_table()
        if (dock := self._props_dock) is not None and not dock.isClosed():
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
