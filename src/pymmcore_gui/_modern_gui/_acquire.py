"""Acquire tab: camera controls, live preview, and a dockable MDA/tools layout."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

from pymmcore_plus import CMMCorePlus

from pymmcore_gui._array_viewer import (
    ensure_visible_icon,
    set_icon_tint,
    unstyle_widgets,
)
from pymmcore_gui._layouts import (
    DEFAULT_LAYOUT_NAME,
    RESERVED_LAYOUT_NAMES,
    AcquireLayout,
    available_layouts,
    delete_layout,
    is_valid_layout_name,
    list_layouts,
    resolve_layout,
    save_layout,
)
from pymmcore_gui._qt.QtAds import CDockManager, CDockWidget, DockWidgetArea
from pymmcore_gui._qt.QtCore import (
    QEvent,
    QObject,
    QPoint,
    QSignalBlocker,
    Qt,
    QTimer,
    Signal,
)
from pymmcore_gui._qt.QtGui import QCursor, QFont
from pymmcore_gui._qt.QtWidgets import (
    QWIDGETSIZE_MAX,
    QAbstractButton,
    QAbstractSlider,
    QInputDialog,
    QMessageBox,
    QPushButton,
    QSplitter,
    QWidget,
)

from ._acquire_toolbar import (
    LayoutMenuButton,
    LiveButton,
    PanelButtonBar,
    ShuttersBar,
    SnapButton,
    toolbar_separator,
)
from ._acquire_viewers import AcquireViewersManager
from ._camera_roi_sync import CameraRoiSyncController
from ._panels import PANELS, PanelInfo, PanelKey
from ._tab_page import TabPage
from ._theme import qcolor, theme

if TYPE_CHECKING:
    from collections.abc import Iterable

    import useq

    from pymmcore_gui._qt.QtAds import CDockAreaWidget
    from pymmcore_gui._qt.QtGui import QResizeEvent, QShowEvent
    from pymmcore_gui.widgets._mda_widget import MemoryMDAWidget
    from pymmcore_gui.widgets._stage_explorer import ThemedStageExplorer

_DOCK_MIN_WIDTH = 0
_MDA_DOCK_WIDTH = 700
_RIGHT_DOCK_MAX_WIDTH = 500
_RIGHT_DOCK_MIN_USABLE_WIDTH = 200
_WIDTH_SETTLE_DELAY_MS = 200
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
_REMOVED_PANEL_KEYS = frozenset({"camera_roi"})

_ads_configured = False


def _configure_ads() -> None:
    """Apply the process-wide ADS flags, once.

    Mirrors ``_main_window.py``'s classic-GUI setup so both GUIs' dock
    managers behave identically, plus two additions beyond what the classic
    GUI sets: without ``DockAreaHasAutoHideButton``, pinning a dock to a side
    bar is drag-only and undiscoverable; without ``EqualSplitOnInsertion``,
    dragging a viewer tab out to split it gives the new area only a sliver
    (a fixed minimum width) while the original area keeps the rest, rather
    than splitting the space evenly. These setters are static -- they affect
    every ``CDockManager`` in the process, and only the *first* call before
    any manager is constructed has an effect.

    Note: emptying an area in the outer tools manager by moving its last dock
    reproduces an offscreen pytest-qt crash on PyQt6Ads 4.4.0.post2 and 5.0.0,
    despite working interactively on a real display. Automated tests therefore
    verify outer-manager membership and allowed drop areas rather than simulate
    those drags. Viewer splitting is exercised directly because it happens in
    the isolated nested manager and is stable under the harness.

    A second, related harness-only quirk: running *several* of the
    ``restore_layout`` tests together via a partial ``-k`` selection can
    segfault in pytest-qt's between-test widget teardown (repainting a
    half-destroyed widget). It is not a ``restoreState`` bug -- 8 full
    save/restore cycles over 16 live ``AcquirePage``s in one process are
    fine, and the full test files and the full suite both pass repeatedly.
    Only contrived subsets trip it, so run whole files rather than narrow
    ``-k`` filters when working on this area.
    """
    global _ads_configured
    if _ads_configured:
        return
    _ads_configured = True
    CDockManager.setConfigFlag(CDockManager.eConfigFlag.DockAreaHasCloseButton, False)
    CDockManager.setConfigFlag(CDockManager.eConfigFlag.OpaqueSplitterResize, True)
    CDockManager.setConfigFlag(CDockManager.eConfigFlag.EqualSplitOnInsertion, True)
    CDockManager.setAutoHideConfigFlag(
        CDockManager.eAutoHideFlag.AutoHideFeatureEnabled, True
    )
    CDockManager.setAutoHideConfigFlag(
        CDockManager.eAutoHideFlag.DockAreaHasAutoHideButton, True
    )


@dataclass
class _Panel:
    """Runtime state for one registry panel.

    Holds its toggle button, and -- once opened -- its widget and dock.
    """

    info: PanelInfo
    button: QPushButton
    widget: QWidget | None = None
    dock: CDockWidget | None = None


class AcquirePage(TabPage):
    """Acquisition controls with a dockable MDA/tools layout and image viewers.

    Every panel below the toolbar -- MDA, Groups and Presets, the Property
    Browser, the console, and so on -- is a QtAds dock widget built from the
    registry in ``_panels.py``, so the user can rearrange them, pin any of
    them to a side bar, or move them to the bottom. Only ``toolbar`` is
    inherited from ``TabPage``; the dock manager fills the whole content
    area, replacing the fixed left/right sidebar split the other pages
    still use.
    """

    layoutReset = Signal()
    """Emitted after :meth:`reset_layout` restores the default arrangement."""

    layoutNameChanged = Signal(str)
    """Emitted with the new name whenever the selected layout changes."""

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
        # ADS's own stylesheet drives each tab close button's icon via a
        # ``qproperty-icon`` rule keyed on the dynamic ``activeTab`` property
        # (see _apply_dock_style). Switching which tab is current in an area
        # flips that property on both the tab losing and the tab gaining
        # focus, and Qt re-polishing either one re-applies that qproperty-icon
        # rule -- silently wiping out our red tint on both, not just the one
        # that changed. There's no manager-wide "current tab changed" signal,
        # so hook each area's own ``currentChanged`` as it's created.
        self._dock_manager.dockAreaCreated.connect(self._connect_dock_area_tab_switch)
        self.add_content_widget(self._dock_manager)
        self._base_dock_style = self._dock_manager.styleSheet()

        # The outer manager owns every movable tool panel.  Its central widget
        # is a second, independent manager that owns viewers only.  Changes to
        # the inner viewer splitter tree therefore cannot make the outer
        # manager recompute the MDA/tools widths, while MDA and every tool can
        # still be dragged to any side of the outer workspace.
        self._viewer_dock_manager = CDockManager()
        self._viewer_dock_manager.dockWidgetAdded.connect(self._queue_dock_icon_refresh)
        self._viewer_dock_manager.dockAreaCreated.connect(
            self._connect_dock_area_tab_switch
        )
        self._viewer_base_dock_style = self._viewer_dock_manager.styleSheet()

        # The outer central dock is only a non-movable shell for the viewer
        # manager.  Restricting its area to outer drops prevents an MDA/tool
        # panel from being tabbed into the viewer workspace: it may only dock
        # around it on the left, right, top, or bottom.
        self._central = CDockWidget(self._dock_manager, "Viewers", self)
        self._central.setObjectName("acquire_viewers")
        self._central.setFeature(CDockWidget.DockWidgetFeature.NoTab, True)
        self._central.setWidget(
            self._viewer_dock_manager,
            CDockWidget.eInsertMode.ForceNoScrollArea,
        )
        central_dock_area = self._dock_manager.setCentralWidget(self._central)
        assert central_dock_area is not None
        central_dock_area.setAllowedAreas(DockWidgetArea.OuterDockAreas)
        self._central_dock_area = central_dock_area
        self._apply_dock_style()

        self._viewers = AcquireViewersManager(
            self._viewer_dock_manager,
            self._core,
            parent=self,
        )
        # Connect before the MDA panel constructs CameraRoiWidget.  Its roiSet
        # handler performs Auto Snap synchronously, so a lazy Preview must be
        # created by an earlier listener in order to receive imageSnapped.
        self._core.events.roiSet.connect(self._ensure_preview_for_roi_auto_snap)

        self._right_dock_area: CDockAreaWidget | None = None
        # Values are usually the CDockAreaWidget itself, but see
        # ``_column_widget`` -- once a column holds more than one
        # stacked/split-apart area, the locked widget is the wrapping
        # QSplitter instead.
        self._width_locked_areas: dict[QObject, QWidget] = {}
        # Keep PySide6's wrapper for each QtAds-owned splitter alive alongside
        # its handle. Otherwise the temporary parentWidget() wrapper may be
        # collected at the end of _install_width_lock and invalidate the handle.
        self._width_lock_splitters: dict[QObject, QSplitter] = {}
        self._dragging_width_handles: set[QObject] = set()
        self._mda_width_locked_at_real_size = False
        self._layout_restored = False
        # Bumped whenever the whole arrangement is replaced (restore or
        # reset). See ``_pin_dock_widths_for_epoch``.
        self._layout_epoch = 0
        # Debounced rather than a plain one-shot on the first showEvent:
        # MainWindow requests WindowMaximized before it's ever shown, and on
        # real window managers that maximize is applied asynchronously --
        # often *after* this tab has already been switched to and gotten its
        # first showEvent. Locking immediately there can freeze the columns
        # at a transient, too-small pre-maximize size that nothing ever
        # revisits afterward (see resizeEvent). Restarting this timer on
        # every resize and only acting once it goes quiet for
        # ``_WIDTH_SETTLE_DELAY_MS`` waits out that churn.
        self._width_settle_timer = QTimer(self)
        self._width_settle_timer.setSingleShot(True)
        self._width_settle_timer.setInterval(_WIDTH_SETTLE_DELAY_MS)
        self._width_settle_timer.timeout.connect(self._settle_and_lock_widths)

        # toolbar: snap|live ‖ shutters … [panel buttons]
        self._shutters = ShuttersBar(self._core)
        self._snap_btn = SnapButton(mmcore=self._core)
        self.toolbar.add_widget(self._snap_btn)
        self._live_btn = LiveButton(mmcore=self._core)
        self.toolbar.add_widget(self._live_btn)
        self.toolbar.add_widget(toolbar_separator())
        self.toolbar.add_widget(self._shutters)

        self._layout_name = DEFAULT_LAYOUT_NAME
        self._layout_btn = LayoutMenuButton(DEFAULT_LAYOUT_NAME, self)
        self._layout_btn.layoutSelected.connect(self.select_layout)
        self._layout_btn.saveLayoutRequested.connect(self._prompt_save_layout)
        self._layout_btn.deleteLayoutRequested.connect(self._delete_layout)
        self.refresh_layout_menu()

        self._panel_bar = PanelButtonBar(PANELS, self)
        self._place_panel_bar()

        self._panels: dict[str, _Panel] = {
            info.key: _Panel(info=info, button=self._panel_bar.button_for(info.key))
            for info in PANELS
        }
        for info in PANELS:
            self._panel_bar.button_for(info.key).toggled.connect(
                partial(self._toggle_panel, info.key)
            )
        self._panel_bar.panelVisibilityChanged.connect(self._set_panel_visible)

        # Default-open panels build now, before any width pinning. MDA goes
        # first and is bound immediately: opening any *other* panel creates
        # the right column, which pins the column widths -- and that reads
        # ``self._mda_dock``.
        self._panel_bar.button_for(PanelKey.MDA).setChecked(True)
        self._mda = cast("MemoryMDAWidget", self.panel_widget(PanelKey.MDA))
        self._mda_dock = cast("CDockWidget", self.panel_dock(PanelKey.MDA))
        for info in PANELS:
            if info.default_open and info.key != PanelKey.MDA:
                self._panel_bar.button_for(info.key).setChecked(True)

        self._snap_btn.snapRequested.connect(self._mda.apply_active_channel_for_capture)
        self._snap_btn.snapRequested.connect(self._viewers.ensure_preview)
        self._live_btn.liveStartedRequested.connect(
            self._mda.apply_active_channel_for_capture
        )
        self._live_btn.liveStartedRequested.connect(self._viewers.ensure_preview)
        self._roi_sync = CameraRoiSyncController(
            self._core,
            self._mda,
            self._viewers,
            self._live_btn,
            parent=self,
        )

        self._pin_dock_widths()
        self._lock_default_areas()
        # A fixed-width neighbor may prevent a platform's hit testing from
        # targeting the QSplitterHandle at all, so its Enter/Press filter would
        # never get a chance to unlock it. Polling the cursor while this page is
        # alive lets the real handle become draggable before the click.
        self._width_handle_hover_timer = QTimer(self)
        self._width_handle_hover_timer.setInterval(20)
        self._width_handle_hover_timer.timeout.connect(self._update_width_handle_hover)
        self._width_handle_hover_timer.start()

        # dockWidgetAdded / dockAreaCreated->currentChanged (see __init__ above)
        # give instant correction for the two known triggers, but ADS resets a
        # tab close button's icon by calling setIcon() directly from its own
        # C++ tab-switch code -- confirmed by installing a raw event filter on
        # the button and switching tabs: no Polish/StyleChange/anything else
        # reaches it, and the button is the same instance before and after (it
        # isn't destroyed and recreated either). There is no Qt event to
        # intercept, so -- like ``_width_handle_hover_timer`` above, for the
        # same reason -- this polls instead: a low-cost backstop that self-
        # heals from any cause, including ones this app doesn't know about
        # yet, instead of relying on someone finding and wiring a new signal
        # every time ADS invalidates chrome a new way.
        self._dock_icon_poll_timer = QTimer(self)
        self._dock_icon_poll_timer.setInterval(1500)
        self._dock_icon_poll_timer.timeout.connect(self._refresh_dock_icons)
        self._dock_icon_poll_timer.start()
        self._refresh_dock_icons()

    def _ensure_preview_for_roi_auto_snap(self, *_args: object) -> None:
        """Create the lazy Preview before Camera ROI performs an Auto Snap."""
        if self._viewers.preview is not None:
            return
        mda = getattr(self, "_mda", None)
        if mda is None:
            return
        auto_snap = mda.camera_roi.snap_checkbox
        if auto_snap.isChecked() and auto_snap.isVisible():
            self._viewers.ensure_preview()

    # ------------------------------------------------------------------ panels

    def _place_panel_bar(self) -> None:
        """Host the panel bar. This is the single seam for relocating it.

        Today it shares the acquire toolbar row, right of snap/live/shutters.
        To give it its own row instead, swap the body for::

            row = TabToolBar()
            row.add_widget(self._panel_bar)
            row.add_stretch()
            self.add_toolbar_row(row)

        Moving it into ``MainWindow`` as a draggable ``QToolBar`` is the same
        shape: the bar itself needs no changes, only a different host.

        The layout drop-down shares the row, separated from the panel
        buttons: switching/saving the whole arrangement is a different
        concern from toggling one panel.
        """
        self.toolbar.add_stretch()
        self.toolbar.add_widget(toolbar_separator())
        self.toolbar.add_widget(self._panel_bar)
        self.toolbar.add_widget(toolbar_separator())
        self.toolbar.add_widget(self._layout_btn)
        # Right-clicking anywhere on the host row opens the same customize
        # menu as the bar's own ⋯ button -- the Qt convention for toolbars.
        self.toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.toolbar.customContextMenuRequested.connect(self._popup_panel_menu)

    def _popup_panel_menu(self, pos: QPoint) -> None:
        self._panel_bar.popup_menu(self.toolbar.mapToGlobal(pos))

    def panel_button(self, key: str) -> QPushButton:
        """Return the toolbar toggle button for the panel registered under *key*."""
        return self._panels[key].button

    def panel_widget(self, key: str) -> QWidget | None:
        """Return the panel's widget, or None if it hasn't been created yet."""
        return self._panels[key].widget

    def panel_dock(self, key: str) -> CDockWidget | None:
        """Return the panel's dock, or None if it hasn't been created yet."""
        return self._panels[key].dock

    def open_panels(self) -> set[str]:
        """Return the keys of panels that are currently open."""
        return {
            key
            for key, panel in self._panels.items()
            if panel.dock is not None and not panel.dock.isClosed()
        }

    def hidden_panels(self) -> set[str]:
        """Return the keys whose toolbar buttons the user has hidden."""
        return self._panel_bar.hidden_keys()

    def apply_hidden_panels(self, keys: Iterable[str]) -> None:
        """Hide the toolbar buttons for *keys*, showing all the others.

        The restore-time counterpart to :meth:`hidden_panels`. Unlike the
        interactive menu path (:meth:`_set_panel_visible`), showing a button
        here must *not* open its panel: which panels are open is
        ``restore_layout``'s business, and forcing them open would both fight
        that and defeat the lazy-construction guarantee -- every registered
        panel would be built on every launch.
        """
        hidden = {k for k in keys if k in self._panels}
        for key in self._panels:
            visible = key not in hidden
            self._panel_bar.set_button_visible(key, visible)
            if not visible:
                self._close_panel(key)

    def _set_panel_visible(self, key: str, visible: bool) -> None:
        """Show or hide *key*'s toolbar button, taking its dock along with it.

        The customize-menu path. Hiding a button would otherwise strand its
        panel on screen with no way to close it, so hiding also closes the
        dock; symmetrically, re-adding a button opens its panel, which is the
        whole point of picking it from the menu. The widget itself is kept
        alive either way, matching what plain close/reopen already does.
        """
        self._panel_bar.set_button_visible(key, visible)
        if visible:
            self.panel_button(key).setChecked(True)
        else:
            self._close_panel(key)

    def _close_panel(self, key: str) -> None:
        """Close *key*'s panel if it's open, without ever building it."""
        button = self.panel_button(key)
        if button.isChecked():
            button.setChecked(False)

    def _toggle_panel(self, key: str, checked: bool) -> None:
        panel = self._panels[key]
        if panel.dock is None:
            if not checked:
                return
            self._create_panel(panel)
        dock = panel.dock
        assert dock is not None
        dock.toggleView(checked)
        if checked:
            dock.setAsCurrentTab()
            # Closing the last dock in the right column makes ADS destroy it;
            # re-showing one builds a *new* column, which starts at the same
            # transient sliver width (47--60 px) a freshly created one does.
            # ``_add_side_dock``'s pin only covers first creation -- this path
            # never goes through it, since the dock already exists -- so the
            # column would otherwise stay unusably narrow with no way back
            # except dragging it. Widening only kicks in below the usability
            # floor, so a column the user deliberately sized is left alone.
            self._widen_right_column_soon()
            if panel.info.refresh is not None:
                QTimer.singleShot(0, partial(self._refresh_panel, key))

    def _create_panel(self, panel: _Panel) -> None:
        # Some upstream factories return a QDialog (PropertyBrowser) or set
        # always-on-top window flags (create_exception_log) meant for
        # standalone use. Nothing is done about that here on purpose:
        # ``_add_dock``'s ``dock.setWidget()`` reparents the widget, and
        # QWidget.setParent() clears window flags -- which is exactly how
        # docking a QDialog has always worked here. Pre-emptively calling
        # setWindowFlags() would be redundant *and* harmful: Qt documents
        # that it hides the widget, requiring an explicit show() afterward.
        widget = panel.info.create(self, self._core)
        if panel.info.unstyle:
            unstyle_widgets(widget)
        panel.widget = widget
        name, title = panel.info.dock_name, panel.info.title
        if panel.info.area == DockWidgetArea.LeftDockWidgetArea:
            panel.dock = self._add_dock(name, title, widget, panel.info.area)
        else:
            panel.dock = self._add_side_dock(name, title, widget)
        if panel.info.key == PanelKey.STAGE_EXPLORER:
            explorer = cast("ThemedStageExplorer", widget)
            explorer.sendToMDARequested.connect(self._on_stage_explorer_send_to_mda)
        # Connects viewToggled (not toggleViewAction().toggled): pinning a
        # dock to an auto-hide side bar transiently unchecks and re-checks
        # that action without ever emitting viewToggled, so binding the
        # button to the action would close the dock the moment the user
        # pinned it. Both directions are idempotent (setChecked and
        # toggleView no-op when already in the target state), so this never
        # loops.
        panel.dock.viewToggled.connect(panel.button.setChecked)

    def _on_stage_explorer_send_to_mda(
        self, positions: list[useq.Position], replace: bool
    ) -> None:
        """Copy Stage Explorer regions into the MDA positions table."""
        if not positions:
            return
        table = self._mda.stage_positions
        combined = list(positions)
        if not replace:
            combined = [*table.value(exclude_unchecked=False), *combined]
        table.setValue(combined)

        # Make the result immediately visible and active in the acquisition.
        section = self._mda._collapsible_tabs().section("p")
        section.set_checked(True)
        section.set_expanded(True)
        self._mda._collapsible_tabs().refresh_summaries()
        self.panel_button(PanelKey.MDA).setChecked(True)

    def refresh_stage_explorer_pixel_geometry(self) -> None:
        """Refresh an open Stage Explorer after pixel configs are committed."""
        panel = self._panels[PanelKey.STAGE_EXPLORER]
        if panel.widget is not None:
            explorer = cast("ThemedStageExplorer", panel.widget)
            explorer.refreshPixelGeometry()

    def _refresh_panel(self, key: str) -> None:
        panel = self._panels[key]
        if panel.widget is not None and panel.info.refresh is not None:
            panel.info.refresh(panel.widget)

    # ------------------------------------------------------------------ layout

    def current_layout(self) -> AcquireLayout:
        """Capture the current arrangement (docks, open panels, hidden buttons)."""
        open_keys = self.open_panels()
        if not open_keys:
            return AcquireLayout(hidden_panels=frozenset(self.hidden_panels()))
        return AcquireLayout(
            dock_state=self._dock_manager.saveState().data(),
            panels=frozenset(open_keys),
            hidden_panels=frozenset(self.hidden_panels()),
        )

    def apply_layout(self, layout: AcquireLayout) -> bool:
        """Apply a saved *layout*, falling back to the default if it can't be.

        The single entry point for "make the page look like this", used both
        at startup and when the user picks a layout from the toolbar menu.
        Buttons are applied before the docks for the reason given in
        :meth:`apply_hidden_panels`: hiding a button closes its panel, which
        would undo part of the arrangement being restored.

        Returns True if *layout* was restored, False if the page fell back to
        the built-in arrangement.
        """
        self.apply_hidden_panels(layout.hidden_panels)
        if self.restore_layout(layout.dock_state, layout.panels):
            if self.isVisible():
                # Startup goes through the showEvent/resize settle path with
                # the window not yet shown; a *live* switch has real geometry
                # already, but ADS applies the restored splitter tree in its
                # own deferred pass -- so re-arm the same debounced settle
                # rather than locking widths against a half-applied tree.
                self._mda_width_locked_at_real_size = False
                self._schedule_width_settle()
            return True
        self.reset_layout()
        return False

    # ------------------------------------------------- named layouts

    @property
    def layout_name(self) -> str:
        """Name of the layout currently selected in the toolbar menu."""
        return self._layout_name

    def select_layout(self, name: str) -> None:
        """Switch to the layout called *name*, live.

        An unknown or vanished name resolves to the built-in arrangement
        rather than doing nothing, so a layout deleted outside the app can't
        leave the menu pointing at something unreachable.
        """
        layout = resolve_layout(name)
        if layout is None or layout.is_empty():
            self.reset_layout()
            # reset_layout() reports the built-in arrangement, which is what
            # the page now shows even if *name* was a (broken) saved layout.
            return
        self._set_layout_name(name)
        self.apply_layout(layout)

    def refresh_layout_menu(self) -> None:
        """Re-read the available layouts into the toolbar drop-down."""
        names = available_layouts()
        if self._layout_name not in names:
            self._layout_name = DEFAULT_LAYOUT_NAME
        self._layout_btn.set_layouts(names, self._layout_name)

    def _set_layout_name(self, name: str) -> None:
        self._layout_name = name
        self.refresh_layout_menu()
        self.layoutNameChanged.emit(name)

    def _prompt_save_layout(self) -> None:
        """Ask for a name, then write the current arrangement under it."""
        suggestion = (
            self._layout_name if self._layout_name not in RESERVED_LAYOUT_NAMES else ""
        )
        name, ok = QInputDialog.getText(
            self, "Save Layout", "Layout name:", text=suggestion
        )
        if not ok:
            return
        name = name.strip()
        if not is_valid_layout_name(name):
            QMessageBox.warning(
                self,
                "Save Layout",
                f"{name!r} is not a usable layout name."
                if name
                else "Please enter a layout name.",
            )
            return
        if name in list_layouts() and not self._confirm_overwrite(name):
            return
        save_layout(name, self.current_layout())
        self._set_layout_name(name)

    def _confirm_overwrite(self, name: str) -> bool:
        choice = QMessageBox.question(
            self,
            "Save Layout",
            f"A layout named {name!r} already exists.\n\nReplace it?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return choice == QMessageBox.StandardButton.Yes

    def _delete_layout(self, name: str) -> None:
        choice = QMessageBox.question(
            self,
            "Delete Layout",
            f"Delete the layout {name!r}?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if choice != QMessageBox.StandardButton.Yes:
            return
        delete_layout(name)
        # Deleting the layout doesn't rearrange anything -- the page keeps
        # showing it -- but it's no longer a name that can be selected.
        if self._layout_name == name:
            self._set_layout_name(DEFAULT_LAYOUT_NAME)
        else:
            self.refresh_layout_menu()

    def restore_layout(self, state: bytes | None, keys: Iterable[str]) -> bool:
        """Recreate the given panels and restore a previously saved dock layout.

        Returns True if the layout was restored, False if there was nothing
        to restore or ADS rejected the saved state -- either way, the page
        is left in a working (default) layout.
        """
        requested = set(keys)
        # ADS state stores dock object names as well as our separate key set. An
        # old state containing the removed standalone Camera ROI dock cannot be
        # safely rewritten, so deliberately fall back to the working default once.
        if requested & _REMOVED_PANEL_KEYS:
            return False
        wanted = {k for k in requested if k in self._panels}
        if not state or not wanted:
            return False
        for key in wanted:
            panel = self._panels[key]
            if panel.dock is None:
                self._create_panel(panel)
        self._release_width_locks()
        if not self._dock_manager.restoreState(state):
            self._relock_widths(pin=False)
            return False
        for panel in self._panels.values():
            if panel.dock is not None:
                with QSignalBlocker(panel.button):
                    panel.button.setChecked(not panel.dock.isClosed())
        self._rediscover_areas()
        self._layout_restored = True
        self._layout_epoch += 1
        # Deliberately *not* re-locking widths here: restore_layout runs
        # before the window has ever been shown (``_app`` defers it to a
        # singleShot that itself calls ``show()``), so every dock area is
        # still 0px wide. ``showEvent`` installs the locks once there is a
        # real geometry to lock to.
        return True

    def reset_layout(self) -> None:
        """Restore the out-of-the-box Acquire arrangement.

        This *is* the "Default" layout -- there is no stored record of it, so
        selecting Default from the layout menu comes straight here.

        Un-hides every toolbar button, un-pins anything the user sent to an
        auto-hide side bar, closes every panel except the default-open ones
        (MDA on the left, Groups and Presets on the right -- see
        ``_panels.PANELS``), puts the MDA column back on the left, and
        re-applies the default column widths. Panel *widgets* are kept alive,
        exactly as a normal close does -- this resets the arrangement, not the
        session.

        Only the layout is affected: window geometry, theme and zoom are
        deliberately left alone, since losing those to a "reset layout"
        action would be a surprise.
        """
        for info in PANELS:
            panel = self._panels[info.key]
            self._panel_bar.set_button_visible(info.key, True)
            if (dock := panel.dock) is not None:
                with suppress(RuntimeError):
                    if dock.isAutoHide():
                        dock.setAutoHide(False)
            panel.button.setChecked(info.default_open)

        self._release_width_locks()
        if not self._mda_is_home():
            # Only move MDA when the user actually dragged it elsewhere.
            # Re-docking it unconditionally would empty and destroy its
            # current area even in the common case where it never moved --
            # churn for nothing, and the one ADS operation documented in
            # ``_configure_ads`` as fatal under the offscreen test platform.
            self._dock_manager.addDockWidget(
                DockWidgetArea.LeftDockWidgetArea, self._mda_dock
            )
        self._right_dock_area = None
        # A reset supersedes whatever was restored, so the canonical widths
        # apply again (see showEvent's one-shot pin).
        self._layout_restored = False
        self._layout_epoch += 1
        self._relock_widths(pin=True)
        self._set_layout_name(DEFAULT_LAYOUT_NAME)
        self.layoutReset.emit()

    def _mda_is_home(self) -> bool:
        """True if the MDA dock still occupies the leftmost column."""
        if self._mda_dock.isAutoHide():
            return False
        area = self._mda_dock.dockAreaWidget()
        if area is None or area is self._central_dock_area:
            return False
        splitter = area.parentWidget()
        return isinstance(splitter, QSplitter) and splitter.indexOf(area) == 0

    def _rediscover_areas(self) -> None:
        """Re-point cached area references after ``restoreState`` rebuilt the tree."""
        central = self._dock_manager.centralWidget()
        if central is not None and (area := central.dockAreaWidget()) is not None:
            self._central_dock_area = area
            area.setAllowedAreas(DockWidgetArea.OuterDockAreas)

        # Drop the stale cache and re-derive from the rebuilt tree, so the
        # next panel opened tabs into the restored right column instead of
        # spawning a second one beside it.
        self._right_dock_area = None
        self._resolve_right_dock_area()

    # ------------------------------------------------------------------ dock helpers

    def _pin_dock_widths(self) -> None:
        """Set the MDA / right-column widths for the current manager width.

        Called once at startup (before the MDA column's width lock -- see
        ``_install_width_lock`` -- takes over) and whenever the right column
        is first created.
        """
        mda_area = self._mda_dock.dockAreaWidget()
        if mda_area is None:
            return
        # ``setSplitterSizes`` needs exactly one entry per *current* child of
        # that splitter. Deriving the count from the live splitter rather than
        # from ``_right_dock_area`` keeps this correct in the in-between
        # states -- notably right after ``reset_layout`` closes the side
        # panels, when ADS has not yet collapsed the emptied right column, so
        # the tree still has three children while the cache says two.
        sizes = list(self._dock_manager.splitterSizes(mda_area))
        if len(sizes) < 2:
            return
        total = self._dock_manager.width()
        new = [0] * len(sizes)
        new[0] = _MDA_DOCK_WIDTH
        middle = range(1, len(sizes))
        if len(sizes) > 2:
            new[-1] = min(total // 4, _RIGHT_DOCK_MAX_WIDTH)
            middle = range(1, len(sizes) - 1)
        remaining = max(total - sum(new), _DOCK_MIN_WIDTH)
        for idx in middle:
            new[idx] = remaining // len(middle)
        self._dock_manager.setSplitterSizes(mda_area, new)

    def _widen_right_column_soon(self, attempts: int = 5) -> None:
        """Widen an unusably narrow right column once QtADS has laid it out.

        Both callers act on a column that does not exist yet at the moment
        they run -- reopening a dock into a column ADS destroyed when it was
        emptied, or restoring one -- so this has to wait for ADS's own
        deferred layout pass. Bounded retries rather than a single shot
        because that pass may take more than one event-loop turn on a slower
        machine, and no retries at all would silently leave the sliver.
        """

        def _attempt() -> None:
            # The page (or its docks) may be gone by the next turn.
            with suppress(RuntimeError):
                if not self._widen_unusable_right_column() and attempts > 1:
                    self._widen_right_column_soon(attempts - 1)

        QTimer.singleShot(0, _attempt)

    def _widen_unusable_right_column(self) -> bool:
        """Expand an unusably narrow tools column to a workable width.

        Widths at or above the usability floor are user choices and remain
        untouched. Below it, the column is one of QtADS's transient 47--60 px
        slivers -- what a freshly created (or re-shown, or restored) column
        starts at before anything sizes it, and what older sessions persisted
        when a lazy dock was frozen before QtADS laid it out. Widen those,
        taking space from the largest non-MDA sibling in the outer horizontal
        splitter.

        Returns whether the caller can stop retrying -- *not* whether the
        column reached the width it aimed for. Only a dock tree QtADS has not
        finished rebuilding reports False, so the startup settle timer runs
        again after its next deferred pass. Every other outcome is final,
        including "the donor column had less to spare than we asked for":
        retrying recomputes identical numbers forever, which would leave the
        settle timer firing for the life of the page and -- because
        ``_settle_and_lock_widths`` bails before setting it -- the MDA width
        lock never marked as installed. A CI runner (or any small display)
        that clamps the window below the requested size lands in exactly that
        case, so it has to be a normal outcome rather than a retry.
        """
        area = self._resolve_right_dock_area()
        if area is None:
            return True
        column = self._column_widget(area)
        if column.width() >= _RIGHT_DOCK_MIN_USABLE_WIDTH:
            return True

        splitter = column.parentWidget()
        if not isinstance(splitter, QSplitter):
            return False
        idx = splitter.indexOf(column)
        sizes = list(splitter.sizes())
        total = self._dock_manager.width()
        # A manager still at its pre-layout width gives nothing to size
        # against -- that is genuinely "not ready", so retry.
        if idx < 0 or len(sizes) < 2 or total <= 0:
            return False

        excluded = {idx}
        if (mda_area := self._mda_dock.dockAreaWidget()) is not None:
            mda_column = self._column_widget(mda_area)
            if mda_column.parentWidget() is splitter:
                excluded.add(splitter.indexOf(mda_column))
        donors = [i for i in range(len(sizes)) if i not in excluded]
        if not donors:
            return True

        # Aim for the canonical width but settle for whatever the donor can
        # actually spare, so a too-small window still gets the widest usable
        # column available instead of staying at the corrupt one. Asking for
        # more than the donor's own minimum allows is harmless either way --
        # the splitter clamps the request.
        target = min(total // 4, _RIGHT_DOCK_MAX_WIDTH)
        donor = max(donors, key=sizes.__getitem__)
        delta = min(target - sizes[idx], sizes[donor])
        if delta <= 0:
            return True
        sizes[idx] += delta
        sizes[donor] -= delta
        splitter.setSizes(sizes)
        return True

    def _lock_default_areas(self) -> bool:
        """(Re)install the width lock on the MDA area, if available.

        The right tools column deliberately remains unlocked: locking it as
        soon as its first lazy dock is created can freeze QtADS's transient
        pre-layout width (often only 47--60 px).  It also must remain directly
        resizable without depending on hover detection to release a lock.

        Returns whether the MDA lock took -- see ``_install_width_lock``.
        """
        if (mda_area := self._mda_dock.dockAreaWidget()) is not None:
            return self._install_width_lock(mda_area)
        return True

    def _column_widget(self, area: CDockAreaWidget) -> QWidget:
        """Return the outer splitter's direct child that *area* lives under.

        ``_add_side_dock`` tabs every new right-side panel into one shared
        area by default, in which case *area* already is that direct child --
        and the same is true of the MDA column, ``_add_dock``'s
        ``LeftDockWidgetArea`` default. But ADS lets the user drag *any* tab
        out into its own area stacked (or split) alongside the others in the
        same column -- MDA's included, nothing here is special-cased to it --
        and then that column is a nested splitter wrapping one area per stack
        slot, with *area* somewhere inside it. A splitter forces every
        stacked child to share its width, so locking one of those inner areas
        to a fixed width would transitively lock the *whole* column to that
        width forever, with no handle left to escape through (the
        eventFilter below is attached to the boundary of whatever gets
        locked, and there is no boundary between stacked siblings and the
        rest of the window). Locking the wrapping splitter itself instead
        keeps the lock on the actual column-width boundary regardless of how
        many areas the user has split that column into, or which column it
        is.

        Climbs from *area* until the next splitter up is no longer nested in
        another splitter -- i.e. until it reaches the outermost (MDA / center
        / right) splitter -- and returns whichever ancestor sits directly
        under *that*. ADS wraps even an unsplit column in a chain of
        single-child splitters (observed for the center/viewer column), so
        "no splitter above" is what actually identifies the outermost level,
        not "only one splitter up" -- and unlike comparing against a
        separately-fetched reference with ``is``, this never depends on
        Python wrapper identity being preserved across repeated
        ``.parentWidget()`` calls on the same C++ object, which PySide6 does
        not guarantee (PyQt6 does, but relying on that is what let a PySide6-
        only regression through here once already).
        """
        widget: QWidget = area
        parent = widget.parentWidget()
        while isinstance(parent, QSplitter) and isinstance(
            parent.parentWidget(), QSplitter
        ):
            widget = parent
            parent = widget.parentWidget()
        return widget

    def _install_width_lock(self, area: CDockAreaWidget) -> bool:
        """Freeze *area*'s column width except while it's actively being dragged.

        Viewer changes cannot reach this splitter because viewers now live in
        their own nested manager. The outer manager can still recompute its
        proportions when an MDA/tool area is opened, closed, moved, restored,
        or pinned, however. A hard
        ``minimumWidth == maximumWidth`` constraint is the only thing a
        splitter must respect in *every* layout pass it computes, regardless
        of what triggers that pass or when, so this locks the column there
        instead. Locking permanently would also block genuine user resizing,
        though, so the constraint is lifted for exactly the duration of a
        real drag on the handle adjacent to it (mouse press to release,
        caught via ``eventFilter``) and re-applied at whatever width the
        user leaves it at -- preserving free resizing while remaining
        immune to every other cause of unwanted relayout.

        Returns whether the lock actually took. Right after a layout
        restore, ADS is still applying the restored tree to the live
        widgets in its own deferred pass (see ``_settle_and_lock_widths``),
        so the expected handle -- or even *area*'s width -- may not exist
        yet; the caller is expected to retry rather than treat that as
        permanent.
        """
        widget = self._column_widget(area)
        splitter = widget.parentWidget()
        if not isinstance(splitter, QSplitter):
            return False
        # The leftmost column (MDA) has no handle to its left; every other
        # column (e.g. the right/Groups & Presets column) has no handle to
        # its right, since it's the last splitter child -- use whichever
        # boundary actually exists.
        idx = splitter.indexOf(widget)
        handle = splitter.handle(idx if idx > 0 else idx + 1)
        if handle is None:
            return False
        handle.installEventFilter(self)
        self._width_locked_areas[handle] = widget
        if type(splitter).__module__.startswith("PySide6"):
            # PySide6 can invalidate the handle when its temporary wrapper for
            # this QtAds-owned splitter is collected. PyQt6 does not have that
            # ownership issue, so preserve its original wrapper lifecycle.
            self._width_lock_splitters[handle] = splitter
        return self._lock_width(widget)

    def _release_width_locks(self) -> None:
        """Drop every lock before the objects holding them are destroyed.

        ``CDockManager.restoreState`` tears down and rebuilds the entire
        splitter/dock-area tree, so both the ``QSplitterHandle`` keys and the
        ``CDockAreaWidget`` values in ``_width_locked_areas`` become dangling
        C++ wrappers afterward -- touching one then raises ``RuntimeError``.
        Unhook the event filters and lift the constraints first, so nothing
        stale survives the rebuild.
        """
        for handle, area in self._width_locked_areas.items():
            # PySide6 raises as soon as a wrapper's C++ object has already
            # disappeared; ADS may replace a splitter handle during an earlier
            # relayout, before restoreState itself starts rebuilding the tree.
            with suppress(RuntimeError):
                handle.removeEventFilter(self)
            with suppress(RuntimeError):
                self._unlock_width(area)
        self._width_locked_areas.clear()
        self._width_lock_splitters.clear()
        self._dragging_width_handles.clear()

    def _relock_widths(self, *, pin: bool) -> bool:
        """Lift every width lock, optionally re-pin, then lock again.

        Used by the one-shot ``showEvent`` refresh: the constraints have to
        come off before ``_pin_dock_widths`` (when *pin* is true) can resize
        anything, and go back on at the resulting width. Rebuilding the locks
        from scratch rather than reusing the old set also covers the restore
        path, where ``restore_layout`` released them and ADS then built an
        entirely new set of dock areas.

        *pin* must be False after a layout restore -- the restored splitter
        sizes are the widths the user left, and ``_pin_dock_widths`` assumes
        the canonical 2-/3-column arrangement, which an arbitrary restored
        layout need not have.

        Returns whether every attempted lock actually took -- see
        ``_lock_default_areas``.
        """
        self._release_width_locks()
        if pin:
            self._pin_dock_widths()
        return self._lock_default_areas()

    def _lock_width(self, area: QWidget) -> bool:
        width = area.width()
        if width <= 0:
            # Nothing meaningful to lock to yet -- the page hasn't been laid
            # out at its real geometry. Freezing min == max == 0 here would
            # pin the column shut permanently (it survives every later
            # layout pass, which is the whole point of the lock), so leave it
            # unconstrained and let showEvent lock it once it has a width.
            return False
        area.setMinimumWidth(width)
        area.setMaximumWidth(width)
        return True

    def _unlock_width(self, area: QWidget) -> None:
        area.setMinimumWidth(_DOCK_MIN_WIDTH)
        area.setMaximumWidth(QWIDGETSIZE_MAX)

    def _update_width_handle_hover(self) -> None:
        """Unlock a fixed column while the pointer is near its resize handle."""
        cursor = QCursor.pos()
        for handle, area in tuple(self._width_locked_areas.items()):
            if not isinstance(handle, QWidget):
                continue
            with suppress(RuntimeError):
                local_pos = handle.mapFromGlobal(cursor)
                is_near = handle.rect().adjusted(-6, 0, 6, 0).contains(local_pos)
                is_unlocked = area.minimumWidth() != area.maximumWidth()
                if is_near and not is_unlocked:
                    self._unlock_width(area)
                elif (
                    not is_near
                    and is_unlocked
                    and handle not in self._dragging_width_handles
                ):
                    self._lock_width(area)

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        """Unlock a column while its splitter handle is hovered or dragged."""
        area = None if a0 is None else self._width_locked_areas.get(a0)
        if a0 is not None and a1 is not None and area is not None:
            event_type = a1.type()
            if event_type == QEvent.Type.Enter:
                # Unlock before the press: on some real platform plugins a
                # fixed-size neighbor prevents the handle from beginning a
                # drag, so waiting for MouseButtonPress is too late.
                self._unlock_width(area)
            elif event_type == QEvent.Type.MouseButtonPress:
                self._dragging_width_handles.add(a0)
                self._unlock_width(area)
            elif event_type == QEvent.Type.MouseButtonRelease:
                self._dragging_width_handles.discard(a0)
                self._lock_width(area)
            elif (
                event_type == QEvent.Type.Leave
                and a0 not in self._dragging_width_handles
            ):
                self._lock_width(area)
        return super().eventFilter(a0, a1)

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

    def _resolve_right_dock_area(self) -> CDockAreaWidget | None:
        """Return the live right-column area, re-deriving it if the cache is stale.

        ADS destroys a dock area once its last dock widget leaves it, and
        ``restoreState`` rebuilds the whole tree -- either way the cached
        reference can end up wrapping a deleted C++ object, and docking *into*
        that would crash rather than tab. Re-derive from whichever side panel
        is currently open before concluding there is no right column yet.
        """
        area = self._right_dock_area
        if area is not None:
            with suppress(RuntimeError):
                if area.dockWidgetsCount() > 0:
                    return area
        self._right_dock_area = None
        for key, panel in self._panels.items():
            if key == PanelKey.MDA or panel.dock is None:
                continue
            with suppress(RuntimeError):
                if panel.dock.isClosed() or panel.dock.isAutoHide():
                    continue
                candidate = panel.dock.dockAreaWidget()
                if candidate is not None and candidate is not self._central_dock_area:
                    self._right_dock_area = candidate
                    return candidate
        return None

    def _add_side_dock(self, name: str, title: str, widget: QWidget) -> CDockWidget:
        """Add a dock to the right column, tabbing into it if one already exists.

        Every panel that isn't explicitly placed elsewhere lands here, so a
        widget being opened for the first time always appears in the right
        sidebar -- as a new column if it's the only one there, tabbed
        alongside its neighbours otherwise. The first call caps the column's
        initial width at ``_RIGHT_DOCK_MAX_WIDTH``.
        """
        if (right_area := self._resolve_right_dock_area()) is not None:
            return self._add_dock(
                name,
                title,
                widget,
                DockWidgetArea.CenterDockWidgetArea,
                right_area,
            )
        dock = self._add_dock(name, title, widget, DockWidgetArea.RightDockWidgetArea)
        self._right_dock_area = dock.dockAreaWidget()
        self._pin_dock_widths()
        # addDockWidget() has created the area, but QtADS does not insert and
        # lay out its new splitter child until control returns to the event
        # loop.  Re-pin once that deferred insertion is complete; otherwise
        # the request above is discarded and the new tools column stays at
        # its transient minimum width (typically 47--60 px).
        pin = partial(self._pin_dock_widths_for_epoch, self._layout_epoch)
        QTimer.singleShot(0, pin)
        return dock

    def _pin_dock_widths_for_epoch(self, epoch: int) -> None:
        """Run the deferred pin only if the layout hasn't changed under it.

        A default-open side panel queues this from ``__init__``, so it can
        still be pending when ``restore_layout`` runs on the very next line
        (that's the launch sequence: construct, restore, show). Firing it
        then would overwrite the restored splitter sizes with the canonical
        defaults -- the user's dragged column widths, silently reset on every
        launch. Comparing epochs skips exactly that case while leaving a pin
        queued by a panel opened *after* a restore free to do its job.
        """
        # A default-open panel queues this from ``__init__``, so a page that
        # is built and torn down inside a single event-loop turn (common in
        # tests) reaches here with its docks already destroyed.
        with suppress(RuntimeError):
            if epoch == self._layout_epoch:
                self._pin_dock_widths()

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
        override = f"""
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
        self._dock_manager.setStyleSheet(self._base_dock_style + override)
        self._viewer_dock_manager.setStyleSheet(self._viewer_base_dock_style + override)

    def _connect_dock_area_tab_switch(self, area: CDockAreaWidget) -> None:
        """Refresh dock icons whenever *area*'s current tab changes.

        There's no dock-manager-wide signal for this (only per-area), so this
        is wired for every area as it's created -- see the comment where
        ``dockAreaCreated`` is connected, and ``_refresh_dock_icons`` for why
        a tab switch needs this at all.
        """
        with suppress(RuntimeError):
            area.currentChanged.connect(self._queue_dock_icon_refresh)

    def _queue_dock_icon_refresh(self, *_args: object) -> None:
        """Refresh once ADS has finished whatever chrome change triggered this.

        Shared by ``dockWidgetAdded`` (a newly added dock's chrome is still
        being constructed) and each dock area's ``currentChanged`` (switching
        tabs makes ADS overwrite the outgoing and incoming tab's close-button
        icon -- see ``_refresh_dock_icons``); queuing to the next event-loop
        turn covers both callers' in-progress state the same way.
        """
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
        white again on the next theme change -- and so this same method, when
        re-run, can restore it after ADS's own C++ tab-switch code calls
        ``setIcon()`` directly on both the outgoing and incoming tab's close
        button, reverting it to the fixed black source. That call isn't
        reachable through Qt's event system at all (confirmed: the same
        button instance survives a switch -- it isn't destroyed and recreated
        -- and installing a raw event filter on it sees no Polish,
        StyleChange, or any other event when the icon flips), which is why
        this is invoked from a *timer* (``_dock_icon_poll_timer``) as a
        backstop in addition to the two known-trigger signals above: it
        self-heals from any cause, including ones not enumerated here.
        """
        # _queue_dock_icon_refresh defers to the next event-loop turn. A
        # short-lived window may be gone by then, especially under PySide6,
        # whose wrappers immediately reject access to deleted C++ objects.
        with suppress(RuntimeError):
            red = qcolor(theme().status_red)
            for btn in self.findChildren(QAbstractButton):
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
        for dm in (self._dock_manager, self._viewer_dock_manager):
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
        if not self._mda_width_locked_at_real_size:
            # AcquirePage is constructed eagerly in MainWindow.__init__, before
            # the window has been shown/resized to its real on-screen geometry
            # -- so the one-time initial pin (and the width lock installed
            # right after it) may have captured too small a width. Schedule
            # the (one-time) settle-and-lock now that this tab has *a* real
            # geometry; resizeEvent keeps pushing it back out while that
            # geometry is still changing (e.g. an in-progress WindowMaximized
            # completing after this showEvent), so it only fires once things
            # have actually settled.
            self._schedule_width_settle()
        self._shutters.refresh()
        for key, panel in self._panels.items():
            if (
                panel.widget is not None
                and panel.dock is not None
                and not panel.dock.isClosed()
                and panel.info.refresh is not None
            ):
                QTimer.singleShot(0, partial(self._refresh_panel, key))

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        """Keep pushing the initial width-settle out while still resizing.

        Covers the same startup race as ``showEvent``'s call into
        ``_schedule_width_settle``, for the common case where this tab's
        first real resize (e.g. an async ``WindowMaximized`` completing)
        lands *after* that first showEvent already scheduled a lock -- this
        restarts the same timer so the lock still waits for the resizing to
        stop rather than firing mid-resize.
        """
        super().resizeEvent(a0)
        if not self._mda_width_locked_at_real_size:
            self._schedule_width_settle()

    def _schedule_width_settle(self) -> None:
        """(Re)start the debounce timer that runs ``_settle_and_lock_widths`` once."""
        if not self._mda_width_locked_at_real_size:
            self._width_settle_timer.start()

    def _settle_and_lock_widths(self) -> None:
        """One-time pin/lock once this tab's geometry has stopped changing.

        Mirrors what the old one-shot ``showEvent`` handler did, just fired
        from the debounce timer instead: unlike the refreshes in
        ``showEvent``, this must run only once ever -- repeating it on every
        later resize would wipe out any width the user has since dragged a
        locked column to (or, after a layout restore, the width the user
        left it at last session -- hence ``pin=False`` on that path).

        This tab's *own* geometry settling (what the debounce above waits
        for) is necessary but not sufficient after a restore: ADS applies
        the restored splitter sizes to the live dock-area tree -- rebuilding
        the splitter/handle structure along the way -- in its own deferred
        pass, decoupled from this tab's resize events. The debounce can fire
        while that pass is still mid-flight, when the MDA area may be
        genuinely 0px wide, or its column's expected splitter handle may not
        exist yet at all (``_install_width_lock`` needs a specific handle
        index, which requires every sibling column to have already been
        re-added to the tree). Either way ``_relock_widths`` reports it
        rather than lock nothing and call it done: a width of 0 wouldn't
        actually get frozen there (``_lock_width`` guards against that), and
        a missing handle just skips the lock outright -- both
        indistinguishable, from the outside, from having settled with
        nothing to lock. Retrying on that signal instead of a plain width
        check catches the missing-handle case too, and waits out however
        long ADS's pass actually takes instead of gambling that one fixed
        timeout covers it -- a race that a slower CI runner or platform
        binding can still lose.
        """
        ok = self._relock_widths(pin=not self._layout_restored)
        if ok and self._layout_restored:
            ok = self._widen_unusable_right_column()
        if not ok:
            self._width_settle_timer.start()
            return
        self._mda_width_locked_at_real_size = True
