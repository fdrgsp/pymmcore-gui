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
from pymmcore_gui._qt.QtGui import QFont
from pymmcore_gui._qt.QtWidgets import (
    QWIDGETSIZE_MAX,
    QAbstractButton,
    QAbstractSlider,
    QPushButton,
    QSplitter,
    QWidget,
)

from ._acquire_toolbar import (
    LiveButton,
    PanelButtonBar,
    ShuttersBar,
    SnapButton,
    toolbar_separator,
)
from ._acquire_viewers import AcquireViewersManager
from ._panels import PANELS, PanelInfo, PanelKey
from ._tab_page import TabPage
from ._theme import qcolor, theme

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pymmcore_gui._qt.QtAds import CDockAreaWidget
    from pymmcore_gui._qt.QtGui import QShowEvent
    from pymmcore_gui.widgets._mda_widget import MemoryMDAWidget

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
    managers behave identically, plus two additions beyond what the classic
    GUI sets: without ``DockAreaHasAutoHideButton``, pinning a dock to a side
    bar is drag-only and undiscoverable; without ``EqualSplitOnInsertion``,
    dragging a viewer tab out to split it gives the new area only a sliver
    (a fixed minimum width) while the original area keeps the rest, rather
    than splitting the space evenly. These setters are static -- they affect
    every ``CDockManager`` in the process, and only the *first* call before
    any manager is constructed has an effect.

    Note: emptying a dock area (moving its last widget elsewhere, whether to
    another regular area or to auto-hide) reproducibly segfaults under
    ``QT_QPA_PLATFORM=offscreen`` + pytest-qt, independent of any app code,
    on both PyQt6Ads 4.4.0.post2 and 5.0.0 -- confirmed test-harness-only,
    since interactive drag-and-drop on a real display does not reproduce it.
    Automated tests therefore stick to what doesn't empty an area (open/close
    toggling, tabbing a dock into an existing one); actual rearranging is a
    manual smoke-test item (see the PR description).

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

        self._right_dock_area: CDockAreaWidget | None = None
        self._width_locked_areas: dict[QObject, CDockAreaWidget] = {}
        self._mda_width_locked_at_real_size = False
        self._layout_restored = False

        # toolbar: snap|live ‖ shutters … [panel buttons]
        self._shutters = ShuttersBar(self._core)
        self._snap_btn = SnapButton(mmcore=self._core)
        self.toolbar.add_widget(self._snap_btn)
        self._live_btn = LiveButton(mmcore=self._core)
        self.toolbar.add_widget(self._live_btn)
        self.toolbar.add_widget(toolbar_separator())
        self.toolbar.add_widget(self._shutters)

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
        self._panel_bar.resetLayoutRequested.connect(self.reset_layout)

        # Default-open panels (MDA) build now, before any width pinning.
        for info in PANELS:
            if info.default_open:
                self._panel_bar.button_for(info.key).setChecked(True)

        self._mda = cast("MemoryMDAWidget", self.panel_widget(PanelKey.MDA))
        self._mda_dock = cast("CDockWidget", self.panel_dock(PanelKey.MDA))
        self._snap_btn.snapRequested.connect(self._mda.apply_active_channel_for_capture)
        self._snap_btn.snapRequested.connect(self._viewers.ensure_preview)
        self._live_btn.liveStartedRequested.connect(
            self._mda.apply_active_channel_for_capture
        )
        self._live_btn.liveStartedRequested.connect(self._viewers.ensure_preview)

        self._pin_dock_widths()
        self._lock_default_areas()
        self._refresh_dock_icons()

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
        """
        self.toolbar.add_stretch()
        self.toolbar.add_widget(toolbar_separator())
        self.toolbar.add_widget(self._panel_bar)
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
        # Connects viewToggled (not toggleViewAction().toggled): pinning a
        # dock to an auto-hide side bar transiently unchecks and re-checks
        # that action without ever emitting viewToggled, so binding the
        # button to the action would close the dock the moment the user
        # pinned it. Both directions are idempotent (setChecked and
        # toggleView no-op when already in the target state), so this never
        # loops.
        panel.dock.viewToggled.connect(panel.button.setChecked)

    def _refresh_panel(self, key: str) -> None:
        panel = self._panels[key]
        if panel.widget is not None and panel.info.refresh is not None:
            panel.info.refresh(panel.widget)

    # ------------------------------------------------------------------ layout

    def save_layout(self) -> tuple[bytes | None, set[str]]:
        """Return ``(dock_manager_state, open_panel_keys)`` for persistence."""
        open_keys = self.open_panels()
        if not open_keys:
            return None, set()
        return self._dock_manager.saveState().data(), open_keys

    def restore_layout(self, state: bytes | None, keys: Iterable[str]) -> bool:
        """Recreate the given panels and restore a previously saved dock layout.

        Returns True if the layout was restored, False if there was nothing
        to restore or ADS rejected the saved state -- either way, the page
        is left in a working (default) layout.
        """
        wanted = {k for k in keys if k in self._panels}
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
        # Deliberately *not* re-locking widths here: restore_layout runs
        # before the window has ever been shown (``_app`` defers it to a
        # singleShot that itself calls ``show()``), so every dock area is
        # still 0px wide. ``showEvent`` installs the locks once there is a
        # real geometry to lock to.
        return True

    def reset_layout(self) -> None:
        """Restore the out-of-the-box Acquire arrangement.

        Un-hides every toolbar button, un-pins anything the user sent to an
        auto-hide side bar, closes every panel except the default-open ones,
        puts the MDA column back on the left, and re-applies the default
        column widths. Panel *widgets* are kept alive, exactly as a normal
        close does -- this resets the arrangement, not the session.

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
        self._pin_dock_widths()
        self._lock_default_areas()
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
            self._viewers.set_central_dock_area(area)

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

    def _lock_default_areas(self) -> None:
        """(Re)install width locks on the MDA area and the right column, if any."""
        if (mda_area := self._mda_dock.dockAreaWidget()) is not None:
            self._install_width_lock(mda_area)
        if self._right_dock_area is not None:
            self._install_width_lock(self._right_dock_area)

    def _install_width_lock(self, area: CDockAreaWidget) -> None:
        """Freeze *area*'s width except while it's actively being dragged.

        ADS recomputes splitter proportions for the *whole* manager whenever
        any dock area's visibility changes anywhere in it -- not just areas
        adjacent to the one that changed -- so a central viewer opening or
        closing can silently resize the MDA or right (Groups & Presets /
        Properties / Console) columns even though the user never touched
        them. Reactively re-applying a width after the fact (via a dock's
        ``closed`` signal, even retried across several deferred event-loop
        turns) is a race against ADS's own relayout passes, which can be
        arbitrarily delayed -- proved unreliable in practice. A hard
        ``minimumWidth == maximumWidth`` constraint is the only thing a
        splitter must respect in *every* layout pass it computes, regardless
        of what triggers that pass or when, so this locks the column there
        instead. Locking permanently would also block genuine user resizing,
        though, so the constraint is lifted for exactly the duration of a
        real drag on the handle adjacent to it (mouse press to release,
        caught via ``eventFilter``) and re-applied at whatever width the
        user leaves it at -- preserving free resizing while remaining
        immune to every other cause of unwanted relayout.
        """
        splitter = area.parentWidget()
        if not isinstance(splitter, QSplitter):
            return
        # The leftmost column (MDA) has no handle to its left; every other
        # column (e.g. the right/Groups & Presets column) has no handle to
        # its right, since it's the last splitter child -- use whichever
        # boundary actually exists.
        idx = splitter.indexOf(area)
        handle = splitter.handle(idx if idx > 0 else idx + 1)
        if handle is None:
            return
        handle.installEventFilter(self)
        self._width_locked_areas[handle] = area
        self._lock_width(area)

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
            handle.removeEventFilter(self)
            self._unlock_width(area)
        self._width_locked_areas.clear()

    def _relock_widths(self, *, pin: bool) -> None:
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
        """
        self._release_width_locks()
        if pin:
            self._pin_dock_widths()
        self._lock_default_areas()

    def _lock_width(self, area: CDockAreaWidget) -> None:
        width = area.width()
        if width <= 0:
            # Nothing meaningful to lock to yet -- the page hasn't been laid
            # out at its real geometry. Freezing min == max == 0 here would
            # pin the column shut permanently (it survives every later
            # layout pass, which is the whole point of the lock), so leave it
            # unconstrained and let showEvent lock it once it has a width.
            return
        area.setMinimumWidth(width)
        area.setMaximumWidth(width)

    def _unlock_width(self, area: CDockAreaWidget) -> None:
        area.setMinimumWidth(_DOCK_MIN_WIDTH)
        area.setMaximumWidth(QWIDGETSIZE_MAX)

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        """Unlock a locked column's width for the duration of a live handle drag."""
        area = None if a0 is None else self._width_locked_areas.get(a0)
        if a1 is not None and area is not None:
            if a1.type() == QEvent.Type.MouseButtonPress:
                self._unlock_width(area)
            elif a1.type() == QEvent.Type.MouseButtonRelease:
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
        if self._right_dock_area is not None:
            self._install_width_lock(self._right_dock_area)
        return dock

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
            # right after it) may have captured too small a width. Re-run both
            # now that this tab has real geometry, but only once: unlike the
            # refreshes below, repeating this on every tab switch would wipe
            # out any width the user had since dragged a locked column to (or,
            # after a layout restore, the width the user left it at last
            # session -- hence pin=False on that path).
            self._mda_width_locked_at_real_size = True
            if self._layout_restored:
                # A restored layout has no widths to pin, so there is nothing
                # to force the areas to their final size synchronously -- ADS
                # only applies the restored splitter sizes on the layout pass
                # that follows this event. Locking here would freeze the
                # still-unresized (0px) columns, so wait one turn.
                QTimer.singleShot(0, partial(self._relock_widths, pin=False))
            else:
                self._relock_widths(pin=True)
        self._shutters.refresh()
        for key, panel in self._panels.items():
            if (
                panel.widget is not None
                and panel.dock is not None
                and not panel.dock.isClosed()
                and panel.info.refresh is not None
            ):
                QTimer.singleShot(0, partial(self._refresh_panel, key))
