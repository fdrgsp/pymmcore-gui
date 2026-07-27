"""Configurations tab: property browser, group editor and pixel configuration.

These are the upstream ``pymmcore_widgets`` editors; this module only arranges
them as sub-tabs and keeps them in step with the core.
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from pymmcore_plus import CMMCorePlus
from pymmcore_widgets import (
    ConfigGroupsEditor,
    PixelConfigurationWidget,
)
from pymmcore_widgets._util import block_core
from superqt.iconify import QIconifyIcon

from pymmcore_gui._array_viewer import unstyle_widgets
from pymmcore_gui._qt.QtCore import QEvent, QTimer, pyqtSignal
from pymmcore_gui._qt.QtGui import QFont, QPalette
from pymmcore_gui._qt.QtWidgets import (
    QAbstractSlider,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ._busy import BusyOverlay, busy
from ._tab_page import TabPage
from ._theme import qcolor, theme

SAVING_MSG = "Saving configuration to core…"

if TYPE_CHECKING:
    from pymmcore_gui._qt.QtGui import QShowEvent


class _GroupEditorTab(QWidget):
    """A ConfigGroupsEditor adapted for embedding in the configurations page."""

    def __init__(self, core: CMMCorePlus, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._core = core
        self.editor = ConfigGroupsEditor.create_from_core(core)
        self._overlay = BusyOverlay(self)

        # ConfigGroupsEditor has its own internal "Apply" button (bottom-right,
        # wired to an `applyRequested` signal we don't listen to). Our own
        # "Save to core" toolbar button is the one save action; hide theirs so
        # there's no dead, duplicate control.
        for btn in self.editor.findChildren(QPushButton):
            if btn.text() == "Apply":
                btn.hide()
        # It also has its own status icon/label next to that button ("Unsaved
        # changes" / "No changes") -- redundant with our own toolbar dirty
        # label, which already covers this editor and Pixel Configuration
        # together. Hide both (guarded in case of a future rename upstream).
        for attr in ("_status_icon", "_status_label"):
            if (w := getattr(self.editor, attr, None)) is not None:
                w.hide()
        unstyle_widgets(self.editor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.editor, 1)

    def changeEvent(self, event: QEvent | None) -> None:
        # ConfigGroupsEditor.__init__ calls self.setStyleSheet(...), and
        # applying *any* stylesheet makes Qt resolve and *freeze* the font of
        # the widget's whole subtree -- so it stops following later
        # QApplication.setFont() changes, which is exactly how this app's zoom
        # (Cmd+Shift+±) works. The frozen tree keeps whatever size was current
        # when the stylesheet was first applied and never scales again.
        #
        # set_zoom() sets the new app font, then sends every widget a
        # StyleChange event (which also re-freezes this subtree). Riding that
        # same event, reset each descendant's font to a default-constructed
        # (unresolved) QFont so it re-inherits the now-current app font --
        # matching the pattern _toolbar.py / the dirty label already use to
        # stay theme/zoom reactive.
        if event is not None and event.type() == QEvent.Type.StyleChange:
            for w in (self.editor, *self.editor.findChildren(QWidget)):
                if not isinstance(w, QAbstractSlider):
                    w.setFont(QFont())
        super().changeEvent(event)

    def save(self) -> None:
        """Replace the core's config groups with the editor's contents."""
        groups = list(self.editor.data())
        desired = {g.name for g in groups}
        try:
            with busy(self._overlay, SAVING_MSG):
                # Suppress core events for the whole bulk rewrite. Without
                # this, every single defineConfig() call below fires a live
                # configDefined signal — and any OTHER widget reacting to
                # that in real time (e.g. GroupPresetTableWidget in the
                # Acquire sidebar) sees a group mid-rebuild, treats that
                # incomplete snapshot as ground truth, and "helpfully"
                # issues its own deleteConfigGroup()/defineConfig() calls —
                # silently wiping properties this loop already wrote, or
                # hasn't gotten to yet. Two independent listeners mutating
                # the same live core concurrently was causing real data loss.
                with block_core(self._core.events):
                    # drop groups the editor no longer has
                    for name in list(self._core.getAvailableConfigGroups()):
                        if name not in desired:
                            self._core.deleteConfigGroup(name)
                    # redefine each group (delete first to clear stale presets)
                    for group in groups:
                        if group.name in self._core.getAvailableConfigGroups():
                            self._core.deleteConfigGroup(group.name)
                        self._core.defineConfigGroup(group.name)
                        for preset in group.presets.values():
                            for s in preset.settings:
                                self._core.defineConfig(
                                    group.name,
                                    # preset.name, not the dict key — after a
                                    # rename the editor leaves the key stale
                                    preset.name,
                                    # the device LABEL, not s.device.name
                                    # (which is the adapter name)
                                    s.device.label,
                                    s.property_name,
                                    s.value,
                                )
                    # deleteConfigGroup() above silently clears the core's
                    # channel-group designation whenever it happens to target
                    # the current channel group (MMCore has no concept of a
                    # group's designation surviving its own deletion) — and
                    # the editor's own reassignment (via its "Set Channel
                    # Group" action) otherwise never reaches the core at all.
                    # Restore it from the editor's marked group every save.
                    channel_group = next(
                        (g for g in groups if getattr(g, "is_channel_group", False)),
                        None,
                    )
                    self._core.setChannelGroup(
                        channel_group.name if channel_group else ""
                    )
        except Exception as e:
            QMessageBox.warning(
                self, "Save configuration groups", f"Failed to save:\n\n{e}"
            )
            return
        # NOTE: deliberately do NOT emit systemConfigurationLoaded here. Doing so
        # reloads the editor from the core, which drops any preset that couldn't
        # be persisted — MMCore has no concept of an empty preset (a preset only
        # exists through its defineConfig settings). The editor stays the source
        # of truth so in-progress (settings-less) presets aren't lost on save.
        #
        # Other widgets (e.g. GroupPresetTableWidget) missed every event we
        # just suppressed and are now stale; they already refresh defensively
        # when their tab is shown (the same pattern used elsewhere in the app
        # for cross-tab staleness).

        # ConfigGroupsEditor tracks edits via its own QUndoStack but never marks
        # it clean itself (that was the hidden "Apply" button's job). Do it here
        # so undoStack().cleanChanged reflects *our* save action.
        self.editor.undoStack().setClean()


class _EmbeddedPixelConfig(PixelConfigurationWidget):
    """PixelConfigurationWidget adapted for embedding.

    Upstream, "Apply and Close" applies the pixel configs and then closes the
    widget; embedded in a tab there is nothing to close, so Apply just applies.
    """

    changed = pyqtSignal()  # emitted on any user edit (no public signal upstream)

    def __init__(
        self, mmcore: CMMCorePlus | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent, mmcore=mmcore)
        self._suppress_close = False
        self._suppress_changed = False
        self._overlay = BusyOverlay(self)
        for btn in self.findChildren(QPushButton):
            if btn.text() in {"Apply and Close", "Cancel"}:
                # Save actions live in the page toolbar when embedded.
                btn.hide()
        unstyle_widgets(self)
        # bridge the widget's internal edit signals to a public `changed` one
        for attr in ("_px_table", "_affine_table", "_props_selector"):
            owner = getattr(self, attr, None)
            if (sig := getattr(owner, "valueChanged", None)) is not None:
                with suppress(Exception):
                    sig.connect(self._on_value_changed)

    def apply(self) -> None:
        """Apply the pixel configurations to the core (without closing)."""
        self._on_apply()

    def _on_apply(self) -> None:
        # _on_apply ends by calling self.close(); keep the widget open instead.
        self._suppress_close = True
        try:
            with busy(self._overlay, SAVING_MSG):
                super()._on_apply()
        finally:
            self._suppress_close = False

    def _on_px_table_selection_changed(self) -> None:
        # Selecting a different resolution row makes upstream call
        # _props_selector.setValue(...) purely to *display* that row's
        # settings — but setValue() unconditionally emits valueChanged with
        # no way to tell a display refresh apart from a genuine edit.
        # Suppress our bridge for the duration so merely clicking a row
        # doesn't mark this page dirty.
        self._suppress_changed = True
        try:
            super()._on_px_table_selection_changed()
        finally:
            self._suppress_changed = False

    def _on_value_changed(self, *_: object) -> None:
        if not self._suppress_changed:
            self.changed.emit()

    def close(self) -> bool:
        if self._suppress_close:
            return False
        return super().close()


class ConfigurationsPage(TabPage):
    """Sub-tabbed editors for config groups and pixel sizes."""

    # "Save to core" commits to the live core only; this asks the window to
    # write the whole configuration (hardware + groups + pixel) to a .cfg.
    saveToFileRequested = pyqtSignal()

    def __init__(
        self, mmcore: CMMCorePlus | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._core = mmcore or CMMCorePlus.instance()

        self._group_tab = _GroupEditorTab(self._core)
        self._group_editor = self._group_tab.editor
        self._pixel_config = _EmbeddedPixelConfig(mmcore=self._core)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._group_tab, "Group Editor")
        self._tabs.addTab(self._pixel_config, "Pixel Configuration")

        self.add_content_widget(self._tabs)
        # these editors fill the page; the docks would only crowd them
        self.left.hide()
        self.right.hide()
        self.bottom.hide()

        self._save_core_btn = QPushButton("Save to core")
        self._save_core_btn.setProperty("variant", "primary")
        self._save_core_btn.setToolTip(
            "Apply edits from the selected configuration tab to the live core"
        )
        self._save_core_btn.clicked.connect(self.commit_current_to_core)
        self.toolbar.add_widget(self._save_core_btn)

        self._save_file_btn = QPushButton("Save to file…")
        self._save_file_btn.setProperty("variant", "primary")
        self._save_file_btn.setToolTip(
            "Apply edits from the selected configuration tab to the live core, "
            "then save the full configuration to a .cfg file"
        )
        self._save_file_btn.clicked.connect(self.saveToFileRequested.emit)
        self.toolbar.add_widget(self._save_file_btn)

        self._dirty_icon = QLabel()
        self._dirty_text = QLabel()
        self._dirty_label = QWidget()
        dirty_layout = QHBoxLayout(self._dirty_label)
        dirty_layout.setContentsMargins(0, 0, 0, 0)
        dirty_layout.setSpacing(4)
        dirty_layout.addWidget(self._dirty_icon)
        dirty_layout.addWidget(self._dirty_text)
        self._apply_dirty_style()
        self._dirty_label.hide()
        self.toolbar.add_stretch()
        self.toolbar.add_widget(self._dirty_label)

        # Track the editors independently: saving the selected tab must not
        # silently mark changes in the other tab as persisted.
        #
        # The group editor already tracks edits via its own QUndoStack, which
        # is more accurate than watching for a "changed" signal — e.g. it
        # correctly goes clean again if the user undoes back to the original
        # state. It never marks itself clean on save though (that was the
        # hidden "Apply" button's job) — _GroupEditorTab.save() does it now.
        #
        # Pixel Configuration has no such undo stack, so `_pixel_dirty` is
        # still tracked manually from its bridged `changed` signal.
        # `_suppress` is a depth counter guarding only that manual pixel path
        # against our own programmatic refresh/commit calls.
        self._group_dirty = False
        self._pixel_dirty = False
        self._suppress = 0
        self._group_editor.undoStack().cleanChanged.connect(
            self._on_group_clean_changed
        )
        self._pixel_config.changed.connect(self._mark_pixel_dirty)

        # a configuration may be loaded after this page is built
        self._core.events.systemConfigurationLoaded.connect(
            self._on_system_config_loaded
        )

    # ── unsaved-changes API ───────────────────────────────────────

    def is_dirty(self) -> bool:
        """Whether either editor has edits not yet persisted to a file."""
        return self._group_dirty or self._pixel_dirty

    def mark_saved(self) -> None:
        """Mark both editors as persisted to the configuration file."""
        with suppress(Exception):
            self._group_editor.undoStack().setClean()
        self._group_dirty = False
        self._pixel_dirty = False
        self._update_dirty_label()

    def mark_current_saved(self) -> None:
        """Mark only the selected editor as persisted to the file."""
        if self._tabs.currentWidget() is self._group_tab:
            with suppress(Exception):
                self._group_editor.undoStack().setClean()
            self._group_dirty = False
        elif self._tabs.currentWidget() is self._pixel_config:
            self._pixel_dirty = False
        self._update_dirty_label()

    def commit_current_to_core(self) -> None:
        """Write the selected editor's contents into the live core."""
        self._suppress += 1
        try:
            if self._tabs.currentWidget() is self._group_tab:
                self._group_tab.save()
            elif self._tabs.currentWidget() is self._pixel_config:
                self._pixel_config.apply()
        finally:
            self._suppress -= 1

    def commit_to_core(self) -> None:
        """Write the group and pixel editors' contents into the core."""
        self._suppress += 1
        try:
            self._group_tab.save()
            self._pixel_config.apply()
        finally:
            self._suppress -= 1

    def _on_group_clean_changed(self, clean: bool) -> None:
        """QUndoStack.cleanChanged — the accurate source for group-dirty state.

        Unlike a plain "something changed" signal, this correctly goes clean
        again if the user undoes back to the original state, not just on save.
        """
        self._group_dirty = not clean
        self._update_dirty_label()

    def _mark_group_dirty(self, *_: object) -> None:
        """Force the group tab dirty, bypassing the undo stack.

        Used by tests that want to simulate an edit without driving the real
        editor UI; real edits are tracked via _on_group_clean_changed instead.
        """
        if self._suppress == 0:
            self._group_dirty = True
            self._update_dirty_label()

    def _mark_pixel_dirty(self, *_: object) -> None:
        if self._suppress == 0:
            self._pixel_dirty = True
            self._update_dirty_label()

    def _update_dirty_label(self) -> None:
        parts = []
        if self._group_dirty:
            parts.append("Group Editor")
        if self._pixel_dirty:
            parts.append("Pixel Configuration")
        if parts:
            self._dirty_text.setText(f"Unsaved changes: {', '.join(parts)}")
            self._dirty_label.show()
        else:
            self._dirty_label.hide()

    def _apply_dirty_style(self) -> None:
        """(Re)apply the amber warning icon/color from the *current* theme.

        A theme toggle overwrites every widget's QPalette wholesale (see
        set_theme()'s app.allWidgets() sweep in _gui/_theme/__init__.py),
        which would silently reset this label back to the default
        WindowText color. changeEvent() below re-runs this on every
        StyleChange, which set_zoom() (always called by set_theme()) sends
        to every widget — the same mechanism _toolbar.py/_sidebar.py rely on.
        """
        color = qcolor(theme().status_amber)
        icon = QIconifyIcon("mdi:alert", color=color.name())
        size = self._dirty_text.fontMetrics().height()
        self._dirty_icon.setPixmap(icon.pixmap(size, size))
        pal = self._dirty_text.palette()
        pal.setColor(QPalette.ColorRole.WindowText, color)
        self._dirty_text.setPalette(pal)

    def changeEvent(self, event: QEvent | None) -> None:
        if event is not None and event.type() == QEvent.Type.StyleChange:
            self._apply_dirty_style()
        super().changeEvent(event)

    def _on_system_config_loaded(self) -> None:
        """A whole new configuration was loaded — reload everything."""
        # the widget may already be torn down on the C++ side
        with suppress(RuntimeError):
            self._refresh(reload_configs=True)
            # a freshly loaded config is the new clean baseline
            self.mark_saved()

    def showEvent(self, event: QShowEvent | None) -> None:
        # Editing hardware on another tab loads devices into the core but does
        # not fire systemConfigurationLoaded, so these editors would otherwise
        # be stale. Refresh devices whenever this page is shown — but NOT the
        # config groups, or in-progress group edits would be clobbered on every
        # revisit.
        #
        # Defer to the next event-loop turn: rebuilding the property table
        # synchronously here happens before the tab's geometry has settled on
        # first show, leaving the table sized to a stale (narrow) width until
        # the user switches sub-tabs. Deferring lets the layout settle first.
        super().showEvent(event)
        QTimer.singleShot(0, self._refresh_on_show)

    def _refresh_on_show(self) -> None:
        # the widget may already be torn down on the C++ side
        with suppress(RuntimeError):
            self._refresh(reload_configs=False)

    def refresh(self) -> None:
        """Re-read the editors from the current state of the core."""
        self._refresh(reload_configs=True)

    def _refresh(self, *, reload_configs: bool) -> None:
        # programmatic refresh: don't let the resulting signals set the pixel
        # dirty flag (unless a real config was loaded, which is a genuine
        # change). The group side doesn't need this guard — update_from_core's
        # setData() doesn't touch the undo stack, so cleanChanged only ever
        # fires from a genuine user edit.
        self._suppress += 1
        try:
            with suppress(Exception):
                self._group_editor.update_from_core(
                    self._core, update_configs=reload_configs
                )
            if reload_configs:
                # a full reload replaces the model wholesale — any undo
                # history now refers to a superseded state, and the freshly
                # loaded data is by definition the new clean baseline.
                with suppress(Exception):
                    self._group_editor.undoStack().clear()
            # PixelConfigurationWidget only rebuilds on the core's
            # systemConfigurationLoaded event and exposes no public refresh, so
            # invoke its internal rebuild directly (guarded against renames).
            if callable(
                fn := getattr(self._pixel_config, "_on_sys_config_loaded", None)
            ):
                with suppress(Exception):
                    fn()
                    # The rebuild replaces every row's cell widget (fresh
                    # TableDoubleSpinBox instances) with the upstream default
                    # stylesheet intact -- the one-time sweep in
                    # _EmbeddedPixelConfig.__init__ only ever covered the rows
                    # that existed at construction, so anything rebuilt here
                    # (every time this page is shown, or on a config reload)
                    # would otherwise keep a stale, theme-unaware appearance.
                    unstyle_widgets(self._pixel_config)
        finally:
            self._suppress -= 1
