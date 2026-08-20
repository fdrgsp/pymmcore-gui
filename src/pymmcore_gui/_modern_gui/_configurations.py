"""Configurations tab: property browser, group editor and pixel configuration.

These are the upstream ``pymmcore_widgets`` editors; this module only arranges
them as sub-tabs and keeps them in step with the core.
"""

from __future__ import annotations

from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING

from pymmcore_plus import CMMCorePlus
from pymmcore_widgets import ConfigGroupsEditor
from pymmcore_widgets._icons import StandardIcon
from pymmcore_widgets._util import block_core
from superqt.iconify import QIconifyIcon

from pymmcore_gui._array_viewer import set_source_icon, unstyle_widgets
from pymmcore_gui._qt.QtCore import QEvent, QSize, QTimer, Signal
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
from pymmcore_gui.widgets._pixel_configuration import PixelConfigurationWidget

from ._busy import BusyOverlay, busy
from ._tab_page import TabPage
from ._theme import qcolor, theme

SAVING_MSG = "Saving configuration to core…"

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pymmcore_gui._qt.QtGui import QShowEvent


def _group_matches_core(core: CMMCorePlus, group: object) -> bool:
    """Whether `group` (an editor ConfigGroup) is already what the core holds.

    Compared as ordered (preset, settings) tuples, since MMCore preserves the
    order in which both were defined. Anything the comparison cannot express
    the same way in both — an in-progress preset with no settings, which MMCore
    cannot represent at all, or a value it normalizes on the way in — simply
    reports "different" and gets rewritten, which is the status quo.
    """
    name = getattr(group, "name", None)
    if not name or name not in core.getAvailableConfigGroups():
        return False
    try:
        in_core = [
            (preset, tuple((d, p, v) for d, p, v in core.getConfigData(name, preset)))
            for preset in core.getAvailableConfigs(name)
        ]
        in_editor = [
            (
                preset.name,
                tuple(
                    (s.device.label, s.property_name, s.value) for s in preset.settings
                ),
            )
            for preset in group.presets.values()  # type: ignore[attr-defined]
        ]
    except Exception:
        return False
    return in_core == in_editor


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
        self._apply_themed_action_icons()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.editor, 1)

    def changeEvent(self, a0: QEvent | None) -> None:
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
        if a0 is not None and a0.type() == QEvent.Type.StyleChange:
            for w in (self.editor, *self.editor.findChildren(QWidget)):
                if not isinstance(w, QAbstractSlider):
                    w.setFont(QFont())
            self._apply_themed_action_icons()
        super().changeEvent(a0)

    def _apply_themed_action_icons(self) -> None:
        """Color constructive/destructive Group Editor actions semantically."""
        green = qcolor(theme().status_green).name()
        red = qcolor(theme().status_red).name()
        glyphs = {
            "Add Group": (StandardIcon.FOLDER_ADD, green),
            "Add Preset": (StandardIcon.DOCUMENT_ADD, green),
            "Edit Properties": (StandardIcon.PROPERTY_ADD, green),
            "Duplicate": (StandardIcon.COPY, green),
            "Remove": (StandardIcon.DELETE, red),
        }
        for action in self.editor._tb.actions():
            if themed := glyphs.get(action.text()):
                glyph, color = themed
                action.setIcon(glyph.icon(color))

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
                        if _group_matches_core(self._core, group):
                            # Already exactly what the core holds. Skipping is
                            # not just faster: deleteConfigGroup() below is
                            # destructive (it drops the channel-group
                            # designation, and anything the editor could not
                            # represent), so a save that changed one preset has
                            # no business tearing down every other group.
                            continue
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

    # Redeclare the inherited signal alongside this subclass's own signal.
    # PySide6 otherwise appends the inherited Python signal after inherited
    # slots in _EmbeddedPixelConfig's dynamic QMetaObject, making emit() a
    # no-op (and producing a "Signals and slots ... not ordered" warning).
    calibrationRunningChanged = Signal(bool)
    applied = Signal()

    def __init__(
        self, mmcore: CMMCorePlus | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent, mmcore=mmcore)
        self._suppress_close = False
        self._overlay = BusyOverlay(self)
        for btn in self.findChildren(QPushButton):
            if btn.text() in {"Apply and Close", "Cancel"}:
                # Save actions live in the page toolbar when embedded.
                btn.hide()
        # Upstream's own status icon/label next to those buttons is redundant
        # with the page's toolbar dirty label, which covers both editors.
        for attr in ("_status_icon", "_status_label"):
            if (w := getattr(self, attr, None)) is not None:
                w.hide()
        unstyle_widgets(self)

    def apply(self) -> None:
        """Apply the pixel configurations to the core (without closing)."""
        self._on_apply()
        # Validation failures leave the editor dirty. Only announce a real
        # commit so Stage Explorer is not reset for an apply that never landed.
        if self.isClean():
            self.applied.emit()

    def _on_apply(self) -> None:
        # _on_apply ends by calling self.close(); keep the widget open instead.
        self._suppress_close = True
        try:
            with busy(self._overlay, SAVING_MSG):
                # Suppress core events for the whole bulk rewrite, exactly as
                # _GroupEditorTab.save() does — upstream's _on_apply deletes
                # and redefines every preset, and pymmcore-plus emits
                # pixelSizeChanged from deletePixelSizeConfig(),
                # definePixelSizeConfig() AND setPixelSizeUm(): three events
                # per preset. Every listener (Stage Explorer's AffineState,
                # the camera ROI widget, the MDA grid/position plans) answers
                # each one by calling getPixelSizeUm()/getPixelSizeAffine(),
                # which MMCore resolves by reading the objective's *live*
                # property values. On a serial device (an ASI Tiger, say)
                # every one of those reads is a round trip over the COM port,
                # so the cost is presets x listeners round trips — long enough
                # for the OS to mark the window unresponsive. And each answer
                # is computed from a half-rewritten config anyway, so none of
                # them is worth having.
                with block_core(self._mmc.events):
                    super()._on_apply()
                # Those listeners are now stale. Give them the single event
                # that matters, with the final configuration in place — still
                # under the overlay, since this read can be slow too.
                if self.isClean():
                    self._mmc.events.pixelSizeChanged.emit(self._mmc.getPixelSizeUm())
        finally:
            self._suppress_close = False

    def close(self) -> bool:
        if self._suppress_close:
            return False
        return super().close()


class ConfigurationsPage(TabPage):
    """Sub-tabbed editors for config groups and pixel sizes."""

    # "Save to core" commits to the live core only; this asks the window to
    # write the whole configuration (hardware + groups + pixel) to a .cfg.
    saveToFileRequested = Signal()
    calibrationRunningChanged = Signal(bool)
    pixelConfigurationsApplied = Signal()

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
        # these editors fill the page; the left dock would only crowd them
        self.left.hide()

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
        self._apply_save_button_icons()
        self._pixel_config.calibrationRunningChanged.connect(
            self._on_pixel_calibration_running
        )
        self._pixel_config.applied.connect(self.pixelConfigurationsApplied)

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
        # Both editors expose the same clean-state interface — the group
        # editor backs it with a QUndoStack, Pixel Configuration with a
        # baseline snapshot — so each reports dirtiness accurately, including
        # going clean again when an edit is reverted rather than saved. Neither
        # marks itself clean on *our* save (the group editor's was the hidden
        # "Apply" button's job), so `mark_saved`/`mark_current_saved` do that.
        self._group_dirty = False
        self._pixel_dirty = False
        # guards against a second commit starting while one is in flight
        self._commit_in_progress = False
        self._group_editor.undoStack().cleanChanged.connect(
            self._on_group_clean_changed
        )
        self._pixel_config.cleanChanged.connect(self._on_pixel_clean_changed)

        # a configuration may be loaded after this page is built
        self._core.events.systemConfigurationLoaded.connect(
            self._on_system_config_loaded
        )

    # ── unsaved-changes API ───────────────────────────────────────

    def is_dirty(self) -> bool:
        """Whether either editor has edits not yet persisted to a file."""
        return self._group_dirty or self._pixel_dirty

    def dirty_parts(self) -> list[str]:
        """Names of the editors that currently have unsaved edits."""
        parts = []
        if self._group_dirty:
            parts.append("Group Editor")
        if self._pixel_dirty:
            parts.append("Pixel Configuration")
        return parts

    def mark_saved(self) -> None:
        """Mark both editors as persisted to the configuration file."""
        with suppress(Exception):
            self._group_editor.undoStack().setClean()
        with suppress(Exception):
            self._pixel_config.setClean()
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
            with suppress(Exception):
                self._pixel_config.setClean()
            self._pixel_dirty = False
        self._update_dirty_label()

    def discard_changes(self) -> None:
        """Replace unsaved group and pixel edits with the live core state."""
        self._refresh(reload_configs=True)
        self.mark_saved()

    @contextmanager
    def _committing(self) -> Iterator[None]:
        """Hold the save actions shut for the duration of a commit.

        A commit blocks the GUI thread, so on real hardware the user has
        seconds in which to click "Save" again. Disabling the buttons both
        says so and makes Qt drop clicks aimed at them, while `_committing`
        stops any *other* route (the leave-the-page prompt, closeEvent)
        starting a second rewrite of a core the first is still halfway
        through. Prior enabled state is restored rather than assumed, since
        pixel calibration disables these buttons for its own reasons.
        """
        buttons = (self._save_core_btn, self._save_file_btn)
        was_enabled = [btn.isEnabled() for btn in buttons]
        for btn in buttons:
            btn.setEnabled(False)
        self._commit_in_progress = True
        try:
            yield
        finally:
            self._commit_in_progress = False
            for btn, enabled in zip(buttons, was_enabled, strict=False):
                btn.setEnabled(enabled)

    def commit_current_to_core(self) -> None:
        """Write the selected editor's contents into the live core."""
        if self._commit_in_progress:
            return
        with self._committing():
            if self._tabs.currentWidget() is self._group_tab:
                self._group_tab.save()
            elif self._tabs.currentWidget() is self._pixel_config:
                self._pixel_config.apply()

    def commit_to_core(self) -> None:
        """Write the group and pixel editors' contents into the core."""
        if self._commit_in_progress:
            return
        with self._committing():
            self._group_tab.save()
            self._pixel_config.apply()

    def _on_pixel_calibration_running(self, running: bool) -> None:
        """Prevent configuration rewrites or tab changes during stage motion."""
        self._save_core_btn.setEnabled(not running)
        self._save_file_btn.setEnabled(not running)
        self._tabs.setTabEnabled(self._tabs.indexOf(self._group_tab), not running)
        self.calibrationRunningChanged.emit(running)

    def shutdownCalibration(self) -> None:
        """Cancel calibration and wait for stage/capture-state restoration."""
        self._pixel_config.shutdownCalibration()

    def _on_group_clean_changed(self, clean: bool) -> None:
        """QUndoStack.cleanChanged — the accurate source for group-dirty state.

        Unlike a plain "something changed" signal, this correctly goes clean
        again if the user undoes back to the original state, not just on save.
        """
        self._group_dirty = not clean
        self._update_dirty_label()

    def _on_pixel_clean_changed(self, clean: bool) -> None:
        """PixelConfigurationWidget.cleanChanged — same contract as the group tab.

        Backed by a baseline snapshot rather than an undo stack, so it too goes
        clean again when an edit is reverted by hand, and merely selecting a
        different resolution row never reports a change.
        """
        self._pixel_dirty = not clean
        self._update_dirty_label()

    def _mark_group_dirty(self, *_: object) -> None:
        """Force the group tab dirty, bypassing the undo stack.

        Used by tests that want to simulate an edit without driving the real
        editor UI; real edits are tracked via _on_group_clean_changed instead.
        """
        self._group_dirty = True
        self._update_dirty_label()

    def _mark_pixel_dirty(self, *_: object) -> None:
        """Force the pixel tab dirty; test-only counterpart of the above."""
        self._pixel_dirty = True
        self._update_dirty_label()

    def _update_dirty_label(self) -> None:
        parts = self.dirty_parts()
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

    def _apply_save_button_icons(self) -> None:
        color = qcolor(theme().text_secondary).name()
        size = theme().scaled(16)
        for btn, icon in (
            (self._save_core_btn, "material-symbols:upload-file-outline-rounded"),
            (self._save_file_btn, "material-symbols:file-save-outline-rounded"),
        ):
            set_source_icon(btn, QIconifyIcon(icon, color=color))
            btn.setIconSize(QSize(size, size))

    def changeEvent(self, a0: QEvent | None) -> None:
        if a0 is not None and a0.type() == QEvent.Type.StyleChange:
            self._apply_dirty_style()
            self._apply_save_button_icons()
        super().changeEvent(a0)

    def _on_system_config_loaded(self) -> None:
        """A whole new configuration was loaded — reload everything."""
        # the widget may already be torn down on the C++ side
        with suppress(RuntimeError):
            self._refresh(reload_configs=True)
            # a freshly loaded config is the new clean baseline
            self.mark_saved()

    def showEvent(self, a0: QShowEvent | None) -> None:
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
        super().showEvent(a0)
        QTimer.singleShot(0, self._refresh_on_show)

    def _refresh_on_show(self) -> None:
        # the widget may already be torn down on the C++ side
        with suppress(RuntimeError):
            self._refresh(reload_configs=False)

    def refresh(self) -> None:
        """Re-read the editors from the current state of the core."""
        self._refresh(reload_configs=True)

    def _refresh(self, *, reload_configs: bool) -> None:
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
        # That rebuild reads the core wholesale, so skip it while the user has
        # unsaved pixel edits — for the same reason the group editor's configs
        # aren't reloaded on a plain revisit (see showEvent). A real config
        # load (reload_configs) supersedes those edits and does rebuild.
        if reload_configs or self._pixel_config.isClean():
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
