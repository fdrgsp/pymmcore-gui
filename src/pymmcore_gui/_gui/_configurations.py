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
    PropertyBrowser,
)

from pymmcore_gui._qt.QtCore import QTimer, pyqtSignal
from pymmcore_gui._qt.QtWidgets import (
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ._tab_page import TabPage
from ._theme import theme

if TYPE_CHECKING:
    from pymmcore_gui._qt.QtGui import QShowEvent


class _GroupEditorTab(QWidget):
    """A ConfigGroupsEditor with a Save button that writes back to the core."""

    def __init__(self, core: CMMCorePlus, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._core = core
        self.editor = ConfigGroupsEditor.create_from_core(core)

        self._save_btn = QPushButton("Save to core")
        self._save_btn.setProperty("variant", "primary")
        self._save_btn.clicked.connect(self.save)

        m = theme().sp_md
        row = QHBoxLayout()
        row.setContentsMargins(m, m, m, m)
        row.addStretch()
        row.addWidget(self._save_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.editor, 1)
        layout.addLayout(row)

    def save(self) -> None:
        """Replace the core's config groups with the editor's contents."""
        groups = list(self.editor.data())
        desired = {g.name for g in groups}
        try:
            # drop groups the editor no longer has
            for name in list(self._core.getAvailableConfigGroups()):
                if name not in desired:
                    self._core.deleteConfigGroup(name)
            # redefine each edited group (delete first to clear stale presets)
            for group in groups:
                if group.name in self._core.getAvailableConfigGroups():
                    self._core.deleteConfigGroup(group.name)
                self._core.defineConfigGroup(group.name)
                for preset_name, preset in group.presets.items():
                    for s in preset.settings:
                        self._core.defineConfig(
                            group.name,
                            preset_name,
                            # the device LABEL, not s.device.name (adapter name)
                            s.device.label,
                            s.property_name,
                            s.value,
                        )
        except Exception as e:
            QMessageBox.warning(
                self, "Save configuration groups", f"Failed to save:\n\n{e}"
            )
            return
        self._core.events.systemConfigurationLoaded.emit()


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
        for btn in self.findChildren(QPushButton):
            if btn.text() == "Apply and Close":
                # this already writes the pixel configs to the core; match the
                # group editor's button in both label and style
                btn.setText("Save to core")
                btn.setProperty("variant", "primary")
            elif btn.text() == "Cancel":
                btn.hide()
        # bridge the widget's internal edit signals to a public `changed` one
        for attr in ("_px_table", "_affine_table", "_props_selector"):
            owner = getattr(self, attr, None)
            if (sig := getattr(owner, "valueChanged", None)) is not None:
                with suppress(Exception):
                    sig.connect(self.changed)

    def apply(self) -> None:
        """Apply the pixel configurations to the core (without closing)."""
        self._on_apply()

    def _on_apply(self) -> None:
        # _on_apply ends by calling self.close(); keep the widget open instead.
        self._suppress_close = True
        try:
            super()._on_apply()
        finally:
            self._suppress_close = False

    def close(self) -> bool:
        if self._suppress_close:
            return False
        return super().close()


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
        self._group_tab = _GroupEditorTab(self._core)
        self._group_editor = self._group_tab.editor
        self._pixel_config = _EmbeddedPixelConfig(mmcore=self._core)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._property_browser, "Property Browser")
        self._tabs.addTab(self._group_tab, "Group Editor")
        self._tabs.addTab(self._pixel_config, "Pixel Configuration")

        self.add_content_widget(self._tabs)
        # these editors fill the page; the docks would only crowd them
        self.left.hide()
        self.right.hide()
        self.bottom.hide()
        self.toolbar.hide()

        # unsaved-changes tracking. `_suppress` (a depth counter, so nested
        # refresh/commit stay guarded) keeps programmatic refreshes from being
        # mistaken for user edits.
        self._dirty = False
        self._suppress = 0
        self._group_editor.configChanged.connect(self._mark_dirty)
        self._pixel_config.changed.connect(self._mark_dirty)

        # a configuration may be loaded after this page is built
        self._core.events.systemConfigurationLoaded.connect(
            self._on_system_config_loaded
        )

    # ── unsaved-changes API ───────────────────────────────────────

    def is_dirty(self) -> bool:
        """Whether the group or pixel editors have uncommitted edits."""
        return self._dirty

    def mark_saved(self) -> None:
        self._dirty = False

    def commit_to_core(self) -> None:
        """Write the group and pixel editors' contents into the core."""
        self._suppress += 1
        try:
            self._group_tab.save()
            self._pixel_config.apply()
        finally:
            self._suppress -= 1
        self._dirty = False

    def _mark_dirty(self, *_: object) -> None:
        if self._suppress == 0:
            self._dirty = True

    def _on_system_config_loaded(self) -> None:
        """A whole new configuration was loaded — reload everything."""
        # the widget may already be torn down on the C++ side
        with suppress(RuntimeError):
            self._refresh(reload_configs=True)
            # a freshly loaded config is the new clean baseline
            self._dirty = False

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
        # programmatic refresh: don't let the resulting signals set the dirty
        # flag (unless a real config was loaded, which is a genuine change)
        self._suppress += 1
        try:
            with suppress(Exception):
                self._group_editor.update_from_core(
                    self._core, update_configs=reload_configs
                )
            # PropertyBrowser and PixelConfigurationWidget only rebuild on the
            # core's systemConfigurationLoaded event and expose no public
            # refresh, so invoke their internal rebuild directly (guarded
            # against upstream renames).
            for target, method in (
                (self._property_browser._prop_table, "_rebuild_table"),
                (self._pixel_config, "_on_sys_config_loaded"),
            ):
                if callable(fn := getattr(target, method, None)):
                    with suppress(Exception):
                        fn()
        finally:
            self._suppress -= 1
