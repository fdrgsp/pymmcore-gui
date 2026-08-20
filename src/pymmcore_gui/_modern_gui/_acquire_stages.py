"""The "Stages" panel: an arbitrary number of freely-arranged stage controls.

Stages are added on demand through the panel's own "+ Add Stage" button
rather than being pre-populated, and once added live in a small nested QtAds
dock manager -- exactly like ``AcquireViewersManager``'s viewer area -- so
the user can freely tab, split, or drag them into rows, columns, or any mix
of the two, matching the rest of the app's docking model instead of a fixed
grid.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from pymmcore_plus import CMMCorePlus, DeviceType
from superqt.iconify import QIconifyIcon

from pymmcore_gui._array_viewer import set_source_icon, unstyle_widgets
from pymmcore_gui._qt.QtAds import CDockManager, CDockWidget, DockWidgetArea
from pymmcore_gui._qt.QtCore import QEvent, QSize, Signal
from pymmcore_gui._qt.QtGui import QAction
from pymmcore_gui._qt.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ._panels import create_stage_widget
from ._theme import dock_chrome_stylesheet, qcolor, theme

if TYPE_CHECKING:
    from collections.abc import Iterable

# Both stage flavors share one panel -- a device name is unique across the
# whole loaded device list regardless of type, so a flat dict keyed by name
# is enough to track every open dock.
_STAGE_DEVICE_TYPES = (DeviceType.XYStage, DeviceType.Stage)


class StagesPanel(QWidget):
    """Add-on-demand stage controls, freely arranged in their own dock area."""

    # Emitted each time a new per-device StageWidget is created, so a parent
    # (AcquirePage) can wire up cross-panel behavior -- e.g. ensuring the
    # lazy snap Preview exists before that stage's own "Snap" checkbox can
    # trigger a snap-on-move -- without this panel needing to know about the
    # viewer manager itself.
    stageWidgetAdded = Signal(QWidget)

    def __init__(
        self, *, parent: QWidget | None = None, mmcore: CMMCorePlus | None = None
    ) -> None:
        super().__init__(parent)
        self._core = mmcore or CMMCorePlus.instance()
        self._docks: dict[str, CDockWidget] = {}

        self._dock_manager = CDockManager(self)
        # Captured before any custom stylesheet is applied -- see
        # AcquirePage._apply_dock_style, which this mirrors for this nested
        # manager instead of ADS's unthemed default.
        self._base_style = self._dock_manager.styleSheet()
        self._apply_dock_style()

        self._add_btn = QPushButton(self)
        self._add_btn.setProperty("variant", "subtle")
        self._add_btn.setText(" Add Stage")
        self._add_btn.clicked.connect(self._popup_add_menu)

        self._empty_hint = QLabel("No stages added yet.", self)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self._add_btn)
        header.addWidget(self._empty_hint)
        header.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addLayout(header)
        layout.addWidget(self._dock_manager)

        self._apply_add_icon()

    # ── add / remove ─────────────────────────────────────────────

    def _loaded_devices(self) -> list[str]:
        """Every loaded XY/Z stage device, open or not."""
        devices: list[str] = []
        for device_type in _STAGE_DEVICE_TYPES:
            devices.extend(self._core.getLoadedDevicesOfType(device_type))
        return devices

    def _available_devices(self) -> list[str]:
        """Loaded XY/Z stage devices that don't already have an open dock."""
        return [d for d in self._loaded_devices() if d not in self._docks]

    def open_devices(self) -> frozenset[str]:
        """Every device currently shown as its own dock -- see :meth:`add_stages`."""
        return frozenset(self._docks)

    def add_stages(self, devices: Iterable[str]) -> None:
        """Open each of *devices* that is currently loaded; silently skip the rest.

        Used to restore a saved layout's open stages (see
        ``AcquirePage.restore_layout``) without erroring when the
        now-loaded configuration doesn't have one of them -- e.g. switching
        to a config with fewer stages than the one the layout was saved
        under.
        """
        loaded = set(self._loaded_devices())
        for device in devices:
            if device in loaded:
                self._add_stage(device)

    def _build_add_menu(self) -> QMenu:
        """The "+ Add Stage" menu.

        Split out so tests can inspect it without calling ``exec`` (which
        would block on a real click).
        """
        menu = QMenu(self)
        devices = self._available_devices()
        if not devices:
            action = QAction("No stages available", menu)
            action.setEnabled(False)
            menu.addAction(action)
        for device in devices:
            action = QAction(device, menu)
            action.triggered.connect(partial(self._add_stage, device))
            menu.addAction(action)
        return menu

    def _popup_add_menu(self) -> None:
        menu = self._build_add_menu()
        menu.exec(self._add_btn.mapToGlobal(self._add_btn.rect().bottomLeft()))

    def _add_stage(self, device: str) -> None:
        if device in self._docks:
            return
        widget = create_stage_widget(self, self._core, device)
        unstyle_widgets(widget)

        dock = CDockWidget(self._dock_manager, device, self)
        # This panel lives inside AcquirePage, itself inside MainWindow's
        # QStackedWidget -- a floating dock would be a top-level window that
        # lingers after switching to another mode tab, same reason every
        # other panel dock disallows it (see AcquirePage._add_dock).
        dock.setFeature(CDockWidget.DockWidgetFeature.DockWidgetFloatable, False)
        # Closing a stage (its own tab's X) removes it outright rather than
        # leaving a hidden husk -- re-adding it is one click away in the Add
        # menu, and there's no other way to reveal a merely-hidden one.
        dock.setFeature(CDockWidget.DockWidgetFeature.DockWidgetDeleteOnClose, True)
        dock.setWidget(widget, CDockWidget.eInsertMode.ForceNoScrollArea)
        # No target area: each new stage splits off a fresh column to the
        # right of whatever's already there, rather than tabbing into it --
        # the user can still drag any of them into a tab group or a
        # different row/column afterwards, same as any other ADS dock.
        self._dock_manager.addDockWidget(DockWidgetArea.RightDockWidgetArea, dock)
        dock.closed.connect(partial(self._on_dock_closed, device))

        self._docks[device] = dock
        dock.setAsCurrentTab()
        self._empty_hint.setVisible(False)
        self.stageWidgetAdded.emit(widget)

    def _on_dock_closed(self, device: str) -> None:
        self._docks.pop(device, None)
        self._empty_hint.setVisible(not self._docks)

    # ── theming ──────────────────────────────────────────────────

    def _apply_add_icon(self) -> None:
        color = qcolor(theme().text_secondary).name()
        set_source_icon(self._add_btn, QIconifyIcon("mdi:plus", color=color))
        size = theme().scaled(20)
        self._add_btn.setIconSize(QSize(size, size))

    def _apply_dock_style(self) -> None:
        self._dock_manager.setStyleSheet(self._base_style + dock_chrome_stylesheet())

    def changeEvent(self, a0: QEvent | None) -> None:
        if a0 is not None and a0.type() == QEvent.Type.StyleChange:
            self._apply_dock_style()
            self._apply_add_icon()
        super().changeEvent(a0)
