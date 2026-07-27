"""Application-themed Stage Explorer with MDA position transfer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_widgets import StageExplorer
from superqt.iconify import QIconifyIcon

from pymmcore_gui._array_viewer import (
    ensure_visible_icon,
    set_source_icon,
    unstyle_widgets,
)
from pymmcore_gui._modern_gui._theme import qcolor, theme
from pymmcore_gui._qt.QtCore import QEvent, QSize, Signal
from pymmcore_gui._qt.QtWidgets import QMessageBox, QStyle, QToolButton

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

    from pymmcore_gui._qt.QtWidgets import QWidget


class ThemedStageExplorer(StageExplorer):
    """Stage Explorer adapted to this app's style and MDA editor."""

    sendToMDARequested = Signal(list, bool)

    def __init__(
        self,
        parent: QWidget | None = None,
        mmcore: CMMCorePlus | None = None,
    ) -> None:
        super().__init__(parent=parent, mmcore=mmcore)

        toolbar = self.toolBar()
        # Compatibility for releases predating the cite-branch port. Once the
        # updated dependency is installed, StageExplorer supplies both this
        # action and the sendToMDARequested signal itself.
        action = getattr(toolbar, "send_to_mda_action", None)
        if action is None:
            action = toolbar.addAction(QIconifyIcon("mdi:send"), "Send to MDA")
            action.triggered.connect(self._on_send_to_mda_fallback)
        self._send_to_mda_action = action
        self._send_to_mda_action.setToolTip(
            "Add the Explorer regions to the MDA stage-position plan"
        )

        self._normalize_style()

    def _normalize_style(self) -> None:
        """Remove upstream one-off styling and use the application's QStyle."""
        # Unlike ndv's functional contrast-limit stylesheet, StageExplorer's
        # slider stylesheet hardcodes its own colors, handle, label font and
        # geometry. The application style supplies all of those consistently.
        slider = getattr(getattr(self, "_contrast_slider", None), "_slider", None)
        if slider is not None:
            slider.setStyleSheet("")

        unstyle_widgets(self)

        toolbar = self.toolBar()
        toolbar.setMovable(False)
        toolbar.setContentsMargins(0, 0, theme().sp_xs, 0)
        if style := self.style():
            metric = style.pixelMetric(QStyle.PixelMetric.PM_ToolBarIconSize)
            toolbar.setIconSize(QSize(metric, metric))

        # QToolBar normally makes its buttons auto-raise (ghost style). The
        # rest of this app's action buttons use the persistent subtle frame.
        for action in toolbar.actions():
            button = toolbar.widgetForAction(action)
            if isinstance(button, QToolButton):
                button.setAutoRaise(False)
                button.setProperty("variant", "subtle")
                ensure_visible_icon(button)

        self._apply_themed_icons()

    def _apply_themed_icons(self) -> None:
        green = qcolor(theme().status_green).name()
        icon = QIconifyIcon("mdi:send", color=green)
        self._send_to_mda_action.setIcon(icon)
        button = self.toolBar().widgetForAction(self._send_to_mda_action)
        if isinstance(button, QToolButton):
            set_source_icon(button, icon)

    def _on_send_to_mda_fallback(self) -> None:
        """Provide the cite-branch behavior for older installed releases."""
        fov_w, fov_h = self._fov_w_h()
        z_pos = self._mmc.getZPosition() if self._mmc.getFocusDevice() else None
        manager = self.roi_manager
        positions = [
            roi.create_useq_position(
                fov_w,
                fov_h,
                z_pos=z_pos,
                overlap=manager.scan_overlap,
                mode=manager.scan_mode,
            )
            for roi in manager.all_rois()
        ]
        if positions and (replace := self._choose_mda_update()) is not None:
            self.sendToMDARequested.emit(positions, replace)

    def _choose_mda_update(self) -> bool | None:
        """Return True for Replace, False for Add, and None for Cancel."""
        msg = QMessageBox(self)
        msg.setWindowTitle("Send to MDA")
        msg.setText("Replace existing stage positions or add to them?")
        replace_btn = msg.addButton("Replace", QMessageBox.ButtonRole.AcceptRole)
        add_btn = msg.addButton("Add", QMessageBox.ButtonRole.AcceptRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked is replace_btn:
            return True
        if clicked is add_btn:
            return False
        if clicked is cancel_btn or clicked is None:
            return None
        return None  # pragma: no cover

    def changeEvent(self, a0: QEvent | None) -> None:
        super().changeEvent(a0)
        if (
            a0 is not None
            and a0.type() == QEvent.Type.StyleChange
            and hasattr(self, "_send_to_mda_action")
        ):
            self._normalize_style()
