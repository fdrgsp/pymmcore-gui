"""Application-themed Stage Explorer with MDA position transfer."""

from __future__ import annotations

import math
from contextlib import suppress
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
from pymmcore_gui._qt.QtWidgets import QMessageBox, QToolButton

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

    def _fov_w_h(self) -> tuple[float, float]:
        """Return camera-axis FOV dimensions from the active affine transform."""
        width = self._mmc.getImageWidth()
        height = self._mmc.getImageHeight()
        matrix = self._affine_state.system_affine
        # Each affine column is the stage-space vector for one camera pixel.
        # Its norm remains correct for rotated, mirrored, and sheared
        # calibrations, unlike getPixelSizeUm() alone.
        pixel_width = math.hypot(float(matrix[0, 0]), float(matrix[1, 0]))
        pixel_height = math.hypot(float(matrix[0, 1]), float(matrix[1, 1]))
        return width * pixel_width, height * pixel_height

    def _on_roi_changed(self) -> None:
        """Refresh ROI tiling with affine-aware FOV dimensions."""
        super()._on_roi_changed()
        if self._mmc.getImageWidth() and self._mmc.getImageHeight():
            self.roi_manager.update_fovs(self._fov_w_h())

    def refreshPixelGeometry(self) -> None:
        """Recompute every Stage Explorer visual derived from pixel calibration."""
        self._affine_state.refresh()
        self._on_roi_changed()

        # Updating the cached affine is not enough: the marker retains the old
        # Vispy transform until the stage poller next reports movement. Apply
        # the new calibration immediately at the current stage position.
        if self._stage_pos_marker is not None:
            stage_x = stage_y = 0.0
            if self._mmc.getXYStageDevice():
                with suppress(Exception):
                    stage_x, stage_y = self._mmc.getXYPosition()
            matrix = self._affine_state.system_affine_translated(stage_x, stage_y)
            self._stage_pos_marker.apply_transform(matrix.T)

        # ROI data-change signals redraw their FOV divisions; explicitly ask
        # the canvas for a frame as well so an idle Explorer updates at once.
        with suppress(Exception):
            self._stage_viewer.canvas.update()
        if self._auto_zoom_to_fit:
            self.zoom_to_fit()

    def _on_pixel_size_changed(self, value: float) -> None:
        del value
        self.refreshPixelGeometry()

    def _on_pixel_size_affine_changed(self) -> None:
        self.refreshPixelGeometry()

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
        # Match the rest of the app's action buttons (Snap/Live/etc, see
        # _acquire_toolbar.py's _icon_size()) rather than the native QStyle's
        # PM_ToolBarIconSize, which renders noticeably larger (30px vs 20px).
        # Scaled with the theme -- this is re-applied on every StyleChange
        # (below), which is also when the app's zoom pass would otherwise
        # reset every QToolBar's icon size back to PM_ToolBarIconSize.
        icon_size = theme().scaled(20)
        toolbar.setIconSize(QSize(icon_size, icon_size))

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
        foreground = qcolor(theme().text_primary).name()
        marker_icon = QIconifyIcon("mdi:map-marker-outline", color=foreground)
        marker_action = self.toolBar().poll_stage_action
        marker_action.setIcon(marker_icon)
        marker_button = self.toolBar().widgetForAction(marker_action)
        if isinstance(marker_button, QToolButton):
            set_source_icon(marker_button, marker_icon)

        marker_mode_icons = {
            "FOV Rectangle": "ic:outline-check-box-outline-blank",
            "FOV Center": "ic:baseline-plus",
            "Both": "ic:outline-add-box",
        }
        for action in self.toolBar().marker_mode_action_group.actions():
            if glyph := marker_mode_icons.get(action.text()):
                action.setIcon(QIconifyIcon(glyph, color=foreground))

        green = qcolor(theme().status_green).name()
        icon = QIconifyIcon("mdi:send", color=green)
        self._send_to_mda_action.setIcon(icon)
        button = self.toolBar().widgetForAction(self._send_to_mda_action)
        if isinstance(button, QToolButton):
            set_source_icon(button, icon)

        red = qcolor(theme().status_red).name()
        stop_action = self.toolBar().stop_scan_action
        stop_icon = QIconifyIcon("bi:sign-stop", color=red)
        stop_action.setIcon(stop_icon)
        stop_button = self.toolBar().widgetForAction(stop_action)
        if isinstance(stop_button, QToolButton):
            set_source_icon(stop_button, stop_icon)

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
