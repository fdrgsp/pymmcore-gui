"""Application-themed Stage Explorer with MDA position transfer."""

from __future__ import annotations

import math
from contextlib import suppress
from typing import TYPE_CHECKING, ClassVar

from pymmcore_widgets import StageExplorer
from superqt.iconify import QIconifyIcon

from pymmcore_gui._array_viewer import (
    ensure_visible_icon,
    set_icon_tint,
    set_source_icon,
    unstyle_widgets,
)
from pymmcore_gui._modern_gui._theme import qcolor, theme
from pymmcore_gui._qt.QtCore import QEvent, QSize, QTimer, Signal
from pymmcore_gui._qt.QtWidgets import QMessageBox, QToolButton

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from vispy.app.canvas import MouseEvent

    from pymmcore_gui._qt.QtGui import QAction
    from pymmcore_gui._qt.QtWidgets import QWidget


class ThemedStageExplorer(StageExplorer):
    """Stage Explorer adapted to this app's style and MDA editor."""

    sendToMDARequested = Signal(list, bool)

    MDA_ALLOWED_ACTIONS: ClassVar[tuple[str, ...]] = (
        "zoom_to_fit_action",
        "auto_zoom_to_fit_action",
        "show_grid_action",
        "map_memory_action",
    )
    """Toolbar actions that stay usable while an acquisition is running.

    Everything else on the toolbar either drives the hardware (snap, stage
    polling, ROI scanning) or edits state a running acquisition owns (Clear
    View, Delete ROIs, Send to MDA), so :meth:`setMdaLocked` shuts it off.
    What is left is pure navigation of what has already been mapped -- the
    reason the panel stays reachable at all during a run.
    """

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

        self._mda_locked = False
        # Enabled state each locked action had before the lock, so releasing
        # it can't hand back an action that was already unavailable for its
        # own reasons (e.g. ``_update_actions_enabled`` with no devices).
        self._pre_lock_action_states: dict[QAction, bool] = {}

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

    def setMdaLocked(self, locked: bool) -> None:
        """Restrict the Explorer to viewing while an acquisition is running.

        The panel is deliberately not disabled outright: watching the map fill
        in -- and zooming/panning around it -- is the one thing this widget is
        useful for mid-run. Only the actions that would touch the microscope
        or the acquisition are taken away (see :attr:`MDA_ALLOWED_ACTIONS`),
        along with double-click-to-move-stage.
        """
        if locked == self._mda_locked:
            return
        self._mda_locked = locked
        if locked:
            self._pre_lock_action_states = {}
            for action in self._locked_actions():
                self._pre_lock_action_states[action] = action.isEnabled()
                action.setEnabled(False)
        else:
            for action, enabled in self._pre_lock_action_states.items():
                action.setEnabled(enabled)
            self._pre_lock_action_states = {}

    def _locked_actions(self) -> list[QAction]:
        """Return every action that an acquisition takes away."""
        toolbar = self.toolBar()
        allowed = {
            action
            for name in self.MDA_ALLOWED_ACTIONS
            if (action := getattr(toolbar, name, None)) is not None
        }
        # The ROI drawing modes are already among the toolbar's actions;
        # ``mode_actions`` is included anyway so they stay covered if upstream
        # ever moves them into a popup menu the way marker mode is. Marker
        # mode is left alone: it only changes what is drawn on the map.
        candidates = [*toolbar.actions(), *self.roi_manager.mode_actions.actions()]
        return [a for a in dict.fromkeys(candidates) if a not in allowed]

    def _update_actions_enabled(self) -> None:
        """Keep an acquisition's restrictions in place across upstream refreshes.

        ``StageExplorer`` re-derives every action's enabled state from the
        loaded devices (on construction and on ``systemConfigurationLoaded``),
        which would otherwise hand back actions the lock had taken away.
        """
        super()._update_actions_enabled()
        if getattr(self, "_mda_locked", False):
            for action in self._locked_actions():
                self._pre_lock_action_states[action] = action.isEnabled()
                action.setEnabled(False)

    def _on_mouse_double_click(self, event: MouseEvent) -> None:
        """Ignore double-click-to-move-stage while an acquisition is running."""
        if self._mda_locked:
            return
        super()._on_mouse_double_click(event)

    def _normalize_style(self) -> None:
        """Remove upstream one-off styling and use the application's QStyle."""
        slider = getattr(getattr(self, "_contrast_slider", None), "_slider", None)
        if slider is not None and not slider.styleSheet():
            # Compatibility for pymmcore-widgets releases from before the
            # Stage Explorer adopted ndv's functional contrast-slider style.
            from ndv.views._qt._array_view import SLIDER_STYLE

            slider.setStyleSheet(
                SLIDER_STYLE + "SliderLabel { font-size: 10px; color: white;}"
            )

        unstyle_widgets(self)
        QTimer.singleShot(0, self._reposition_contrast_labels)

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

    def _reposition_contrast_labels(self) -> None:
        slider = getattr(getattr(self, "_contrast_slider", None), "_slider", None)
        reposition = getattr(slider, "_reposition_labels", None)
        if slider is not None and callable(reposition):
            if layout := slider.layout():
                layout.activate()
            reposition()

    def _apply_themed_icons(self) -> None:
        """Recolor every neutral toolbar icon to match the app's chrome.

        Upstream bakes these in as a fixed ``#666`` gray (see
        ``pymmcore_widgets.control._rois.roi_manager.GRAY``), which has no
        light/dark theme awareness and doesn't match this app's other
        toolbar icons (e.g. the gear button in ``_preferences.py``, which
        uses ``theme().text_secondary``). Re-deriving every neutral icon
        from that same token keeps this toolbar consistent with the rest of
        the app in both themes; only the semantic (green/red) actions below
        are deliberately left off-token.
        """
        toolbar = self.toolBar()
        muted = qcolor(theme().text_secondary).name()

        def _recolor(action: QAction, glyph: str) -> None:
            icon = QIconifyIcon(glyph, color=muted)
            action.setIcon(icon)
            button = toolbar.widgetForAction(action)
            if isinstance(button, QToolButton):
                set_source_icon(button, icon)

        _recolor(toolbar.clear_action, "mdi:close")
        _recolor(toolbar.zoom_to_fit_action, "mdi:fullscreen")
        _recolor(toolbar.snap_action, "mdi:camera-outline")
        _recolor(toolbar.poll_stage_action, "mdi:map-marker-outline")
        _recolor(toolbar.show_grid_action, "mdi:grid")
        _recolor(toolbar.map_memory_action, "mdi:memory")
        _recolor(toolbar.delete_rois_action, "mdi:vector-square-remove")

        # Auto Zoom to Fit's icon is a static SVG with a baked-in fill, not a
        # QIconifyIcon glyph, so it's tinted after the fact instead.
        auto_zoom_button = toolbar.widgetForAction(toolbar.auto_zoom_to_fit_action)
        if isinstance(auto_zoom_button, QToolButton):
            set_icon_tint(auto_zoom_button, qcolor(theme().text_secondary))

        marker_mode_icons = {
            "FOV Rectangle": "ic:outline-check-box-outline-blank",
            "FOV Center": "ic:baseline-plus",
            "Both": "ic:outline-add-box",
        }
        for action in toolbar.marker_mode_action_group.actions():
            if glyph := marker_mode_icons.get(action.text()):
                action.setIcon(QIconifyIcon(glyph, color=muted))

        roi_mode_icons = {
            "Select": "mdi:cursor-default-outline",
            "Rectangle": "mdi:vector-square",
            "Polygon": "mdi:vector-polygon",
        }
        for action in self.roi_manager.mode_actions.actions():
            if glyph := roi_mode_icons.get(action.text()):
                action.setIcon(QIconifyIcon(glyph, color=muted))

        green = qcolor(theme().status_green).name()
        icon = QIconifyIcon("mdi:send", color=green)
        self._send_to_mda_action.setIcon(icon)
        button = self.toolBar().widgetForAction(self._send_to_mda_action)
        if isinstance(button, QToolButton):
            set_source_icon(button, icon)

        scan_action = self.toolBar().scan_action
        scan_icon = QIconifyIcon("ph:path-duotone", color=green)
        scan_action.setIcon(scan_icon)
        scan_button = self.toolBar().widgetForAction(scan_action)
        if isinstance(scan_button, QToolButton):
            set_source_icon(scan_button, scan_icon)

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
