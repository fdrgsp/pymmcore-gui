"""Application-themed Stage Explorer with MDA position transfer."""

from __future__ import annotations

import math
from contextlib import suppress
from typing import TYPE_CHECKING, Any, ClassVar, cast

from pymmcore_widgets import StageExplorer
from pymmcore_widgets._util import _CAMERA_GEOMETRY, pixel_size_camera_warning
from superqt.iconify import QIconifyIcon

from pymmcore_gui._array_viewer import (
    ensure_visible_icon,
    set_icon_tint,
    set_source_icon,
    unstyle_widgets,
)
from pymmcore_gui._qt.QtCore import QEvent, QSize, Qt, QTimer
from pymmcore_gui._qt.QtGui import QColor, QPainter, QPalette, QPixmap
from pymmcore_gui._qt.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from pymmcore_gui._theme import qcolor, theme

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from vispy.app.canvas import MouseEvent

    from pymmcore_gui._qt.QtGui import QAction, QActionGroup, QPaintEvent

_NO_PIXEL_SIZE_MESSAGE = (
    "The Stage Explorer is unavailable: the current objective/resolution preset "
    "has no pixel size. Select a calibrated preset, or set a pixel size for this "
    "one in the Pixel Configuration Editor (Configurations tab)."
)


class _PixmapWidget(QWidget):
    """Paints a pixmap as-is; ``QLabel`` grays its pixmap while disabled."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap = QPixmap()

    def setPixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self.setFixedSize(pixmap.size() / pixmap.devicePixelRatio())
        self.update()

    def pixmap(self) -> QPixmap:
        return self._pixmap

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._pixmap)


class _NoPixelSizeBanner(QWidget):
    """Warning shown in place of a usable Explorer when there is no pixel size.

    Themed through the palette (all three color groups) rather than a
    stylesheet: the Explorer is disabled while this is visible, which would
    otherwise gray it out, and ``unstyle_widgets`` clears descendant stylesheets
    on every style change anyway.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(True)
        self._icon = _PixmapWidget()
        self._text = QLabel(_NO_PIXEL_SIZE_MESSAGE)
        self._text.setWordWrap(True)

        layout = QHBoxLayout(self)
        margin = theme().sp_xs
        layout.setContentsMargins(margin, margin // 2, margin, margin // 2)
        layout.setSpacing(margin)
        layout.addWidget(self._icon, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._text, 1)
        self.apply_theme()

    def apply_theme(self) -> None:
        """Re-derive the warning colors from the active theme."""
        amber = qcolor(theme().status_amber)
        text = qcolor(theme().text_primary)
        # Tint the explorer's background with the warning color, opaque so it
        # needs no alpha blending against whatever is behind the banner.
        parent = self.parentWidget()
        base = (parent or self).palette().color(QPalette.ColorRole.Window)
        tint = 0.18
        background = QColor(
            round(base.red() * (1 - tint) + amber.red() * tint),
            round(base.green() * (1 - tint) + amber.green() * tint),
            round(base.blue() * (1 - tint) + amber.blue() * tint),
        )
        palette = self.palette()
        for group in (
            QPalette.ColorGroup.Active,
            QPalette.ColorGroup.Inactive,
            QPalette.ColorGroup.Disabled,
        ):
            palette.setColor(group, QPalette.ColorRole.Window, background)
            palette.setColor(group, QPalette.ColorRole.WindowText, text)
        self.setPalette(palette)
        size = theme().scaled(20)
        # A pixmap, not the icon: the icon would be drawn in its Disabled mode.
        pixmap = QIconifyIcon("mdi:alert").pixmap(QSize(size, size))
        painter = QPainter(pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(pixmap.rect(), amber)
        painter.end()
        self._icon.setPixmap(pixmap)


class ThemedStageExplorer(StageExplorer):
    """Stage Explorer adapted to this app's style and MDA editor."""

    MDA_ALLOWED_ACTIONS: ClassVar[tuple[str, ...]] = (
        "clear_action",
        "zoom_to_fit_action",
        "auto_zoom_to_fit_action",
        "poll_stage_action",
        "show_grid_action",
        "show_mda_positions_action",
        "map_memory_action",
        "stop_scan_action",
    )
    """Toolbar actions that stay usable while an acquisition is running.

    Everything else on the toolbar either drives the hardware (snap, ROI
    scanning) or edits state a running acquisition owns (Delete ROIs, Send to
    MDA), so :meth:`setMdaLocked` shuts it off. What is left is navigation of
    what has been mapped -- the reason the panel stays reachable at all during
    a run -- Clear View (display only, like the memory banner's clear), and the
    stage position marker, which only reads the stage. Stop Scan is the one
    exception that does touch the run: it must stay live so a scan started from
    the Explorer can be cancelled.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        mmcore: CMMCorePlus | None = None,
    ) -> None:
        self._pixel_calibrated = False
        super().__init__(parent=parent, mmcore=mmcore)

        self.toolBar().send_to_mda_action.setToolTip(
            "Add the Explorer regions to the MDA stage-position plan"
        )

        self._mda_locked = False
        # Enabled state each locked action had before the lock, so releasing
        # it can't hand back an action that was already unavailable for its
        # own reasons (e.g. ``_update_actions_enabled`` with no devices).
        self._pre_lock_action_states: dict[QAction, bool] = {}

        self._no_pixel_banner = _NoPixelSizeBanner(self)
        self._no_pixel_banner.setVisible(False)
        if isinstance(layout := self.layout(), QVBoxLayout):
            # Right under the toolbar, where the map's own banner sits.
            layout.insertWidget(1, self._no_pixel_banner)

        self._normalize_style()
        self._update_calibration_state()
        # ``pixelSizeChanged`` is emitted from a core callback thread and can
        # arrive late or out of order, so also re-check once ``configSet`` --
        # emitted with the settled state -- reports a calibration change.
        self._mmc.events.configSet.connect(self._on_config_set)

    def _unavailable_reason(self) -> str:
        """Return why the Explorer cannot place images, or "" if it can.

        Without a pixel size the core reports 0 and an all-zero affine, which
        the Explorer cannot turn into a stage-to-pixel transform: the resulting
        matrix is singular and crashes Vispy. With several cameras, the pixel
        size must also be the current camera's.
        """
        if self._mmc.getPixelSizeUm() <= 0:
            return _NO_PIXEL_SIZE_MESSAGE
        if warning := pixel_size_camera_warning(self._mmc):
            return f"The Stage Explorer is unavailable: {warning}"
        return ""

    def _is_pixel_calibrated(self) -> bool:
        """Return whether the current pixel size can place images."""
        return not self._unavailable_reason()

    def _update_calibration_state(self) -> None:
        """Disable the whole Explorer while there is no pixel calibration."""
        reason = self._unavailable_reason()
        self._pixel_calibrated = not reason
        self.setEnabled(not reason)
        self.setToolTip(reason)
        self._no_pixel_banner._text.setText(reason)
        self._no_pixel_banner.setVisible(bool(reason))

    def _on_config_set(self, group: str, preset: str) -> None:
        properties = {
            (device, prop) for device, prop, _ in self._mmc.getConfigData(group, preset)
        }
        geometry_properties = {
            (device, prop)
            for name in self._mmc.getAvailablePixelSizeConfigs()
            for device, prop, _ in self._mmc.getPixelSizeConfigData(name)
        }
        camera = self._mmc.getCameraDevice()
        geometry_properties.update((camera, prop) for prop in _CAMERA_GEOMETRY)
        if properties.isdisjoint(geometry_properties):
            return
        self._fov_observer.request()

    def _fov_w_h(self) -> tuple[float, float]:
        """Return camera-axis FOV dimensions from the active affine transform."""
        width, height = -2 * self._half_img_shift[:2, 3]
        matrix = self._affine_state.system_affine
        # Each affine column is the stage-space vector for one camera pixel.
        # Its norm remains correct for rotated, mirrored, and sheared
        # calibrations, unlike getPixelSizeUm() alone.
        pixel_width = math.hypot(float(matrix[0, 0]), float(matrix[1, 0]))
        pixel_height = math.hypot(float(matrix[0, 1]), float(matrix[1, 1]))
        return width * pixel_width, height * pixel_height

    def _on_roi_changed(self) -> None:
        """Refresh ROI tiling with affine-aware FOV dimensions."""
        if not self._is_pixel_calibrated():
            return
        super()._on_roi_changed()
        if self._mmc.getImageWidth() and self._mmc.getImageHeight():
            self.roi_manager.update_fovs(self._fov_w_h())

    def refreshPixelGeometry(self) -> None:
        """Recompute every Stage Explorer visual derived from pixel calibration."""
        self._update_calibration_state()
        if not self._pixel_calibrated:
            # Keep the last good geometry: refreshing from a missing
            # calibration would build a singular transform.
            return
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

    def _on_sys_config_loaded(self) -> None:
        # Upstream refreshes the affine and rebuilds the marker unconditionally;
        # the singular matrix only reaches Vispy once something applies it, and
        # every path that does is guarded below.
        super()._on_sys_config_loaded()
        self._update_calibration_state()

    def _update_stage_pos_marker(self, stage_x: float, stage_y: float) -> None:
        if self._pixel_calibrated:
            super()._update_stage_pos_marker(stage_x, stage_y)
        else:
            self._stage_pos_label.setText(f"X: {stage_x:.2f} µm  Y: {stage_y:.2f} µm")

    def _add_image_and_update_widget(self, *args: Any, **kwargs: Any) -> None:
        """Drop snapped and MDA frames that cannot be placed without a pixel size."""
        if self._pixel_calibrated:
            super()._add_image_and_update_widget(*args, **kwargs)

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
        unstyle_widgets(self)
        QTimer.singleShot(0, self._reposition_contrast_labels)

        toolbar = self.toolBar()
        toolbar.setMovable(False)
        toolbar.setContentsMargins(0, 0, theme().sp_xs, 0)
        # Match the rest of the app's action buttons (Snap/Live/etc, see
        # widgets/_toolbars.py's toolbar_icon_size()) rather than the native QStyle's
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
        self._no_pixel_banner.apply_theme()

    def _reposition_contrast_labels(self) -> None:
        slider = self._contrast_slider._slider
        # this runs from a timer, which the slider may not have outlived
        with suppress(RuntimeError):
            if layout := slider.layout():
                layout.activate()
            slider._reposition_labels()

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
        _recolor(toolbar.show_mda_positions_action, "mdi:map-marker-multiple-outline")
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
        marker_modes = cast("QActionGroup", toolbar.marker_mode_action_group)
        for action in marker_modes.actions():
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
        red = qcolor(theme().status_red).name()
        for action, glyph, color in (
            (toolbar.send_to_mda_action, "mdi:send", green),
            (toolbar.scan_action, "ph:path-duotone", green),
            (toolbar.stop_scan_action, "bi:sign-stop", red),
        ):
            icon = QIconifyIcon(glyph, color=color)
            action.setIcon(icon)
            button = toolbar.widgetForAction(action)
            if isinstance(button, QToolButton):
                set_source_icon(button, icon)

    def changeEvent(self, a0: QEvent | None) -> None:
        super().changeEvent(a0)
        if (
            a0 is not None
            and a0.type() == QEvent.Type.StyleChange
            # not during StageExplorer.__init__, before this class's own setup
            and hasattr(self, "_no_pixel_banner")
        ):
            self._normalize_style()
