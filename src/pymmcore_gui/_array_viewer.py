"""Custom ndv.ArrayViewer subclass for pymmcore-gui.

Adapted from the Christina viewer implementation.
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any

import ndv
import numpy as np
import tifffile
from superqt import QIconifyIcon

from pymmcore_gui._qt.QtCore import QEvent, QObject, QSize, Qt
from pymmcore_gui._qt.QtGui import QColor, QIcon, QPainter, QPalette
from pymmcore_gui._qt.QtWidgets import QAbstractButton, QFileDialog, QPushButton
from pymmcore_gui.actions.widget_actions import WidgetAction, _get_mm_main_window

# icons this dark (or darker) are effectively invisible against a dark
# theme's background and get recolored; see `ensure_visible_icon`.
_MIN_ICON_BG_CONTRAST = 60
# icons with channels further apart than this have real hue (e.g. green,
# magenta) and are left alone regardless of luminance -- only a near-gray
# icon (small spread) is treated as the "hardcoded black" bug.
_MAX_ICON_GRAYSCALE_SPREAD = 30


class _KeyFilter(QObject):
    def __init__(self, viewer: MMArrayViewer) -> None:
        super().__init__()
        self._viewer = viewer

    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:
        if event is None or obj is None:
            return False

        event_key = getattr(event, "key", lambda: None)
        if event.type() == QEvent.Type.KeyPress and event_key() == Qt.Key.Key_M:
            stats_key = getattr(WidgetAction, "STATS_TABLE", None)
            if stats_key is not None and (main_win := _get_mm_main_window(obj)):
                with suppress(KeyError):
                    table = main_win.get_widget(stats_key)
                    if (data := self._viewer._get_roi_data()) is not None:
                        table.add_stats(data)
            return True
        return False


_ORTHO_VIEWS = [("y", "x"), ("z", "x"), ("z", "y")]


class MMArrayViewer(ndv.ArrayViewer):
    """ArrayViewer with OME-TIFF saving and orthogonal-axis rotation."""

    def __init__(self, data: Any = None, /, **kwargs: Any) -> None:
        opts = kwargs.pop("viewer_options", None) or {}
        opts.setdefault("show_roi_button", True)
        kwargs["viewer_options"] = opts
        super().__init__(data, **kwargs)

        self._key_filter = _KeyFilter(self)
        widget = self.widget()
        widget.installEventFilter(self._key_filter)
        if canvas := getattr(widget, "_canvas_widget", None):
            canvas.installEventFilter(self._key_filter)

        with suppress(Exception):
            _add_save_button(self)
        with suppress(Exception):
            _add_roll_axes_button(self)
        with suppress(Exception):
            _unstyle_buttons(widget)

    def _roll_axes(self) -> None:
        """Cycle visible axes through the three orthogonal ZYX views."""
        wrapper = self.data_wrapper
        if wrapper is None:
            return
        keys = set(wrapper.sizes())
        if not {"x", "y", "z"}.issubset(keys):
            return

        current = self.display_model.visible_axes
        try:
            idx = _ORTHO_VIEWS.index(current)  # type: ignore[arg-type]
        except ValueError:
            idx = 0
        self.display_model.visible_axes = _ORTHO_VIEWS[(idx + 1) % len(_ORTHO_VIEWS)]

    def _save_data(self) -> None:
        """Save the viewer data as one or more OME-TIFF files."""
        data = self.data
        if data is None:
            return

        arr = np.asarray(data)
        if arr.size == 0:
            return

        path, _ = QFileDialog.getSaveFileName(
            self.widget(),
            "Save Image",
            "",
            "OME-TIFF (*.ome.tif);;All Files (*)",
        )
        if not path:
            return

        sizes: dict[str, int] = {}
        with suppress(Exception):
            if wrapper := self.data_wrapper:
                sizes = {str(key): value for key, value in wrapper.sizes().items()}

        scales = self.display_model.scales
        pixel_size_um = scales.get("x") or scales.get("y")
        z_step_um = scales.get("z")
        non_yx = [axis for axis in sizes if axis.lower() in "tcz"]
        axes = "".join(axis.upper() for axis in non_yx) + "YX" if non_yx else ""

        if sizes.get("p", 0) > 1:
            _save_multiposition(
                arr, sizes, path, pixel_size_um, z_step_um, axes
            )
        else:
            if "p" in sizes:
                p_idx = list(sizes).index("p")
                arr = np.squeeze(arr, axis=p_idx)
            _save_as_tiff(arr, path, pixel_size_um, z_step_um, axes)

    def _get_roi_data(self) -> np.ndarray | None:
        """Extract data under the current ROI bounding box."""
        if self.data is None or (roi := self.roi) is None:
            return None
        bbox = roi.bounding_box
        if bbox == ((0, 0), (0, 0)):
            return None

        try:
            resolved = self._resolved
        except AttributeError:
            return None
        if len(resolved.visible_axes) < 2:
            return None

        (x0, y0), (x1, y1) = bbox
        x0i, y0i = max(int(np.floor(x0)), 0), max(int(np.floor(y0)), 0)
        x1i, y1i = int(np.ceil(x1)), int(np.ceil(y1))
        if x1i <= x0i or y1i <= y0i:
            return None

        nd_index = dict(resolved.current_index)
        nd_index[resolved.visible_axes[-2]] = slice(y0i, y1i)
        nd_index[resolved.visible_axes[-1]] = slice(x0i, x1i)

        ndim = len(self.data.shape)
        idx = tuple(nd_index.get(i, slice(None)) for i in range(ndim))
        arr = np.asarray(self.data[idx])
        return arr if arr.size > 0 else None


def _luminance(color: QColor) -> float:
    return 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()


def _icon_avg_rgb(icon: QIcon, size: QSize) -> tuple[float, float, float] | None:
    """Average (r, g, b) of an icon's opaque pixels, or None if fully transparent."""
    image = icon.pixmap(size).toImage()
    r_total = g_total = b_total = 0.0
    count = 0
    for y in range(image.height()):
        for x in range(image.width()):
            pixel = image.pixelColor(x, y)
            if pixel.alpha() > 32:
                r_total += pixel.red()
                g_total += pixel.green()
                b_total += pixel.blue()
                count += 1
    if not count:
        return None
    return (r_total / count, g_total / count, b_total / count)


def _recolor_icon(icon: QIcon, color: QColor) -> QIcon:
    """Return a copy of `icon` with every opaque pixel recolored to `color`."""
    new_icon = QIcon()
    for size in icon.availableSizes() or [QSize(24, 24)]:
        pixmap = icon.pixmap(size)
        painter = QPainter(pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(pixmap.rect(), color)
        painter.end()
        new_icon.addPixmap(pixmap)
    return new_icon


def ensure_visible_icon(btn: QAbstractButton) -> None:
    """Recolor a button's icon if it's a too-dark gray to see against its background.

    Several third-party buttons (ndv in particular) hardcode a dark gray icon
    color with no theme awareness, meant for a light background — against
    this app's dark theme they're nearly invisible. This only touches icons
    that are themselves close to grayscale (no real hue) *and* close in
    luminance to the button's background; a meaningfully-colored icon (e.g. a
    green/magenta/red state indicator) is left alone even if it happens to be
    on the darker side, since its hue still reads fine against a dark bg.
    """
    icon = btn.icon()
    if icon.isNull():
        return
    size = btn.iconSize()
    if not size.isValid() or size.isEmpty():
        sizes = icon.availableSizes()
        size = sizes[0] if sizes else QSize(24, 24)
    rgb = _icon_avg_rgb(icon, size)
    if rgb is None:
        return
    r, g, b = rgb
    if max(r, g, b) - min(r, g, b) > _MAX_ICON_GRAYSCALE_SPREAD:
        return  # has real hue -- not the "invisible black icon" case
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    pal = btn.palette()
    bg_lum = _luminance(pal.color(QPalette.ColorRole.Window))
    if abs(lum - bg_lum) < _MIN_ICON_BG_CONTRAST:
        btn.setIcon(_recolor_icon(icon, pal.color(QPalette.ColorRole.WindowText)))


def _unstyle_buttons(widget: Any) -> None:
    """Normalize ndv's buttons to the app's themed button look.

    ndv gives a few buttons (e.g. the play/pause toggle) their own hardcoded
    ``QPushButton {...}`` stylesheet. A widget-level stylesheet takes over
    that widget's rendering entirely, bypassing the app's themed QStyle, so
    those buttons look inconsistent with the rest of the GUI. Clearing it
    lets them fall back to the same themed style as every other button.

    Most of these are small icon-only buttons with no text label, which are
    easy to miss under the default "ghost" variant (no visible box until
    hovered) — the "subtle" variant gives them a persistently visible box
    instead, matching Snap/Live/Shutters in the Acquire toolbar. Icons that
    are too dark to see against that box are recolored (see
    `ensure_visible_icon`).

    Only buttons are touched — other ndv widgets (e.g. the contrast-limits
    slider) rely on their own stylesheet for correct display and are left
    alone.
    """
    for btn in widget.findChildren(QAbstractButton):
        if btn.styleSheet():
            btn.setStyleSheet("")
        if not btn.property("variant"):
            btn.setProperty("variant", "subtle")
        ensure_visible_icon(btn)


def _add_save_button(viewer: MMArrayViewer) -> QPushButton:
    q_widget = viewer.widget()
    btn_layout = q_widget._btn_layout

    btn = QPushButton(q_widget)
    btn.setIcon(QIconifyIcon("mdi:content-save-outline"))
    btn.setToolTip("Save as OME-TIFF")
    btn.clicked.connect(viewer._save_data)

    ndims_idx = btn_layout.indexOf(q_widget.ndims_btn)
    btn_layout.insertWidget(ndims_idx + 1, btn)
    return btn


def _add_roll_axes_button(viewer: MMArrayViewer) -> QPushButton:
    q_widget = viewer.widget()
    btn_layout = q_widget._btn_layout

    btn = QPushButton(q_widget)
    btn.setIcon(QIconifyIcon("fluent:cube-rotate-20-regular"))
    btn.setToolTip("Cycle orthogonal views")
    btn.clicked.connect(viewer._roll_axes)

    ndims_idx = btn_layout.indexOf(q_widget.ndims_btn)
    btn_layout.insertWidget(ndims_idx + 1, btn)
    return btn


def _save_multiposition(
    arr: Any,
    sizes: dict[str, int],
    path: str,
    pixel_size_um: float | None,
    z_step_um: float | None,
    axes: str,
) -> None:
    """Save a multi-position array as one OME-TIFF per position."""
    p_idx = list(sizes).index("p")
    base = str(Path(path).with_suffix("").with_suffix(""))
    for i in range(arr.shape[p_idx]):
        _save_as_tiff(
            np.take(arr, i, axis=p_idx),
            f"{base}_p{i:03d}.ome.tif",
            pixel_size_um,
            z_step_um,
            axes,
        )


def _save_as_tiff(
    arr: Any,
    path: str,
    pixel_size_um: float | None = None,
    z_step_um: float | None = None,
    axes: str = "",
) -> None:
    """Save an array as OME-TIFF with physical-size metadata."""
    array = np.asarray(arr)
    metadata: dict[str, Any] = {}
    if axes:
        metadata["axes"] = axes
    if pixel_size_um:
        metadata.update(
            {
                "PhysicalSizeX": pixel_size_um,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": pixel_size_um,
                "PhysicalSizeYUnit": "µm",
            }
        )
    if z_step_um:
        metadata["PhysicalSizeZ"] = z_step_um
        metadata["PhysicalSizeZUnit"] = "µm"

    tifffile.imwrite(
        path,
        array,
        ome=True,
        photometric="minisblack",
        metadata=metadata or None,
    )
