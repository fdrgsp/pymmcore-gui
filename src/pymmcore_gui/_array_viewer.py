"""Custom ndv.ArrayViewer subclass for pymmcore-gui.

Adapted from the Christina viewer implementation.
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import ndv
import numpy as np
import tifffile
from ndv.models import ChannelMode
from ndv.models._viewer_model import InteractionMode
from ome_writers import AcquisitionSettings, Dimension
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.metadata import summary_metadata
from superqt import QIconifyIcon

from pymmcore_gui._mda_export import AcquisitionRecord, export_acquisition
from pymmcore_gui._qt.QtCore import QEvent, QObject, QSize, Qt
from pymmcore_gui._qt.QtGui import QColor, QIcon, QPainter, QPalette
from pymmcore_gui._qt.QtWidgets import (
    QAbstractButton,
    QAbstractSlider,
    QApplication,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QWidget,
)
from pymmcore_gui.actions.widget_actions import WidgetAction, _get_mm_main_window

if TYPE_CHECKING:
    from pymmcore_gui._mda_export import ExportFormat

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

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if a1 is None or a0 is None:
            return False

        event_key = getattr(a1, "key", lambda: None)
        if a1.type() == QEvent.Type.KeyPress and event_key() == Qt.Key.Key_M:
            stats_key = getattr(WidgetAction, "STATS_TABLE", None)
            if stats_key is not None and (main_win := _get_mm_main_window(a0)):
                with suppress(KeyError):
                    table = main_win.get_widget(stats_key)
                    if (data := self._viewer._get_roi_data()) is not None:
                        table.add_stats(data)
            return True
        return False


def _disable_vispy_backspace_reset(canvas: Any) -> None:
    """Stop vispy's camera from resetting the view to its pre-data state.

    ``vispy.scene.cameras.BaseCamera.viewbox_key_event`` resets the camera to
    whatever range was set *before* any image was ever loaded whenever
    Backspace reaches the canvas (macOS labels this key "delete", and it
    reliably reaches the canvas since nothing else claims it). The reset
    target is essentially an empty 1x1 rect, so the image appears to vanish
    even though no data was touched. Qt-level event filtering can't prevent
    this -- vispy's key handling isn't reachable through the normal
    QWidget/eventFilter chain -- so the camera's listener is disconnected
    from vispy's own key-press emitter directly.
    """
    camera = getattr(canvas, "_camera", None)
    vispy_canvas = getattr(canvas, "_canvas", None)
    if camera is None or vispy_canvas is None:
        return
    with suppress(Exception):
        vispy_canvas.events.key_press.disconnect(camera.viewbox_key_event)
    with suppress(Exception):
        vispy_canvas.events.key_release.disconnect(camera.viewbox_key_event)


def _guard_vispy_camera_resets(canvas: Any) -> None:
    """Apply :func:`_disable_vispy_backspace_reset` to every camera vispy creates.

    Channel-mode changes (grayscale vs. composite) never touch the camera,
    but toggling ndv's 2D/3D view does: ``VispyArrayCanvas.set_ndim`` is the
    only place that swaps in a new vispy camera (2D ``PanZoomCamera`` <-> 3D
    ``ArcballCamera``), and each new camera reconnects its own Backspace-reset
    listener independently of any previous one that was disarmed. Wrapping
    ``set_ndim`` re-disarms whichever camera comes out of it, so the fix
    survives 2D/3D toggling instead of only covering the camera that existed
    at viewer construction.
    """
    _disable_vispy_backspace_reset(canvas)
    set_ndim = getattr(canvas, "set_ndim", None)
    if set_ndim is None:
        return

    def _set_ndim_and_guard(*args: Any, **kwargs: Any) -> Any:
        result = set_ndim(*args, **kwargs)
        _disable_vispy_backspace_reset(canvas)
        return result

    with suppress(Exception):
        canvas.set_ndim = _set_ndim_and_guard


_ORTHO_VIEWS = [("y", "x"), ("z", "x"), ("z", "y")]


class MMArrayViewer(ndv.ArrayViewer):
    """ArrayViewer with OME-TIFF/OME-Zarr saving and orthogonal-axis rotation."""

    def __init__(self, data: Any = None, /, **kwargs: Any) -> None:
        show_save_button = bool(kwargs.pop("show_save_button", True))
        show_roll_axes_button = bool(kwargs.pop("show_roll_axes_button", True))
        opts = kwargs.pop("viewer_options", None) or {}
        opts.setdefault("show_roi_button", True)
        kwargs["viewer_options"] = opts
        super().__init__(data, **kwargs)

        # Set by the viewer manager (e.g. AcquireViewersManager) right after
        # construction, for MDA-backed viewers: a snapshot of the sink's
        # resolved settings + summary metadata, plus live per-frame metadata
        # appended as the acquisition progresses. When absent (e.g. the
        # snap/live Preview, which isn't backed by an MDA at all), _save_data
        # falls back to synthesizing a minimal one from what's on screen.
        self._acquisition_record: AcquisitionRecord | None = None

        self._key_filter = _KeyFilter(self)
        widget = self.widget()
        widget.installEventFilter(self._key_filter)
        if canvas := getattr(widget, "_canvas_widget", None):
            canvas.installEventFilter(self._key_filter)

        _guard_vispy_camera_resets(self._canvas)

        if show_save_button:
            with suppress(Exception):
                _add_save_button(self)
        if show_roll_axes_button:
            with suppress(Exception):
                _add_roll_axes_button(self)
        with suppress(Exception):
            unstyle_widgets(widget)

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

    def set_roi_selection_active(self, active: bool) -> None:
        """Enter or leave ndv's rectangular ROI interaction mode."""
        mode = InteractionMode.CREATE_ROI if active else InteractionMode.PAN_ZOOM
        if self._viewer_model.interaction_mode != mode:
            self._viewer_model.interaction_mode = mode

    def roi_selection_active(self) -> bool:
        """Return whether ndv's rectangular ROI interaction mode is active."""
        return self._viewer_model.interaction_mode is InteractionMode.CREATE_ROI

    def set_existing_roi_editing_active(self, active: bool) -> None:
        """Show handles for the existing ROI without entering creation mode.

        ndv's ``CREATE_ROI`` mode intentionally uses the next mouse press to
        start a brand-new rectangle. Existing rectangle handles, however, are
        selected and dragged in ``PAN_ZOOM`` mode.
        """
        if self._viewer_model.interaction_mode is not InteractionMode.PAN_ZOOM:
            self._viewer_model.interaction_mode = InteractionMode.PAN_ZOOM
        if active and self.roi is not None:
            if self._roi_view is None:
                self._create_roi_view()
            self._synchronize_roi()
        self.set_roi_visual_selected(active)

    def existing_roi_editing_active(self) -> bool:
        """Return whether an existing ROI is selected for handle editing."""
        return (
            self._viewer_model.interaction_mode is InteractionMode.PAN_ZOOM
            and self.roi is not None
            and self.roi_visual_selected()
        )

    def set_roi_visual_selected(self, selected: bool) -> None:
        """Set the current ndv ROI visual's selected/handle state."""
        if self._roi_view is not None:
            self._roi_view.set_selected(selected)

    def roi_visual_selected(self) -> bool:
        """Return whether the current ndv ROI visual is visibly selected."""
        return self._roi_view is not None and self._roi_view.selected()

    def roi_visual_visible(self) -> bool:
        """Return whether the current ndv ROI visual is visible."""
        return self._roi_view is not None and self._roi_view.visible()

    def reset_zoom(self) -> None:
        """Fit the canvas camera to the currently displayed image."""
        self._on_view_reset_zoom_clicked()

    def clear_roi(self) -> None:
        """Remove both the ndv ROI model and its canvas visual."""
        self.roi = None
        if self._roi_view is not None:
            self._roi_view.remove()
            self._roi_view = None

    def connect_roi_selection_changed(self, callback: Any) -> None:
        """Connect to ndv interaction-mode changes through one compatibility seam."""
        self._viewer_model.events.interaction_mode.connect(callback)

    def disconnect_roi_selection_changed(self, callback: Any) -> None:
        """Disconnect a callback registered by :meth:`connect_roi_selection_changed`."""
        with suppress(Exception):
            self._viewer_model.events.interaction_mode.disconnect(callback)

    def _save_data(self) -> None:
        """Export the viewer's data as a metadata-complete OME-TIFF or OME-Zarr.

        Streams frame-by-frame through `ome_writers` -- the same writer a live,
        disk-backed acquisition uses -- rather than materializing the whole
        array and hand-writing a TIFF, so channel names, timestamps,
        exposures, stage positions and the summary metadata all survive.
        """
        if self.data is None:
            return

        if self.display_model.channel_mode is ChannelMode.RGBA:
            # ome_writers models a written frame as a plain 2D (Y, X) plane; it
            # has no concept of an RGB/RGBA sample axis. This only affects the
            # snap/live Preview (the only place an RGB frame can appear), so
            # fall back to a direct, non-metadata TIFF write for that one case.
            self._save_rgb_snapshot()
            return

        prompt = _prompt_save_path(self.widget())
        if prompt is None:
            return
        path, fmt = prompt

        record = self._acquisition_record or _synthesize_record(self)
        if record is None:
            QMessageBox.warning(self.widget(), "Save", "No data available to save.")
            return

        try:
            self._export_with_overwrite_prompt(record, path, fmt)
        except Exception as e:
            QMessageBox.critical(
                self.widget(), "Save failed", f"Failed to save data:\n\n{e}"
            )

    def _export_with_overwrite_prompt(
        self, record: AcquisitionRecord, path: Path, fmt: ExportFormat
    ) -> None:
        """Run `export_acquisition`, confirming before clobbering an existing path.

        `ome_writers` is the authority on whether the *resolved* output path
        (which, for a multi-position OME-TIFF, is a directory distinct from
        the file path the user picked) already exists -- so the first attempt
        always goes in with `overwrite=False`, and only on the resulting
        `FileExistsError` do we ask and retry. `create_stream` raises that
        error before any frame is written, so the retry never double-writes.
        """

        def _run(*, overwrite: bool) -> None:
            dlg = QProgressDialog("Saving…", "Cancel", 0, 0, self.widget())
            dlg.setWindowModality(Qt.WindowModality.WindowModal)
            dlg.setMinimumDuration(0)
            dlg.setAutoClose(True)
            dlg.setValue(0)

            def _progress(done: int, total: int) -> bool:
                dlg.setMaximum(total)
                dlg.setValue(done)
                QApplication.processEvents()
                return not dlg.wasCanceled()

            try:
                export_acquisition(
                    record, path, fmt, overwrite=overwrite, progress=_progress
                )
            finally:
                dlg.close()

        try:
            _run(overwrite=False)
        except FileExistsError:
            reply = QMessageBox.question(
                self.widget(),
                "Overwrite existing data?",
                f"{path} already exists.\n\nOverwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                _run(overwrite=True)

    def _save_rgb_snapshot(self) -> None:
        """Save a single RGB/RGBA frame directly as TIFF (no ome_writers path)."""
        arr = np.asarray(self.data)
        if arr.size == 0:
            return
        # Drop leading singleton axes (e.g. the Preview's 1-frame ring-buffer
        # axis) so tifffile sees a plain (Y, X, samples) plane.
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        path, _ = QFileDialog.getSaveFileName(
            self.widget(), "Save Image", "", "TIFF (*.tiff *.tif)"
        )
        if not path:
            return
        if not path.lower().endswith((".tif", ".tiff")):
            path += ".tiff"
        try:
            tifffile.imwrite(path, arr, photometric="rgb")
        except Exception as e:
            QMessageBox.critical(
                self.widget(), "Save failed", f"Failed to save data:\n\n{e}"
            )

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


_ORIGINAL_ICON_PROPERTY = "_pymmcore_gui_original_icon"
_ICON_TINT_PROPERTY = "_pymmcore_gui_icon_tint"


def set_source_icon(btn: QAbstractButton, icon: QIcon) -> None:
    """Set a button icon and remember it as the source for theme recoloring."""
    btn.setProperty(_ORIGINAL_ICON_PROPERTY, icon)
    btn.setIcon(icon)


def set_icon_tint(btn: QAbstractButton, color: QColor) -> None:
    """Force a button icon to *color*, retaining its pristine source icon.

    The tint is stored as a dynamic property so subsequent calls to
    :func:`ensure_visible_icon` (notably the application-wide theme-change
    sweep) preserve the requested semantic color instead of replacing it with
    the generic foreground color.
    """
    icon = btn.property(_ORIGINAL_ICON_PROPERTY)
    if icon is None:
        icon = btn.icon()
        if icon.isNull():
            return
        btn.setProperty(_ORIGINAL_ICON_PROPERTY, icon)
    btn.setProperty(_ICON_TINT_PROPERTY, QColor(color))
    btn.setIcon(_recolor_icon(icon, color))


def ensure_visible_icon(btn: QAbstractButton) -> None:
    """Recolor a button's icon if it's a too-dark gray to see against its background.

    Several third-party buttons (ndv in particular) hardcode a dark gray icon
    color with no theme awareness, meant for a light background — against
    this app's dark theme they're nearly invisible. This only touches icons
    that are themselves close to grayscale (no real hue) *and* close in
    luminance to the button's background; a meaningfully-colored icon (e.g. a
    green/magenta/red state indicator) is left alone even if it happens to be
    on the darker side, since its hue still reads fine against a dark bg.

    Re-callable any time the theme changes (light <-> dark): the button's
    *original* icon is stashed on first call and every subsequent call
    re-evaluates from that pristine copy, never from an already-recolored
    one -- otherwise a dark-theme recolor (e.g. tinted white) would get
    baked in permanently and turn invisible again after switching to light
    mode, since a static pixmap doesn't know the theme changed.
    """
    icon = btn.property(_ORIGINAL_ICON_PROPERTY)
    if icon is None:
        icon = btn.icon()
        if icon.isNull():
            return
        btn.setProperty(_ORIGINAL_ICON_PROPERTY, icon)

    if isinstance(tint := btn.property(_ICON_TINT_PROPERTY), QColor):
        btn.setIcon(_recolor_icon(icon, tint))
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
        btn.setIcon(icon)  # has real hue -- always show the original
        return
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    # QApplication.palette(), not btn.palette(): right after
    # QApplication.setPalette() the *application* palette is updated
    # immediately, but an already-constructed widget's own .palette() is
    # only refreshed once its queued PaletteChange event is processed --
    # reading it synchronously (as the theme-change sweep does) would still
    # see the *previous* theme's colors.
    pal = QApplication.palette() if QApplication.instance() else btn.palette()
    bg_lum = _luminance(pal.color(QPalette.ColorRole.Window))
    if abs(lum - bg_lum) < _MIN_ICON_BG_CONTRAST:
        btn.setIcon(_recolor_icon(icon, pal.color(QPalette.ColorRole.WindowText)))
    else:
        btn.setIcon(icon)  # original already contrasts fine against this bg


def unstyle_widgets(widget: Any) -> None:
    """Normalize third-party widgets to the app's themed look.

    Third-party code sometimes hardcodes a one-off ``setStyleSheet(...)``
    (e.g. ndv's play/pause button, or useq_widgets' border-less table
    spinboxes and gray range labels) -- and not always directly on the
    styled leaf widget: several cases apply a type-selector rule (e.g.
    ``"QLabel {...}"``) to a *wrapper* container instead, relying on Qt's
    stylesheet cascade to reach the actual descendant. A stylesheet anywhere
    in that chain takes over rendering for whatever properties it sets,
    bypassing the app's themed QStyle entirely, so clearing only specific
    leaf types misses these. Walking every descendant and clearing
    unconditionally catches both cases.

    The one deliberate exception is `QAbstractSlider` (covers `QSlider` and
    superqt's `QLabeledSlider`/`QLabeledRangeSlider` family): ndv's
    contrast-limits slider sets a stylesheet that defines its actual
    groove/handle rendering and handle-label color, which is functional, not
    cosmetic, and would look broken if cleared.

    Buttons additionally get the "subtle" variant (a persistently visible
    box, rather than only on hover -- most are small icon-only buttons with
    no text label, easy to miss under the default "ghost" variant) and a
    pass through `ensure_visible_icon`.

    ``widget`` itself is swept too (not just ``findChildren``), since the
    offending stylesheet is sometimes on the third-party root -- e.g.
    ConfigGroupsEditor sets one on itself.
    """
    for w in (widget, *widget.findChildren(QWidget)):
        if isinstance(w, QAbstractSlider):
            continue
        if w.styleSheet():
            w.setStyleSheet("")
        if isinstance(w, QAbstractButton):
            if not w.property("variant"):
                w.setProperty("variant", "subtle")
            ensure_visible_icon(w)


def _add_save_button(viewer: MMArrayViewer) -> QPushButton:
    q_widget = viewer.widget()
    btn_layout = q_widget._btn_layout

    btn = QPushButton(q_widget)
    btn.setIcon(QIconifyIcon("mdi:content-save-outline"))
    btn.setToolTip("Save as OME-TIFF / OME-Zarr")
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


_SAVE_FILTERS = "OME-TIFF (*.ome.tiff *.ome.tif);;OME-Zarr (*.ome.zarr)"

_DimType = Literal["space", "time", "channel", "position", "other"]
_TYPE_BY_AXIS_NAME: dict[str, _DimType] = {
    "t": "time",
    "c": "channel",
    "z": "space",
    "p": "position",
}


class _RecordSource(Protocol):
    """Structural type for `_synthesize_record`'s input.

    Matches the subset of `ndv.ArrayViewer` (which `MMArrayViewer` inherits)
    that the function actually reads, so lightweight test fakes can duck-type
    against it without depending on the concrete `MMArrayViewer` class.
    """

    @property
    def data(self) -> Any: ...
    @property
    def data_wrapper(self) -> Any: ...
    @property
    def display_model(self) -> Any: ...


def _prompt_save_path(parent: QWidget) -> tuple[Path, ExportFormat] | None:
    """Ask for a destination path and format via one native save dialog."""
    path_str, selected_filter = QFileDialog.getSaveFileName(
        parent, "Save Acquisition", "", _SAVE_FILTERS
    )
    if not path_str:
        return None

    fmt: ExportFormat = "ome-zarr" if "Zarr" in selected_filter else "ome-tiff"
    name = path_str.lower()
    if fmt == "ome-zarr":
        if not name.endswith(".zarr"):
            path_str += ".ome.zarr"
    elif not name.endswith((".tif", ".tiff")):
        path_str += ".ome.tiff"
    return Path(path_str), fmt


def _synthesize_record(viewer: _RecordSource) -> AcquisitionRecord | None:
    """Build a minimal record straight from what the viewer is currently displaying.

    Used whenever no live `AcquisitionRecord` was attached at acquisition time
    -- e.g. the snap/live Preview (not backed by an MDA at all), or an MDA
    viewer whose manager didn't attach one. Per-frame metadata is unavailable
    here, and non-spatial axes get sequential (not necessarily named)
    coordinates; physical scale comes only from the viewer's current
    `display_model.scales`. `viewer.data` (never materialized) is used
    directly as the record's view -- it already supports the same
    acquisition-order tuple indexing `export_acquisition` relies on, whether
    it's a live `StreamView` or an `ndv` `RingBuffer`.
    """
    data = viewer.data
    wrapper = viewer.data_wrapper
    if data is None or wrapper is None:
        return None

    sizes = dict(wrapper.sizes())
    if len(sizes) < 2:
        return None
    names = list(sizes)
    n = len(names)
    scales = dict(viewer.display_model.scales)

    dims: list[Dimension] = []
    for i, name in enumerate(names):
        is_frame_axis = i >= n - 2
        axis_name = ("y", "x")[i - (n - 2)] if is_frame_axis else str(name)
        dim_type: _DimType = (
            "space" if is_frame_axis else _TYPE_BY_AXIS_NAME.get(axis_name, "other")
        )
        scale = scales.get(axis_name)
        dims.append(
            Dimension(
                name=axis_name,
                count=sizes[name],
                type=dim_type,
                scale=scale,
                unit="micrometer" if (dim_type == "space" and scale) else None,
            )
        )

    summary_meta = None
    with suppress(Exception):
        summary_meta = summary_metadata(CMMCorePlus.instance())

    settings = AcquisitionSettings(
        dimensions=tuple(dims), dtype=str(np.dtype(wrapper.dtype))
    )
    return AcquisitionRecord(settings=settings, summary_meta=summary_meta, view=data)
