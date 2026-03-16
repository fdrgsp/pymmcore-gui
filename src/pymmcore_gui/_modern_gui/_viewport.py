"""Image Viewport Widget."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from pymmcore_widgets import ImagePreview

from pymmcore_gui._modern_gui._utils import current_core
from pymmcore_gui._qt.QtCore import (
    QEvent,
    QPointF,
    QRectF,
    Qt,
    QTimer,
    Signal,
)
from pymmcore_gui._qt.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QMouseEvent,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
)
from pymmcore_gui._qt.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ._theme import ROW_HEIGHT, mono_font, qcolor, theme, ui_font

# Viewport-specific colors not covered by the shared theme
OVERLAY = QColor(255, 255, 255, 102)  # ~40% white
OVERLAY_DIM = QColor(255, 255, 255, 64)  # ~25% white
CH_DAPI = QColor(0x44, 0x72, 0xC4)
CH_GFP = QColor(0x00, 0xCC, 0x66)
CH_CY5 = QColor(0xCC, 0x44, 0xCC)


# ═══════════════════════════════════════════════════════════════
# ToolbarButton — custom painted, no QSS
# ═══════════════════════════════════════════════════════════════


class ToolbarButton(QWidget):
    """A small toolbar button with text/icon, hover state, and optional
    toggle/active styling.  Painted entirely via QPainter.
    """

    clicked = Signal()

    def __init__(
        self,
        text: str,
        *,
        checkable: bool = False,
        accent_color: QColor | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._text = text
        self._checkable = checkable
        self._checked = False
        self._hovered = False
        self._accent = accent_color

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self.setFixedHeight(26)

        fm = QFontMetricsF(ui_font(8, QFont.Weight.Medium))
        text_w = fm.horizontalAdvance(text)
        self.setFixedWidth(int(text_w) + 20)

    @property
    def checked(self) -> bool:
        return self._checked

    @checked.setter
    def checked(self, val: bool) -> None:
        self._checked = val
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(0, 0, -1, -1)
        t = theme()

        # Background
        if self._accent and self._checked:
            bg = QColor(self._accent)
            bg.setAlpha(40)
            border = QColor(self._accent)
            border.setAlpha(76)
            text_color = self._accent
        elif self._checked:
            bg = qcolor(t.accent_muted)
            border = qcolor(t.accent)
            border.setAlpha(76)
            text_color = qcolor(t.accent)
        elif self._hovered:
            bg = qcolor(t.bg_hover)
            border = Qt.GlobalColor.transparent
            text_color = qcolor(t.text_primary)
        else:
            bg = Qt.GlobalColor.transparent
            border = Qt.GlobalColor.transparent
            text_color = qcolor(t.text_secondary)

        p.setPen(
            QPen(QColor(border), 1)
            if border != Qt.GlobalColor.transparent
            else QPen(Qt.PenStyle.NoPen)
        )
        p.setBrush(
            QBrush(QColor(bg))
            if bg != Qt.GlobalColor.transparent
            else QBrush(Qt.BrushStyle.NoBrush)
        )
        p.drawRoundedRect(QRectF(r), 3, 3)

        # Text
        p.setPen(QColor(text_color))
        p.setFont(ui_font(8, QFont.Weight.Medium))
        p.drawText(QRectF(r), Qt.AlignmentFlag.AlignCenter, self._text)
        p.end()

    def enterEvent(self, event) -> None:
        self._hovered = True
        self.update()

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._checkable:
                self._checked = not self._checked
            self.clicked.emit()
            self.update()


# ═══════════════════════════════════════════════════════════════
# Toolbar Separator
# ═══════════════════════════════════════════════════════════════


class ToolbarSep(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(1, 18)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setPen(QPen(qcolor(theme().border_subtle), 1))
        p.drawLine(0, 0, 0, self.height())
        p.end()


# ═══════════════════════════════════════════════════════════════
# ViewportOverlay — transparent HUD drawn on top of the image
# ═══════════════════════════════════════════════════════════════


@dataclass
class OverlayInfo:
    """Data displayed as overlays on the viewport."""

    label: str = "Live"
    detail: str = "DAPI+GFP+Cy5"
    dimensions: str = "2048 × 2048"
    scale_bar_um: float = 100.0
    scale_bar_px: int = 80
    cursor_x: int = 0
    cursor_y: int = 0
    cursor_intensity: int = 0


class ViewportOverlay(QWidget):
    """Transparent overlay painted on top of the image canvas."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(parent.rect())
        self.raise_()
        parent.installEventFilter(self)

        self.info = OverlayInfo()

    def eventFilter(self, obj: object, event: object) -> bool:
        if isinstance(event, QEvent) and event.type() == QEvent.Type.Resize:
            self.setGeometry(obj.rect())  # type: ignore[union-attr]
        return False

    def paintEvent(self, event: object) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        ov = self.info
        margin = 16

        # ── Top-left: label ──
        p.setFont(mono_font(8))
        p.setPen(OVERLAY_DIM)
        p.drawText(margin, margin + 12, ov.label)

        # ── Top-right: detail info ──
        p.setFont(mono_font(8))
        p.setPen(OVERLAY_DIM)
        lines = [
            f"{ov.label} · {ov.detail}",
            ov.dimensions,
        ]
        y = margin
        for line in lines:
            fm = QFontMetricsF(p.font())
            tw = fm.horizontalAdvance(line)
            p.drawText(int(w - margin - tw), y + 12, line)
            y += 16

        # ── Bottom-left: scale bar ──
        sb_x = margin
        sb_y = h - margin - 18

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(255, 255, 255, 204)))
        p.drawRoundedRect(QRectF(sb_x, sb_y, ov.scale_bar_px, 3), 1.5, 1.5)

        p.setFont(mono_font(8))
        p.setPen(OVERLAY)
        p.drawText(sb_x, sb_y + 16, f"{ov.scale_bar_um:.0f} μm")

        # ── Bottom-right: cursor info ──
        cursor_text = f"X: {ov.cursor_x}  Y: {ov.cursor_y}  ·  I: {ov.cursor_intensity}"
        p.setFont(mono_font(8))
        p.setPen(OVERLAY_DIM)
        fm = QFontMetricsF(p.font())
        tw = fm.horizontalAdvance(cursor_text)
        p.drawText(int(w - margin - tw), h - margin, cursor_text)

        p.end()


# ═══════════════════════════════════════════════════════════════
# ChannelButton — single channel toggle in the strip
# ═══════════════════════════════════════════════════════════════


class ChannelButton(QWidget):
    """A channel toggle button with a colored dot."""

    toggled = Signal(bool)

    def __init__(
        self,
        name: str,
        color: QColor,
        *,
        active: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._name = name
        self._color = color
        self._active = active
        self._hovered = False

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self.setFixedHeight(22)

        fm = QFontMetricsF(ui_font(8, QFont.Weight.Medium))
        self.setFixedWidth(int(fm.horizontalAdvance(name)) + 28)

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, val: bool) -> None:
        self._active = val
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(0, 0, -1, -1)
        t = theme()

        # Border when active
        if self._active:
            p.setPen(QPen(qcolor(t.border_default), 1))
        else:
            p.setPen(QPen(Qt.PenStyle.NoPen))

        if self._hovered:
            p.setBrush(QBrush(qcolor(t.bg_hover)))
        else:
            p.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        p.drawRoundedRect(QRectF(r), 2, 2)

        # Dot
        dot_r = 3
        dot_x = 8
        dot_y = r.height() / 2
        dot_color = QColor(self._color) if self._active else qcolor(t.text_disabled)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(dot_color))
        p.drawEllipse(QPointF(dot_x, dot_y), dot_r, dot_r)

        # Name
        text_color = qcolor(t.text_primary if self._active else t.text_secondary)
        p.setPen(text_color)
        p.setFont(ui_font(8, QFont.Weight.Medium))
        p.drawText(
            QRectF(dot_x + dot_r + 5, 0, r.width() - dot_x - dot_r - 10, r.height()),
            Qt.AlignmentFlag.AlignVCenter,
            self._name,
        )
        p.end()

    def enterEvent(self, event) -> None:
        self._hovered = True
        self.update()

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._active = not self._active
            self.toggled.emit(self._active)
            self.update()


# ═══════════════════════════════════════════════════════════════
# ChannelStrip — row of channel toggles
# ═══════════════════════════════════════════════════════════════


class ChannelStrip(QWidget):
    """Horizontal strip of channel toggle buttons."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(28)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 3, 12, 3)
        layout.setSpacing(2)

        self._channels: dict[str, ChannelButton] = {}

        # Default channels
        for name, color in [
            ("DAPI", CH_DAPI),
            ("GFP", CH_GFP),
            ("Cy5", CH_CY5),
        ]:
            btn = ChannelButton(name, color)
            layout.addWidget(btn)
            self._channels[name] = btn

        layout.addStretch()

        self._merge_btn = ToolbarButton("Merge", checkable=True)
        self._merge_btn.checked = True
        layout.addWidget(self._merge_btn)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setPen(QPen(qcolor(theme().border_subtle), 1))
        w, h = self.width(), self.height()
        p.drawLine(0, 0, w, 0)
        p.drawLine(0, h - 1, w, h - 1)
        p.end()
        super().paintEvent(event)


# ═══════════════════════════════════════════════════════════════
# SnapThumbnail — single thumbnail in the filmstrip
# ═══════════════════════════════════════════════════════════════


class SnapThumbnail(QWidget):
    """A single snap thumbnail with timestamp and star toggle."""

    THUMB_SIZE = 52

    clicked = Signal()

    def __init__(
        self,
        pixmap: QPixmap,
        timestamp: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._pixmap = pixmap.scaled(
            self.THUMB_SIZE,
            self.THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._timestamp = timestamp
        self._hovered = False
        self._active = False
        self._starred = False

        self.setFixedSize(self.THUMB_SIZE, self.THUMB_SIZE + 14)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, val: bool) -> None:
        self._active = val
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s = self.THUMB_SIZE

        t = theme()

        # Border
        if self._active:
            p.setPen(QPen(qcolor(t.accent), 2))
        elif self._hovered:
            p.setPen(QPen(qcolor(t.border_default), 2))
        else:
            p.setPen(QPen(Qt.PenStyle.NoPen))

        p.setBrush(QBrush(qcolor(t.bg_raised)))
        p.drawRoundedRect(QRectF(0, 0, s, s), 3, 3)

        # Image
        p.setClipRect(QRectF(1, 1, s - 2, s - 2))
        p.drawPixmap(1, 1, self._pixmap)
        p.setClipping(False)

        # Timestamp bar
        ts_h = 12
        ts_y = s - ts_h
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, 153)))
        p.drawRect(QRectF(0, ts_y, s, ts_h))

        p.setFont(mono_font(5.5))
        p.setPen(QColor(255, 255, 255, 153))
        p.drawText(
            QRectF(0, ts_y, s, ts_h),
            Qt.AlignmentFlag.AlignCenter,
            self._timestamp,
        )

        # Star (top-right, on hover)
        if self._hovered or self._starred:
            star_color = (
                QColor(0xFF, 0xD5, 0x4F) if self._starred else QColor(255, 255, 255, 76)
            )
            p.setFont(ui_font(7))
            p.setPen(star_color)
            p.drawText(QRectF(s - 14, 1, 12, 12), Qt.AlignmentFlag.AlignCenter, "★")

        p.end()

    def enterEvent(self, event) -> None:
        self._hovered = True
        self.update()

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if star area clicked
            s = self.THUMB_SIZE
            star_rect = QRectF(s - 14, 1, 14, 14)
            if star_rect.contains(event.position()):
                self._starred = not self._starred
                self.update()
            else:
                self.clicked.emit()


class _RotatedLabel(QWidget):
    """Text label rotated 90 degrees counter-clockwise."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text = text
        self._font = ui_font(11, QFont.Weight.DemiBold)
        fm = QFontMetricsF(self._font)
        text_h = int(fm.horizontalAdvance(text))
        self.setFixedWidth(int(fm.height()) + 4)
        self.setMinimumHeight(text_h + 8)

    def paintEvent(self, event: object) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setFont(self._font)
        p.setPen(qcolor(theme().text_disabled))
        p.translate(self.width() / 2, self.height() / 2)
        p.rotate(-90)
        fm = QFontMetricsF(self._font)
        tw = fm.horizontalAdvance(self._text)
        p.drawText(
            QRectF(-tw / 2, -fm.height() / 2, tw, fm.height()),
            Qt.AlignmentFlag.AlignCenter,
            self._text,
        )
        p.end()


# ═══════════════════════════════════════════════════════════════
# SnapFilmstrip — horizontal scrollable row of snap thumbnails
# ═══════════════════════════════════════════════════════════════


class SnapFilmstrip(QWidget):
    """Horizontal filmstrip of snap thumbnails with a clear button."""

    STRIP_HEIGHT = 72

    snap_selected = Signal(int)  # index

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(self.STRIP_HEIGHT)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 0, 12, 0)
        outer.setSpacing(8)

        # Rotated label
        self._label = _RotatedLabel("SNAPS")
        outer.addWidget(self._label)

        # Scroll area for thumbnails
        self._scroll = QScrollArea()
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFixedHeight(self.STRIP_HEIGHT)
        outer.addWidget(self._scroll, 1)

        # Inner widget with horizontal layout
        self._container = QWidget()
        self._thumb_layout = QHBoxLayout(self._container)
        self._thumb_layout.setContentsMargins(0, 6, 0, 0)
        self._thumb_layout.setSpacing(6)
        self._thumb_layout.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._scroll.setWidget(self._container)

        self._thumbnails: list[SnapThumbnail] = []

        # Clear button
        self._clear_btn = ToolbarButton("Clear")
        self._clear_btn.setFixedHeight(22)
        self._clear_btn.clicked.connect(self.clear)
        outer.addWidget(self._clear_btn, 0, Qt.AlignmentFlag.AlignVCenter)

    def add_snap(self, data: np.ndarray, timestamp: str | None = None) -> None:
        """Add a snap image to the filmstrip.

        Parameters
        ----------
        data : np.ndarray
            Image data. Accepts (H, W), (H, W, 3), or (H, W, 4).
            Will be converted to QPixmap internally.
        timestamp : str, optional
            Display timestamp. Defaults to current time HH:MM:SS.
        """
        if timestamp is None:
            timestamp = time.strftime("%H:%M:%S")

        pixmap = self._ndarray_to_pixmap(data)
        thumb = SnapThumbnail(pixmap, timestamp)
        thumb.clicked.connect(
            lambda idx=len(self._thumbnails): self._on_thumb_clicked(idx)
        )
        self._thumb_layout.addWidget(thumb)
        self._thumbnails.append(thumb)

        # Scroll to end
        QTimer.singleShot(10, self._scroll_to_end)

    def clear(self) -> None:
        """Remove all snap thumbnails."""
        for thumb in self._thumbnails:
            self._thumb_layout.removeWidget(thumb)
            thumb.deleteLater()
        self._thumbnails.clear()

    def _on_thumb_clicked(self, index: int) -> None:
        for i, t in enumerate(self._thumbnails):
            t.active = i == index
        self.snap_selected.emit(index)

    def _scroll_to_end(self) -> None:
        sb = self._scroll.horizontalScrollBar()
        if sb:
            sb.setValue(sb.maximum())

    @staticmethod
    def _ndarray_to_pixmap(data: np.ndarray) -> QPixmap:
        """Convert a numpy array to QPixmap."""
        if data.ndim == 2:
            h, w = data.shape
            # Normalize to 8-bit if needed
            if data.dtype != np.uint8:
                d = data.astype(np.float32)
                lo, hi = d.min(), d.max()
                if hi > lo:
                    d = ((d - lo) / (hi - lo) * 255).astype(np.uint8)
                else:
                    d = np.zeros_like(data, dtype=np.uint8)
            else:
                d = data
            # Grayscale → RGB
            rgb = np.stack([d, d, d], axis=-1)
            img = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(img.copy())

        elif data.ndim == 3:
            h, w, ch = data.shape
            d = (
                data
                if data.dtype == np.uint8
                else (data / data.max() * 255).astype(np.uint8)
            )
            if ch == 3:
                img = QImage(d.data, w, h, w * 3, QImage.Format.Format_RGB888)
            elif ch == 4:
                img = QImage(d.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
            else:
                raise ValueError(f"Unsupported channel count: {ch}")
            return QPixmap.fromImage(img.copy())

        raise ValueError(f"Unsupported ndarray shape: {data.shape}")

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setPen(QPen(qcolor(theme().border_subtle), 1))
        p.drawLine(0, 0, self.width(), 0)
        p.end()
        super().paintEvent(event)


# ═══════════════════════════════════════════════════════════════
# ViewportToolbar
# ═══════════════════════════════════════════════════════════════


class ViewportToolbar(QWidget):
    """Top toolbar: Snap, Live, zoom controls, Fit, Range, zoom %."""

    snap_clicked = Signal()
    live_toggled = Signal(bool)
    fit_clicked = Signal()
    range_toggled = Signal(bool)

    TOOLBAR_HEIGHT = ROW_HEIGHT

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(self.TOOLBAR_HEIGHT)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(6)

        self._snap_btn = ToolbarButton("📷 Snap")
        self._snap_btn.clicked.connect(self.snap_clicked)
        layout.addWidget(self._snap_btn)

        # Live
        self._live_btn = ToolbarButton(
            "● Live", accent_color=qcolor(theme().status_red), checkable=True
        )
        self._live_btn.checked = False
        self._live_btn.clicked.connect(
            lambda: self.live_toggled.emit(self._live_btn.checked)
        )
        layout.addWidget(self._live_btn)

        layout.addWidget(ToolbarSep())

        # Zoom
        self._zoom_in = ToolbarButton("🔍+")
        layout.addWidget(self._zoom_in)
        self._zoom_out = ToolbarButton("🔍−")
        layout.addWidget(self._zoom_out)
        self._fit_btn = ToolbarButton("Fit")
        self._fit_btn.clicked.connect(self.fit_clicked)
        layout.addWidget(self._fit_btn)

        layout.addWidget(ToolbarSep())

        # Range indicator
        self._range_btn = ToolbarButton("◐ Range", checkable=True)
        self._range_btn.clicked.connect(
            lambda: self.range_toggled.emit(self._range_btn.checked)
        )
        layout.addWidget(self._range_btn)

        layout.addStretch()

        # Zoom percentage
        self._zoom_label = QLabel("32.08%")
        self._zoom_label.setFont(mono_font(8))
        zpal = self._zoom_label.palette()
        zpal.setColor(QPalette.ColorRole.WindowText, qcolor(theme().text_secondary))
        self._zoom_label.setPalette(zpal)
        layout.addWidget(self._zoom_label)

    @property
    def zoom_label(self) -> QLabel:
        return self._zoom_label

    @property
    def live_button(self) -> ToolbarButton:
        return self._live_btn

    @property
    def snap_button(self) -> ToolbarButton:
        return self._snap_btn

    def set_zoom_text(self, text: str) -> None:
        self._zoom_label.setText(text)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setPen(QPen(qcolor(theme().border_subtle), 1))
        p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        p.end()
        super().paintEvent(event)


# ═══════════════════════════════════════════════════════════════
# ImageViewport — the full composite widget
# ═══════════════════════════════════════════════════════════════


class ImageViewport(QWidget):
    """Complete image viewport area: toolbar + canvas + channels + filmstrip.

    Public API:
        add_snap(data, timestamp)  — add a snap to the filmstrip
        clear_snaps()              — clear all snaps
        canvas                     — access the ImagePreview
        toolbar                    — access the ViewportToolbar
        channel_strip              — access the ChannelStrip
        filmstrip                  — access the SnapFilmstrip
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        self.toolbar = ViewportToolbar()
        layout.addWidget(self.toolbar)

        # Image canvas (expands to fill)
        self.canvas = ImagePreview(mmcore=current_core(self))
        self.overlay = ViewportOverlay(self.canvas)
        layout.addWidget(self.canvas, 1)

        # Channel strip
        self.channel_strip = ChannelStrip()
        layout.addWidget(self.channel_strip)

        # Snap filmstrip
        self.filmstrip = SnapFilmstrip()
        layout.addWidget(self.filmstrip)

        # ── Wire up toolbar signals ──
        self.toolbar.live_toggled.connect(self._on_live_toggled)
        self.toolbar.snap_clicked.connect(self._on_snap_clicked)

    def clear_snaps(self) -> None:
        """Clear all snaps from the filmstrip."""
        self.filmstrip.clear()

    def _on_snap_clicked(self) -> None:
        # In real implementation, would trigger actual snap capture
        if core := current_core(self):
            if core.isSequenceRunning():
                core.stopSequenceAcquisition()
            img = core.snap()
            # timestamp in HH:MM:SS:msec format
            timestamp = (
                time.strftime("%H:%M:%S") + f":{int(time.time() * 1000) % 1000:03d}"
            )
            self.filmstrip.add_snap(img, timestamp)

    def _on_live_toggled(self, live: bool) -> None:
        # In real implementation, would start/stop live view
        if core := current_core(self):
            if live:
                core.startContinuousSequenceAcquisition()  # dummy interval
            else:
                core.stopSequenceAcquisition()
