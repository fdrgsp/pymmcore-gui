"""Image Viewport Widget."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from pymmcore_widgets import ImagePreview

from pymmcore_gui._modern_gui._utils import current_core
from pymmcore_gui._qt.QtCore import (
    QEvent,
    QObject,
    QPointF,
    QRectF,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from pymmcore_gui._qt.QtGui import (
    QBrush,
    QColor,
    QEnterEvent,
    QFont,
    QFontMetricsF,
    QImage,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPalette,
    QPen,
    QPixmap,
)
from pymmcore_gui._qt.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ._theme import mono_font, qcolor, theme, ui_font

# Viewport-specific colors not covered by the shared theme
OVERLAY = QColor(255, 255, 255, 102)  # ~40% white
OVERLAY_DIM = QColor(255, 255, 255, 64)  # ~25% white
CH_DAPI = QColor(0x44, 0x72, 0xC4)
CH_GFP = QColor(0x00, 0xCC, 0x66)
CH_CY5 = QColor(0xCC, 0x44, 0xCC)


# ═══════════════════════════════════════════════════════════════
# ViewportOverlay — transparent HUD drawn on top of the image
# ═══════════════════════════════════════════════════════════════


@dataclass
class OverlayInfo:
    """Data displayed as overlays on the viewport."""

    label: str = "Live"
    detail: str = "DAPI+GFP+Cy5"
    dimensions: str = "2048 × 2048"  # noqa: RUF001
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

    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:
        if event is not None and event.type() == QEvent.Type.Resize and obj is not None:
            self.setGeometry(obj.rect())  # type: ignore[union-attr]
        return False

    def paintEvent(self, event: QPaintEvent | None) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        ov = self.info
        t = theme()
        margin = t.sp_md

        # ── Top-left: label ──
        p.setFont(mono_font(8))
        p.setPen(OVERLAY_DIM)
        p.drawText(margin, margin + t.sp_sm, ov.label)

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
            p.drawText(int(w - margin - tw), y + t.sp_sm, line)
            y += t.sp_md

        # ── Bottom-left: scale bar ──
        sb_x = margin
        sb_y = h - margin - t.scaled(18)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(255, 255, 255, 204)))
        bar_h = t.scaled(3)
        p.drawRoundedRect(
            QRectF(sb_x, sb_y, ov.scale_bar_px, bar_h), bar_h / 2, bar_h / 2
        )

        p.setFont(mono_font(8))
        p.setPen(OVERLAY)
        p.drawText(sb_x, sb_y + t.sp_md, f"{ov.scale_bar_um:.0f} μm")

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

    _BASE_HEIGHT = 22

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
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        t = theme()
        fm = QFontMetricsF(ui_font(10, QFont.Weight.Medium))
        w = int(fm.horizontalAdvance(self._name)) + t.sp_xl
        return QSize(w, t.scaled(self._BASE_HEIGHT))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, val: bool) -> None:
        self._active = val
        self.update()

    def paintEvent(self, event: QPaintEvent | None) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(0, 0, -1, -1)
        t = theme()

        if self._active:
            p.setPen(QPen(qcolor(t.border_default), 1))
        else:
            p.setPen(QPen(Qt.PenStyle.NoPen))

        if self._hovered:
            p.setBrush(QBrush(qcolor(t.bg_hover)))
        else:
            p.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        p.drawRoundedRect(QRectF(r), t.radius, t.radius)

        # Dot
        dot_r = t.scaled(3)
        dot_x = t.sp_xs
        dot_y = r.height() / 2
        dot_color = QColor(self._color) if self._active else qcolor(t.text_disabled)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(dot_color))
        p.drawEllipse(QPointF(dot_x, dot_y), dot_r, dot_r)

        # Name
        text_color = qcolor(t.text_primary if self._active else t.text_secondary)
        p.setPen(text_color)
        p.setFont(ui_font(10, QFont.Weight.Medium))
        p.drawText(
            QRectF(
                dot_x + dot_r + t.scaled(5),
                0,
                r.width() - dot_x - dot_r - t.scaled(10),
                r.height(),
            ),
            Qt.AlignmentFlag.AlignVCenter,
            self._name,
        )
        p.end()

    def enterEvent(self, event: QEnterEvent | None) -> None:
        self._hovered = True
        self.update()

    def leaveEvent(self, event: QEvent | None) -> None:
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

    _BASE_HEIGHT = 28

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        t = theme()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(t.sp_sm, t.scaled(3), t.sp_sm, t.scaled(3))
        layout.setSpacing(t.scaled(2))

        self._channels: dict[str, ChannelButton] = {}

        for name, color in [
            ("DAPI", CH_DAPI),
            ("GFP", CH_GFP),
            ("Cy5", CH_CY5),
        ]:
            btn = ChannelButton(name, color)
            layout.addWidget(btn)
            self._channels[name] = btn

        layout.addStretch()

        self._merge_btn = QPushButton("Merge")
        self._merge_btn.setCheckable(True)
        self._merge_btn.setChecked(True)
        self._merge_btn.setFont(ui_font(10, QFont.Weight.Medium))
        layout.addWidget(self._merge_btn)

    def sizeHint(self) -> QSize:
        return QSize(super().sizeHint().width(), theme().scaled(self._BASE_HEIGHT))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def changeEvent(self, event: QEvent | None) -> None:
        if event is not None and event.type() == QEvent.Type.StyleChange:
            t = theme()
            if lay := self.layout():
                lay.setContentsMargins(t.sp_sm, t.scaled(3), t.sp_sm, t.scaled(3))
                lay.setSpacing(t.scaled(2))
        super().changeEvent(event)

    def paintEvent(self, event: QPaintEvent | None) -> None:
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

    _BASE_THUMB = 52
    _BASE_LABEL_H = 14

    clicked = Signal()

    def __init__(
        self,
        pixmap: QPixmap,
        timestamp: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._original_pixmap = pixmap
        self._pixmap: QPixmap | None = None
        self._timestamp = timestamp
        self._hovered = False
        self._active = False
        self._starred = False

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._rescale_pixmap()

    def _rescale_pixmap(self) -> None:
        s = theme().scaled(self._BASE_THUMB)
        self._pixmap = self._original_pixmap.scaled(
            s,
            s,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )

    def sizeHint(self) -> QSize:
        t = theme()
        s = t.scaled(self._BASE_THUMB)
        return QSize(s, s + t.scaled(self._BASE_LABEL_H))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def changeEvent(self, event: QEvent | None) -> None:
        if event is not None and event.type() == QEvent.Type.StyleChange:
            self._rescale_pixmap()
        super().changeEvent(event)

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, val: bool) -> None:
        self._active = val
        self.update()

    def paintEvent(self, event: QPaintEvent | None) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        t = theme()
        s = t.scaled(self._BASE_THUMB)

        # Border
        if self._active:
            p.setPen(QPen(qcolor(t.accent), 2))
        elif self._hovered:
            p.setPen(QPen(qcolor(t.border_default), 2))
        else:
            p.setPen(QPen(Qt.PenStyle.NoPen))

        p.setBrush(QBrush(qcolor(t.bg_raised)))
        p.drawRoundedRect(QRectF(0, 0, s, s), t.radius, t.radius)

        # Image
        if self._pixmap is not None:
            p.setClipRect(QRectF(1, 1, s - 2, s - 2))
            p.drawPixmap(1, 1, self._pixmap)
        p.setClipping(False)

        # Timestamp bar
        ts_h = t.sp_sm
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
            star_sz = t.sp_sm
            p.drawText(
                QRectF(s - star_sz - 2, 1, star_sz, star_sz),
                Qt.AlignmentFlag.AlignCenter,
                "★",
            )

        p.end()

    def enterEvent(self, event: QEnterEvent | None) -> None:
        self._hovered = True
        self.update()

    def leaveEvent(self, event: QEvent | None) -> None:
        self._hovered = False
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            t = theme()
            s = t.scaled(self._BASE_THUMB)
            star_sz = t.sp_sm
            star_rect = QRectF(s - star_sz - 2, 1, star_sz + 2, star_sz + 2)
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
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)

    def _font(self) -> QFont:
        return ui_font(11, QFont.Weight.DemiBold)

    def sizeHint(self) -> QSize:
        fm = QFontMetricsF(self._font())
        w = int(fm.height()) + 4
        h = int(fm.horizontalAdvance(self._text)) + 8
        return QSize(w, h)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event: QPaintEvent | None) -> None:
        font = self._font()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setFont(font)
        p.setPen(qcolor(theme().text_disabled))
        p.translate(self.width() / 2, self.height() / 2)
        p.rotate(-90)
        fm = QFontMetricsF(font)
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

    _BASE_STRIP_HEIGHT = 72

    snap_selected = Signal(int)  # index

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        t = theme()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(t.sp_sm, 0, t.sp_sm, 0)
        outer.setSpacing(t.sp_xs)

        # Rotated label
        self._label = _RotatedLabel("SNAPS")
        outer.addWidget(self._label)

        # Scroll area for thumbnails
        self._scroll = QScrollArea()
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFixedHeight(t.scaled(self._BASE_STRIP_HEIGHT))
        outer.addWidget(self._scroll, 1)

        # Inner widget with horizontal layout
        self._container = QWidget()
        self._thumb_layout = QHBoxLayout(self._container)
        self._thumb_layout.setContentsMargins(0, t.scaled(6), 0, 0)
        self._thumb_layout.setSpacing(t.scaled(6))
        self._thumb_layout.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._scroll.setWidget(self._container)

        self._thumbnails: list[SnapThumbnail] = []

        # Clear button
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setProperty("variant", "danger")
        self._clear_btn.setFont(ui_font(10, QFont.Weight.Medium))
        self._clear_btn.setFixedHeight(t.scaled(22))
        self._clear_btn.clicked.connect(self.clear)
        outer.addWidget(self._clear_btn, 0, Qt.AlignmentFlag.AlignVCenter)

    def sizeHint(self) -> QSize:
        return QSize(
            super().sizeHint().width(),
            theme().scaled(self._BASE_STRIP_HEIGHT),
        )

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def changeEvent(self, event: QEvent | None) -> None:
        if event is not None and event.type() == QEvent.Type.StyleChange:
            t = theme()
            self._scroll.setFixedHeight(t.scaled(self._BASE_STRIP_HEIGHT))
            if lay := self.layout():
                lay.setContentsMargins(t.sp_sm, 0, t.sp_sm, 0)
                lay.setSpacing(t.sp_xs)
            self._thumb_layout.setContentsMargins(0, t.scaled(6), 0, 0)
            self._thumb_layout.setSpacing(t.scaled(6))
        super().changeEvent(event)

    def add_snap(self, data: np.ndarray, timestamp: str | None = None) -> None:
        """Add a snap image to the filmstrip."""
        if timestamp is None:
            timestamp = time.strftime("%H:%M:%S")

        pixmap = self._ndarray_to_pixmap(data)
        idx = len(self._thumbnails)
        thumb = SnapThumbnail(pixmap, timestamp)
        thumb.clicked.connect(lambda _idx=idx: self._on_thumb_clicked(_idx))
        self._thumb_layout.addWidget(thumb)
        self._thumbnails.append(thumb)

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
            if data.dtype != np.uint8:
                d = data.astype(np.float32)
                lo, hi = d.min(), d.max()
                if hi > lo:
                    d = ((d - lo) / (hi - lo) * 255).astype(np.uint8)
                else:
                    d = np.zeros_like(data, dtype=np.uint8)
            else:
                d = data
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

    def paintEvent(self, event: QPaintEvent | None) -> None:
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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        t = theme()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(t.sp_sm, t.scaled(4), t.sp_sm, t.scaled(4))
        layout.setSpacing(t.scaled(6))

        btn_font = ui_font(10, QFont.Weight.Medium)

        self._snap_btn = QPushButton("📷 Snap")
        self._snap_btn.setFont(btn_font)
        self._snap_btn.clicked.connect(self.snap_clicked)
        layout.addWidget(self._snap_btn)

        self._live_btn = QPushButton("● Live")
        self._live_btn.setFont(btn_font)
        self._live_btn.setCheckable(True)
        self._live_btn.setProperty("accent", qcolor(t.status_red))
        self._live_btn.clicked.connect(
            lambda: self.live_toggled.emit(self._live_btn.isChecked())
        )
        layout.addWidget(self._live_btn)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        layout.addWidget(sep1)

        self._zoom_in = QPushButton("🔍+")
        self._zoom_in.setFont(btn_font)
        layout.addWidget(self._zoom_in)
        self._zoom_out = QPushButton("🔍-")
        self._zoom_out.setFont(btn_font)
        layout.addWidget(self._zoom_out)
        self._fit_btn = QPushButton("Fit")
        self._fit_btn.setFont(btn_font)
        self._fit_btn.clicked.connect(self.fit_clicked)
        layout.addWidget(self._fit_btn)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        layout.addWidget(sep2)

        self._range_btn = QPushButton("◐ Range")
        self._range_btn.setFont(btn_font)
        self._range_btn.setCheckable(True)
        self._range_btn.clicked.connect(
            lambda: self.range_toggled.emit(self._range_btn.isChecked())
        )
        layout.addWidget(self._range_btn)

        layout.addStretch()

        self._zoom_label = QLabel("32.08%")
        self._zoom_label.setFont(mono_font(8))
        zpal = self._zoom_label.palette()
        zpal.setColor(QPalette.ColorRole.WindowText, qcolor(t.text_secondary))
        self._zoom_label.setPalette(zpal)
        layout.addWidget(self._zoom_label)

    def sizeHint(self) -> QSize:
        return QSize(super().sizeHint().width(), theme().row_height)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def changeEvent(self, event: QEvent | None) -> None:
        if event is not None and event.type() == QEvent.Type.StyleChange:
            t = theme()
            if lay := self.layout():
                lay.setContentsMargins(t.sp_sm, t.scaled(4), t.sp_sm, t.scaled(4))
                lay.setSpacing(t.scaled(6))
            self._zoom_label.setFont(mono_font(8))
        super().changeEvent(event)

    @property
    def zoom_label(self) -> QLabel:
        return self._zoom_label

    @property
    def live_button(self) -> QPushButton:
        return self._live_btn

    @property
    def snap_button(self) -> QPushButton:
        return self._snap_btn

    def set_zoom_text(self, text: str) -> None:
        self._zoom_label.setText(text)

    def paintEvent(self, event: QPaintEvent | None) -> None:
        p = QPainter(self)
        p.setPen(QPen(qcolor(theme().border_subtle), 1))
        p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        p.end()
        super().paintEvent(event)


# ═══════════════════════════════════════════════════════════════
# ImageViewport — the full composite widget
# ═══════════════════════════════════════════════════════════════


class ImageViewport(QWidget):
    """Complete image viewport area: toolbar + canvas + channels + filmstrip."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.toolbar = ViewportToolbar()
        layout.addWidget(self.toolbar)

        self.canvas = ImagePreview(mmcore=current_core(self))
        self.overlay = ViewportOverlay(self.canvas)
        layout.addWidget(self.canvas, 1)

        self.channel_strip = ChannelStrip()
        layout.addWidget(self.channel_strip)

        self.filmstrip = SnapFilmstrip()
        layout.addWidget(self.filmstrip)

        self.toolbar.live_toggled.connect(self._on_live_toggled)
        self.toolbar.snap_clicked.connect(self._on_snap_clicked)

    def clear_snaps(self) -> None:
        self.filmstrip.clear()

    def _on_snap_clicked(self) -> None:
        if core := current_core(self):
            if core.isSequenceRunning():
                core.stopSequenceAcquisition()
            img = core.snap()
            timestamp = (
                time.strftime("%H:%M:%S") + f":{int(time.time() * 1000) % 1000:03d}"
            )
            self.filmstrip.add_snap(img, timestamp)

    def _on_live_toggled(self, live: bool) -> None:
        if core := current_core(self):
            if live:
                core.startContinuousSequenceAcquisition()
            else:
                core.stopSequenceAcquisition()
