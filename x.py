"""
Collapsible Panel Sidebar — Qt Implementation.

Scrollable left panel with collapsible sections, matching the HTML mockup.
No QSS. Only QPalette + QProxyStyle(Fusion) + custom QWidget painting.

Usage:
    python collapsible_panels.py
"""

from __future__ import annotations

import sys
from enum import Enum, auto

from PyQt6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRect,
    Qt,
    pyqtProperty,
)
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QFontMetrics,
    QPainter,
    QPalette,
    QPen,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProxyStyle,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStyleOption,
    QVBoxLayout,
    QWidget,
)

# ═══════════════════════════════════════════════════════════════════
# Design Tokens
# ═══════════════════════════════════════════════════════════════════
SP_EXPANDING = QSizePolicy.Policy.Expanding
SP_PREFERRED = QSizePolicy.Policy.Preferred
SP_MAXIMUM = QSizePolicy.Policy.Maximum


class Clr:
    """Color tokens — matches the HTML mockup exactly."""

    BG_DEEPEST = QColor(0x12, 0x12, 0x12)
    BG_BASE = QColor(0x1E, 0x1E, 0x1E)
    BG_RAISED = QColor(0x25, 0x25, 0x25)
    BG_SURFACE = QColor(0x2D, 0x2D, 0x2D)
    BG_HOVER = QColor(0x35, 0x35, 0x35)
    BG_ACTIVE = QColor(0x40, 0x40, 0x40)

    TEXT_PRIMARY = QColor(0xE0, 0xE0, 0xE0)
    TEXT_SECONDARY = QColor(0xA0, 0xA0, 0xA0)
    TEXT_DISABLED = QColor(0x70, 0x70, 0x70)

    BORDER_SUBTLE = QColor(0x33, 0x33, 0x33)
    BORDER_DEFAULT = QColor(0x44, 0x44, 0x44)
    BORDER_FOCUS = QColor(0x4A, 0x9E, 0xFF)

    ACCENT = QColor(0x4A, 0x9E, 0xFF)
    ACCENT_MUTED = QColor(0x4A, 0x9E, 0xFF, 0x26)  # ~15%

    GREEN = QColor(0x4C, 0xAF, 0x50)
    RED = QColor(0xEF, 0x53, 0x50)
    AMBER = QColor(0xFF, 0xA7, 0x26)


class Sp:
    """Spacing tokens (px). 4px base grid."""

    XXS = 4
    XS = 8
    SM = 12
    MD = 16
    LG = 24
    XL = 32


RADIUS = 3
RADIUS_LG = 6
SIDEBAR_W = 300


# ═══════════════════════════════════════════════════════════════════
# Fonts
# ═══════════════════════════════════════════════════════════════════


def ui_font(size_pt: float = 10, weight: int = QFont.Weight.Normal) -> QFont:
    """System UI font."""
    f = QFont()
    f.setPointSizeF(size_pt)
    f.setWeight(weight)
    return f


def mono_font(size_pt: float = 10, weight: int = QFont.Weight.Normal) -> QFont:
    """Monospace font — try JetBrains Mono, fall back to system mono."""
    families = ["JetBrains Mono", "SF Mono", "Cascadia Code", "Fira Code", "Consolas"]
    for fam in families:
        if fam in QFontDatabase.families():
            f = QFont(fam)
            f.setPointSizeF(size_pt)
            f.setWeight(weight)
            return f
    f = QFont()
    f.setStyleHint(QFont.StyleHint.Monospace)
    f.setPointSizeF(size_pt)
    f.setWeight(weight)
    return f


# ═══════════════════════════════════════════════════════════════════
# QPalette
# ═══════════════════════════════════════════════════════════════════


def make_dark_palette() -> QPalette:
    p = QPalette()

    p.setColor(QPalette.ColorRole.Window, Clr.BG_BASE)
    p.setColor(QPalette.ColorRole.WindowText, Clr.TEXT_PRIMARY)
    p.setColor(QPalette.ColorRole.Base, Clr.BG_DEEPEST)
    p.setColor(QPalette.ColorRole.AlternateBase, Clr.BG_RAISED)
    p.setColor(QPalette.ColorRole.Button, Clr.BG_SURFACE)
    p.setColor(QPalette.ColorRole.ButtonText, Clr.TEXT_PRIMARY)
    p.setColor(QPalette.ColorRole.Highlight, Clr.ACCENT)
    p.setColor(QPalette.ColorRole.HighlightedText, Clr.TEXT_PRIMARY)
    p.setColor(QPalette.ColorRole.ToolTipBase, Clr.BG_RAISED)
    p.setColor(QPalette.ColorRole.ToolTipText, Clr.TEXT_PRIMARY)
    p.setColor(QPalette.ColorRole.PlaceholderText, Clr.TEXT_DISABLED)
    p.setColor(QPalette.ColorRole.BrightText, Clr.TEXT_PRIMARY)
    p.setColor(QPalette.ColorRole.Link, Clr.ACCENT)
    p.setColor(QPalette.ColorRole.Mid, Clr.BORDER_DEFAULT)
    p.setColor(QPalette.ColorRole.Dark, Clr.BG_DEEPEST)
    p.setColor(QPalette.ColorRole.Midlight, Clr.BG_HOVER)
    p.setColor(QPalette.ColorRole.Shadow, QColor(0, 0, 0))
    p.setColor(QPalette.ColorRole.Light, Clr.BG_HOVER)

    # Disabled
    p.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.WindowText,
        Clr.TEXT_DISABLED,
    )
    p.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.ButtonText,
        Clr.TEXT_DISABLED,
    )

    return p


# ═══════════════════════════════════════════════════════════════════
# QProxyStyle — wraps Fusion, tweaks geometry and painting
# ═══════════════════════════════════════════════════════════════════


class MicroscopeStyle(QProxyStyle):
    """Thin proxy over Fusion. Overrides only what we need.

    - Global border-radius → 3px
    - Scrollbar width → 5px, minimal styling
    - Frame painting for our custom containers.
    """

    def __init__(self) -> None:
        super().__init__("Fusion")

    def pixelMetric(
        self,
        metric: QStyle.PixelMetric,
        option: QStyleOption = None,
        widget: QWidget = None,
    ) -> int:
        if metric == QStyle.PixelMetric.PM_ScrollBarExtent:
            return 5
        if metric == QStyle.PixelMetric.PM_ScrollBarSliderMin:
            return 20
        if metric == QStyle.PixelMetric.PM_LayoutHorizontalSpacing:
            return Sp.SM
        if metric == QStyle.PixelMetric.PM_LayoutVerticalSpacing:
            return Sp.XXS
        return super().pixelMetric(metric, option, widget)

    def drawPrimitive(
        self,
        element: QStyle.PrimitiveElement,
        option: QStyleOption,
        painter: QPainter,
        widget: QWidget = None,
    ) -> None:
        # Suppress default frame drawing for scroll areas — we draw our own
        if element == QStyle.PrimitiveElement.PE_Frame and isinstance(
            widget, QScrollArea
        ):
            return
        super().drawPrimitive(element, option, painter, widget)


# ═══════════════════════════════════════════════════════════════════
# DeviceStatus — the little colored dot
# ═══════════════════════════════════════════════════════════════════


class DeviceStatus(Enum):
    CONNECTED = auto()
    DISCONNECTED = auto()
    BUSY = auto()
    ERROR = auto()


# ═══════════════════════════════════════════════════════════════════
# CollapsiblePanelHeader — custom-painted clickable header
# ═══════════════════════════════════════════════════════════════════


class CollapsiblePanelHeader(QWidget):
    """
    Custom-painted header bar for a collapsible panel.

    Draws: chevron (▶ / ▼), status dot, title, summary text.
    All via QPainter — no child widgets, no QSS.
    """

    HEADER_HEIGHT = 36

    def __init__(
        self,
        title: str,
        summary: str = "",
        status: DeviceStatus = DeviceStatus.CONNECTED,
        show_status_dot: bool = True,
        parent: QWidget = None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._summary = summary
        self._status = status
        self._show_dot = show_status_dot
        self._expanded = False
        self._hovered = False
        self._chevron_angle = 0.0  # 0 = collapsed (right), 90 = expanded (down)

        self.setFixedHeight(self.HEADER_HEIGHT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

        # Chevron animation
        self._chevron_anim = QPropertyAnimation(self, b"chevronAngle")
        self._chevron_anim.setDuration(200)
        self._chevron_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    # -- Properties for animation --

    def _get_chevron_angle(self) -> float:
        return self._chevron_angle

    def _set_chevron_angle(self, val: float) -> None:
        self._chevron_angle = val
        self.update()

    chevronAngle = pyqtProperty(float, _get_chevron_angle, _set_chevron_angle)

    # -- Public API --

    @property
    def expanded(self) -> bool:
        return self._expanded

    @expanded.setter
    def expanded(self, val: bool) -> None:
        self._expanded = val
        self._chevron_anim.stop()
        self._chevron_anim.setStartValue(self._chevron_angle)
        self._chevron_anim.setEndValue(90.0 if val else 0.0)
        self._chevron_anim.start()

    @property
    def summary(self) -> str:
        return self._summary

    @summary.setter
    def summary(self, val: str) -> None:
        self._summary = val
        self.update()

    @property
    def status(self) -> DeviceStatus:
        return self._status

    @status.setter
    def status(self, val: DeviceStatus) -> None:
        self._status = val
        self.update()

    # -- Events --

    def enterEvent(self, event) -> None:
        self._hovered = True
        self.update()

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self.update()

    # -- Painting --

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()

        # Background
        bg = Clr.BG_RAISED if self._hovered else Clr.BG_BASE
        p.fillRect(0, 0, w, h, bg)

        # Bottom border
        p.setPen(QPen(Clr.BORDER_SUBTLE, 1))
        p.drawLine(0, h - 1, w, h - 1)

        x = Sp.SM  # running x cursor

        # ── Chevron ──
        p.save()
        p.translate(x + 6, h / 2)
        p.rotate(self._chevron_angle)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(Clr.TEXT_DISABLED)
        # Small right-pointing triangle
        tri = [
            (-3, -4),
            (4, 0),
            (-3, 4),
        ]
        from PyQt6.QtCore import QPointF
        from PyQt6.QtGui import QPolygonF

        poly = QPolygonF([QPointF(tx, ty) for tx, ty in tri])
        p.drawPolygon(poly)
        p.restore()
        x += 12 + Sp.XS

        # ── Status dot ──
        if self._show_dot:
            dot_color = {
                DeviceStatus.CONNECTED: Clr.GREEN,
                DeviceStatus.DISCONNECTED: Clr.RED,
                DeviceStatus.BUSY: Clr.AMBER,
                DeviceStatus.ERROR: Clr.RED,
            }.get(self._status, Clr.TEXT_DISABLED)

            dot_r = 3.5
            dot_cx = x + dot_r
            dot_cy = h / 2

            # Glow
            p.setPen(Qt.PenStyle.NoPen)
            glow = QColor(dot_color)
            glow.setAlphaF(0.3)
            p.setBrush(glow)
            p.drawEllipse(QPointF(dot_cx, dot_cy), dot_r + 2, dot_r + 2)

            # Dot
            p.setBrush(dot_color)
            p.drawEllipse(QPointF(dot_cx, dot_cy), dot_r, dot_r)

            x += int(dot_r * 2) + Sp.XS

        # ── Title ──
        title_font = ui_font(10, QFont.Weight.DemiBold)
        p.setFont(title_font)
        p.setPen(Clr.TEXT_PRIMARY)
        title_rect = QRect(x, 0, w - x - Sp.SM, h)
        fm = QFontMetrics(title_font)
        title_width = fm.horizontalAdvance(self._title)
        p.drawText(title_rect, Qt.AlignmentFlag.AlignVCenter, self._title)

        # ── Summary (shown when collapsed, fades based on chevron angle) ──
        if self._summary:
            # Opacity: 1 when chevron=0 (collapsed), 0 when chevron=90 (expanded)
            opacity = 1.0 - (self._chevron_angle / 90.0)
            if opacity > 0.01:
                summary_font = mono_font(8)
                p.setFont(summary_font)
                summary_color = QColor(Clr.TEXT_SECONDARY)
                summary_color.setAlphaF(opacity)
                p.setPen(summary_color)

                summary_x = x + title_width + Sp.SM
                summary_rect = QRect(summary_x, 0, w - summary_x - Sp.SM, h)
                p.drawText(
                    summary_rect,
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                    self._summary,
                )

        p.end()


# ═══════════════════════════════════════════════════════════════════
# CollapsiblePanel — header + animated body
# ═══════════════════════════════════════════════════════════════════


class CollapsiblePanel(QWidget):
    """A collapsible panel with an animated body."""

    HEADER_H = CollapsiblePanelHeader.HEADER_HEIGHT

    def __init__(
        self,
        title: str,
        summary: str = "",
        status: DeviceStatus = DeviceStatus.CONNECTED,
        show_status_dot: bool = True,
        expanded: bool = False,
        parent: QWidget = None,
    ) -> None:
        super().__init__(parent)

        self._expanded = expanded

        # Main layout — zero margins, zero spacing
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        self._header = CollapsiblePanelHeader(
            title=title,
            summary=summary,
            status=status,
            show_status_dot=show_status_dot,
        )
        self._header.mousePressEvent = self._on_header_click
        layout.addWidget(self._header)

        # Body container
        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(Sp.SM, Sp.XS, Sp.SM, Sp.SM)
        self._body_layout.setSpacing(Sp.SM)
        layout.addWidget(self._body)

        # Animation targets the panel's own fixedHeight.
        # This avoids touching body constraints during animation, sidestepping
        # layout-propagation glitches through the sizeHint chain.
        self._anim = QPropertyAnimation(self, b"panelHeight")
        self._anim.setDuration(250)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.finished.connect(self._on_anim_finished)

        if not expanded:
            self._body.setMaximumHeight(0)

        # Set initial state
        if expanded:
            self._header.expanded = True

        self.setSizePolicy(SP_PREFERRED, SP_MAXIMUM)

    # -- Animated property: panel's own fixed height --

    def _get_panel_height(self) -> int:
        return self.height()

    def _set_panel_height(self, h: int) -> None:
        self.setFixedHeight(h)

    panelHeight = pyqtProperty(int, _get_panel_height, _set_panel_height)

    @property
    def body_layout(self) -> QVBoxLayout:
        return self._body_layout

    @property
    def header(self) -> CollapsiblePanelHeader:
        return self._header

    @property
    def expanded(self) -> bool:
        return self._expanded

    @expanded.setter
    def expanded(self, val: bool) -> None:
        if val == self._expanded:
            return
        self._expanded = val
        self._header.expanded = val
        self._animate(val)

    def toggle(self) -> None:
        self.expanded = not self._expanded

    def _on_header_click(self, event) -> None:
        self.toggle()

    def _animate(self, expanding: bool) -> None:
        self._anim.stop()
        current_h = self.height()

        if expanding:
            # Unconstrain body so we can measure the natural panel height
            self._body.setMaximumHeight(16777215)
            target = self.HEADER_H + self._body.sizeHint().height()
            # Lock at current collapsed height, then animate to target
            self.setFixedHeight(current_h)
            self._anim.setStartValue(current_h)
            self._anim.setEndValue(target)
        else:
            # Lock at current expanded height, then animate to header-only
            self.setFixedHeight(current_h)
            self._anim.setStartValue(current_h)
            self._anim.setEndValue(self.HEADER_H)

        self._anim.start()

    def _on_anim_finished(self) -> None:
        # Remove fixed-height constraint so the layout manages us normally
        self.setMinimumHeight(0)
        self.setMaximumHeight(16777215)
        if not self._expanded:
            self._body.setMaximumHeight(0)


# ═══════════════════════════════════════════════════════════════════
# Placeholder content widgets (just labels for now)
# ═══════════════════════════════════════════════════════════════════


class PlaceholderContent(QWidget):
    """Simple placeholder showing lines of text to represent panel content."""

    def __init__(self, lines: list[str], height: int = 80, parent: QWidget = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Sp.XXS)
        for line in lines:
            lbl = QLabel(line)
            lbl.setFont(mono_font(8))
            # Use palette for color — no stylesheet
            pal = lbl.palette()
            pal.setColor(QPalette.ColorRole.WindowText, Clr.TEXT_SECONDARY)
            lbl.setPalette(pal)
            layout.addWidget(lbl)


# ═══════════════════════════════════════════════════════════════════
# Sidebar assembly
# ═══════════════════════════════════════════════════════════════════


class Sidebar(QWidget):
    """Scrollable sidebar containing collapsible panels."""

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(SIDEBAR_W)

        # Scroll content
        content = QWidget()
        self._layout = QVBoxLayout(content)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)

        # Outer layout holds the scroll area
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(scroll)

        # ── Positioning ──
        pos = CollapsiblePanel(
            title="Positioning",
            summary="74.4, -147.6, 12.3",
            expanded=True,
        )
        pos.body_layout.addWidget(
            PlaceholderContent(
                [
                    "X:    74.40  μm",
                    "Y:  -147.60  μm",
                    "Z:    12.34  μm",
                    "",
                    "Step: [0.1] [1] [10] [100] [1k]",
                    "",
                    "┌─── Joystick ───┐",
                    "│                │",
                    "│       +        │",
                    "│                │",
                    "└────────────────┘",
                ]
            )
        )
        self._layout.addWidget(pos)

        # ── Objective ──
        obj = CollapsiblePanel(
            title="Objective",
            summary="40× Oil 1.30",
        )
        obj.body_layout.addWidget(
            PlaceholderContent(
                [
                    "[4×] [10×] [20×] [40× Oil] [63×] [100×]",
                    "Pixel size: 0.162 μm",
                ]
            )
        )
        self._layout.addWidget(obj)

        # ── Camera ──
        cam = CollapsiblePanel(
            title="Camera",
            summary="100 ms · 1×1",
        )
        cam.body_layout.addWidget(
            PlaceholderContent(
                [
                    "Exposure:  100 ms",
                    "Gain:      1.0",
                    "Binning:   [1×1] [2×2] [4×4]",
                    "Format:    2048 × 2048 · 16 bit",
                ]
            )
        )
        self._layout.addWidget(cam)

        # ── Channels ──
        ch = CollapsiblePanel(
            title="Channels",
            summary="DAPI · GFP · Cy5",
        )
        ch.body_layout.addWidget(
            PlaceholderContent(
                [
                    "■ DAPI    50 ms   👁",
                    "■ GFP    100 ms   👁",
                    "■ Cy5    200 ms   👁",
                ]
            )
        )
        self._layout.addWidget(ch)

        # ── Histogram ──
        hist = CollapsiblePanel(
            title="Histogram",
            summary="0 – 4095",
            show_status_dot=False,
        )
        hist.body_layout.addWidget(
            PlaceholderContent(
                [
                    "┌──── histogram ────┐",
                    "│▓▓░░░░░░░░░░░░░░░░│",
                    "└──────────────────-┘",
                    "[⚡ Auto] [↺ Reset] [㏒] [◐]",
                ]
            )
        )
        self._layout.addWidget(hist)

        # ── Acquisition ──
        acq = CollapsiblePanel(
            title="Acquisition",
            summary="Z×50 · T×100",
            show_status_dot=False,
        )
        acq.body_layout.addWidget(
            PlaceholderContent(
                [
                    "☑ Z-Stack      50 sl · 0.5 μm",
                    "☑ Time Series  100 × 30 s",
                    "☐ Tile Scan    —",
                    "☐ Positions    —",
                    "",
                    "Frames:    15,000",
                    "Duration:  50 min",
                    "Storage:   11.2 GB",
                ]
            )
        )
        self._layout.addWidget(acq)

    def paintEvent(self, event) -> None:
        """Draw right border."""
        p = QPainter(self)
        p.setPen(QPen(Clr.BORDER_SUBTLE, 1))
        p.drawLine(self.width() - 1, 0, self.width() - 1, self.height())
        p.end()


# ═══════════════════════════════════════════════════════════════════
# Main Window
# ═══════════════════════════════════════════════════════════════════


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Microscope Control — Panel Mockup")
        self.resize(900, 700)

        central = QWidget()
        self.setCentralWidget(central)

        # Viewport placeholder
        viewport = QWidget()
        viewport.setSizePolicy(SP_EXPANDING, SP_EXPANDING)

        # Paint the viewport area black
        pal = viewport.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(0, 0, 0))
        viewport.setPalette(pal)
        viewport.setAutoFillBackground(True)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(Sidebar())
        layout.addWidget(viewport)


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    app = QApplication(sys.argv)

    # Apply our style and palette
    app.setStyle(MicroscopeStyle())
    app.setPalette(make_dark_palette())

    # Global font
    app.setFont(ui_font(10))

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
