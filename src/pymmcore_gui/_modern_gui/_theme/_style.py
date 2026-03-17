"""
MicroscopeStyle — QProxyStyle over Fusion.

Makes standard Qt widgets match the design system automatically.
Contributors use QPushButton, QSlider, QCheckBox, etc. normally.
No QSS. No custom widget subclasses required.

Button variants via dynamic property:
    btn = QPushButton("Snap")                              # → Ghost (default)
    btn.setProperty("variant", "subtle")                   # → Subtle
    btn.setProperty("variant", "primary")                  # → Primary
    btn.setProperty("variant", "danger")                   # → Danger
    btn.setProperty("accent", QColor("#4CAF50"))           # → Custom accent

Convenience helpers (optional, not required):
    set_variant(btn, "primary")
    set_variant(btn, "danger")

Usage:
    app.setStyle(MicroscopeStyle())
    app.setPalette(make_dark_palette())
"""

from __future__ import annotations

from pymmcore_gui._qt.QtCore import QPointF, QRect, QRectF, QSize, Qt
from pymmcore_gui._qt.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
)
from pymmcore_gui._qt.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QFrame,
    QProxyStyle,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStyleOption,
    QStyleOptionButton,
    QStyleOptionComboBox,
    QStyleOptionComplex,
    QStyleOptionSlider,
    QStyleOptionSpinBox,
    QWidget,
)

# ═══════════════════════════════════════════════════════════════
# Color helpers — these read from the widget's palette so they
# work with both light and dark themes automatically.
# ═══════════════════════════════════════════════════════════════


def _with_alpha(color: QColor, alpha: int) -> QColor:
    c = QColor(color)
    c.setAlpha(alpha)
    return c


def _lerp_color(a: QColor, b: QColor, t: float) -> QColor:
    """Linearly interpolate between two colors."""
    return QColor(
        int(a.red() + (b.red() - a.red()) * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue() + (b.blue() - a.blue()) * t),
        int(a.alpha() + (b.alpha() - a.alpha()) * t),
    )


# ═══════════════════════════════════════════════════════════════
# Button variant resolution
# ═══════════════════════════════════════════════════════════════


def _get_variant(widget: QWidget | None) -> str:
    """Read the 'variant' dynamic property, defaulting to 'ghost'."""
    if widget is None:
        return "ghost"
    v = widget.property("variant")
    if isinstance(v, str) and v in ("ghost", "subtle", "primary", "danger"):
        return v
    return "ghost"


def _get_accent(widget: QWidget | None) -> QColor:
    """Read optional 'accent' dynamic property, else use palette highlight."""
    if widget is not None:
        a = widget.property("accent")
        if isinstance(a, QColor):
            return a
    pal = widget.palette() if widget else QPalette()
    return pal.color(QPalette.ColorRole.Highlight)


def _button_colors(
    opt: QStyleOptionButton,
    widget: QWidget | None,
) -> tuple[QColor, QColor, QColor]:
    """Resolve (background, border, text) for a QPushButton.

    Uses the widget's palette + variant property. Returns colors that
    work for both light and dark themes because they're derived from
    palette roles, not hardcoded hex values.
    """
    pal = opt.palette
    variant = _get_variant(widget)
    accent = _get_accent(widget)

    hovered = bool(opt.state & QStyle.StateFlag.State_MouseOver)
    pressed = bool(opt.state & QStyle.StateFlag.State_Sunken)
    checked = bool(opt.state & QStyle.StateFlag.State_On)
    enabled = bool(opt.state & QStyle.StateFlag.State_Enabled)

    if not enabled:
        return (
            QColor(0, 0, 0, 0),
            pal.color(QPalette.ColorRole.Mid),
            pal.color(QPalette.ColorRole.PlaceholderText),
        )

    transparent = QColor(0, 0, 0, 0)
    muted_text = _lerp_color(
        pal.color(QPalette.ColorRole.WindowText),
        pal.color(QPalette.ColorRole.PlaceholderText),
        0.4,
    )

    match variant:
        case "ghost":
            if checked:
                return (_with_alpha(accent, 38), _with_alpha(accent, 76), accent)
            if pressed:
                return (_with_alpha(accent, 50), _with_alpha(accent, 76), accent)
            if hovered:
                return (
                    pal.color(QPalette.ColorRole.Midlight),
                    transparent,
                    pal.color(QPalette.ColorRole.WindowText),
                )
            return (transparent, transparent, muted_text)

        case "subtle":
            if checked:
                return (_with_alpha(accent, 38), _with_alpha(accent, 76), accent)
            if pressed:
                return (
                    pal.color(QPalette.ColorRole.Dark),
                    pal.color(QPalette.ColorRole.Mid),
                    pal.color(QPalette.ColorRole.WindowText),
                )
            if hovered:
                return (
                    pal.color(QPalette.ColorRole.Midlight),
                    pal.color(QPalette.ColorRole.Mid),
                    pal.color(QPalette.ColorRole.WindowText),
                )
            return (
                pal.color(QPalette.ColorRole.Button),
                pal.color(QPalette.ColorRole.Mid),
                muted_text,
            )

        case "primary":
            if pressed or hovered:
                return (accent, accent, QColor(0xFF, 0xFF, 0xFF))
            return (_with_alpha(accent, 38), _with_alpha(accent, 76), accent)

        case "danger":
            red = QColor(0xEF, 0x53, 0x50)
            if pressed:
                return (red, red, QColor(0xFF, 0xFF, 0xFF))
            if hovered:
                return (_with_alpha(red, 38), _with_alpha(red, 76), red)
            return (
                pal.color(QPalette.ColorRole.Button),
                pal.color(QPalette.ColorRole.Mid),
                muted_text,
            )

        case _:
            return (transparent, transparent, pal.color(QPalette.ColorRole.WindowText))


# ═══════════════════════════════════════════════════════════════
# The Style
# ═══════════════════════════════════════════════════════════════

# Design constants
RADIUS = 3
RADIUS_LG = 6
SCROLLBAR_WIDTH = 7
SLIDER_GROOVE_H = 4
SLIDER_HANDLE_SIZE = 14
CHECKBOX_SIZE = 16
COMBO_ARROW_W = 20
SPLITTER_HANDLE_W = 4
TOOLBAR_HANDLE_W = 0  # no drag handle


class MicroscopeStyle(QProxyStyle):
    """Zoomable proxy over Fusion.

    makes standard Qt widgets match the microscope GUI design system.

    All pixel metrics are scaled by `zoom_factor`. Custom base overrides
    (e.g. thin scrollbars) are preserved via `_BASE_OVERRIDES` and still
    route through zoom.

    Covered widgets:
        QPushButton    — variant system via 'variant' property
        QCheckBox      — accent-filled box, drawn checkmark
        QRadioButton   — accent ring + inner dot
        QSlider        — slim groove, accent fill, circular handle
        QSpinBox       — rounded frame, focus glow, drawn arrows
        QComboBox      — rounded frame, drawn chevron
        QScrollBar     — minimal 5px, no arrows, rounded thumb
        QLineEdit      — rounded frame, accent focus border + glow
        QToolTip       — dark inverted, rounded corners, shadow
        QFrame         — 1px subtle lines, replaces etched/sunken
        QToolBar       — borderless, flush background, no drag handle
        QSplitter      — thin hairline handle, hover grip dots

    Button variant styling via dynamic property:
        btn.setProperty("variant", "primary")

    All colors derived from QPalette — works with both light and dark themes.
    """

    def __init__(self) -> None:
        super().__init__("Fusion")
        self._zoom: float = 1.0

    @property
    def zoom_factor(self) -> float:
        return self._zoom

    @zoom_factor.setter
    def zoom_factor(self, value: float) -> None:
        self._zoom = max(0.25, min(value, 4.0))

    # ── Scaled metrics ────────────────────────────────────────────

    def layoutSpacing(
        self,
        control1: QSizePolicy.ControlType,
        control2: QSizePolicy.ControlType,
        orientation: Qt.Orientation,
        option: QStyleOption | None = None,
        widget: QWidget | None = None,
    ) -> int:
        val = super().layoutSpacing(control1, control2, orientation, option, widget)
        if val < 0:
            return val  # -1 means "use pixelMetric-based spacing"
        return max(1, round(val * self._zoom))

    def _border_color(self) -> QPen:
        """Resolve the border-subtle pen (late import avoids circular ref)."""
        from . import qcolor, theme

        return QPen(qcolor(theme().border_subtle), 1)

    # ── Pixel Metrics (zoom-scaled) ──

    def pixelMetric(
        self,
        metric: QStyle.PixelMetric,
        option: QStyleOption | None = None,
        widget: QWidget | None = None,
    ) -> int:
        PM = QStyle.PixelMetric
        match metric:
            case PM.PM_ScrollBarExtent:
                base = SCROLLBAR_WIDTH
            case PM.PM_ScrollBarSliderMin:
                base = 20
            case PM.PM_ButtonMargin:
                base = 6
            case PM.PM_DefaultFrameWidth:
                base = 1
            case (
                PM.PM_IndicatorWidth
                | PM.PM_IndicatorHeight
                | PM.PM_ExclusiveIndicatorWidth
                | PM.PM_ExclusiveIndicatorHeight
            ):
                base = CHECKBOX_SIZE
            case PM.PM_SliderThickness | PM.PM_SliderLength:
                base = SLIDER_HANDLE_SIZE
            case PM.PM_FocusFrameHMargin | PM.PM_FocusFrameVMargin:
                return 0  # We draw our own focus indicator
            case PM.PM_SplitterWidth:
                base = SPLITTER_HANDLE_W
            case PM.PM_ToolBarHandleExtent:
                base = TOOLBAR_HANDLE_W
            case PM.PM_ToolBarSeparatorExtent:
                base = 8
            case PM.PM_ToolBarFrameWidth:
                base = 0
            case PM.PM_ToolBarItemMargin:
                base = 2
            case PM.PM_ToolBarItemSpacing:
                base = 4
            case _:
                base = super().pixelMetric(metric, option, widget)
        if base <= 0:
            return base  # preserve sentinels and intentional zeros
        return max(1, round(base * self._zoom))

    # ── Size Hints ──

    def sizeFromContents(
        self,
        type: QStyle.ContentsType,
        option: QStyleOption | None,
        size: QSize,
        widget: QWidget | None,
    ) -> QSize:
        s = super().sizeFromContents(type, option, size, widget)
        if type == QStyle.ContentsType.CT_PushButton:
            # Ensure minimum button height and horizontal padding
            s.setHeight(max(s.height(), 26))
            s.setWidth(s.width() + 8)
        return s

    # ── Primitive Elements ──

    def drawPrimitive(
        self,
        element: QStyle.PrimitiveElement,
        option: QStyleOption | None,
        painter: QPainter | None,
        widget: QWidget | None = None,
    ) -> None:
        if option is None or painter is None:
            return
        PE = QStyle.PrimitiveElement
        pal = option.palette

        match element:
            case PE.PE_FrameFocusRect:
                # Subtle accent focus ring instead of dotted rect
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                r = QRectF(option.rect).adjusted(0.5, 0.5, -0.5, -0.5)
                pen = QPen(
                    _with_alpha(pal.color(QPalette.ColorRole.Highlight), 120), 1.5
                )
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(r, RADIUS + 1, RADIUS + 1)

            case PE.PE_PanelScrollAreaCorner | PE.PE_FrameDefaultButton:
                pass  # suppress default panel/frame backgrounds

            case PE.PE_PanelStatusBar:
                super().drawPrimitive(element, option, painter, widget)
                painter.setPen(self._border_color())
                painter.drawLine(
                    option.rect.left(),
                    option.rect.top(),
                    option.rect.right(),
                    option.rect.top(),
                )

            case PE.PE_PanelLineEdit:
                self._draw_lineedit(option, painter, widget)

            case PE.PE_Frame:
                if not isinstance(widget, QScrollArea):
                    self._draw_frame(option, painter, widget)

            case PE.PE_PanelTipLabel:
                self._draw_tooltip(option, painter, widget)

            case PE.PE_PanelToolBar:
                painter.fillRect(option.rect, pal.color(QPalette.ColorRole.Window))

            case PE.PE_IndicatorDockWidgetResizeHandle:
                self._draw_splitter_handle(option, painter, widget)

            case PE.PE_IndicatorCheckBox:
                self._draw_checkbox(option, painter, widget)

            case PE.PE_IndicatorRadioButton:
                self._draw_radio(option, painter, widget)

            case _:
                super().drawPrimitive(element, option, painter, widget)

    # ── Control Elements ──

    def drawControl(
        self,
        element: QStyle.ControlElement,
        option: QStyleOption | None,
        painter: QPainter | None,
        widget: QWidget | None = None,
    ) -> None:
        if option is None or painter is None:
            return
        CE = QStyle.ControlElement

        match element:
            case CE.CE_PushButton if isinstance(option, QStyleOptionButton):
                self._draw_push_button(option, painter, widget)

            case CE.CE_PushButtonLabel if isinstance(option, QStyleOptionButton):
                self._draw_push_button_label(option, painter, widget)

            case CE.CE_ScrollBarAddLine | CE.CE_ScrollBarSubLine:
                pass  # hide arrow buttons

            case CE.CE_ScrollBarSlider:
                self._draw_scrollbar_slider(option, painter, widget)

            case CE.CE_ToolBar:
                super().drawControl(element, option, painter, widget)
                r = option.rect
                painter.setPen(self._border_color())
                painter.drawLine(r.left(), r.bottom(), r.right(), r.bottom())

            case CE.CE_Splitter:
                self._draw_splitter_handle(option, painter, widget)

            case _:
                super().drawControl(element, option, painter, widget)

    # ── Complex Controls ──

    def drawComplexControl(
        self,
        control: QStyle.ComplexControl,
        option: QStyleOptionComplex | None,
        painter: QPainter | None,
        widget: QWidget | None = None,
    ) -> None:
        if option is None or painter is None:
            return
        CC = QStyle.ComplexControl

        match control:
            case CC.CC_Slider if isinstance(option, QStyleOptionSlider):
                self._draw_slider(option, painter, widget)

            case CC.CC_SpinBox if isinstance(option, QStyleOptionSpinBox):
                self._draw_spinbox(option, painter, widget)

            case CC.CC_ComboBox if isinstance(option, QStyleOptionComboBox):
                self._draw_combobox(option, painter, widget)

            case _:
                super().drawComplexControl(control, option, painter, widget)

    def subControlRect(
        self,
        cc: QStyle.ComplexControl,
        opt: QStyleOptionComplex | None,
        sc: QStyle.SubControl,
        widget: QWidget | None = None,
    ) -> QRect:
        if opt is None:
            return super().subControlRect(cc, opt, sc, widget)
        if cc == QStyle.ComplexControl.CC_SpinBox:
            r = opt.rect
            arrow_w = 20
            if sc == QStyle.SubControl.SC_SpinBoxEditField:
                # Text field takes everything except the arrow column
                return QRect(r.left() + 2, r.top(), r.width() - arrow_w - 2, r.height())
            if sc == QStyle.SubControl.SC_SpinBoxUp:
                # Top half of the arrow column
                return QRect(r.right() - arrow_w, r.top(), arrow_w, r.height() // 2)
            if sc == QStyle.SubControl.SC_SpinBoxDown:
                # Bottom half of the arrow column
                return QRect(
                    r.right() - arrow_w,
                    r.top() + r.height() // 2,
                    arrow_w,
                    r.height() // 2,
                )
            if sc == QStyle.SubControl.SC_SpinBoxFrame:
                return r
        return super().subControlRect(cc, opt, sc, widget)

    # ═══════════════════════════════════════════════════════════
    # Push Button
    # ═══════════════════════════════════════════════════════════

    def _draw_push_button(
        self,
        opt: QStyleOptionButton,
        p: QPainter,
        widget: QWidget | None,
    ) -> None:
        """Draw the complete push button: background + label."""
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(opt.rect).adjusted(0.5, 0.5, -0.5, -0.5)

        bg, border, _ = _button_colors(opt, widget)

        # Background
        if bg.alpha() > 0:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(bg))
            p.drawRoundedRect(r, RADIUS, RADIUS)

        # Border
        if border.alpha() > 0:
            p.setPen(QPen(border, 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r, RADIUS, RADIUS)

        # Draw label via the standard path
        label_opt = QStyleOptionButton(opt)
        self._draw_push_button_label(label_opt, p, widget)

    def _draw_push_button_label(
        self,
        opt: QStyleOptionButton,
        p: QPainter,
        widget: QWidget | None,
    ) -> None:
        """Draw push button text with the correct color."""
        _, _, text_color = _button_colors(opt, widget)
        p.setPen(text_color)

        r = opt.rect

        # Icon
        if not opt.icon.isNull():
            icon_size = opt.iconSize
            icon_rect = QRect(
                r.left() + 8,
                r.top() + (r.height() - icon_size.height()) // 2,
                icon_size.width(),
                icon_size.height(),
            )
            opt.icon.paint(p, icon_rect)
            r = QRect(
                icon_rect.right() + 4,
                r.top(),
                r.width() - icon_rect.width() - 12,
                r.height(),
            )

        # Text
        if opt.text:
            p.drawText(QRectF(r), Qt.AlignmentFlag.AlignCenter, opt.text)

    # ═══════════════════════════════════════════════════════════
    # Checkbox
    # ═══════════════════════════════════════════════════════════

    def _draw_checkbox(
        self,
        opt: QStyleOption,
        p: QPainter,
        widget: QWidget | None,
    ) -> None:
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = opt.palette
        r = QRectF(opt.rect)

        checked = bool(opt.state & QStyle.StateFlag.State_On)
        hovered = bool(opt.state & QStyle.StateFlag.State_MouseOver)
        enabled = bool(opt.state & QStyle.StateFlag.State_Enabled)
        accent = pal.color(QPalette.ColorRole.Highlight)

        # Box
        if checked:
            p.setPen(QPen(accent, 1))
            p.setBrush(QBrush(accent))
        else:
            border = pal.color(QPalette.ColorRole.Mid)
            if hovered:
                border = pal.color(QPalette.ColorRole.WindowText)
            p.setPen(QPen(border, 1))
            bg = pal.color(QPalette.ColorRole.AlternateBase)
            p.setBrush(QBrush(bg))

        if not enabled:
            p.setOpacity(0.4)

        p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 2, 2)

        # Checkmark
        if checked:
            p.setPen(
                QPen(
                    QColor(0xFF, 0xFF, 0xFF),
                    1.8,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
            )
            cx, cy = r.center().x(), r.center().y()
            s = r.width() * 0.22
            path = QPainterPath()
            path.moveTo(cx - s * 1.1, cy - s * 0.1)
            path.lineTo(cx - s * 0.2, cy + s * 0.9)
            path.lineTo(cx + s * 1.3, cy - s * 0.9)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)

        p.setOpacity(1.0)

    # ═══════════════════════════════════════════════════════════
    # Radio Button
    # ═══════════════════════════════════════════════════════════

    def _draw_radio(
        self,
        opt: QStyleOption,
        p: QPainter,
        widget: QWidget | None,
    ) -> None:
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = opt.palette
        r = QRectF(opt.rect).adjusted(0.5, 0.5, -0.5, -0.5)

        checked = bool(opt.state & QStyle.StateFlag.State_On)
        hovered = bool(opt.state & QStyle.StateFlag.State_MouseOver)
        accent = pal.color(QPalette.ColorRole.Highlight)

        # Circle
        if checked:
            p.setPen(QPen(accent, 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
        else:
            border = pal.color(QPalette.ColorRole.Mid)
            if hovered:
                border = pal.color(QPalette.ColorRole.WindowText)
            p.setPen(QPen(border, 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)

        cx, cy = r.center().x(), r.center().y()
        radius = min(r.width(), r.height()) / 2 - 1
        p.drawEllipse(QPointF(cx, cy), radius, radius)

        # Inner dot
        if checked:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(accent))
            p.drawEllipse(QPointF(cx, cy), radius * 0.45, radius * 0.45)

    # ═══════════════════════════════════════════════════════════
    # Slider
    # ═══════════════════════════════════════════════════════════

    def _draw_slider(
        self,
        opt: QStyleOptionSlider,
        p: QPainter,
        widget: QWidget | None,
    ) -> None:
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = opt.palette

        # Groove rect
        groove_rect = self.subControlRect(
            QStyle.ComplexControl.CC_Slider,
            opt,
            QStyle.SubControl.SC_SliderGroove,
            widget,
        )
        # Handle rect
        handle_rect = self.subControlRect(
            QStyle.ComplexControl.CC_Slider,
            opt,
            QStyle.SubControl.SC_SliderHandle,
            widget,
        )

        horiz = opt.orientation == Qt.Orientation.Horizontal
        accent = pal.color(QPalette.ColorRole.Highlight)
        groove_bg = pal.color(QPalette.ColorRole.Button)
        hovered = bool(opt.state & QStyle.StateFlag.State_MouseOver)

        # ── Groove ──
        if horiz:
            gy = groove_rect.center().y() - SLIDER_GROOVE_H // 2
            gr = QRectF(groove_rect.left(), gy, groove_rect.width(), SLIDER_GROOVE_H)
        else:
            gx = groove_rect.center().x() - SLIDER_GROOVE_H // 2
            gr = QRectF(gx, groove_rect.top(), SLIDER_GROOVE_H, groove_rect.height())

        # Background groove
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(groove_bg))
        p.drawRoundedRect(gr, SLIDER_GROOVE_H / 2, SLIDER_GROOVE_H / 2)

        # Filled portion
        if horiz:
            filled = QRectF(
                gr.left(), gr.top(), handle_rect.center().x() - gr.left(), gr.height()
            )
        else:
            filled = QRectF(
                gr.left(),
                handle_rect.center().y(),
                gr.width(),
                gr.bottom() - handle_rect.center().y(),
            )

        p.setBrush(QBrush(accent))
        p.drawRoundedRect(filled, SLIDER_GROOVE_H / 2, SLIDER_GROOVE_H / 2)

        # ── Handle ──
        hx = handle_rect.center().x()
        # Force vertical center on the groove, not the handle_rect
        hy = gr.center().y()
        hr = 8  # radius — bigger than SLIDER_HANDLE_SIZE/2

        # Shadow
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, 35)))
        p.drawEllipse(QPointF(hx, hy + 1), hr, hr)

        # Handle body
        if hovered:
            handle_bg = pal.color(QPalette.ColorRole.WindowText)
        else:
            handle_bg = _lerp_color(
                pal.color(QPalette.ColorRole.PlaceholderText),
                pal.color(QPalette.ColorRole.WindowText),
                0.4,
            )

        p.setPen(QPen(pal.color(QPalette.ColorRole.Window), 2))
        p.setBrush(QBrush(handle_bg))
        p.drawEllipse(QPointF(hx, hy), hr, hr)

    # ═══════════════════════════════════════════════════════════
    # SpinBox
    # ═══════════════════════════════════════════════════════════

    def _draw_spinbox(
        self,
        opt: QStyleOptionSpinBox,
        p: QPainter,
        widget: QWidget | None,
    ) -> None:
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = opt.palette

        r = QRectF(opt.rect).adjusted(0.5, 0.5, -0.5, -0.5)
        focused = bool(opt.state & QStyle.StateFlag.State_HasFocus)
        hovered = bool(opt.state & QStyle.StateFlag.State_MouseOver)
        enabled = bool(opt.state & QStyle.StateFlag.State_Enabled)

        active_sc = opt.activeSubControls
        up_hovered = hovered and bool(active_sc & QStyle.SubControl.SC_SpinBoxUp)
        dn_hovered = hovered and bool(active_sc & QStyle.SubControl.SC_SpinBoxDown)
        up_pressed = bool(opt.state & QStyle.StateFlag.State_Sunken) and up_hovered
        dn_pressed = bool(opt.state & QStyle.StateFlag.State_Sunken) and dn_hovered

        # ── Unified frame ──
        bg = pal.color(QPalette.ColorRole.AlternateBase)
        if not enabled:
            bg = pal.color(QPalette.ColorRole.Window)
        if focused:
            border = pal.color(QPalette.ColorRole.Highlight)
        elif hovered:
            border = _lerp_color(
                pal.color(QPalette.ColorRole.Mid),
                pal.color(QPalette.ColorRole.WindowText),
                0.3,
            )
        else:
            border = pal.color(QPalette.ColorRole.Mid)

        p.setPen(QPen(border, 1))
        p.setBrush(QBrush(bg))
        p.drawRoundedRect(r, RADIUS, RADIUS)

        # Focus glow
        if focused:
            glow = QRectF(r).adjusted(-1.5, -1.5, 1.5, 1.5)
            p.setPen(
                QPen(_with_alpha(pal.color(QPalette.ColorRole.Highlight), 35), 2.5)
            )
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(glow, RADIUS + 1.5, RADIUS + 1.5)

        # ── Arrow area ──
        arrow_w = 20
        arrow_x = r.right() - arrow_w
        mid_y = r.center().y()

        # Subtle highlight on hovered/pressed arrow half
        if up_hovered or up_pressed or dn_hovered or dn_pressed:
            is_top = up_hovered or up_pressed
            highlight_r = QRectF(
                arrow_x,
                r.top() + 1 if is_top else mid_y,
                arrow_w - 1,
                (mid_y - r.top() - 1) if is_top else (r.bottom() - mid_y - 1),
            )
            p.setPen(Qt.PenStyle.NoPen)
            bg_hl = (
                pal.color(QPalette.ColorRole.Button)
                if (up_pressed or dn_pressed)
                else pal.color(QPalette.ColorRole.Midlight)
            )
            p.setBrush(QBrush(bg_hl))
            p.drawRoundedRect(highlight_r, 2, 2)

        # Arrow colors — brightest on press, brighter on hover, muted at rest
        if up_pressed:
            up_color = pal.color(QPalette.ColorRole.Highlight)
        elif up_hovered:
            up_color = pal.color(QPalette.ColorRole.WindowText)
        else:
            up_color = pal.color(QPalette.ColorRole.PlaceholderText)

        if dn_pressed:
            dn_color = pal.color(QPalette.ColorRole.Highlight)
        elif dn_hovered:
            dn_color = pal.color(QPalette.ColorRole.WindowText)
        else:
            dn_color = pal.color(QPalette.ColorRole.PlaceholderText)

        if not enabled:
            up_color = dn_color = pal.color(QPalette.ColorRole.PlaceholderText)

        ax = arrow_x + arrow_w / 2
        s = 3.5  # half-width of chevron

        # Up chevron
        p.setPen(
            QPen(
                up_color,
                1.4,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        ay_up = mid_y - 5
        p.drawLine(QPointF(ax - s, ay_up + s * 0.7), QPointF(ax, ay_up - s * 0.3))
        p.drawLine(QPointF(ax, ay_up - s * 0.3), QPointF(ax + s, ay_up + s * 0.7))

        # Down chevron
        p.setPen(
            QPen(
                dn_color,
                1.4,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        ay_dn = mid_y + 5
        p.drawLine(QPointF(ax - s, ay_dn - s * 0.7), QPointF(ax, ay_dn + s * 0.3))
        p.drawLine(QPointF(ax, ay_dn + s * 0.3), QPointF(ax + s, ay_dn - s * 0.7))

    # ═══════════════════════════════════════════════════════════
    # ComboBox
    # ═══════════════════════════════════════════════════════════

    def _draw_combobox(
        self,
        opt: QStyleOptionComboBox,
        p: QPainter,
        widget: QWidget | None,
    ) -> None:
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = opt.palette

        r = QRectF(opt.rect).adjusted(0.5, 0.5, -0.5, -0.5)
        hovered = bool(opt.state & QStyle.StateFlag.State_MouseOver)
        focused = bool(opt.state & QStyle.StateFlag.State_HasFocus)
        bool(opt.state & QStyle.StateFlag.State_Sunken)

        # Frame
        bg = pal.color(QPalette.ColorRole.AlternateBase)
        if focused:
            border = pal.color(QPalette.ColorRole.Highlight)
        elif hovered:
            border = pal.color(QPalette.ColorRole.WindowText)
        else:
            border = pal.color(QPalette.ColorRole.Mid)

        p.setPen(QPen(border, 1))
        p.setBrush(QBrush(bg))
        p.drawRoundedRect(r, RADIUS, RADIUS)

        # Dropdown arrow
        arrow_w = COMBO_ARROW_W
        ax = int(r.right()) - arrow_w // 2 - 2
        ay = int(r.center().y())
        s = 3

        p.setPen(
            QPen(
                pal.color(QPalette.ColorRole.WindowText)
                if hovered
                else pal.color(QPalette.ColorRole.PlaceholderText),
                1.5,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawLine(QPointF(ax - s, ay - 1), QPointF(ax, ay + s - 1))
        p.drawLine(QPointF(ax, ay + s - 1), QPointF(ax + s, ay - 1))

    # ═══════════════════════════════════════════════════════════
    # Scrollbar (minimal)
    # ═══════════════════════════════════════════════════════════

    def _draw_scrollbar_slider(
        self,
        opt: QStyleOption,
        p: QPainter,
        widget: QWidget | None,
    ) -> None:
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = opt.palette

        r = QRectF(opt.rect)
        hovered = bool(opt.state & QStyle.StateFlag.State_MouseOver)
        pressed = bool(opt.state & QStyle.StateFlag.State_Sunken)

        if pressed:
            color = pal.color(QPalette.ColorRole.WindowText)
        elif hovered:
            color = _lerp_color(
                pal.color(QPalette.ColorRole.Mid),
                pal.color(QPalette.ColorRole.WindowText),
                0.3,
            )
        else:
            color = pal.color(QPalette.ColorRole.Mid)

        # Slim rounded bar
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(color))

        margin = 1
        if r.width() < r.height():
            # Vertical scrollbar
            slim = QRectF(
                r.left() + margin, r.top(), r.width() - margin * 2, r.height()
            )
        else:
            # Horizontal scrollbar
            slim = QRectF(
                r.left(), r.top() + margin, r.width(), r.height() - margin * 2
            )

        radius = min(slim.width(), slim.height()) / 2
        p.drawRoundedRect(slim, radius, radius)

    # ═══════════════════════════════════════════════════════════
    # QLineEdit
    # ═══════════════════════════════════════════════════════════

    def _draw_lineedit(
        self,
        opt: QStyleOption,
        p: QPainter,
        widget: QWidget | None,
    ) -> None:
        """Rounded line edit with accent focus border and subtle glow."""
        # Skip — the parent spinbox/combobox already drew the frame
        if widget is not None and widget.parent() is not None:
            parent = widget.parent()
            if isinstance(parent, (QAbstractSpinBox, QComboBox)):
                return

        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = opt.palette

        r = QRectF(opt.rect).adjusted(0.5, 0.5, -0.5, -0.5)
        focused = bool(opt.state & QStyle.StateFlag.State_HasFocus)
        hovered = bool(opt.state & QStyle.StateFlag.State_MouseOver)
        enabled = bool(opt.state & QStyle.StateFlag.State_Enabled)

        # Background
        bg = pal.color(QPalette.ColorRole.AlternateBase)
        if not enabled:
            bg = pal.color(QPalette.ColorRole.Window)

        # Border color
        if focused:
            border = pal.color(QPalette.ColorRole.Highlight)
        elif hovered and enabled:
            border = _lerp_color(
                pal.color(QPalette.ColorRole.Mid),
                pal.color(QPalette.ColorRole.WindowText),
                0.3,
            )
        else:
            border = pal.color(QPalette.ColorRole.Mid)

        p.setPen(QPen(border, 1))
        p.setBrush(QBrush(bg))
        p.drawRoundedRect(r, RADIUS, RADIUS)

        # Focus glow — subtle accent halo around the frame
        if focused:
            glow_r = QRectF(r).adjusted(-1.5, -1.5, 1.5, 1.5)
            glow_color = _with_alpha(pal.color(QPalette.ColorRole.Highlight), 35)
            p.setPen(QPen(glow_color, 2.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(glow_r, RADIUS + 1.5, RADIUS + 1.5)

    # ═══════════════════════════════════════════════════════════
    # QFrame (generic separators and box frames)
    # ═══════════════════════════════════════════════════════════

    def _draw_frame(
        self,
        opt: QStyleOption,
        p: QPainter,
        widget: QWidget | None,
    ) -> None:
        """Draw frames as clean 1px lines using border-subtle.

        Handles HLine, VLine, and box-style frames. Replaces Fusion's
        etched/sunken multi-pixel frames.
        """
        pal = opt.palette
        r = opt.rect

        # Determine frame shape from the widget itself
        shape = QFrame.Shape.NoFrame
        if isinstance(widget, QFrame):
            shape = widget.frameShape()

        # Determine the line color
        line_color = pal.color(QPalette.ColorRole.Mid)
        # Softer color for separator lines
        if shape in (QFrame.Shape.HLine, QFrame.Shape.VLine):
            line_color = _lerp_color(
                pal.color(QPalette.ColorRole.Window),
                pal.color(QPalette.ColorRole.Mid),
                0.5,
            )

        p.setPen(QPen(line_color, 1))
        S = QFrame.Shape

        match shape:
            case S.HLine:
                y = r.top() + r.height() // 2
                p.drawLine(r.left(), y, r.right(), y)
            case S.VLine:
                x = r.left() + r.width() // 2
                p.drawLine(x, r.top(), x, r.bottom())
            case S.StyledPanel | S.Panel | S.Box:
                p.setRenderHint(QPainter.RenderHint.Antialiasing)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(
                    QRectF(r).adjusted(0.5, 0.5, -0.5, -0.5), RADIUS, RADIUS
                )

    # ═══════════════════════════════════════════════════════════
    # QToolTip
    # ═══════════════════════════════════════════════════════════

    def _draw_tooltip(
        self,
        opt: QStyleOption,
        p: QPainter,
        widget: QWidget | None,
    ) -> None:
        """Dark inverted tooltip with rounded corners and subtle border.

        Always dark regardless of light/dark theme — tooltips use the
        ToolTipBase/ToolTipText palette roles which we set to dark values
        in both themes.
        """
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = opt.palette

        r = QRectF(opt.rect).adjusted(0.5, 0.5, -0.5, -0.5)

        # Background — from ToolTipBase
        bg = pal.color(QPalette.ColorRole.ToolTipBase)
        # Border — slightly lighter than background
        border = _lerp_color(bg, QColor(255, 255, 255), 0.12)

        # Shadow (offset fill behind the main rect)
        shadow_r = QRectF(r).adjusted(0, 1, 0, 1)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, 80)))
        p.drawRoundedRect(shadow_r, RADIUS_LG, RADIUS_LG)

        # Main body
        p.setPen(QPen(border, 1))
        p.setBrush(QBrush(bg))
        p.drawRoundedRect(r, RADIUS_LG, RADIUS_LG)

    # ═══════════════════════════════════════════════════════════
    # QSplitter handle
    # ═══════════════════════════════════════════════════════════

    def _draw_splitter_handle(
        self,
        opt: QStyleOption,
        p: QPainter,
        widget: QWidget | None,
    ) -> None:
        """Thin subtle line for splitter handles.

        Replaces Fusion's dotted grip pattern with a single hairline.
        Brightens slightly on hover.
        """
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = opt.palette
        r = opt.rect

        hovered = bool(opt.state & QStyle.StateFlag.State_MouseOver)

        if hovered:
            color = _lerp_color(
                pal.color(QPalette.ColorRole.Mid),
                pal.color(QPalette.ColorRole.WindowText),
                0.2,
            )
        else:
            color = _lerp_color(
                pal.color(QPalette.ColorRole.Window),
                pal.color(QPalette.ColorRole.Mid),
                0.5,
            )

        # Determine orientation from rect shape
        is_horizontal = r.width() > r.height()

        p.setPen(QPen(color, 1))

        if is_horizontal:
            # Horizontal splitter — draw a centered horizontal line
            y = r.top() + r.height() // 2
            # Inset slightly from edges
            inset = min(r.width() // 4, 20)
            p.drawLine(r.left() + inset, y, r.right() - inset, y)
        else:
            # Vertical splitter — draw a centered vertical line
            x = r.left() + r.width() // 2
            inset = min(r.height() // 4, 20)
            p.drawLine(x, r.top() + inset, x, r.bottom() - inset)

        # On hover, draw small center grip dots for discoverability
        if hovered:
            dot_color = _with_alpha(color, 180)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(dot_color))
            cx = r.left() + r.width() // 2
            cy = r.top() + r.height() // 2
            dot_r = 1.5

            if is_horizontal:
                for dx in (-6, 0, 6):
                    p.drawEllipse(QPointF(cx + dx, cy), dot_r, dot_r)
            else:
                for dy in (-6, 0, 6):
                    p.drawEllipse(QPointF(cx, cy + dy), dot_r, dot_r)


# ═══════════════════════════════════════════════════════════════
# Convenience helpers (optional — just set properties)
# ═══════════════════════════════════════════════════════════════


def set_variant(widget: QWidget, variant: str) -> None:
    """Set the button variant property.

    Equivalent to widget.setProperty("variant", variant), but validates.
    """
    if variant not in ("ghost", "subtle", "primary", "danger"):
        raise ValueError(f"Unknown variant: {variant!r}")
    widget.setProperty("variant", variant)
    widget.update()


def set_accent(widget: QWidget, color: QColor) -> None:
    """Set a custom accent color for a specific widget."""
    widget.setProperty("accent", color)
    widget.update()
