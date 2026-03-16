"""Styled Button System.

One button class, four variants, zero QSS.

Variants:
    GHOST   — transparent resting, tinted hover. Toolbars, viewport controls.
    SUBTLE  — bg-surface + border. Panel actions, secondary controls.
    PRIMARY — accent-colored. One promoted action per context.
    DANGER  — red on hover/active. Destructive actions (clear, stop, delete).

Special widgets:
    SegmentedControl — exclusive toggle group (step sizes, binning, modes).
"""

from __future__ import annotations

from enum import Enum, auto

from PyQt6.QtCore import QEvent, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QEnterEvent,
    QFont,
    QFontMetricsF,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
)
from PyQt6.QtWidgets import QSizePolicy, QWidget


def _with_alpha(color: QColor, alpha: int) -> QColor:
    c = QColor(color)
    c.setAlpha(alpha)
    return c


def _qc(attr: str) -> QColor:
    """Get a theme color as QColor (lazy import avoids circular ref)."""
    from . import qcolor, theme

    return qcolor(getattr(theme(), attr))


# ═══════════════════════════════════════════════════════════════
# Button variant enum
# ═══════════════════════════════════════════════════════════════


class ButtonVariant(Enum):
    GHOST = auto()
    SUBTLE = auto()
    PRIMARY = auto()
    DANGER = auto()


# ═══════════════════════════════════════════════════════════════
# Color resolver — one function, all states
# ═══════════════════════════════════════════════════════════════

_TRANSPARENT = QColor(0, 0, 0, 0)
_WHITE = QColor(0xFF, 0xFF, 0xFF)


def _resolve_colors(
    variant: ButtonVariant,
    *,
    hovered: bool,
    pressed: bool,
    checked: bool,
    accent: QColor,
) -> tuple[QColor, QColor, QColor]:
    """Return (background, border, text) for the current state."""
    # ── GHOST ──
    if variant == ButtonVariant.GHOST:
        if checked:
            return (
                _with_alpha(accent, 38),
                _with_alpha(accent, 76),
                accent,
            )
        if pressed:
            return (
                _with_alpha(accent, 50),
                _with_alpha(accent, 76),
                accent,
            )
        if hovered:
            return (_qc("bg_hover"), _TRANSPARENT, _qc("text_primary"))
        return (_TRANSPARENT, _TRANSPARENT, _qc("text_secondary"))

    # ── SUBTLE ──
    if variant == ButtonVariant.SUBTLE:
        if checked:
            return (
                _with_alpha(accent, 38),
                _with_alpha(accent, 76),
                accent,
            )
        if pressed:
            return (_qc("bg_active"), _qc("border_default"), _qc("text_primary"))
        if hovered:
            return (_qc("bg_hover"), _qc("border_default"), _qc("text_primary"))
        return (_qc("bg_surface"), _qc("border_subtle"), _qc("text_secondary"))

    # ── PRIMARY ──
    if variant == ButtonVariant.PRIMARY:
        if pressed:
            return (accent, accent, _WHITE)
        if hovered:
            return (accent, accent, _WHITE)
        return (_with_alpha(accent, 38), _with_alpha(accent, 76), accent)

    # ── DANGER ──
    if variant == ButtonVariant.DANGER:
        red = _qc("status_red")
        if pressed:
            return (red, red, _WHITE)
        if hovered:
            return (_with_alpha(red, 38), _with_alpha(red, 76), red)
        return (_qc("bg_surface"), _qc("border_subtle"), _qc("text_secondary"))

    # Fallback
    return (_TRANSPARENT, _TRANSPARENT, _qc("text_secondary"))


# ═══════════════════════════════════════════════════════════════
# StyledButton
# ═══════════════════════════════════════════════════════════════


class StyledButton(QWidget):
    """A single button class for the entire application."""

    clicked = pyqtSignal()
    toggled = pyqtSignal(bool)

    def __init__(
        self,
        text: str,
        *,
        variant: ButtonVariant = ButtonVariant.GHOST,
        checkable: bool = False,
        checked: bool = False,
        accent_override: QColor | None = None,
        font_size: float = 8,
        monospace: bool = False,
        min_width: int | None = None,
        height: int = 26,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._text = text
        self._variant = variant
        self._checkable = checkable
        self._checked = checked
        self._accent = accent_override or _qc("accent")
        self._font_size = font_size
        self._monospace = monospace
        self._min_width = min_width
        self._hovered = False
        self._pressed = False

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self._update_size(height)

    def _update_size(self, height: int | None = None) -> None:
        """(Re)calculate fixed size from font metrics + theme tokens."""
        from . import theme

        t = theme()
        if height is not None:
            self._height = height
        self.setFixedHeight(self._height)

        font = self._make_font()
        fm = QFontMetricsF(font)
        text_w = fm.horizontalAdvance(self._text)
        w = max(int(text_w) + t.sp_lg, self._min_width or 0)
        self.setMinimumWidth(w)
        if self._min_width is None:
            self.setFixedWidth(w)

    # ── Properties ──

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, val: str) -> None:
        self._text = val
        self.update()

    @property
    def variant(self) -> ButtonVariant:
        return self._variant

    @variant.setter
    def variant(self, val: ButtonVariant) -> None:
        self._variant = val
        self.update()

    @property
    def checked(self) -> bool:
        return self._checked

    @checked.setter
    def checked(self, val: bool) -> None:
        if self._checked != val:
            self._checked = val
            self.toggled.emit(val)
            self.update()

    @property
    def accent(self) -> QColor:
        return self._accent

    @accent.setter
    def accent(self, val: QColor) -> None:
        self._accent = val
        self.update()

    # ── Font helper ──

    def _make_font(self) -> QFont:
        from . import mono_font, ui_font

        if self._monospace:
            return mono_font(self._font_size, QFont.Weight.Medium)
        return ui_font(self._font_size, QFont.Weight.Medium)

    # ── Painting ──

    def paintEvent(self, event: QPaintEvent | None) -> None:
        from . import theme

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)

        bg, border, text_color = _resolve_colors(
            self._variant,
            hovered=self._hovered,
            pressed=self._pressed,
            checked=self._checked,
            accent=self._accent,
        )

        if border.alpha() > 0:
            p.setPen(QPen(border, 1))
        else:
            p.setPen(QPen(Qt.PenStyle.NoPen))

        if bg.alpha() > 0:
            p.setBrush(QBrush(bg))
        else:
            p.setBrush(QBrush(Qt.BrushStyle.NoBrush))

        rad = theme().radius
        p.drawRoundedRect(r, rad, rad)

        p.setPen(text_color)
        p.setFont(self._make_font())
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, self._text)
        p.end()

    # ── Events ──

    def changeEvent(self, event: QEvent | None) -> None:
        if event is not None and event.type() == QEvent.Type.StyleChange:
            self._update_size()
            self.update()
        super().changeEvent(event)

    def enterEvent(self, event: QEnterEvent | None) -> None:
        self._hovered = True
        self.update()

    def leaveEvent(self, event: QEvent | None) -> None:
        self._hovered = False
        self._pressed = False
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressed = True
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._pressed:
            self._pressed = False
            if self.rect().contains(event.pos()):
                if self._checkable:
                    self.checked = not self._checked
                self.clicked.emit()
            self.update()


# ═══════════════════════════════════════════════════════════════
# SegmentedControl
# ═══════════════════════════════════════════════════════════════


class SegmentedControl(QWidget):
    """Exclusive toggle group for discrete values."""

    selection_changed = pyqtSignal(int, str)  # index, label

    def __init__(
        self,
        options: list[str],
        *,
        selected: int = 0,
        monospace: bool = True,
        font_size: float = 8,
        height: int = 26,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._options = list(options)
        self._selected = selected
        self._hovered_idx = -1
        self._pressed_idx = -1
        self._monospace = monospace
        self._font_size = font_size
        self._seg_w: int = 0

        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_size(height)

    def _update_size(self, height: int | None = None) -> None:
        """(Re)calculate segment widths from font metrics."""
        from . import theme

        t = theme()
        if height is not None:
            self._height = height
        self.setFixedHeight(self._height)

        font = self._make_font()
        fm = QFontMetricsF(font)
        max_label_w = max(fm.horizontalAdvance(o) for o in self._options)
        self._seg_w = max(int(max_label_w) + t.sp_sm, t.sp_xl)
        total_w = self._seg_w * len(self._options) + 4
        self.setFixedWidth(total_w)

    @property
    def selected(self) -> int:
        return self._selected

    @selected.setter
    def selected(self, idx: int) -> None:
        if 0 <= idx < len(self._options) and idx != self._selected:
            self._selected = idx
            self.selection_changed.emit(idx, self._options[idx])
            self.update()

    @property
    def selected_value(self) -> str:
        return self._options[self._selected]

    def _make_font(self) -> QFont:
        from . import mono_font, ui_font

        if self._monospace:
            return mono_font(self._font_size, QFont.Weight.Medium)
        return ui_font(self._font_size, QFont.Weight.Medium)

    def _idx_at(self, x: float) -> int:
        """Return segment index at x pixel, or -1."""
        inner_x = x - 2
        if inner_x < 0:
            return -1
        idx = int(inner_x // self._seg_w) if self._seg_w > 0 else -1
        return idx if 0 <= idx < len(self._options) else -1

    def paintEvent(self, event: QPaintEvent | None) -> None:
        from . import qcolor, theme

        t = theme()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        full = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)

        p.setPen(QPen(qcolor(t.border_subtle), 1))
        p.setBrush(QBrush(qcolor(t.bg_raised)))
        p.drawRoundedRect(full, t.radius, t.radius)

        font = self._make_font()
        p.setFont(font)

        accent = qcolor(t.accent)
        pad = 2.0
        for i, label in enumerate(self._options):
            seg_r = QRectF(
                pad + i * self._seg_w,
                pad,
                self._seg_w,
                self.height() - pad * 2,
            )

            is_sel = i == self._selected
            is_hov = i == self._hovered_idx and not is_sel
            is_press = i == self._pressed_idx

            if is_sel:
                p.setPen(QPen(Qt.PenStyle.NoPen))
                p.setBrush(QBrush(_with_alpha(accent, 38)))
                p.drawRoundedRect(seg_r, 2, 2)
                text_color = accent
            elif is_press:
                p.setPen(QPen(Qt.PenStyle.NoPen))
                p.setBrush(QBrush(qcolor(t.bg_active)))
                p.drawRoundedRect(seg_r, 2, 2)
                text_color = qcolor(t.text_primary)
            elif is_hov:
                p.setPen(QPen(Qt.PenStyle.NoPen))
                p.setBrush(QBrush(qcolor(t.bg_hover)))
                p.drawRoundedRect(seg_r, 2, 2)
                text_color = qcolor(t.text_primary)
            else:
                text_color = qcolor(t.text_secondary)

            p.setPen(text_color)
            font_to_use = self._make_font()
            if is_sel:
                font_to_use.setWeight(QFont.Weight.DemiBold)
            p.setFont(font_to_use)
            p.drawText(seg_r, Qt.AlignmentFlag.AlignCenter, label)

        p.end()

    def changeEvent(self, event: QEvent | None) -> None:
        if event is not None and event.type() == QEvent.Type.StyleChange:
            self._update_size()
            self.update()
        super().changeEvent(event)

    def enterEvent(self, event: QEnterEvent | None) -> None:
        pass  # handled by mouseMoveEvent

    def leaveEvent(self, event: QEvent | None) -> None:
        self._hovered_idx = -1
        self._pressed_idx = -1
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        idx = self._idx_at(event.position().x())
        if idx != self._hovered_idx:
            self._hovered_idx = idx
            self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressed_idx = self._idx_at(event.position().x())
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            idx = self._idx_at(event.position().x())
            if idx >= 0 and idx == self._pressed_idx:
                self.selected = idx
            self._pressed_idx = -1
            self.update()


# ═══════════════════════════════════════════════════════════════
# ToolbarSeparator
# ═══════════════════════════════════════════════════════════════


class ToolbarSep(QWidget):
    """Thin vertical separator for toolbars."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._update_size()

    def _update_size(self) -> None:
        from . import theme

        self.setFixedSize(1, theme().scaled(18))

    def paintEvent(self, event: QPaintEvent | None) -> None:
        p = QPainter(self)
        p.setPen(QPen(_qc("border_subtle"), 1))
        p.drawLine(0, 0, 0, self.height())
        p.end()

    def changeEvent(self, event: QEvent | None) -> None:
        if event is not None and event.type() == QEvent.Type.StyleChange:
            self._update_size()
        super().changeEvent(event)
