"""
Styled Button System.

One button class, four variants, zero QSS.
Every button in the microscope GUI should come from here.

Variants:
    GHOST   — transparent resting, tinted hover. Toolbars, viewport controls.
    SUBTLE  — bg-surface + border. Panel actions, secondary controls.
    PRIMARY — accent-colored. One promoted action per context.
    DANGER  — red on hover/active. Destructive actions (clear, stop, delete).

Special widgets:
    SegmentedControl — exclusive toggle group (step sizes, binning, modes).

Usage:
    btn = StyledButton("📷 Snap", variant=ButtonVariant.GHOST)
    btn = StyledButton("⚡ Auto", variant=ButtonVariant.PRIMARY)
    btn = StyledButton("Clear", variant=ButtonVariant.DANGER)
    btn = StyledButton("● Live", variant=ButtonVariant.GHOST,
                       checkable=True, accent_override=Clr.RED)

    seg = SegmentedControl(["0.1", "1", "10", "100", "1k"], selected=2)
    seg.selection_changed.connect(on_step_changed)

Assumes app-level:
    app.setStyle(MicroscopeStyle())
    app.setPalette(make_dark_palette())
"""

from __future__ import annotations

from enum import Enum, auto

from PyQt6.QtCore import (
    QRectF,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QFontMetricsF,
    QMouseEvent,
    QPainter,
    QPen,
)
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QSizePolicy,
    QWidget,
)

# ═══════════════════════════════════════════════════════════════
# Tokens (would normally import from shared module)
# ═══════════════════════════════════════════════════════════════


class Clr:
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

    ACCENT = QColor(0x4A, 0x9E, 0xFF)
    GREEN = QColor(0x4C, 0xAF, 0x50)
    RED = QColor(0xEF, 0x53, 0x50)


def _with_alpha(color: QColor, alpha: int) -> QColor:
    c = QColor(color)
    c.setAlpha(alpha)
    return c


def _ui_font(size: float = 10, weight: int = QFont.Weight.Normal) -> QFont:
    f = QFont()
    f.setPointSizeF(size)
    f.setWeight(weight)
    return f


def _mono_font(size: float = 10, weight: int = QFont.Weight.Normal) -> QFont:
    for fam in ("JetBrains Mono", "SF Mono", "Cascadia Code", "Consolas"):
        if fam in QFontDatabase.families():
            f = QFont(fam)
            f.setPointSizeF(size)
            f.setWeight(weight)
            return f
    f = QFont()
    f.setStyleHint(QFont.StyleHint.Monospace)
    f.setPointSizeF(size)
    f.setWeight(weight)
    return f


RADIUS = 3.0


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


def _resolve_colors(
    variant: ButtonVariant,
    *,
    hovered: bool,
    pressed: bool,
    checked: bool,
    accent: QColor,
) -> tuple[QColor, QColor, QColor]:
    """Return (background, border, text) colors for the current state.

    This is the single source of truth for button appearance.
    """
    # ── GHOST ──
    if variant == ButtonVariant.GHOST:
        if checked:
            return (
                _with_alpha(accent, 38),  # bg: accent ~15%
                _with_alpha(accent, 76),  # border: accent ~30%
                accent,  # text: accent
            )
        if pressed:
            return (
                _with_alpha(accent, 50),
                _with_alpha(accent, 76),
                accent,
            )
        if hovered:
            return (
                Clr.BG_HOVER,  # bg: hover tint
                QColor(0, 0, 0, 0),  # border: none
                Clr.TEXT_PRIMARY,  # text: primary
            )
        return (
            QColor(0, 0, 0, 0),  # bg: transparent
            QColor(0, 0, 0, 0),  # border: none
            Clr.TEXT_SECONDARY,  # text: secondary
        )

    # ── SUBTLE ──
    if variant == ButtonVariant.SUBTLE:
        if checked:
            return (
                _with_alpha(accent, 38),
                _with_alpha(accent, 76),
                accent,
            )
        if pressed:
            return (Clr.BG_ACTIVE, Clr.BORDER_DEFAULT, Clr.TEXT_PRIMARY)
        if hovered:
            return (Clr.BG_HOVER, Clr.BORDER_DEFAULT, Clr.TEXT_PRIMARY)
        return (
            Clr.BG_SURFACE,
            Clr.BORDER_SUBTLE,
            Clr.TEXT_SECONDARY,
        )

    # ── PRIMARY ──
    if variant == ButtonVariant.PRIMARY:
        if pressed:
            return (accent, accent, QColor(0xFF, 0xFF, 0xFF))
        if hovered:
            return (accent, accent, QColor(0xFF, 0xFF, 0xFF))
        return (
            _with_alpha(accent, 38),
            _with_alpha(accent, 76),
            accent,
        )

    # ── DANGER ──
    if variant == ButtonVariant.DANGER:
        if pressed:
            return (Clr.RED, Clr.RED, QColor(0xFF, 0xFF, 0xFF))
        if hovered:
            return (
                _with_alpha(Clr.RED, 38),
                _with_alpha(Clr.RED, 76),
                Clr.RED,
            )
        return (
            Clr.BG_SURFACE,
            Clr.BORDER_SUBTLE,
            Clr.TEXT_SECONDARY,
        )

    # Fallback
    return (QColor(0, 0, 0, 0), QColor(0, 0, 0, 0), Clr.TEXT_SECONDARY)


# ═══════════════════════════════════════════════════════════════
# StyledButton
# ═══════════════════════════════════════════════════════════════


class StyledButton(QWidget):
    """A single button class for the entire application.

    Parameters
    ----------
    text : str
        Button label. May include emoji/unicode icons.
    variant : ButtonVariant
        Visual style: GHOST, SUBTLE, PRIMARY, or DANGER.
    checkable : bool
        If True, clicking toggles checked state.
    checked : bool
        Initial checked state (only meaningful if checkable).
    accent_override : QColor | None
        Override the accent color (e.g., red for Live button).
    font_size : float
        Font size in points.
    monospace : bool
        Use monospace font (for numeric labels in segmented controls).
    min_width : int | None
        Minimum width. If None, auto-sized from text.
    height : int
        Fixed height.
    """

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
        self._accent = accent_override or Clr.ACCENT
        self._font_size = font_size
        self._monospace = monospace
        self._hovered = False
        self._pressed = False

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self.setFixedHeight(height)

        # Auto-width from text
        font = self._make_font()
        fm = QFontMetricsF(font)
        text_w = fm.horizontalAdvance(text)
        w = max(int(text_w) + 20, min_width or 0)
        self.setMinimumWidth(w)
        if min_width is None:
            self.setFixedWidth(w)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

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
        if self._monospace:
            return _mono_font(self._font_size, QFont.Weight.Medium)
        return _ui_font(self._font_size, QFont.Weight.Medium)

    # ── Painting ──

    def paintEvent(self, event) -> None:
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

        # Background + border
        if border.alpha() > 0:
            p.setPen(QPen(border, 1))
        else:
            p.setPen(QPen(Qt.PenStyle.NoPen))

        if bg.alpha() > 0:
            p.setBrush(QBrush(bg))
        else:
            p.setBrush(QBrush(Qt.BrushStyle.NoBrush))

        p.drawRoundedRect(r, RADIUS, RADIUS)

        # Text
        p.setPen(text_color)
        p.setFont(self._make_font())
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, self._text)
        p.end()

    # ── Events ──

    def enterEvent(self, event) -> None:
        self._hovered = True
        self.update()

    def leaveEvent(self, event) -> None:
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
    """Exclusive toggle group for discrete values.

    Renders as a single pill-shaped container with segments.
    Used for step sizes, binning, modes, etc.

    Parameters
    ----------
    options : list[str]
        Labels for each segment.
    selected : int
        Initially selected index.
    monospace : bool
        Use monospace font for labels (numeric values).
    font_size : float
        Label font size.
    height : int
        Fixed height.
    """

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

        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(height)

        # Compute width
        font = self._make_font()
        fm = QFontMetricsF(font)
        max_label_w = max(fm.horizontalAdvance(o) for o in options)
        self._seg_w = max(int(max_label_w) + 12, 32)
        total_w = self._seg_w * len(options) + 4  # 2px padding each side
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
        if self._monospace:
            return _mono_font(self._font_size, QFont.Weight.Medium)
        return _ui_font(self._font_size, QFont.Weight.Medium)

    def _idx_at(self, x: float) -> int:
        """Return segment index at x pixel, or -1."""
        inner_x = x - 2  # account for padding
        if inner_x < 0:
            return -1
        idx = int(inner_x // self._seg_w)
        return idx if 0 <= idx < len(self._options) else -1

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        full = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)

        # Outer container
        p.setPen(QPen(Clr.BORDER_SUBTLE, 1))
        p.setBrush(QBrush(Clr.BG_RAISED))
        p.drawRoundedRect(full, RADIUS, RADIUS)

        font = self._make_font()
        p.setFont(font)

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

            # Segment background
            if is_sel:
                p.setPen(QPen(Qt.PenStyle.NoPen))
                p.setBrush(QBrush(_with_alpha(Clr.ACCENT, 38)))
                p.drawRoundedRect(seg_r, 2, 2)
                text_color = Clr.ACCENT
            elif is_press:
                p.setPen(QPen(Qt.PenStyle.NoPen))
                p.setBrush(QBrush(Clr.BG_ACTIVE))
                p.drawRoundedRect(seg_r, 2, 2)
                text_color = Clr.TEXT_PRIMARY
            elif is_hov:
                p.setPen(QPen(Qt.PenStyle.NoPen))
                p.setBrush(QBrush(Clr.BG_HOVER))
                p.drawRoundedRect(seg_r, 2, 2)
                text_color = Clr.TEXT_PRIMARY
            else:
                text_color = Clr.TEXT_SECONDARY

            # Label
            p.setPen(text_color)
            font_to_use = self._make_font()
            if is_sel:
                font_to_use.setWeight(QFont.Weight.DemiBold)
            p.setFont(font_to_use)
            p.drawText(seg_r, Qt.AlignmentFlag.AlignCenter, label)

        p.end()

    def enterEvent(self, event) -> None:
        pass  # handled by mouseMoveEvent

    def leaveEvent(self, event) -> None:
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
        self.setFixedSize(1, 18)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setPen(QPen(Clr.BORDER_SUBTLE, 1))
        p.drawLine(0, 0, 0, self.height())
        p.end()


# ═══════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════


def _make_palette():
    from PyQt6.QtGui import QPalette

    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, Clr.BG_BASE)
    p.setColor(QPalette.ColorRole.WindowText, Clr.TEXT_PRIMARY)
    p.setColor(QPalette.ColorRole.Base, QColor(0x12, 0x12, 0x12))
    p.setColor(QPalette.ColorRole.Button, Clr.BG_SURFACE)
    p.setColor(QPalette.ColorRole.ButtonText, Clr.TEXT_PRIMARY)
    p.setColor(QPalette.ColorRole.Highlight, Clr.ACCENT)
    p.setColor(QPalette.ColorRole.HighlightedText, Clr.TEXT_PRIMARY)
    p.setColor(QPalette.ColorRole.Mid, Clr.BORDER_DEFAULT)
    p.setColor(QPalette.ColorRole.Dark, QColor(0x12, 0x12, 0x12))
    p.setColor(QPalette.ColorRole.Shadow, QColor(0, 0, 0))
    p.setColor(QPalette.ColorRole.Light, Clr.BG_HOVER)
    p.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.ButtonText,
        Clr.TEXT_DISABLED,
    )
    return p


def main() -> None:
    import sys

    from PyQt6.QtGui import QPalette
    from PyQt6.QtWidgets import (
        QApplication,
        QLabel,
        QMainWindow,
        QVBoxLayout,
    )

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_make_palette())
    app.setFont(_ui_font(10))

    win = QMainWindow()
    win.setWindowTitle("Button System — Demo")
    win.resize(500, 600)

    central = QWidget()
    win.setCentralWidget(central)
    root = QVBoxLayout(central)
    root.setContentsMargins(24, 24, 24, 24)
    root.setSpacing(24)

    # ── Ghost ──
    ghost_group = QVBoxLayout()
    ghost_label = QLabel("GHOST — toolbars, viewport controls")
    ghost_label.setFont(_ui_font(8, QFont.Weight.DemiBold))
    gpal = ghost_label.palette()
    gpal.setColor(QPalette.ColorRole.WindowText, Clr.TEXT_DISABLED)
    ghost_label.setPalette(gpal)
    ghost_group.addWidget(ghost_label)

    ghost_row = QHBoxLayout()
    ghost_row.setSpacing(6)
    ghost_row.addWidget(StyledButton("📷 Snap", variant=ButtonVariant.GHOST))
    ghost_row.addWidget(StyledButton("🔍+", variant=ButtonVariant.GHOST))
    ghost_row.addWidget(StyledButton("Fit", variant=ButtonVariant.GHOST))
    ghost_row.addWidget(
        StyledButton("◐ Range", variant=ButtonVariant.GHOST, checkable=True)
    )
    ghost_row.addWidget(
        StyledButton(
            "● Live",
            variant=ButtonVariant.GHOST,
            checkable=True,
            checked=True,
            accent_override=Clr.RED,
        )
    )
    ghost_row.addStretch()
    ghost_group.addLayout(ghost_row)
    root.addLayout(ghost_group)

    # ── Subtle ──
    subtle_group = QVBoxLayout()
    subtle_label = QLabel("SUBTLE — panel actions, secondary controls")
    subtle_label.setFont(_ui_font(8, QFont.Weight.DemiBold))
    spal = subtle_label.palette()
    spal.setColor(QPalette.ColorRole.WindowText, Clr.TEXT_DISABLED)
    subtle_label.setPalette(spal)
    subtle_group.addWidget(subtle_label)

    subtle_row = QHBoxLayout()
    subtle_row.setSpacing(6)
    subtle_row.addWidget(StyledButton("↺ Reset", variant=ButtonVariant.SUBTLE))
    subtle_row.addWidget(
        StyledButton("㏒ Log", variant=ButtonVariant.SUBTLE, checkable=True)
    )
    subtle_row.addWidget(StyledButton("+ Add", variant=ButtonVariant.SUBTLE))
    subtle_row.addWidget(StyledButton("Edit", variant=ButtonVariant.SUBTLE))
    subtle_row.addStretch()
    subtle_group.addLayout(subtle_row)
    root.addLayout(subtle_group)

    # ── Primary ──
    primary_group = QVBoxLayout()
    primary_label = QLabel("PRIMARY — promoted actions")
    primary_label.setFont(_ui_font(8, QFont.Weight.DemiBold))
    ppal = primary_label.palette()
    ppal.setColor(QPalette.ColorRole.WindowText, Clr.TEXT_DISABLED)
    primary_label.setPalette(ppal)
    primary_group.addWidget(primary_label)

    primary_row = QHBoxLayout()
    primary_row.setSpacing(6)
    primary_row.addWidget(StyledButton("⚡ Auto", variant=ButtonVariant.PRIMARY))
    primary_row.addWidget(
        StyledButton(
            "Start 50m Acquisition (11.2 GB)",
            variant=ButtonVariant.PRIMARY,
            accent_override=Clr.GREEN,
            font_size=9,
            height=34,
        )
    )
    primary_row.addStretch()
    primary_group.addLayout(primary_row)
    root.addLayout(primary_group)

    # ── Danger ──
    danger_group = QVBoxLayout()
    danger_label = QLabel("DANGER — destructive actions")
    danger_label.setFont(_ui_font(8, QFont.Weight.DemiBold))
    dpal = danger_label.palette()
    dpal.setColor(QPalette.ColorRole.WindowText, Clr.TEXT_DISABLED)
    danger_label.setPalette(dpal)
    danger_group.addWidget(danger_label)

    danger_row = QHBoxLayout()
    danger_row.setSpacing(6)
    danger_row.addWidget(StyledButton("Clear all", variant=ButtonVariant.DANGER))
    danger_row.addWidget(StyledButton("■ Stop", variant=ButtonVariant.DANGER))
    danger_row.addWidget(StyledButton("✕ Remove", variant=ButtonVariant.DANGER))
    danger_row.addStretch()
    danger_group.addLayout(danger_row)
    root.addLayout(danger_group)

    # ── Segmented Controls ──
    seg_group = QVBoxLayout()
    seg_label = QLabel("SEGMENTED — exclusive toggles")
    seg_label.setFont(_ui_font(8, QFont.Weight.DemiBold))
    sgpal = seg_label.palette()
    sgpal.setColor(QPalette.ColorRole.WindowText, Clr.TEXT_DISABLED)
    seg_label.setPalette(sgpal)
    seg_group.addWidget(seg_label)

    seg_row = QHBoxLayout()
    seg_row.setSpacing(16)
    seg_row.addWidget(SegmentedControl(["0.1", "1", "10", "100", "1k"], selected=2))
    seg_row.addWidget(SegmentedControl(["1×1", "2×2", "4×4"], selected=0))
    seg_row.addWidget(
        SegmentedControl(["Joystick", "D-Pad"], selected=0, monospace=False)
    )
    seg_row.addStretch()
    seg_group.addLayout(seg_row)
    root.addLayout(seg_group)

    root.addStretch()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
