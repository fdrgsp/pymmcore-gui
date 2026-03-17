"""XY Stage positioning panel for the sidebar."""

from __future__ import annotations

import math
from contextlib import suppress
from typing import TYPE_CHECKING, ClassVar

from pymmcore_widgets.control._q_stage_controller import QStageMoveAccumulator

from pymmcore_gui._modern_gui._utils import current_core
from pymmcore_gui._qt.QtCore import (
    QEvent,
    QPointF,
    QRectF,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from pymmcore_gui._qt.QtGui import (
    QColor,
    QDoubleValidator,
    QFont,
    QFontMetrics,
    QPainter,
    QPen,
)
from pymmcore_gui._qt.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from pymmcore_gui.widgets._joystick import JoystickWidget

from ._theme import mono_font, qcolor, theme, ui_font

if TYPE_CHECKING:
    from pymmcore_gui._qt.QtGui import QKeyEvent, QMouseEvent, QPaintEvent


# Coordinate Display


class CoordinateDisplay(QWidget):
    """Shows current X/Y coordinates with colored axis labels."""

    _BASE_HEIGHT = 60

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._x = 0.0
        self._y = 0.0
        self._moving = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_coordinates(self, x: float, y: float) -> None:
        self._x = x
        self._y = y
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(0, theme().scaled(self._BASE_HEIGHT))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        t = theme()
        w, h = self.width(), self.height()

        # Border box
        p.setPen(QPen(qcolor(t.border_subtle), 1))
        p.setBrush(qcolor(t.bg_raised))
        p.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), t.radius, t.radius)

        pad = t.sp_sm
        row_h = h // 2

        val_color = qcolor(t.status_amber if self._moving else t.text_primary)
        unit_color = qcolor(t.text_disabled)

        for i, (label, val, color) in enumerate(
            [("X", self._x, "#EF6B6B"), ("Y", self._y, "#6BCF6B")]
        ):
            y_off = i * row_h

            # Axis label
            axis_font = mono_font(10, QFont.Weight.DemiBold)
            p.setFont(axis_font)
            p.setPen(QColor(color))
            p.drawText(
                pad,
                y_off,
                t.scaled(20),
                row_h,
                Qt.AlignmentFlag.AlignVCenter,
                label,
            )

            # Value
            value_font = mono_font(12, QFont.Weight.Medium)
            p.setFont(value_font)
            p.setPen(val_color)
            fm = QFontMetrics(value_font)
            val_text = f"{val:.2f}"
            val_w = fm.horizontalAdvance(val_text)

            # Unit
            unit_font = ui_font(9)
            ufm = QFontMetrics(unit_font)
            unit_w = ufm.horizontalAdvance("\u03bcm")
            right_edge = w - pad

            p.drawText(
                right_edge - unit_w - t.sp_xs - val_w,
                y_off,
                val_w,
                row_h,
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                val_text,
            )

            p.setFont(unit_font)
            p.setPen(unit_color)
            p.drawText(
                right_edge - unit_w,
                y_off,
                unit_w,
                row_h,
                Qt.AlignmentFlag.AlignVCenter,
                "\u03bcm",
            )

        p.end()


# Mode Tab Bar (Joystick / D-Pad)


class ModeTabBar(QWidget):
    """Two-button toggle: Joystick / D-Pad."""

    _BASE_HEIGHT = 28

    modeChanged = Signal(int)  # 0 = joystick, 1 = dpad

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._active = 0
        self._hovered = -1
        self._labels = ["Joystick", "D-Pad"]
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)

    @property
    def active_mode(self) -> int:
        return self._active

    def sizeHint(self) -> QSize:
        return QSize(0, theme().scaled(self._BASE_HEIGHT))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def _tab_rect(self, index: int) -> QRectF:
        pad = theme().scaled(2)
        w = (self.width() - 2 * pad) / 2
        return QRectF(pad + index * w, pad, w, self.height() - 2 * pad)

    def _index_at(self, x: float) -> int:
        for i in range(2):
            if self._tab_rect(i).contains(QPointF(x, self.height() / 2)):
                return i
        return -1

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        idx = self._index_at(ev.position().x())
        if idx >= 0 and idx != self._active:
            self._active = idx
            self.modeChanged.emit(idx)
            self.update()

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        idx = self._index_at(ev.position().x())
        if idx != self._hovered:
            self._hovered = idx
            self.update()

    def leaveEvent(self, a0: object) -> None:
        self._hovered = -1
        self.update()

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        t = theme()
        w, h = self.width(), self.height()

        # Outer container
        p.setPen(QPen(qcolor(t.border_subtle), 1))
        p.setBrush(qcolor(t.bg_raised))
        p.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), t.radius, t.radius)

        font = ui_font(9)
        p.setFont(font)

        for i, label in enumerate(self._labels):
            rect = self._tab_rect(i)
            if i == self._active:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(qcolor(t.accent_muted))
                p.drawRoundedRect(rect.adjusted(1, 1, -1, -1), t.scaled(2), t.scaled(2))
                p.setPen(qcolor(t.accent))
                p.setFont(ui_font(9, QFont.Weight.Medium))
            elif i == self._hovered:
                p.setPen(qcolor(t.text_primary))
                p.setFont(ui_font(9))
            else:
                p.setPen(qcolor(t.text_secondary))
                p.setFont(ui_font(9))

            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

        p.end()


# D-Pad with integrated step size selector

STEP_SIZES = [0.1, 1, 10, 100, 1000]
STEP_LABELS = ["0.1", "1", "10", "100", "1k"]

# Layout constants (base px, scaled at paint time)
_LABEL_H = 16
_STEP_H = 32
_DPAD_BTN = 44
_DPAD_GAP = 2
_HINT_H = 16


class DPadWidget(QWidget):
    """D-Pad with built-in step size selector. Emits pre-scaled move values."""

    stepRequested = Signal(float, float)  # pre-scaled (dx * step, dy * step)
    homeRequested = Signal()

    _KEY_TO_BTN: ClassVar[dict[Qt.Key, int]] = {
        Qt.Key.Key_Up: 1,
        Qt.Key.Key_Down: 7,
        Qt.Key.Key_Left: 3,
        Qt.Key.Key_Right: 5,
        Qt.Key.Key_Home: 4,
    }

    _DPAD_BUTTONS: ClassVar[dict[int, tuple[str, int, int]]] = {
        0: ("\u2196", -1, 1),  # top-left
        1: ("\u25b2", 0, 1),  # up
        2: ("\u2197", 1, 1),  # top-right
        3: ("\u25c0", -1, 0),  # left
        4: ("\u25ce", 0, 0),  # center/home
        5: ("\u25b6", 1, 0),  # right
        6: ("\u2199", -1, -1),  # bottom-left
        7: ("\u25bc", 0, -1),  # down
        8: ("\u2198", 1, -1),  # bottom-right
    }
    _DIAGONALS: ClassVar[set[int]] = {0, 2, 6, 8}

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._step_index = 2  # default: 10
        self._hovered_step = -1
        self._hovered_dpad = -1
        self._pressed_dpad = -1
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    @property
    def step_size(self) -> float:
        return STEP_SIZES[self._step_index]

    # ── geometry helpers ──

    def _s(self, base: int | float) -> int:
        return theme().scaled(base)

    def _label_rect(self) -> QRectF:
        return QRectF(0, 0, self.width(), self._s(_LABEL_H))

    def _step_row_rect(self) -> QRectF:
        y = self._s(_LABEL_H + 4)  # label + sp_xxs gap
        return QRectF(0, y, self.width(), self._s(_STEP_H))

    def _step_btn_rect(self, index: int) -> QRectF:
        row = self._step_row_rect()
        pad = self._s(2)
        inner_w = row.width() - 2 * pad
        btn_w = inner_w / len(STEP_SIZES)
        return QRectF(
            row.x() + pad + index * btn_w,
            row.y() + pad,
            btn_w,
            row.height() - 2 * pad,
        )

    def _grid_origin_y(self) -> float:
        return self._step_row_rect().bottom() + self._s(8)

    def _grid_size(self) -> int:
        return 3 * self._s(_DPAD_BTN) + 2 * self._s(_DPAD_GAP)

    def _dpad_cell_rect(self, flat: int) -> QRectF:
        row, col = divmod(flat, 3)
        bs = self._s(_DPAD_BTN)
        g = self._s(_DPAD_GAP)
        gs = self._grid_size()
        x_off = (self.width() - gs) / 2  # center the grid
        y = self._grid_origin_y() + row * (bs + g)
        return QRectF(x_off + col * (bs + g), y, bs, bs)

    def _hint_rect(self) -> QRectF:
        y = self._grid_origin_y() + self._grid_size() + self._s(4)
        return QRectF(0, y, self.width(), self._s(_HINT_H))

    def _total_height(self) -> int:
        return int(self._hint_rect().bottom())

    def sizeHint(self) -> QSize:
        return QSize(0, self._total_height())

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    # ── hit testing ──

    def _step_index_at(self, pos: QPointF) -> int:
        for i in range(len(STEP_SIZES)):
            if self._step_btn_rect(i).contains(pos):
                return i
        return -1

    def _dpad_index_at(self, pos: QPointF) -> int:
        for idx in self._DPAD_BUTTONS:
            if self._dpad_cell_rect(idx).contains(pos):
                return idx
        return -1

    # ── mouse events ──

    def mousePressEvent(self, a0: QMouseEvent) -> None:
        pos = a0.position()
        # Step buttons: select on press
        si = self._step_index_at(pos)
        if si >= 0:
            self._step_index = si
            self.update()
            return
        # D-pad buttons: track press
        di = self._dpad_index_at(pos)
        if di >= 0:
            self._pressed_dpad = di
            self.update()

    def mouseReleaseEvent(self, a0: QMouseEvent) -> None:
        pos = a0.position()
        di = self._dpad_index_at(pos)
        if di >= 0 and di == self._pressed_dpad:
            _, dx, dy = self._DPAD_BUTTONS[di]
            if di == 4:
                self.homeRequested.emit()
            else:
                step = self.step_size
                self.stepRequested.emit(dx * step, dy * step)
        self._pressed_dpad = -1
        self.update()

    def mouseMoveEvent(self, a0: QMouseEvent) -> None:
        pos = a0.position()
        si = self._step_index_at(pos)
        di = self._dpad_index_at(pos)
        changed = False
        if si != self._hovered_step:
            self._hovered_step = si
            changed = True
        if di != self._hovered_dpad:
            self._hovered_dpad = di
            changed = True
        if changed:
            self.update()

    def leaveEvent(self, a0: object) -> None:
        self._hovered_step = -1
        self._hovered_dpad = -1
        self._pressed_dpad = -1
        self.update()

    def trigger_button(self, btn_index: int) -> None:
        """Programmatically activate a d-pad button with visual feedback."""
        if btn_index not in self._DPAD_BUTTONS:
            return
        _, dx, dy = self._DPAD_BUTTONS[btn_index]
        if btn_index == 4:
            self.homeRequested.emit()
        else:
            step = self.step_size
            self.stepRequested.emit(dx * step, dy * step)
        # Visual press highlight
        self._pressed_dpad = btn_index
        self.update()
        QTimer.singleShot(100, self._clear_press)

    def _clear_press(self) -> None:
        self._pressed_dpad = -1
        self.update()

    def keyPressEvent(self, a0: QKeyEvent | None) -> None:
        if a0 is not None:
            btn = self._KEY_TO_BTN.get(Qt.Key(a0.key()))
            if btn is not None:
                self.trigger_button(btn)
                return
        super().keyPressEvent(a0)  # type: ignore[arg-type]

    # ── painting ──

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        t = theme()

        # Section label
        p.setFont(ui_font(9, QFont.Weight.Medium))
        p.setPen(qcolor(t.text_secondary))
        p.drawText(self._label_rect(), Qt.AlignmentFlag.AlignVCenter, "STEP SIZE")

        # Step size row background
        row = self._step_row_rect()
        p.setPen(QPen(qcolor(t.border_subtle), 1))
        p.setBrush(qcolor(t.bg_raised))
        p.drawRoundedRect(row.adjusted(0.5, 0.5, -0.5, -0.5), t.radius, t.radius)

        p.setFont(mono_font(9))
        for i, label in enumerate(STEP_LABELS):
            rect = self._step_btn_rect(i)
            if i == self._step_index:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(qcolor(t.accent_muted))
                p.drawRoundedRect(rect.adjusted(1, 1, -1, -1), self._s(2), self._s(2))
                p.setPen(qcolor(t.accent))
            elif i == self._hovered_step:
                p.setPen(qcolor(t.text_primary))
            else:
                p.setPen(qcolor(t.text_secondary))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

        # D-pad buttons
        for idx, (label, _dx, _dy) in self._DPAD_BUTTONS.items():
            rect = self._dpad_cell_rect(idx)
            is_center = idx == 4
            is_hovered = idx == self._hovered_dpad
            is_pressed = idx == self._pressed_dpad

            if is_pressed:
                bg = qcolor(t.accent_muted)
                border = qcolor(t.accent)
                fg = qcolor(t.accent)
            elif is_center and is_hovered:
                bg = QColor(239, 83, 80, 40)
                border = QColor(239, 83, 80)
                fg = QColor(255, 255, 255)
            elif is_hovered:
                bg = qcolor(t.bg_hover)
                border = qcolor(t.border_default)
                fg = qcolor(t.text_primary)
            elif is_center:
                bg = qcolor(t.bg_raised)
                border = qcolor(t.border_default)
                fg = qcolor(t.text_disabled)
            else:
                bg = qcolor(t.bg_surface)
                border = qcolor(t.border_subtle)
                fg = qcolor(t.text_secondary)

            p.setPen(QPen(border, 1))
            p.setBrush(bg)
            p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), t.radius, t.radius)

            p.setPen(fg)
            is_diag = idx in self._DIAGONALS
            font_size = 10 if is_center else (11 if is_diag else 14)
            p.setFont(ui_font(font_size))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

        # Hint text
        p.setFont(ui_font(8))
        p.setPen(qcolor(t.text_disabled))
        p.drawText(
            self._hint_rect(),
            Qt.AlignmentFlag.AlignCenter,
            "\u2191 \u2193 \u2190 \u2192 arrow keys",
        )

        p.end()


# Joystick wrapper (reuses JoystickWidget from widgets/_joystick.py)


class JoystickPanel(QWidget):
    """Wraps JoystickWidget with speed indicator and hint text."""

    moveRequested = Signal(float, float)  # pre-computed (dx_um, dy_um)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._joystick = JoystickWidget(self)
        self._joystick.deflectionChanged.connect(self._on_deflection)
        self._joystick.released.connect(self._on_release)

        self._dx = 0.0
        self._dy = 0.0
        self._tick_ms = 50
        self._speed_exponent = 2.0
        self._max_speed = 500.0

        self._speed_label = QLabel("Speed")
        self._speed_value = QLabel("0 \u03bcm/s")
        self._speed_bar = _SpeedBar()

        self._hint = QLabel("Click & drag to move \u00b7 distance from center = speed")
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)

        speed_row = QHBoxLayout()
        speed_row.setContentsMargins(0, 0, 0, 0)
        speed_row.addWidget(self._speed_label)
        speed_row.addWidget(self._speed_bar, 1)
        speed_row.addWidget(self._speed_value)

        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(self._tick_ms)
        self._tick_timer.timeout.connect(self._on_tick)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._joystick)
        layout.addLayout(speed_row)
        layout.addWidget(self._hint)

    def changeEvent(self, a0: object) -> None:
        if isinstance(a0, QEvent) and a0.type() == QEvent.Type.StyleChange:
            self._setup_styles()
        super().changeEvent(a0)  # type: ignore[arg-type]

    def _on_deflection(self, dx: float, dy: float) -> None:
        mag = min(math.hypot(dx, dy), 1.0)
        self._speed_bar.set_fraction(mag)
        self._speed_value.setText(f"{mag * 100:.0f}%")
        self._dx = dx
        self._dy = dy
        if not self._tick_timer.isActive() and (dx or dy):
            self._tick_timer.start()

    def _on_release(self) -> None:
        self._tick_timer.stop()

    def _on_tick(self) -> None:
        mag = min(math.hypot(self._dx, self._dy), 1.0)
        if mag < 0.05:
            return
        speed = mag**self._speed_exponent * self._max_speed
        ux, uy = self._dx / mag, self._dy / mag
        dt = self._tick_ms / 1000.0
        self.moveRequested.emit(ux * speed * dt, uy * speed * dt)

    def _setup_styles(self) -> None:
        self._speed_label.setFont(ui_font(8))
        self._speed_value.setFont(mono_font(8))
        self._hint.setFont(ui_font(8))
        pal = self._speed_label.palette()
        pal.setColor(pal.ColorRole.WindowText, qcolor(theme().text_disabled))
        self._speed_label.setPalette(pal)
        self._speed_value.setPalette(pal)
        self._hint.setPalette(pal)

    def showEvent(self, a0: object) -> None:
        self._setup_styles()
        super().showEvent(a0)  # type: ignore[arg-type]


class _SpeedBar(QWidget):
    """Thin horizontal bar showing joystick speed fraction."""

    _BASE_HEIGHT = 3

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fraction = 0.0
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_fraction(self, f: float) -> None:
        self._fraction = max(0.0, min(1.0, f))
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(0, theme().scaled(self._BASE_HEIGHT))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        t = theme()
        w, h = self.width(), self.height()
        r = h / 2

        # Track
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qcolor(t.bg_surface))
        p.drawRoundedRect(QRectF(0, 0, w, h), r, r)

        # Fill
        if self._fraction > 0.01:
            fill_w = self._fraction * w
            color = qcolor(t.status_amber if self._fraction > 0.7 else t.accent)
            p.setBrush(color)
            p.drawRoundedRect(QRectF(0, 0, fill_w, h), r, r)

        p.end()


# Go-To Position


class GoToSection(QWidget):
    """X/Y coordinate inputs with a Move button."""

    moveRequested = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._x_label = QLabel("X")
        self._y_label = QLabel("Y")
        self._x_input = QLineEdit("0")
        self._y_input = QLineEdit("0")
        self._move_btn = QPushButton("\u2197 Move to position")

        self._x_input.setValidator(QDoubleValidator())
        self._y_input.setValidator(QDoubleValidator())

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(self._x_label, 0, 0)
        grid.addWidget(self._x_input, 0, 1)
        grid.addWidget(self._y_label, 0, 2)
        grid.addWidget(self._y_input, 0, 3)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(grid)
        layout.addWidget(self._move_btn)

        self._move_btn.clicked.connect(self._on_move)

    def _on_move(self) -> None:
        try:
            x = float(self._x_input.text() or "0")
            y = float(self._y_input.text() or "0")
        except ValueError:
            return
        self.moveRequested.emit(x, y)

    def changeEvent(self, a0: object) -> None:
        from pymmcore_gui._qt.QtCore import QEvent

        if isinstance(a0, QEvent) and a0.type() == QEvent.Type.StyleChange:
            self._apply_styles()
        super().changeEvent(a0)  # type: ignore[arg-type]

    def _apply_styles(self) -> None:
        t = theme()
        for label, color in [(self._x_label, "#EF6B6B"), (self._y_label, "#6BCF6B")]:
            label.setFont(mono_font(9, QFont.Weight.DemiBold))
            pal = label.palette()
            pal.setColor(pal.ColorRole.WindowText, QColor(color))
            label.setPalette(pal)

        for inp in (self._x_input, self._y_input):
            inp.setFont(mono_font(10))

        self._move_btn.setFont(ui_font(9, QFont.Weight.Medium))

        # Stylesheet for inputs to match theme
        inp_style = (
            f"QLineEdit {{"
            f"  background: {_css_color(t.bg_raised)};"
            f"  border: 1px solid {_css_color(t.border_default)};"
            f"  border-radius: {t.radius}px;"
            f"  color: {_css_color(t.text_primary)};"
            f"  padding: {t.sp_xxs}px {t.sp_xs}px;"
            f"}}"
            f"QLineEdit:focus {{"
            f"  border-color: {_css_color(t.accent)};"
            f"}}"
        )
        self._x_input.setStyleSheet(inp_style)
        self._y_input.setStyleSheet(inp_style)

        btn_style = (
            f"QPushButton {{"
            f"  background: {_css_color(t.bg_surface)};"
            f"  border: 1px solid {_css_color(t.border_default)};"
            f"  border-radius: {t.radius}px;"
            f"  color: {_css_color(t.text_secondary)};"
            f"  padding: {t.sp_xxs + 2}px {t.sp_md}px;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background: {_css_color(t.accent_muted)};"
            f"  color: {_css_color(t.accent)};"
            f"  border-color: {_css_color(t.accent)};"
            f"}}"
        )
        self._move_btn.setStyleSheet(btn_style)

    def showEvent(self, a0: object) -> None:
        self._apply_styles()
        super().showEvent(a0)  # type: ignore[arg-type]


# Saved Positions


class SavedPositionItem(QWidget):
    """Single row in the saved positions list."""

    _BASE_HEIGHT = 32

    goRequested = Signal(float, float)

    def __init__(
        self, name: str, x: float, y: float, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._name = name
        self._x = x
        self._y = y
        self._hovered = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def sizeHint(self) -> QSize:
        return QSize(0, theme().scaled(self._BASE_HEIGHT))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def enterEvent(self, a0: object) -> None:
        self._hovered = True
        self.update()

    def leaveEvent(self, a0: object) -> None:
        self._hovered = False
        self.update()

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        self.goRequested.emit(self._x, self._y)

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        t = theme()
        w, h = self.width(), self.height()

        bg = qcolor(t.bg_hover if self._hovered else t.bg_raised)
        border = qcolor(t.border_subtle if self._hovered else t.bg_raised)
        p.setPen(QPen(border, 1))
        p.setBrush(bg)
        p.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), t.radius, t.radius)

        pad = t.sp_sm

        # Name
        p.setFont(ui_font(9, QFont.Weight.Medium))
        p.setPen(qcolor(t.text_primary))
        p.drawText(pad, 0, w // 2, h, Qt.AlignmentFlag.AlignVCenter, self._name)

        # Coordinates
        p.setFont(mono_font(8))
        p.setPen(qcolor(t.text_secondary))

        def _fmt(v: float) -> str:
            if abs(v) >= 10000:
                return f"{v / 1000:.1f}k"
            return f"{v:.1f}"

        coord_text = f"{_fmt(self._x)},  {_fmt(self._y)}"
        p.drawText(
            w // 2,
            0,
            w // 2 - pad,
            h,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            coord_text,
        )

        p.end()


class SavedPositionsSection(QWidget):
    """List of saved positions with save/clear buttons."""

    goRequested = Signal(float, float)
    saveRequested = Signal()
    clearRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._list_layout = QVBoxLayout()
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(theme().scaled(2))

        self._save_btn = QPushButton("+ Save current")
        self._clear_btn = QPushButton("Clear all")

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.addWidget(self._save_btn)
        btn_row.addWidget(self._clear_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._list_layout)
        layout.addLayout(btn_row)

        self._save_btn.clicked.connect(self.saveRequested)
        self._clear_btn.clicked.connect(self._on_clear)

        # Demo positions
        self._positions: list[tuple[str, float, float]] = [("Origin", 0.0, 0.0)]
        self._rebuild_list()

    def add_position(self, name: str, x: float, y: float) -> None:
        self._positions.append((name, x, y))
        self._rebuild_list()

    def clear_positions(self) -> None:
        self._positions.clear()
        self._rebuild_list()

    def _on_clear(self) -> None:
        self.clear_positions()
        self.clearRequested.emit()

    def _rebuild_list(self) -> None:
        # Clear existing items
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            if item and (w := item.widget()):
                w.deleteLater()

        for name, x, y in self._positions:
            item = SavedPositionItem(name, x, y)
            item.goRequested.connect(self.goRequested)
            self._list_layout.addWidget(item)

    def changeEvent(self, a0: object) -> None:
        from pymmcore_gui._qt.QtCore import QEvent

        if isinstance(a0, QEvent) and a0.type() == QEvent.Type.StyleChange:
            self._list_layout.setSpacing(theme().scaled(2))
            self._apply_btn_styles()
        super().changeEvent(a0)  # type: ignore[arg-type]

    def _apply_btn_styles(self) -> None:
        t = theme()
        btn_style = (
            f"QPushButton {{"
            f"  background: {_css_color(t.bg_surface)};"
            f"  border: 1px solid {_css_color(t.border_subtle)};"
            f"  border-radius: {t.radius}px;"
            f"  color: {_css_color(t.text_secondary)};"
            f"  padding: {t.sp_xxs}px;"
            f"  font-size: {t.scaled(8)}pt;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background: {_css_color(t.bg_hover)};"
            f"  color: {_css_color(t.text_primary)};"
            f"}}"
        )
        self._save_btn.setStyleSheet(btn_style)

        danger_style = (
            f"QPushButton {{"
            f"  background: {_css_color(t.bg_surface)};"
            f"  border: 1px solid {_css_color(t.border_subtle)};"
            f"  border-radius: {t.radius}px;"
            f"  color: {_css_color(t.text_secondary)};"
            f"  padding: {t.sp_xxs}px;"
            f"  font-size: {t.scaled(8)}pt;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background: rgba(239, 83, 80, 0.15);"
            f"  color: rgb(239, 83, 80);"
            f"  border-color: rgb(239, 83, 80);"
            f"}}"
        )
        self._clear_btn.setStyleSheet(danger_style)

    def showEvent(self, a0: object) -> None:
        self._apply_btn_styles()
        super().showEvent(a0)  # type: ignore[arg-type]


# Section Label


class _SectionLabel(QLabel):
    """Uppercase section header label."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text.upper(), parent)
        self._apply_style()

    def _apply_style(self) -> None:
        self.setFont(ui_font(9, QFont.Weight.Medium))
        pal = self.palette()
        pal.setColor(pal.ColorRole.WindowText, qcolor(theme().text_secondary))
        self.setPalette(pal)

    def changeEvent(self, a0: object) -> None:
        from pymmcore_gui._qt.QtCore import QEvent

        if isinstance(a0, QEvent) and a0.type() == QEvent.Type.StyleChange:
            self._apply_style()
        super().changeEvent(a0)  # type: ignore[arg-type]


# Divider


class _Divider(QWidget):
    """1px horizontal divider line."""

    def sizeHint(self) -> QSize:
        return QSize(0, 1)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        p = QPainter(self)
        p.setPen(QPen(qcolor(theme().border_subtle), 1))
        y = self.height() // 2
        p.drawLine(0, y, self.width(), y)
        p.end()


# Main Panel


class XYStagePanel(QWidget):
    """Complete XY stage control panel for the sidebar."""

    summaryChanged = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._coord_display = CoordinateDisplay()
        self._mode_tabs = ModeTabBar()

        self._joystick_panel = JoystickPanel()
        self._joystick_panel.moveRequested.connect(self._on_move_relative)
        self._dpad = DPadWidget()
        self._dpad.stepRequested.connect(self._on_move_relative)

        self._mode_stack = QStackedWidget()
        self._mode_stack.addWidget(self._joystick_panel)
        self._mode_stack.addWidget(self._dpad)

        self._goto = GoToSection()
        self._goto.moveRequested.connect(self._on_move_absolute)
        self._saved = SavedPositionsSection()
        self._saved.goRequested.connect(self._on_move_absolute)

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._coord_display)
        layout.addWidget(self._mode_tabs)
        layout.addWidget(self._mode_stack)
        layout.addWidget(_Divider())
        layout.addWidget(_SectionLabel("Go to position"))
        layout.addWidget(self._goto)
        layout.addWidget(_Divider())
        layout.addWidget(_SectionLabel("Saved positions"))
        layout.addWidget(self._saved)

        # Connections
        self._mode_tabs.modeChanged.connect(self._mode_stack.setCurrentIndex)
        if core := current_core(self):
            core.events.XYStagePositionChanged.connect(self._on_xy_stage_changed)

    def keyPressEvent(self, a0: QKeyEvent | None) -> None:
        if a0 is not None and self._mode_stack.currentWidget() is self._dpad:
            btn = DPadWidget._KEY_TO_BTN.get(Qt.Key(a0.key()))
            if btn is not None:
                self._dpad.trigger_button(btn)
                return
        super().keyPressEvent(a0)  # type: ignore[arg-type]

    def _on_move_relative(self, dx: float, dy: float) -> None:
        if (core := current_core(self)) and (xy_device := core.getXYStageDevice()):
            _acc = QStageMoveAccumulator.for_device(xy_device, core)
            with suppress(TypeError):  # FIXME
                _acc.moveFinished.connect(
                    self._on_pos_changed, Qt.ConnectionType.UniqueConnection
                )
            _acc.move_relative((dx, dy))

    def _on_move_absolute(self, x: float, y: float) -> None:
        if (core := current_core(self)) and (xy_device := core.getXYStageDevice()):
            _acc = QStageMoveAccumulator.for_device(xy_device, core)
            with suppress(TypeError):  # FIXME
                _acc.moveFinished.connect(
                    self._on_pos_changed, Qt.ConnectionType.UniqueConnection
                )
            _acc.move_absolute((x, y))

    def _update_position(self, x: float, y: float) -> None:
        self._coord_display.set_coordinates(x, y)
        self.summaryChanged.emit(f"{x:.2f}, {y:.2f}")

    def _on_xy_stage_changed(self, name: str, x: float, y: float) -> None:
        self._update_position(x, y)

    def _on_pos_changed(self) -> None:
        if core := current_core(self):
            self._update_position(*core.getXYPosition())


# Collapsible panel subclass


def CollapsibleXYStagePanel(parent: QWidget | None = None) -> QWidget:
    """Create an XY Stage panel wrapped in a collapsible header."""
    from ._sidebar import CollapsiblePanel

    panel = CollapsiblePanel(
        title="XY Stage",
        summary="0.0, 0.0",
        expanded=True,
        parent=parent,
    )
    content = XYStagePanel()
    content.summaryChanged.connect(lambda text: setattr(panel.header, "summary", text))
    panel.body_layout.addWidget(content)
    return panel


# Helpers


def _css_color(c: object) -> str:
    """Convert a theme Color to a CSS rgb() string."""
    qc = qcolor(c)  # type: ignore[arg-type]
    return f"rgb({qc.red()}, {qc.green()}, {qc.blue()})"
