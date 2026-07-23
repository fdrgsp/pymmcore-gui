"""Collapsible accordion panel — same interaction style as _modern_gui's sidebar.

A leaner port: chevron-rotation + height animation are kept (the visual
identity), but the status dot and drag-to-reorder features aren't — neither
current use site (group/presets, stage controls) needs them.
"""

from __future__ import annotations

from pymmcore_gui._qt.QtCore import (
    QEasingCurve,
    QEvent,
    QPointF,
    QPropertyAnimation,
    QRectF,
    Qt,
    pyqtProperty,
    pyqtSignal,
)
from pymmcore_gui._qt.QtGui import (
    QEnterEvent,
    QFont,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPolygonF,
)
from pymmcore_gui._qt.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from ._theme import qcolor, theme, ui_font

_QWIDGETSIZE_MAX = 16777215


class CollapsiblePanelHeader(QWidget):
    """Clickable header: an animated chevron plus a title, all custom-painted."""

    clicked = pyqtSignal()

    def __init__(
        self, title: str, expanded: bool, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._hovered = False
        self._chevron_angle = 90.0 if expanded else 0.0

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(theme().row_height)

        self._anim = QPropertyAnimation(self, b"chevronAngle")
        self._anim.setDuration(180)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _get_chevron_angle(self) -> float:
        return self._chevron_angle

    def _set_chevron_angle(self, value: float) -> None:
        self._chevron_angle = value
        self.update()

    chevronAngle = pyqtProperty(float, _get_chevron_angle, _set_chevron_angle)

    def set_expanded(self, expanded: bool, *, animate: bool = True) -> None:
        target = 90.0 if expanded else 0.0
        if not animate:
            self._anim.stop()
            self._set_chevron_angle(target)
            return
        self._anim.stop()
        self._anim.setStartValue(self._chevron_angle)
        self._anim.setEndValue(target)
        self._anim.start()

    def paintEvent(self, event: QPaintEvent | None) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        t = theme()
        w, h = self.width(), self.height()

        p.fillRect(self.rect(), qcolor(t.bg_raised if self._hovered else t.bg_base))
        p.setPen(qcolor(t.border_subtle))
        p.drawLine(0, h - 1, w, h - 1)

        # chevron: a small triangle rotated 0deg (collapsed, pointing right)
        # to 90deg (expanded, pointing down)
        size = t.scaled(4)
        cx, cy = t.sp_lg, h / 2
        p.save()
        p.translate(cx, cy)
        p.rotate(self._chevron_angle)
        triangle = QPolygonF(
            [QPointF(-size, -size), QPointF(size, 0), QPointF(-size, size)]
        )
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qcolor(t.text_secondary))
        p.drawPolygon(triangle)
        p.restore()

        p.setFont(ui_font(11, QFont.Weight.DemiBold))
        p.setPen(qcolor(t.text_primary))
        title_x = t.sp_lg * 2 + size
        p.drawText(
            QRectF(title_x, 0, w - title_x - t.sp_lg, h),
            Qt.AlignmentFlag.AlignVCenter,
            self._title,
        )
        p.end()

    def enterEvent(self, event: QEnterEvent | None) -> None:
        self._hovered = True
        self.update()

    def leaveEvent(self, event: QEvent | None) -> None:
        self._hovered = False
        self.update()

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class CollapsiblePanel(QWidget):
    """A titled section that expands/collapses with an animated height.

    Add body content via :attr:`body_layout`. While collapsed, only the
    header is visible; while expanded (and once its opening animation
    settles) the panel behaves like a normal widget again, so a parent
    layout's stretch factor can still make it grow to fill available space.
    """

    def __init__(
        self,
        title: str,
        *,
        expanded: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._expanded = expanded

        self._header = CollapsiblePanelHeader(title, expanded)
        self._header.clicked.connect(self.toggle)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        m = theme().sp_sm
        self._body_layout.setContentsMargins(m, m, m, m)
        self._body_layout.setSpacing(m)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._header)
        layout.addWidget(self._body)

        self._anim = QPropertyAnimation(self, b"maximumHeight")
        self._anim.setDuration(220)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.finished.connect(self._on_animation_finished)

        # set the initial state instantly — no animation on construction
        if expanded:
            self.setMaximumHeight(_QWIDGETSIZE_MAX)
        else:
            self._body.setMaximumHeight(0)
            self.setMaximumHeight(self._header.height())

    @property
    def body_layout(self) -> QVBoxLayout:
        """Layout to add this panel's content to."""
        return self._body_layout

    @property
    def expanded(self) -> bool:
        return self._expanded

    def toggle(self) -> None:
        self.set_expanded(not self._expanded)

    def set_expanded(self, expand: bool) -> None:
        if expand == self._expanded:
            return
        self._expanded = expand
        self._header.set_expanded(expand)

        self._anim.stop()
        current_h = self.height()
        header_h = self._header.height()
        if expand:
            self._body.setMaximumHeight(_QWIDGETSIZE_MAX)
            target = header_h + self._body.sizeHint().height()
        else:
            target = header_h
        self.setMaximumHeight(current_h)
        self._anim.setStartValue(current_h)
        self._anim.setEndValue(target)
        self._anim.start()

    def _on_animation_finished(self) -> None:
        if self._expanded:
            # release the clamp: let a parent layout's stretch factor take over
            self.setMaximumHeight(_QWIDGETSIZE_MAX)
        else:
            self._body.setMaximumHeight(0)
