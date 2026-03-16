from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pymmcore_gui._qt.QtGui import QPen
from pymmcore_gui._qt.QtWidgets import (
    QProxyStyle,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStyleOption,
    QWidget,
)

if TYPE_CHECKING:
    from pymmcore_gui._qt.QtCore import Qt
    from pymmcore_gui._qt.QtGui import QPainter


class MicroscopeStyle(QProxyStyle):
    """Zoomable proxy over Fusion.

    All pixel metrics are scaled by `zoom_factor`. Custom base overrides
    (e.g. thin scrollbars) are preserved via `_BASE_OVERRIDES` and still
    route through zoom.
    """

    _BASE_OVERRIDES: ClassVar[dict[QStyle.PixelMetric, int]] = {
        QStyle.PixelMetric.PM_ScrollBarExtent: 8,
        QStyle.PixelMetric.PM_ScrollBarSliderMin: 20,
    }

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

    def pixelMetric(
        self,
        metric: QStyle.PixelMetric,
        option: QStyleOption | None = None,
        widget: QWidget | None = None,
    ) -> int:
        base = self._BASE_OVERRIDES.get(metric)
        if base is None:
            base = super().pixelMetric(metric, option, widget)
        if base < 0:
            return base  # preserve sentinels (-1 = disabled)
        return max(1, round(base * self._zoom))

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

    # ── Drawing overrides (unchanged) ─────────────────────────────

    def _border_color(self) -> QPen:
        """Resolve the border-subtle pen (late import avoids circular ref)."""
        from . import qcolor, theme

        return QPen(qcolor(theme().border_subtle), 1)

    def drawPrimitive(
        self,
        element: QStyle.PrimitiveElement,
        option: QStyleOption | None,
        painter: QPainter | None,
        widget: QWidget | None = None,
    ) -> None:
        PE = QStyle.PrimitiveElement
        if element == PE.PE_Frame and isinstance(widget, QScrollArea):
            return
        if element == PE.PE_PanelStatusBar and painter and option:
            super().drawPrimitive(element, option, painter, widget)
            painter.setPen(self._border_color())
            painter.drawLine(
                option.rect.left(),
                option.rect.top(),
                option.rect.right(),
                option.rect.top(),
            )
            return
        super().drawPrimitive(element, option, painter, widget)

    def drawControl(
        self,
        element: QStyle.ControlElement,
        option: QStyleOption | None,
        painter: QPainter | None,
        widget: QWidget | None = None,
    ) -> None:
        CE = QStyle.ControlElement
        if element == CE.CE_ToolBar and painter and option:
            super().drawControl(element, option, painter, widget)
            r = option.rect
            painter.setPen(self._border_color())
            painter.drawLine(r.left(), r.bottom(), r.right(), r.bottom())
            return
        super().drawControl(element, option, painter, widget)
