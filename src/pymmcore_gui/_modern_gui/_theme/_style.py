from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_gui._qt.QtGui import QPen
from pymmcore_gui._qt.QtWidgets import (
    QProxyStyle,
    QScrollArea,
    QStyle,
    QStyleOption,
    QWidget,
)

if TYPE_CHECKING:
    from pymmcore_gui._qt.QtGui import QPainter


class Sp:
    """Spacing tokens (px). 4px base grid."""

    XXS = 4
    XS = 8
    SM = 12
    MD = 16
    LG = 24
    XL = 32


RADIUS = 3
ROW_HEIGHT = 36  # standard height for toolbars, panel headers, etc.


class MicroscopeStyle(QProxyStyle):
    """Thin proxy over Fusion. Overrides only what we need."""

    def __init__(self) -> None:
        super().__init__("Fusion")

    def pixelMetric(
        self,
        metric: QStyle.PixelMetric,
        option: QStyleOption | None = None,
        widget: QWidget | None = None,
    ) -> int:
        """Override a few pixel metrics for our theme."""
        if metric == QStyle.PixelMetric.PM_ScrollBarExtent:
            return 5
        if metric == QStyle.PixelMetric.PM_ScrollBarSliderMin:
            return 20
        if metric == QStyle.PixelMetric.PM_LayoutHorizontalSpacing:
            return Sp.SM
        if metric == QStyle.PixelMetric.PM_LayoutVerticalSpacing:
            return Sp.XXS
        return super().pixelMetric(metric, option, widget)

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
