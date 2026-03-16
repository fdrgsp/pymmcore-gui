from __future__ import annotations

from typing import TYPE_CHECKING

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

    def drawPrimitive(
        self,
        element: QStyle.PrimitiveElement,
        option: QStyleOption | None,
        painter: QPainter | None,
        widget: QWidget | None = None,
    ) -> None:
        """Suppress default frame drawing for scroll areas."""
        if element == QStyle.PrimitiveElement.PE_Frame and isinstance(
            widget, QScrollArea
        ):
            return
        super().drawPrimitive(element, option, painter, widget)
