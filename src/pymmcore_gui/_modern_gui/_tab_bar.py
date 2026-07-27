"""Tab bar whose painting reliably passes through the application style."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_gui._qt.QtCore import Qt
from pymmcore_gui._qt.QtGui import QPalette
from pymmcore_gui._qt.QtWidgets import (
    QStyle,
    QStyleOptionTab,
    QStylePainter,
    QTabBar,
)

if TYPE_CHECKING:
    from pymmcore_gui._qt.QtGui import QPaintEvent
    from pymmcore_gui._qt.QtWidgets import QWidget


class ThemedTabBar(QTabBar):
    """Draw tabs through the Python style override on every paint.

    On macOS, ``QTabBar`` can intermittently bypass or cache the native
    dispatch into a Python ``QProxyStyle`` when a hidden page is first shown
    or a tab is removed and reinserted.  Calling ``drawControl`` here makes
    the dispatch explicit while leaving tab geometry, labels, close buttons,
    mouse handling, and accessibility to Qt.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        painter = QStylePainter(self)
        style = self.style()
        if style is None:
            painter.end()
            super().paintEvent(a0)
            return

        selected: QStyleOptionTab | None = None
        for index in range(self.count()):
            option = QStyleOptionTab()
            self.initStyleOption(option, index)
            self._position_tab_buttons(style, option, index)
            if option.state & QStyle.StateFlag.State_Selected:
                selected = option
            else:
                self._draw_tab(style, option, painter)

        # Match QTabBar's stacking: the selected tab is painted last so its
        # shape and underline are not covered by a neighboring tab.
        if selected is not None:
            self._draw_tab(style, selected, painter)

    def _position_tab_buttons(
        self,
        style: QStyle,
        option: QStyleOptionTab,
        index: int,
    ) -> None:
        """Keep native tab buttons aligned with scrolled tab geometry."""
        sides = (
            (
                QTabBar.ButtonPosition.LeftSide,
                QStyle.SubElement.SE_TabBarTabLeftButton,
            ),
            (
                QTabBar.ButtonPosition.RightSide,
                QStyle.SubElement.SE_TabBarTabRightButton,
            ),
        )
        for side, element in sides:
            if button := self.tabButton(index, side):
                button.setGeometry(style.subElementRect(element, option, self))

    def _draw_tab(
        self,
        style: QStyle,
        option: QStyleOptionTab,
        painter: QStylePainter,
    ) -> None:
        style.drawControl(
            QStyle.ControlElement.CE_TabBarTabShape, option, painter, self
        )
        text_rect = style.subElementRect(
            QStyle.SubElement.SE_TabBarTabText, option, self
        )
        style.drawItemText(
            painter,
            text_rect,
            int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextShowMnemonic),
            option.palette,
            bool(option.state & QStyle.StateFlag.State_Enabled),
            option.text,
            QPalette.ColorRole.WindowText,
        )
