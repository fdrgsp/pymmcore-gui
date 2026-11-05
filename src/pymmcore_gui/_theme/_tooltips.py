"""Readable tooltips for widgets and model-backed dropdowns and tables."""

from __future__ import annotations

import sys
from html import escape

from pymmcore_gui._qt.QtCore import QEvent, QObject, QPoint, QRect, Qt
from pymmcore_gui._qt.QtGui import QHelpEvent, QTextDocument
from pymmcore_gui._qt.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QMenu,
    QToolTip,
    QWidget,
)

_READING_TIME_MS = 30_000
_WIDTH = 360

# With a non-native style (ours is Fusion-based) Qt places tooltips about 40
# logical px below the cursor on macOS, versus ~16 elsewhere, so they look
# detached from what they describe. Qt adds its own offset to the position it
# is given, so shift the anchor up by the difference. Windows/Linux are left
# on Qt's default placement.
_MACOS_TOOLTIP_Y_CORRECTION = 24 if sys.platform == "darwin" else 0


def _tooltip_text(widget: QWidget, event: QHelpEvent) -> tuple[str, QRect]:
    parent = widget.parentWidget()
    if isinstance(parent, QAbstractItemView) and widget is parent.viewport():
        index = parent.indexAt(event.pos())
        text = index.data(Qt.ItemDataRole.ToolTipRole)
        return (text if isinstance(text, str) else "", parent.visualRect(index))
    if isinstance(widget, QMenu):
        action = widget.actionAt(event.pos())
        if widget.toolTipsVisible() and action is not None:
            return action.toolTip(), widget.actionGeometry(action)
        return "", QRect()
    return widget.toolTip(), widget.rect()


def _wrapped_tooltip(text: str, width: int) -> str | None:
    document = QTextDocument()
    document.setDefaultFont(QToolTip.font())
    # Some binding stubs omit this Qt helper even though it is exposed.
    rich_text = Qt.mightBeRichText(text)  # type: ignore[attr-defined]
    if rich_text:
        document.setHtml(text)
    else:
        document.setPlainText(text)
    if document.idealWidth() <= width and len(document.toPlainText()) < 160:
        return None

    # Qt only automatically wraps rich-text tooltips. A table gives its text
    # document a preferred width while retaining existing HTML and newlines.
    body = text if rich_text else escape(text).replace("\n", "<br>")
    return (
        f'<qt><table width="{width}" border="0" cellspacing="0" cellpadding="0">'
        f"<tr><td>{body}</td></tr></table></qt>"
    )


class _TooltipFilter(QObject):
    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        obj, event = a0, a1
        if not isinstance(obj, QWidget) or event is None:
            return False
        if event.type() in (QEvent.Type.Polish, QEvent.Type.Show) and obj.inherits(
            "QTipLabel"
        ):
            # Keep both Qt hit testing and the native tooltip window from
            # taking hover/focus from the source, including combo popups.
            obj.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            obj.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
            if event.type() == QEvent.Type.Polish:
                obj.setWindowFlag(Qt.WindowType.WindowTransparentForInput)
        if event.type() != QEvent.Type.ToolTip or not isinstance(event, QHelpEvent):
            return False

        text, rect = _tooltip_text(obj, event)
        if not text:
            return False
        from . import theme

        width = theme().scaled(_WIDTH)
        if (screen := obj.screen()) is not None:
            width = min(width, screen.availableGeometry().width() - 24)
        wrapped = _wrapped_tooltip(text, max(1, width))
        if wrapped is None and not _MACOS_TOOLTIP_Y_CORRECTION:
            return False
        # Short tooltips keep Qt's default duration; wrapped ones need time to read.
        duration = (
            -1 if wrapped is None else max(_READING_TIME_MS, obj.toolTipDuration())
        )
        QToolTip.showText(
            event.globalPos() - QPoint(0, _MACOS_TOOLTIP_Y_CORRECTION),
            text if wrapped is None else wrapped,
            obj,
            rect,
            duration,
        )
        event.accept()
        return True


_tooltip_filter: _TooltipFilter | None = None


def install_tooltips() -> None:
    """Install the GUI-wide long-tooltip handler once per application."""
    global _tooltip_filter
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        return
    if _tooltip_filter is not None and _tooltip_filter.parent() is app:
        return
    _tooltip_filter = _TooltipFilter(app)
    app.installEventFilter(_tooltip_filter)
