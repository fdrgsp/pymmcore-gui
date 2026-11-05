from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from pymmcore_widgets.useq_widgets._mda_sequence import AutofocusAxis
from pymmcore_widgets.useq_widgets._positions import AF_PER_POS_TOOLTIP

from pymmcore_gui._qt.QtCore import QEvent, QPoint, Qt
from pymmcore_gui._qt.QtGui import QHelpEvent, QTextDocumentFragment
from pymmcore_gui._qt.QtWidgets import QApplication, QLabel, QPushButton, QToolTip
from pymmcore_gui._theme import set_zoom, theme

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytestqt.qtbot import QtBot

    from pymmcore_gui._qt.QtWidgets import QWidget


@pytest.fixture(autouse=True)
def _hide_tooltip(qtbot: QtBot) -> Iterator[None]:
    yield
    QToolTip.hideText()
    qtbot.waitUntil(lambda: not QToolTip.isVisible())


def _show_tooltip(widget: QWidget, pos: QPoint | None = None) -> QLabel:
    pos = widget.rect().center() if pos is None else pos
    event = QHelpEvent(QEvent.Type.ToolTip, pos, widget.mapToGlobal(pos))
    QApplication.sendEvent(widget, event)
    assert QToolTip.isVisible()
    tip = next(
        w for w in QApplication.topLevelWidgets() if w.objectName() == "qtooltip_label"
    )
    assert isinstance(tip, QLabel)
    return tip


@pytest.mark.parametrize("zoom", [1.0, 2.0])
@pytest.mark.parametrize(
    "text",
    [AF_PER_POS_TOOLTIP, "<b>Software autofocus</b><br>" + AF_PER_POS_TOOLTIP * 3],
)
def test_long_tooltips_wrap_without_changing_the_source(
    qtbot: QtBot, zoom: float, text: str
) -> None:
    set_zoom(zoom)
    button = QPushButton("Hover")
    qtbot.addWidget(button)
    button.setToolTip(text)
    button.show()
    qtbot.waitExposed(button)
    qtbot.mouseMove(button)  # type: ignore[no-untyped-call]
    tip = _show_tooltip(button)
    assert tip.wordWrap()
    screen = button.screen()
    assert screen is not None
    assert (
        tip.width()
        <= min(theme().scaled(360), screen.availableGeometry().width() - 24) + 20
    )
    assert tip.height() > button.fontMetrics().height() * 2
    assert button.toolTip() == text
    assert QTextDocumentFragment.fromHtml(QToolTip.text()).toPlainText().strip()


@pytest.mark.parametrize("popup", [False, True])
def test_oughtafocus_tooltip_has_reading_time_without_taking_hover(
    qtbot: QtBot, popup: bool
) -> None:
    editor = AutofocusAxis()
    qtbot.addWidget(editor)
    editor.setChecked(True)
    editor.setKind("software")
    method = editor.method
    index = method.findText("oughtafocus")
    assert index >= 0
    method.setCurrentIndex(index)
    # Long explanations get reading time even if a widget requests a short tip.
    method.setToolTipDuration(50)
    editor.show()
    qtbot.waitExposed(editor)

    if popup:
        method.showPopup()
        view = method.view()
        assert view is not None
        model = view.model()
        assert model is not None
        item = model.index(index, 0)
        viewport = view.viewport()
        assert viewport is not None
        pos = view.visualRect(item).center()
        qtbot.mouseMove(viewport, pos)  # type: ignore[no-untyped-call]
        source, position = viewport, pos
    else:
        qtbot.mouseMove(method)  # type: ignore[no-untyped-call]
        source, position = method, None

    with patch.object(QToolTip, "showText", wraps=QToolTip.showText) as show:
        tip = _show_tooltip(source, position)
    # Native cursor/focus is shared with the other xdist workers. Verify the
    # reading timer passed to Qt instead of assuming none of them moves it.
    assert show.call_args.args[-1] >= 30_000
    assert "sharpest" in QToolTip.text()
    assert tip.wordWrap()
    assert tip.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert tip.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert tip.windowFlags() & Qt.WindowType.WindowTransparentForInput
    if popup:
        method.hidePopup()
