"""Subtle action buttons are drawn as primary ones."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_gui._qt.QtWidgets import QPushButton, QToolButton, QVBoxLayout, QWidget
from pymmcore_gui._theme import keep_subtle_buttons

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


def _button(
    cls: type[QPushButton | QToolButton], parent: QWidget
) -> QPushButton | QToolButton:
    button = cls(parent)
    button.setText("Go")
    button.setProperty("variant", "subtle")
    layout = parent.layout()
    assert layout is not None
    layout.addWidget(button)
    return button


def test_subtle_buttons_become_primary_once_shown(qtbot: QtBot) -> None:
    window = QWidget()
    QVBoxLayout(window)
    qtbot.addWidget(window)
    push = _button(QPushButton, window)
    tool = _button(QToolButton, window)
    toggle = _button(QPushButton, window)
    toggle.setCheckable(True)
    kept_area = QWidget(window)
    QVBoxLayout(kept_area)
    layout = window.layout()
    assert layout is not None
    layout.addWidget(kept_area)
    keep_subtle_buttons(kept_area)
    kept = _button(QPushButton, kept_area)
    plain = QPushButton("Plain", window)
    assert push.property("variant") == "subtle"

    window.show()
    qtbot.waitExposed(window)
    assert push.property("variant") == tool.property("variant") == "primary"
    # primary has no checked look, and this area keeps its own style
    assert toggle.property("variant") == kept.property("variant") == "subtle"
    assert plain.property("variant") is None

    # made subtle while already shown
    plain.setProperty("variant", "subtle")
    assert plain.property("variant") == "primary"


def test_primary_tool_buttons_are_drawn_as_primary(qtbot: QtBot) -> None:
    window = QWidget()
    QVBoxLayout(window)
    qtbot.addWidget(window)
    keep_subtle_buttons(window)
    subtle = _button(QToolButton, window)
    primary = _button(QToolButton, window)
    primary.setProperty("variant", "primary")
    window.show()
    qtbot.waitExposed(window)
    assert subtle.grab().toImage() != primary.grab().toImage()
