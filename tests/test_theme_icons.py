from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from pymmcore_plus.autofocus import DuoSettings, OughtaFocusSettings
from pymmcore_widgets.useq_widgets._autofocus_settings import AutofocusSettingsDialog

from pymmcore_gui._array_viewer import _icon_avg_rgb, ensure_visible_icon
from pymmcore_gui._qt.QtCore import QEvent, QSize
from pymmcore_gui._qt.QtGui import QColor, QIcon, QPixmap
from pymmcore_gui._qt.QtWidgets import QApplication, QPushButton, QWidget
from pymmcore_gui._theme import (
    DARK_THEME,
    LIGHT_THEME,
    _events,
    qcolor,
    set_theme,
    theme,
)

if TYPE_CHECKING:
    import pytest
    from pytestqt.qtbot import QtBot


def test_duo_settings_icons_follow_theme(qtbot: QtBot) -> None:
    dialog = AutofocusSettingsDialog(
        DuoSettings,
        method="duo",
        methods={"oughtafocus": OughtaFocusSettings, "duo": DuoSettings},
    )
    qtbot.addWidget(dialog)
    buttons = [
        button
        for button in dialog.findChildren(QPushButton)
        if button.accessibleName() == "Settings"
    ]
    assert len(buttons) == 2
    dialog.show()
    qtbot.waitExposed(dialog)

    # The dialog is created after theme initialization. Its first show must
    # correct the default black gears without needing a theme toggle.
    def assert_icons(color: QColor) -> None:
        for button in buttons:
            rgb = _icon_avg_rgb(button.icon(), QSize(24, 24))
            assert rgb is not None
            assert all(
                abs(actual - expected) < 2
                for actual, expected in zip(
                    rgb, (color.red(), color.green(), color.blue()), strict=True
                )
            )

    assert_icons(qcolor(theme().text_primary))
    set_theme(LIGHT_THEME)
    assert_icons(QColor("black"))  # the original already contrasts on light
    set_theme(DARK_THEME)
    assert_icons(qcolor(theme().text_primary))


def test_new_button_keeps_its_semantic_icon_color(qtbot: QtBot) -> None:
    pixmap = QPixmap(16, 16)
    pixmap.fill(QColor("green"))
    button = QPushButton()
    button.setIcon(QIcon(pixmap))
    qtbot.addWidget(button)
    button.show()
    qtbot.waitExposed(button)
    assert _icon_avg_rgb(button.icon(), QSize(16, 16)) == (0, 128, 0)


def test_sweep_keeps_an_icon_swapped_in_by_other_code(qtbot: QtBot) -> None:
    """An upstream `setIcon()` becomes the new source, not reverted by a sweep."""

    def icon(color: str) -> QIcon:
        pixmap = QPixmap(16, 16)
        pixmap.fill(QColor(color))
        return QIcon(pixmap)

    button = QPushButton()
    qtbot.addWidget(button)
    button.setIcon(icon("black"))
    ensure_visible_icon(button)  # stashes the black icon, recolors it on dark
    button.setIcon(icon("red"))  # upstream swap, unaware of the stash
    ensure_visible_icon(button)
    assert _icon_avg_rgb(button.icon(), QSize(16, 16)) == (255, 0, 0)


class _EventRecorder(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.received: list[QEvent.Type] = []

    def event(self, a0: QEvent | None) -> bool:
        if a0 is not None:
            self.received.append(a0.type())
        return super().event(a0)


def test_event_hooks_dispatch_by_type_and_consume(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_events, "_hooks", defaultdict(list))
    widget, other = _EventRecorder(), _EventRecorder()
    qtbot.addWidget(widget)
    qtbot.addWidget(other)
    shown: list[QWidget] = []

    def on_show(w: QWidget, _event: QEvent) -> bool:
        shown.append(w)
        return False

    _events.add_event_hook(on_show, QEvent.Type.Show)
    _events.add_event_hook(on_show, QEvent.Type.Show)  # added once
    _events.add_event_hook(lambda w, _e: w is widget, QEvent.Type.User)
    widget.show()
    assert shown.count(widget) == 1
    # a hook returning True consumes the event
    for w in (widget, other):
        QApplication.sendEvent(w, QEvent(QEvent.Type.User))
    assert QEvent.Type.User not in widget.received
    assert QEvent.Type.User in other.received
