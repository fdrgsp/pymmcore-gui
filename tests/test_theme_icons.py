from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_plus.autofocus import DuoSettings, OughtaFocusSettings
from pymmcore_widgets.useq_widgets._autofocus_settings import AutofocusSettingsDialog

from pymmcore_gui._array_viewer import _icon_avg_rgb, ensure_visible_icon
from pymmcore_gui._qt.QtCore import QSize
from pymmcore_gui._qt.QtGui import QColor, QIcon, QPixmap
from pymmcore_gui._qt.QtWidgets import QPushButton
from pymmcore_gui._theme import DARK_THEME, LIGHT_THEME, qcolor, set_theme, theme

if TYPE_CHECKING:
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
