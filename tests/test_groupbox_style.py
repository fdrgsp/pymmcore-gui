"""Checkable card spacing under the application's actual theme."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pymmcore_gui._qt.QtAds import CDockManager, CDockWidget, DockWidgetArea
from pymmcore_gui._qt.QtCore import QCoreApplication, QEvent, Qt
from pymmcore_gui._qt.QtWidgets import QLabel, QMainWindow, QStyle, QStyleOptionGroupBox
from pymmcore_gui._theme import (
    DARK_THEME,
    LIGHT_THEME,
    dock_chrome_stylesheet,
    set_theme,
    set_zoom,
)
from pymmcore_gui.widgets._mda_widget import MemoryMDAWidget

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot

    from pymmcore_gui._theme._types import Theme


@pytest.mark.parametrize("appearance", [DARK_THEME, LIGHT_THEME], ids=["dark", "light"])
@pytest.mark.parametrize("zoom", [1.0, 1.25, 2.0])
@pytest.mark.parametrize("rtl", [False, True])
@pytest.mark.parametrize("docked", [False, True])
def test_autofocus_card_padding_and_title_hit_target(
    qtbot: QtBot,
    mmcore: CMMCorePlus,
    appearance: Theme,
    zoom: float,
    rtl: bool,
    docked: bool,
) -> None:
    set_theme(appearance)
    set_zoom(zoom)
    widget = MemoryMDAWidget(mmcore)
    widget.setLayoutDirection(
        Qt.LayoutDirection.RightToLeft if rtl else Qt.LayoutDirection.LeftToRight
    )
    if docked:
        window = QMainWindow()
        manager = CDockManager(window)
        manager.setStyleSheet(manager.styleSheet() + dock_chrome_stylesheet())
        dock = CDockWidget(manager, "MDA", window)
        dock.setWidget(widget)
        manager.addDockWidget(DockWidgetArea.RightDockWidgetArea, dock)
        qtbot.addWidget(window)
        window.show()
        qtbot.waitExposed(window)
    else:
        qtbot.addWidget(widget)
        widget.show()
        qtbot.waitExposed(widget)
    af = widget.af_axis
    af.setKind("hardware")
    qtbot.waitUntil(
        lambda: (
            af.height() == af.sizeHint().height()
            and af.width() >= af.minimumSizeHint().width()
        )
    )
    # Dock containers may still have a layout request after the card itself
    # reaches its size hint. Settle the ancestor layouts before measuring.
    QCoreApplication.sendPostedEvents(None, QEvent.Type.LayoutRequest)
    size = af.size()
    settings_label = next(
        label
        for label in widget._settings_widget.findChildren(QLabel)
        if label.text() == "Axis Order:"
    )
    settings_edge = settings_label.mapTo(
        widget,
        settings_label.rect().topRight() if rtl else settings_label.rect().topLeft(),
    ).x()

    for kind in ("hardware", "software", "hardware"):
        af.setKind(kind)
        option = QStyleOptionGroupBox()
        af.initStyleOption(option)
        style = af.style()
        assert style is not None
        checkbox = style.subControlRect(
            QStyle.ComplexControl.CC_GroupBox,
            option,
            QStyle.SubControl.SC_GroupBoxCheckBox,
            af,
        )
        margin = af.width() - checkbox.right() - 1 if rtl else checkbox.left()
        assert margin >= 4 * zoom
        control = af.search_below_um if kind == "hardware" else af.settings_button
        bottom = control.mapTo(af, control.rect().bottomLeft()).y()
        assert af.height() - bottom - 1 >= 4 * zoom
        assert af.size() == size
        labels = (
            af.label,
            af._mode_label,
            af._search_label if kind == "hardware" else af._method_label,
        )
        for label in labels:
            edge = label.mapTo(
                widget, label.rect().topRight() if rtl else label.rect().topLeft()
            ).x()
            assert abs(edge - settings_edge) <= 1

        for checked in (False, True):
            with qtbot.waitSignal(af.toggled):
                qtbot.mouseClick(  # type: ignore[no-untyped-call]
                    af, Qt.MouseButton.LeftButton, pos=checkbox.center()
                )
            assert af.isChecked() is checked
            assert af.use_af_p.isEnabled() is checked
