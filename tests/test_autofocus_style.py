"""The Autofocus section under the application's actual theme."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pymmcore_gui._qt.QtAds import CDockManager, CDockWidget, DockWidgetArea
from pymmcore_gui._qt.QtCore import QCoreApplication, QEvent, Qt
from pymmcore_gui._qt.QtWidgets import QMainWindow
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
def test_autofocus_section_keeps_its_layout(
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
    section = widget.tabs.autofocus_section
    section.set_expanded(True)
    af = widget.af_axis
    af.setKind("hardware")
    qtbot.waitUntil(lambda: af.height() == af.sizeHint().height())
    # Dock containers may still have a layout request after the widget reaches
    # its size hint. Settle the ancestor layouts before measuring.
    QCoreApplication.sendPostedEvents(None, QEvent.Type.LayoutRequest)
    size = af.size()
    rows = None

    # switching the kind neither resizes it nor moves its rows
    for kind in ("hardware", "software", "hardware"):
        af.setKind(kind)
        assert af.size() == size
        labels = (
            af._mode_label,
            af.label,
            af._search_label if kind == "hardware" else af._method_label,
        )
        ys = [label.mapTo(af, label.rect().topLeft()).y() for label in labels]
        assert rows is None or ys == rows
        rows = ys

    # the section's checkbox switches it
    for checked in (False, True):
        with qtbot.waitSignal(af.toggled):
            qtbot.mouseClick(  # type: ignore[no-untyped-call]
                section.checkbox, Qt.MouseButton.LeftButton
            )
        assert af.isChecked() is checked
        assert af.use_af_p.isEnabled() is checked
