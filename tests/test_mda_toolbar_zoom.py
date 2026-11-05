from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pymmcore_gui._qt.QtWidgets import QApplication, QToolButton
from pymmcore_gui._theme import reset_zoom, set_zoom, theme
from pymmcore_gui.widgets._mda_widget import MemoryMDAWidget

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot


@pytest.mark.parametrize("order", ["parent_first", "parent_last", "mixed"])
def test_mda_toolbar_sizes_survive_zoom_and_reset(
    qtbot: QtBot,
    mmcore: CMMCorePlus,
    monkeypatch: pytest.MonkeyPatch,
    order: str,
) -> None:
    widget = MemoryMDAWidget(mmcore)
    qtbot.addWidget(widget)
    toolbars = [
        table.toolBar()
        for table in (widget.channels, widget.stage_positions, widget.time_plan)
    ]
    others = [
        w for w in QApplication.allWidgets() if w is not widget and w not in toolbars
    ]
    if order == "parent_first":
        widgets = [widget, *toolbars, *others]
    elif order == "parent_last":
        widgets = [*toolbars, *others, widget]
    else:
        widgets = [toolbars[0], widget, *toolbars[1:], *others]

    with monkeypatch.context() as patch:
        patch.setattr(QApplication, "allWidgets", staticmethod(lambda: widgets))
        for zoom in (1.5, 0.8, 2.0, None, 0.8, None):
            if zoom is None:
                reset_zoom()
            else:
                set_zoom(zoom)
            for toolbar in toolbars:
                assert toolbar.iconSize().width() == theme().scaled(16)
                for action in toolbar.actions():
                    button = toolbar.widgetForAction(action)
                    if isinstance(button, QToolButton):
                        assert button.iconSize() == toolbar.iconSize()
