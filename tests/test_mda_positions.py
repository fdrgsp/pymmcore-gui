"""Position import, local icon updates and icon contrast analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import useq
from pymmcore_widgets.useq_widgets._positions import MDAButton

from pymmcore_gui._array_viewer import _icon_avg_rgb
from pymmcore_gui._qt.QtCore import QSize
from pymmcore_gui._qt.QtGui import QColor, QIcon, QImage, QPixmap
from pymmcore_gui._qt.QtWidgets import QApplication, QWidget
from pymmcore_gui._theme import DARK_THEME, LIGHT_THEME, qcolor, set_theme, theme
from pymmcore_gui.widgets._mda_widget import MemoryMDAWidget, TopbarMemoryMDAWidget

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot


@pytest.mark.parametrize("transparent", [False, True])
def test_icon_average_matches_rendered_pixels(qtbot: QtBot, transparent: bool) -> None:
    image = QImage(3, 2, QImage.Format.Format_RGBA8888)
    colors = [
        QColor(12, 40, 90, 0),
        QColor(250, 20, 70, 32),
        QColor(20, 100, 200, 33),
        QColor(100, 50, 20, 128),
        QColor(30, 40, 50, 255),
        QColor(210, 190, 170, 255),
    ]
    for i, color in enumerate(colors):
        image.setPixelColor(i % 3, i // 3, QColor(0, 0, 0, 0) if transparent else color)
    icon = QIcon(QPixmap.fromImage(image))
    rendered = icon.pixmap(QSize(3, 2)).toImage()
    opaque = [
        rendered.pixelColor(x, y)
        for y in range(rendered.height())
        for x in range(rendered.width())
        if rendered.pixelColor(x, y).alpha() > 32
    ]
    actual = _icon_avg_rgb(icon, QSize(3, 2))
    if opaque:
        expected = np.mean([(c.red(), c.green(), c.blue()) for c in opaque], axis=0)
        assert actual == pytest.approx(expected, abs=1e-10)
    else:
        assert actual is None


def test_icon_average_updates_when_icon_pixels_change(qtbot: QtBot) -> None:
    pixmap = QPixmap(8, 8)
    pixmap.fill(QColor("red"))
    icon = QIcon(pixmap)
    assert _icon_avg_rgb(icon, QSize(8, 8)) == (255, 0, 0)
    pixmap.fill(QColor("blue"))
    icon.addPixmap(pixmap)
    assert _icon_avg_rgb(icon, QSize(8, 8)) == (0, 0, 255)
    assert _icon_avg_rgb(QIcon(), QSize(8, 8)) is None


@pytest.mark.parametrize("widget_cls", [MemoryMDAWidget, TopbarMemoryMDAWidget])
def test_position_insertion_and_grid_edits_only_theme_affected_cells(
    mmcore: CMMCorePlus,
    qtbot: QtBot,
    widget_cls: type[MemoryMDAWidget | TopbarMemoryMDAWidget],
) -> None:
    widget = widget_cls(mmcore)
    qtbot.addWidget(widget)
    sentinel = QWidget(widget)
    sentinel.setStyleSheet("background: magenta")
    editor = widget.stage_positions
    editor.setValue([useq.Position(x=1, y=2)])
    first = editor.table().cellWidget(0, editor.table().indexOf(editor.SEQ))
    assert isinstance(first, MDAButton)
    first_icon = first.seq_btn.icon().cacheKey()

    # Inserting one row must preserve existing icons and unrelated styling.
    editor.table().insertRow(1)
    assert sentinel.styleSheet() == "background: magenta"
    assert first.seq_btn.icon().cacheKey() == first_icon
    second = editor.table().cellWidget(1, editor.table().indexOf(editor.SEQ))
    assert isinstance(second, MDAButton)
    second_icon = second.seq_btn.icon().cacheKey()
    for row in range(2):
        for col in range(editor.table().columnCount()):
            if cell := editor.table().cellWidget(row, col):
                assert not cell.styleSheet()
                for child in cell.findChildren(QWidget):
                    assert not child.styleSheet()
    assert first.seq_btn.property("variant") == "subtle"

    first.setValue(useq.MDASequence(grid_plan=useq.GridRowsColumns(rows=1, columns=2)))
    assert second.seq_btn.icon().cacheKey() == second_icon
    assert _icon_avg_rgb(first.seq_btn.icon(), QSize(24, 24)) == pytest.approx(
        qcolor(theme().status_green).getRgb()[:3], abs=2
    )
    first.setValue(None)
    assert second.seq_btn.icon().cacheKey() == second_icon
    assert _icon_avg_rgb(first.seq_btn.icon(), QSize(24, 24)) == pytest.approx(
        qcolor(theme().text_primary).getRgb()[:3], abs=2
    )


@pytest.mark.parametrize("widget_cls", [MemoryMDAWidget, TopbarMemoryMDAWidget])
def test_hcs_import_preserves_plan_cells_visibility_and_theme(
    mmcore: CMMCorePlus,
    qtbot: QtBot,
    widget_cls: type[MemoryMDAWidget | TopbarMemoryMDAWidget],
) -> None:
    widget = widget_cls(mmcore)
    qtbot.addWidget(widget)
    widget.revealAxis("p")
    widget.show()
    editor = widget.stage_positions
    table = editor.table()
    plan = useq.WellPlatePlan(
        plate=useq.WellPlate.from_str("96-well"),
        a1_center_xy=(1000, 2000),
        selected_wells=((0, 1), (0, 2)),
        well_points_plan=useq.GridRowsColumns(
            rows=1, columns=3, fov_width=100, fov_height=100
        ),
    )
    editor._hcs.setValue(plan)
    with qtbot.waitSignal(editor.valueChanged):
        editor._hcs.accept()
    QApplication.processEvents()
    assert table.isVisible()
    assert editor.value() == plan
    assert table.rowCount() == 6
    for row, pos in enumerate(plan):
        record = table.rowData(row)
        assert (record["name"], record["x"], record["y"]) == (pos.name, pos.x, pos.y)
        x_cell = table.cellWidget(row, table.indexOf(editor.X))
        assert x_cell is not None and not x_cell.isEnabled()

    button = table.cellWidget(0, table.indexOf(editor.SEQ))
    assert isinstance(button, MDAButton)
    for app_theme in (LIGHT_THEME, DARK_THEME):
        set_theme(app_theme)
        QApplication.processEvents()
        assert _icon_avg_rgb(button.seq_btn.icon(), QSize(24, 24)) == pytest.approx(
            qcolor(theme().text_primary).getRgb()[:3], abs=2
        )

    table.hide()
    editor.setValue([useq.Position(x=3, y=4)])
    assert table.isHidden()
    assert editor.value() == (useq.Position(x=3, y=4),)


def test_position_import_error_restores_visibility(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    widget = MemoryMDAWidget(mmcore)
    qtbot.addWidget(widget)
    widget.revealAxis("p")
    widget.show()
    table = widget.stage_positions.table()
    assert table.isVisible()
    with pytest.raises(ValueError):
        widget.stage_positions.setValue([{"x": "invalid"}])  # type: ignore[list-item]
    assert table.isVisible()
