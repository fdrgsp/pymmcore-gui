from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from pymmcore_widgets._light_sources import INTENSITY_KEY, LIGHT_SOURCE_KEY

from pymmcore_gui import _restrictions
from pymmcore_gui._main_window import MicroManagerGUI
from pymmcore_gui._qt.QtCore import QEvent, QPoint
from pymmcore_gui._qt.QtWidgets import QApplication, QGroupBox, QMenu, QWidget
from pymmcore_gui._restrictions import NAMES, Restrictions, restrict, restrictions
from pymmcore_gui.widgets._channel_settings import ChannelSettingsPanel
from pymmcore_gui.widgets._mda_channel_table import MDAChannelTable
from pymmcore_gui.widgets._preferences import PreferencesDialog

if TYPE_CHECKING:
    from pathlib import Path

    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot


@pytest.fixture
def restrictions_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "restrictions.json"
    monkeypatch.setattr(_restrictions, "_restrictions", Restrictions(path))
    return path


def test_the_file_says_what_a_lock_disables(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "restrictions.json"
    unlocked = Restrictions(path)  # no file yet: unlocked, a lock disables all
    assert not unlocked.locked
    assert not unlocked.is_disabled("hardware_tab")
    unlocked.set_locked(True)
    assert all(unlocked.is_disabled(name) for name in NAMES)

    path.write_text(json.dumps({"locked": True, "disable": ["hardware_tab", "typo"]}))
    with caplog.at_level(logging.WARNING):
        partial = Restrictions(path)
    assert "typo" in caplog.text
    assert partial.is_disabled("hardware_tab")
    assert not partial.is_disabled("installation_tab")
    partial.set_locked(False)
    saved = json.loads(path.read_text())
    assert saved == {"locked": False, "disable": ["hardware_tab", "typo"]}

    # broken files lock nothing; "locked" must really be true
    for broken in ("not json", '{"locked": true, "disable": 5}', '{"locked": "x"}'):
        path.write_text(broken)
        assert not Restrictions(path).locked


def test_a_file_without_a_list_keeps_locking_everything(tmp_path: Path) -> None:
    path = tmp_path / "restrictions.json"
    path.write_text('{"locked": true}')
    restricted = Restrictions(path)
    restricted.set_locked(False)
    restricted.set_locked(True)
    assert json.loads(path.read_text()) == {"locked": True}  # no list written in


def test_a_file_that_cannot_be_saved_still_locks(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    restricted = Restrictions(tmp_path / "missing" / "restrictions.json")
    with caplog.at_level(logging.WARNING):
        restricted.set_locked(True)
    assert restricted.locked
    assert "Could not save" in caplog.text


def test_restrict_follows_the_lock_while_its_widget_lives(
    restrictions_file: Path, qtbot: QtBot
) -> None:
    widget = QWidget()  # not given to qtbot: the test deletes it
    seen: list[bool] = []
    restrict("hardware_tab", widget, on_change=seen.append)
    assert widget.isEnabled()
    restrictions().set_locked(True)
    assert not widget.isEnabled()
    assert seen == [True, False]

    widget.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    restrictions().set_locked(False)  # the binding went with its widget
    assert seen == [True, False]


def test_locked_tabs_open_but_their_pages_do_not_respond(
    restrictions_file: Path, mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    window = MicroManagerGUI(mmcore=mmcore)
    qtbot.addWidget(window)
    locked_pages = (window._installation, window._hardware, window._configurations)

    restrictions().set_locked(True)
    assert not any(page.isEnabled() for page in locked_pages)
    assert window._acquire.isEnabled()
    assert all(tab.isEnabled() for tab in window._mode_tabs._tabs)

    restrictions().set_locked(False)
    assert all(page.isEnabled() for page in locked_pages)


def test_hidden_menu_locks_the_gui_but_not_while_acquiring(
    restrictions_file: Path, mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    window = MicroManagerGUI(mmcore=mmcore)
    qtbot.addWidget(window)
    labels: list[str] = []

    def choose(menu: QMenu, *_: object) -> object:
        labels.append(menu.actions()[0].text())
        return menu.actions()[0]

    with patch.object(QMenu, "exec", choose):
        window._show_lock_menu(QPoint())
        assert restrictions().locked
        window._show_lock_menu(QPoint())
    assert labels == ["Lock GUI", "Unlock GUI"]
    assert not restrictions().locked
    assert json.loads(restrictions_file.read_text())["locked"] is False

    with (
        patch.object(mmcore.mda, "is_running", return_value=True),
        patch.object(QMenu, "exec") as exec_menu,
    ):
        window._show_lock_menu(QPoint())
    window._on_pixel_calibration_running(True)  # its Cancel is on a locked page
    with patch.object(QMenu, "exec") as exec_menu_calibrating:
        window._show_lock_menu(QPoint())
    exec_menu.assert_not_called()
    exec_menu_calibrating.assert_not_called()


def _cells(table: MDAChannelTable | ChannelSettingsPanel, key: str) -> list[QWidget]:
    data = table.table()
    col = next(
        c
        for c in range(data.columnCount())
        if (info := data.columnInfo(c)) and info.key == key
    )
    rows = range(data.rowCount())
    return [cell for row in rows if (cell := data.cellWidget(row, col)) is not None]


@pytest.mark.parametrize("cls", [ChannelSettingsPanel, MDAChannelTable])
def test_locked_light_sources_keep_their_intensity(
    restrictions_file: Path, mmcore: CMMCorePlus, qtbot: QtBot, cls: type
) -> None:
    widget = cls(mmcore=mmcore)
    qtbot.addWidget(widget)
    if isinstance(widget, MDAChannelTable):
        widget._add_row()
        checkbox = widget.show_light_source
        widget.setLightSourceVisible(True)
    else:
        checkbox = widget.light_sources
        widget.setLightSourcesVisible(True)
    intensity = [cell.isEnabled() for cell in _cells(widget, INTENSITY_KEY)]

    restrictions().set_locked(True)
    assert not checkbox.isEnabled()
    assert not any(cell.isEnabled() for cell in _cells(widget, LIGHT_SOURCE_KEY))
    assert [cell.isEnabled() for cell in _cells(widget, INTENSITY_KEY)] == intensity
    assert not widget._save_btn.isEnabled()

    restrictions().set_locked(False)
    assert checkbox.isEnabled()
    assert all(cell.isEnabled() for cell in _cells(widget, LIGHT_SOURCE_KEY))
    assert widget._save_btn.isEnabled()

    # saving still needs the light sources shown
    checkbox.setChecked(False)
    restrictions().set_locked(True)
    restrictions().set_locked(False)
    assert not widget._save_btn.isEnabled()


def test_locked_preferences(
    restrictions_file: Path, mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    window = MicroManagerGUI(mmcore=mmcore)
    qtbot.addWidget(window)
    restrictions().set_locked(True)
    dialog = PreferencesDialog(window._acquire)
    qtbot.addWidget(dialog)
    groups = {group.title(): group for group in dialog.findChildren(QGroupBox)}
    assert not groups["Data && Memory"].isEnabled()
    # which widgets show is the user's view, also set by a layout: never locked
    assert groups["Show Widgets"].isEnabled()
    assert groups["Layout"].isEnabled()
