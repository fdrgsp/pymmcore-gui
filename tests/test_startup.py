"""The launch dialog: choosing a layout and a configuration before the window opens.

Replaces the old ``LoadConfigDialog`` ("load the last config?") flow -- see
``pymmcore_gui._modern_gui._startup``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from pymmcore_gui._app import create_mmgui
from pymmcore_gui._layouts import (
    DEFAULT_LAYOUT_NAME,
    LAST_SESSION_LAYOUT_NAME,
    AcquireLayout,
    list_layouts,
    load_layout,
    save_layout,
    store_session_layout,
)
from pymmcore_gui._modern_gui._acquire import AcquirePage
from pymmcore_gui._modern_gui._acquire_toolbar import _LayoutMenuRow
from pymmcore_gui._modern_gui._main_win import MainWindow
from pymmcore_gui._modern_gui._panels import PanelKey
from pymmcore_gui._modern_gui._startup import DEMO_CONFIG, StartupDialog
from pymmcore_gui._qt.QtWidgets import (
    QDialog,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QWidgetAction,
)

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot

    from pymmcore_gui._app import WindowProtocol
    from pymmcore_gui._settings import Settings


def _layout() -> AcquireLayout:
    return AcquireLayout(dock_state=b"state", panels=frozenset({PanelKey.MDA}))


def _combo_items(combo: object) -> list[str]:
    return [combo.itemText(i) for i in range(combo.count())]  # type: ignore[attr-defined]


# ── the dialog itself ─────────────────────────────────────────────


def test_dialog_offers_every_layout_and_the_demo_config(qtbot: QtBot) -> None:
    save_layout("My rig", _layout())
    dialog = StartupDialog()
    qtbot.addWidget(dialog)

    assert _combo_items(dialog._layout_combo) == [DEFAULT_LAYOUT_NAME, "My rig"]
    # First launch: demo is the only real config, so it's preselected.
    assert dialog.value().config == DEMO_CONFIG
    assert dialog.value().layout == DEFAULT_LAYOUT_NAME
    assert _combo_items(dialog._config_combo)[0] == "Demo configuration"


def test_dialog_drops_config_paths_that_no_longer_exist(
    qtbot: QtBot, settings: Settings, tmp_path: Path
) -> None:
    """A moved or deleted config must not be offered and then fail to load."""
    alive = tmp_path / "alive.cfg"
    alive.touch()
    gone = tmp_path / "gone.cfg"
    settings.recent_configs = [gone, alive]

    dialog = StartupDialog()
    qtbot.addWidget(dialog)

    assert "gone.cfg" not in _combo_items(dialog._config_combo)
    assert "alive.cfg" in _combo_items(dialog._config_combo)
    # ...and it is forgotten, not merely hidden this once.
    assert settings.recent_configs == [alive]
    # The most recent surviving config is the likely answer.
    assert dialog.value().config == str(alive)


def test_dialog_preselects_the_remembered_layout(
    qtbot: QtBot, settings: Settings
) -> None:
    save_layout("My rig", _layout())
    settings.modern_window.last_layout = "My rig"

    dialog = StartupDialog()
    qtbot.addWidget(dialog)
    assert dialog.value().layout == "My rig"


def test_dialog_falls_back_to_last_session_when_the_remembered_layout_is_gone(
    qtbot: QtBot, settings: Settings
) -> None:
    """Quitting must never silently discard the arrangement you had."""
    settings.modern_window.last_layout = "deleted-since"
    store_session_layout(_layout())

    dialog = StartupDialog()
    qtbot.addWidget(dialog)
    assert dialog.value().layout == LAST_SESSION_LAYOUT_NAME


def test_l_flag_beats_the_remembered_layout(qtbot: QtBot, settings: Settings) -> None:
    save_layout("My rig", _layout())
    save_layout("Screening", _layout())
    settings.modern_window.last_layout = "My rig"

    dialog = StartupDialog(preselect_layout="Screening")
    qtbot.addWidget(dialog)
    assert dialog.value().layout == "Screening"


def test_browse_adds_the_chosen_file_and_keeps_itself_last(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """Without Browse…, a first launch could only ever choose the demo config."""
    chosen = tmp_path / "picked.cfg"
    chosen.touch()

    dialog = StartupDialog()
    qtbot.addWidget(dialog)
    browse_index = dialog._config_combo.count() - 1

    with patch.object(QFileDialog, "getOpenFileName", return_value=(str(chosen), "")):
        dialog._config_combo.setCurrentIndex(browse_index)
        dialog._config_combo.activated.emit(browse_index)

    assert dialog.value().config == str(chosen)
    assert _combo_items(dialog._config_combo)[-1] == "Browse…"


def test_cancelling_browse_reverts_to_the_previous_choice(qtbot: QtBot) -> None:
    dialog = StartupDialog()
    qtbot.addWidget(dialog)
    before = dialog.value().config
    browse_index = dialog._config_combo.count() - 1

    with patch.object(QFileDialog, "getOpenFileName", return_value=("", "")):
        dialog._config_combo.setCurrentIndex(browse_index)
        dialog._config_combo.activated.emit(browse_index)

    assert dialog.value().config == before


# ── how create_mmgui uses it ──────────────────────────────────────


def test_quitting_the_dialog_exits_without_building_a_window(
    mmcore: CMMCorePlus,
) -> None:
    with (
        patch.object(StartupDialog, "exec", return_value=QDialog.DialogCode.Rejected),
        pytest.raises(SystemExit),
    ):
        create_mmgui(
            mm_config=None,
            mmcore=mmcore,
            install_sys_excepthook=False,
            install_sentry=False,
            exec_app=False,
            window_cls=cast("type[WindowProtocol]", MainWindow),
        )


def test_an_explicit_config_skips_the_dialog_entirely(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """`-c` already answers the config question; `-l` answers the other half."""
    with patch.object(StartupDialog, "exec") as never:
        window = create_mmgui(
            mm_config=Path(__file__).with_name("test_config.cfg"),
            layout=DEFAULT_LAYOUT_NAME,
            mmcore=mmcore,
            install_sys_excepthook=False,
            install_sentry=False,
            exec_app=False,
            window_cls=cast("type[WindowProtocol]", MainWindow),
        )
    never.assert_not_called()
    assert isinstance(window, MainWindow)
    qtbot.addWidget(window)
    qtbot.waitUntil(window.isVisible)
    assert window._acquire.layout_name == DEFAULT_LAYOUT_NAME
    window.close()


def test_the_chosen_layout_is_remembered_for_next_time(
    mmcore: CMMCorePlus, qtbot: QtBot, settings: Settings
) -> None:
    save_layout("My rig", _layout())

    def choose_my_rig(dialog: StartupDialog) -> int:
        dialog._layout_combo.setCurrentText("My rig")
        return QDialog.DialogCode.Accepted

    with patch.object(StartupDialog, "exec", choose_my_rig):
        window = create_mmgui(
            mm_config=None,
            mmcore=mmcore,
            install_sys_excepthook=False,
            install_sentry=False,
            exec_app=False,
            window_cls=cast("type[WindowProtocol]", MainWindow),
        )
    assert isinstance(window, MainWindow)
    qtbot.addWidget(window)
    assert settings.modern_window.last_layout == "My rig"
    window.close()


def test_loading_a_config_records_it_as_recent(
    mmcore: CMMCorePlus, qtbot: QtBot, settings: Settings
) -> None:
    """Every route into the core feeds the dialog's config list.

    ``systemConfigurationLoaded`` is delivered through Qt's event queue, so
    the record lands on the next event-loop turn rather than inside
    ``loadSystemConfiguration`` -- hence waitUntil, not a bare assert.
    """
    config = Path(__file__).with_name("test_config.cfg")
    window = MainWindow(mmcore=mmcore)
    qtbot.addWidget(window)

    mmcore.loadSystemConfiguration(str(config))

    qtbot.waitUntil(lambda: settings.recent_configs == [config.resolve()])
    window.close()


# ── the Acquire toolbar's layout menu ─────────────────────────────


def _layout_menu_names(page: object) -> list[str]:
    menu = page._layout_btn.build_menu()  # type: ignore[attr-defined]
    names = [a.text() for a in menu.actions() if a.text() and not a.isSeparator()]
    menu.deleteLater()
    return names


def test_saving_a_layout_writes_it_and_makes_it_current(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:

    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page.panel_button(PanelKey.EXCEPTION_LOG).click()

    with patch.object(QInputDialog, "getText", return_value=("My rig", True)):
        page._prompt_save_layout()

    saved = load_layout("My rig")
    assert saved is not None
    assert saved.panels == {PanelKey.MDA, PanelKey.PRESETS, PanelKey.EXCEPTION_LOG}
    assert page.layout_name == "My rig"
    assert "My rig" in _layout_menu_names(page)


def test_reserved_layout_names_are_refused(mmcore: CMMCorePlus, qtbot: QtBot) -> None:

    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    with (
        patch.object(QInputDialog, "getText", return_value=(DEFAULT_LAYOUT_NAME, True)),
        patch.object(QMessageBox, "warning") as warned,
    ):
        page._prompt_save_layout()

    warned.assert_called_once()
    assert list_layouts() == []
    assert page.layout_name == DEFAULT_LAYOUT_NAME


def test_cancelling_the_save_prompt_writes_nothing(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:

    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    with patch.object(QInputDialog, "getText", return_value=("Nope", False)):
        page._prompt_save_layout()

    assert list_layouts() == []


def test_layout_rows_offer_a_trash_only_for_saved_layouts(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Default and Last session regenerate themselves; they aren't deletable."""

    save_layout("My rig", _layout())
    store_session_layout(_layout())
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page.refresh_layout_menu()

    menu = page._layout_btn.build_menu()
    rows: dict[str, _LayoutMenuRow] = {
        a.text(): row
        for a in menu.actions()
        if isinstance(a, QWidgetAction)
        and a.text()
        and isinstance(row := a.defaultWidget(), _LayoutMenuRow)
    }
    assert rows[LAST_SESSION_LAYOUT_NAME]._trash.isHidden()
    assert rows[DEFAULT_LAYOUT_NAME]._trash.isHidden()
    assert not rows["My rig"]._trash.isHidden()
    menu.deleteLater()


def test_trash_deletes_the_layout_after_confirmation(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:

    save_layout("My rig", _layout())
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)
    page.refresh_layout_menu()
    page.select_layout("My rig")

    with patch.object(
        QMessageBox, "question", return_value=QMessageBox.StandardButton.No
    ):
        page._delete_layout("My rig")
    assert list_layouts() == ["My rig"]  # declined

    with patch.object(
        QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes
    ):
        page._delete_layout("My rig")

    assert list_layouts() == []
    # The page still *looks* the same, but that name is no longer selectable.
    assert page.layout_name == DEFAULT_LAYOUT_NAME
    assert "My rig" not in _layout_menu_names(page)


def test_selecting_a_vanished_layout_falls_back_to_the_default(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:

    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    with qtbot.waitSignal(page.layoutReset):
        page.select_layout("deleted-behind-our-back")

    assert page.layout_name == DEFAULT_LAYOUT_NAME
    assert page.open_panels() == {PanelKey.MDA, PanelKey.PRESETS}
