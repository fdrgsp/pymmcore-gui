"""Persistence of the modern GUI's window state: geometry, theme, zoom, dock layout.

The autouse ``settings`` fixture (see ``conftest.py``) patches
``pymmcore_gui._settings._GLOBAL_SETTINGS`` with a fresh in-memory
``Settings()`` for every test, so this whole round trip is testable despite
``TESTING`` disabling the settings file on disk -- exactly how
``tests/test_main_window.py::test_save_restore_state`` tests the classic GUI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pymmcore_gui._app import create_mmgui
from pymmcore_gui._modern_gui._main_win import MainWindow
from pymmcore_gui._modern_gui._panels import PanelKey
from pymmcore_gui._modern_gui._theme import zoom_factor, zoom_in

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot

    from pymmcore_gui._app import WindowProtocol
    from pymmcore_gui._settings import Settings


def test_modern_window_save_restore_state(
    mmcore: CMMCorePlus, qtbot: QtBot, settings: Settings
) -> None:
    """geometry/theme/zoom/dock-layout round-trip through Settings, across windows."""
    win1 = MainWindow(mmcore=mmcore)
    qtbot.addWidget(win1)
    win1._acquire.panel_button(PanelKey.PRESETS).click()
    win1._toggle_theme()  # dark -> light
    zoom_in()
    win1._save_state()

    prefs = settings.modern_window
    assert prefs.geometry
    assert prefs.acquire_panels == {PanelKey.MDA, PanelKey.PRESETS}
    assert prefs.theme == "light"
    assert prefs.zoom == zoom_factor()
    win1.close()

    win2 = MainWindow(mmcore=mmcore)
    qtbot.addWidget(win2)
    # Applied in __init__ (_apply_saved_appearance), before restore_state is
    # even called -- so the theme/zoom preference is already live here.
    assert win2._is_dark is False
    assert zoom_factor() == prefs.zoom

    win2.restore_state(show=True)
    assert win2.isVisible()
    assert PanelKey.PRESETS in win2._acquire.open_panels()
    win2.close()


def test_restore_state_on_fresh_settings_opens_only_defaults(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """A first launch (no saved state) must open only the default panels.

    Regression test for the launch path force-opening every registered panel:
    ``restore_state`` -> ``apply_hidden_panels(set())`` used to treat "show
    this button" as "open this panel", so a fresh install eagerly built all
    six panels, tabbed five of them into the right column, and left the MDA
    dock buried. It also poisoned the *next* launch, since the resulting
    state was then persisted on close.
    """
    win = MainWindow(mmcore=mmcore)
    qtbot.addWidget(win)
    win.restore_state(show=True)

    acquire = win._acquire
    assert acquire.open_panels() == {PanelKey.MDA}
    assert sorted(acquire._dock_manager.dockWidgetsMap()) == [
        "acquire_mda",
        "acquire_viewers",
    ]
    assert acquire.hidden_panels() == set()
    assert acquire.panel_widget(PanelKey.CONSOLE) is None
    win.close()


def test_restored_layout_has_real_column_widths_on_second_launch(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """A restored layout must come back with usable widths, not collapsed to 0.

    Regression test for the second-launch blank Acquire page. ``_app`` defers
    ``restore_state`` to a ``singleShot`` that itself calls ``show()``, so
    ``restore_layout`` runs while every dock area is still 0px wide. Installing
    the ``minimumWidth == maximumWidth`` lock there froze the MDA column shut
    permanently -- the lock is deliberately immune to later layout passes, so
    nothing could ever widen it again and the panel rendered as empty space.
    """
    win1 = MainWindow(mmcore=mmcore)
    qtbot.addWidget(win1)
    win1.resize(1400, 900)
    win1.show()
    qtbot.waitExposed(win1)
    win1._save_state()
    # Fully tear the first window down before building the second: two live
    # MainWindows (three pages and a CDockManager each) is enough to segfault
    # the offscreen platform during teardown repaints -- the harness-only ADS
    # fragility documented in ``_configure_ads``.
    win1.close()
    win1.deleteLater()
    qtbot.wait(10)

    # Second launch, in _app's order: build, restore (window never shown yet),
    # and only then show.
    win2 = MainWindow(mmcore=mmcore)
    qtbot.addWidget(win2)
    win2.resize(1400, 900)
    win2.restore_state(show=True)
    qtbot.waitExposed(win2)
    win2._activate_acquire()
    qtbot.wait(100)

    acquire = win2._acquire
    mda_area = acquire._mda_dock.dockAreaWidget()
    assert mda_area is not None
    assert mda_area.width() > 0
    assert acquire._mda.width() > 0
    assert acquire._mda.isVisible()
    assert not acquire._mda_dock.isClosed()
    win2.close()


def test_modern_window_persists_hidden_panel_buttons(
    mmcore: CMMCorePlus, qtbot: QtBot, settings: Settings
) -> None:
    """Buttons hidden via the ⋯ customize menu stay hidden across restarts."""
    win1 = MainWindow(mmcore=mmcore)
    qtbot.addWidget(win1)
    win1._acquire.apply_hidden_panels({PanelKey.CONSOLE, PanelKey.EXCEPTION_LOG})
    win1._save_state()
    assert settings.modern_window.acquire_hidden_panels == {
        PanelKey.CONSOLE,
        PanelKey.EXCEPTION_LOG,
    }
    win1.close()

    win2 = MainWindow(mmcore=mmcore)
    qtbot.addWidget(win2)
    win2.restore_state()
    assert win2._acquire.hidden_panels() == {PanelKey.CONSOLE, PanelKey.EXCEPTION_LOG}
    assert win2._acquire.panel_button(PanelKey.CONSOLE).isHidden()
    assert not win2._acquire.panel_button(PanelKey.PRESETS).isHidden()
    win2.close()


def test_reset_layout_clears_only_the_persisted_layout_keys(
    mmcore: CMMCorePlus, qtbot: QtBot, settings: Settings
) -> None:
    """Reset Layout drops the saved arrangement but keeps geometry/theme/zoom.

    Uses a hidden button rather than an open side panel to trigger a
    non-default state: resetting an open panel would empty its dock area,
    which is fatal under the offscreen test platform (see ``_configure_ads``).
    The dock-level reset behaviour is covered in ``test_new_gui.py``.
    """
    win = MainWindow(mmcore=mmcore)
    qtbot.addWidget(win)
    win._acquire.apply_hidden_panels({PanelKey.CONSOLE})
    win._toggle_theme()  # dark -> light
    win._save_state()

    prefs = settings.modern_window
    assert prefs.acquire_dock_state and prefs.acquire_panels == {PanelKey.MDA}
    assert prefs.acquire_hidden_panels == {PanelKey.CONSOLE}
    geometry, theme, zoom = prefs.geometry, prefs.theme, prefs.zoom

    win._acquire.reset_layout()

    assert prefs.acquire_dock_state is None
    assert prefs.acquire_panels == set()
    assert prefs.acquire_hidden_panels == set()
    # Preferences are not layout -- losing these to a layout reset would be
    # a surprise.
    assert (prefs.geometry, prefs.theme, prefs.zoom) == (geometry, theme, zoom)
    win.close()


def test_modern_window_does_not_touch_classic_window_settings(
    mmcore: CMMCorePlus, qtbot: QtBot, settings: Settings
) -> None:
    """The modern GUI's settings section is independent of the classic GUI's.

    The two GUIs use different ADS dock objectNames for otherwise
    similarly-named panels (e.g. ``acquire_mda`` vs.
    ``docked_pymmcore_gui.mda_widget``), so sharing one settings blob would
    collide -- ``ModernWindowSettingsV1`` must be a separate section.
    """
    win = MainWindow(mmcore=mmcore)
    qtbot.addWidget(win)
    win._save_state()
    win.close()

    assert settings.window.geometry is None
    assert settings.window.dock_manager_state is None


def test_restore_state_shows_the_window(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    """create_mmgui's ``hasattr(win, "restore_state")`` branch actually shows it.

    Adding ``restore_state`` opts ``MainWindow`` out of ``create_mmgui``'s
    direct ``win.show()`` call (see ``_app.py``) in favor of a deferred
    ``QTimer.singleShot(0, lambda: win.restore_state(show=True))`` -- this
    guards that the window still ends up visible.
    """
    window = cast(
        "MainWindow",
        create_mmgui(
            mm_config=False,
            mmcore=mmcore,
            install_sys_excepthook=False,
            install_sentry=False,
            exec_app=False,
            window_cls=cast("type[WindowProtocol]", MainWindow),
        ),
    )
    qtbot.addWidget(window)
    qtbot.waitUntil(window.isVisible)
    window.close()
