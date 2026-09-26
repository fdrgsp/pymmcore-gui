import pytest
from pymmcore_plus import CMMCorePlus
from pytestqt.qtbot import QtBot

from pymmcore_gui import MicroManagerGUI
from pymmcore_gui._qt.QtWidgets import QMenu, QWidget
from pymmcore_gui.actions import ActionInfo, CoreAction, WidgetAction, WidgetActionInfo
from pymmcore_gui.actions.widget_actions import _get_core


def test_action_registry() -> None:
    info = ActionInfo.for_key(CoreAction.SNAP)
    assert info.text == "Snap Image"

    with pytest.raises(KeyError, match=f"Did you mean {CoreAction.LOAD_DEMO.value!r}?"):
        ActionInfo.for_key(CoreAction.LOAD_DEMO.value[:-2])

    with pytest.raises(TypeError, match="is not an instance of"):
        info = WidgetActionInfo.for_key(CoreAction.SNAP)
    info = WidgetActionInfo.for_key(WidgetAction.ABOUT)


def test_actions_in_menus(qtbot: QtBot) -> None:
    # people can add new ones
    text = "My Widget!!!!"
    act = WidgetActionInfo(
        key="mywidget",
        text=text,
        icon="mdi-light:format-list-bulleted",
        create_widget=lambda p: QWidget(p),
    )
    assert "mywidget" in WidgetActionInfo._registry
    assert act in ActionInfo.widget_actions().values()

    win = MicroManagerGUI()
    qtbot.addWidget(win)
    mb = win.menuBar()
    assert mb
    window_menu = next(
        (m for a in mb.actions() if (m := a.menu()) and m.title() == "Window"), None
    )
    assert isinstance(window_menu, QMenu)
    assert any(a.text() == text for a in window_menu.actions())


@pytest.mark.parametrize("window_name", ["MicroManagerGUI", "pyMMGUI"])
def test_get_core_uses_the_hosting_window_core(
    mmcore: CMMCorePlus, qtbot: QtBot, window_name: str
) -> None:
    """Widgets resolve the core of whichever window hosts them.

    Both application windows must be recognized by `objectName`: the classic
    dock-based `MicroManagerGUI` and the modern `pyMMGUI`. Resolving only the
    classic name left every `_get_core`-based factory in the modern GUI
    (e.g. the Property Browser panel, which goes through `_ignoring_core`)
    silently bound to the process-wide singleton instead.
    """
    own_core = CMMCorePlus()
    assert own_core is not CMMCorePlus.instance()  # a real, distinguishable core

    class _Window(QWidget):
        @property
        def mmcore(self) -> CMMCorePlus:
            return own_core

    win = _Window()
    win.setObjectName(window_name)
    qtbot.addWidget(win)
    child = QWidget(win)
    grandchild = QWidget(child)

    assert _get_core(grandchild) is own_core


def test_get_core_falls_back_to_the_singleton(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """A parentless widget, or one under some other window, gets the singleton."""
    orphan = QWidget()
    qtbot.addWidget(orphan)
    assert _get_core(orphan) is CMMCorePlus.instance()

    other = QWidget()
    other.setObjectName("SomeOtherWindow")
    qtbot.addWidget(other)
    assert _get_core(QWidget(other)) is CMMCorePlus.instance()
