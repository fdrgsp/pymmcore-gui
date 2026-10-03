import pytest
from pymmcore_plus import CMMCorePlus, DeviceType
from pytestqt.qtbot import QtBot

from pymmcore_gui._qt.QtWidgets import QWidget
from pymmcore_gui.actions import ActionInfo, CoreAction, WidgetAction, WidgetActionInfo
from pymmcore_gui.actions.widget_actions import _get_core, create_stage_widget
from pymmcore_gui.widgets._acquire import AcquirePage
from pymmcore_gui.widgets._panels import PANELS, PanelInfo, PanelKey


def test_action_registry() -> None:
    info = ActionInfo.for_key(CoreAction.SNAP)
    assert info.text == "Snap Image"

    with pytest.raises(KeyError, match=f"Did you mean {CoreAction.LOAD_DEMO.value!r}?"):
        ActionInfo.for_key(CoreAction.LOAD_DEMO.value[:-2])

    with pytest.raises(TypeError, match="is not an instance of"):
        info = WidgetActionInfo.for_key(CoreAction.SNAP)
    info = WidgetActionInfo.for_key(WidgetAction.ABOUT)


def test_custom_panel_registration(
    qtbot: QtBot, mmcore: CMMCorePlus, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Acquire builds custom tools from the same registry as built-in panels.
    import pymmcore_gui.widgets._acquire as acquire_module

    panel = PanelInfo(
        key="mywidget",
        title="My Widget",
        icon="mdi-light:format-list-bulleted",
        tooltip="Custom panel",
        create=lambda parent, core: QWidget(parent),
    )
    monkeypatch.setattr(acquire_module, "PANELS", (*PANELS, panel))
    page = AcquirePage(mmcore=mmcore)
    qtbot.addWidget(page)
    assert page.panel_widget(panel.key) is None
    page.panel_button(panel.key).click()
    widget = page.panel_widget(panel.key)
    assert isinstance(widget, QWidget)
    dock = page.panel_dock(panel.key)
    assert dock is not None
    assert dock.objectName() == panel.dock_name
    assert page.panel_button(panel.key).isChecked()
    page.shutdown()


def test_get_core_uses_the_hosting_window_core(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Widgets resolve the core of the window that hosts them.

    A window named like the application's resolves its own core instead of
    accidentally using the global singleton.
    """
    own_core = CMMCorePlus()
    assert own_core is not CMMCorePlus.instance()  # a real, distinguishable core

    class _Window(QWidget):
        @property
        def mmcore(self) -> CMMCorePlus:
            return own_core

    win = _Window()
    win.setObjectName("pyMMGUI")
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


def test_stage_action_factory_uses_modern_controls(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    from pymmcore_gui.widgets._stage_control import StagesPanel

    parent = QWidget()
    qtbot.addWidget(parent)
    panel = create_stage_widget(parent)
    assert isinstance(panel, StagesPanel)
    expected = {
        device
        for kind in (DeviceType.XYStage, DeviceType.Stage)
        for device in mmcore.getLoadedDevicesOfType(kind)
    }
    assert panel.open_devices() == expected


def test_properties_panel_uses_the_pages_own_core(qtbot: QtBot) -> None:
    """Panels get the core their page was given, not the global singleton."""
    own_core = CMMCorePlus()
    assert own_core is not CMMCorePlus.instance()

    page = AcquirePage(mmcore=own_core)
    qtbot.addWidget(page)
    page.panel_button(PanelKey.PROPERTIES).click()
    browser = page.panel_widget(PanelKey.PROPERTIES)
    assert browser is not None
    assert browser._mmc is own_core  # type: ignore[attr-defined]
    page.shutdown()
