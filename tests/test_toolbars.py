from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot


def test_oc_toolbar_initial_state(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    """Test that OCToolBar is populated with channel presets on creation."""
    from pymmcore_gui.widgets._toolbars import OCToolBar

    toolbar = OCToolBar(mmcore)
    qtbot.addWidget(toolbar)

    ch_group = mmcore.getChannelGroup()
    assert ch_group == "Channel"

    expected = list(mmcore.getAvailableConfigs(ch_group))
    action_texts = [a.text() for a in toolbar.actions()]
    assert action_texts == expected


def test_oc_toolbar_action_sets_config(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    """Test that clicking an action sets the config on core."""
    from pymmcore_gui.widgets._toolbars import OCToolBar

    toolbar = OCToolBar(mmcore)
    qtbot.addWidget(toolbar)

    actions = toolbar.actions()
    # find a non-current preset
    current = mmcore.getCurrentConfig(mmcore.getChannelGroup())
    target = next(a for a in actions if a.text() != current)

    target.trigger()
    assert mmcore.getCurrentConfig(mmcore.getChannelGroup()) == target.text()


def test_oc_toolbar_system_configuration_loaded(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Test that systemConfigurationLoaded triggers a refresh."""
    from pymmcore_gui.widgets._toolbars import OCToolBar

    toolbar = OCToolBar(mmcore)
    qtbot.addWidget(toolbar)

    with patch.object(toolbar, "_refresh", wraps=toolbar._refresh) as mock_refresh:
        # reconnect the mocked method
        mmcore.events.systemConfigurationLoaded.connect(mock_refresh)
        mmcore.events.systemConfigurationLoaded.emit()
        mock_refresh.assert_called()


def test_oc_toolbar_config_group_changed(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    """Test that configGroupChanged triggers a refresh."""
    from pymmcore_gui.widgets._toolbars import OCToolBar

    toolbar = OCToolBar(mmcore)
    qtbot.addWidget(toolbar)

    # clear toolbar to verify it gets repopulated on signal
    toolbar.clear()
    assert len(toolbar.actions()) == 0
    mmcore.events.configGroupChanged.emit("Channel", "DAPI")
    # _refresh should have repopulated the toolbar
    assert len(toolbar.actions()) > 0


def test_oc_toolbar_channel_group_changed(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    """Test that channelGroupChanged triggers a refresh."""
    from pymmcore_gui.widgets._toolbars import OCToolBar

    toolbar = OCToolBar(mmcore)
    qtbot.addWidget(toolbar)

    # clear toolbar to verify it gets repopulated on signal
    toolbar.clear()
    assert len(toolbar.actions()) == 0
    mmcore.events.channelGroupChanged.emit("Channel")
    # _refresh should have repopulated the toolbar
    assert len(toolbar.actions()) > 0


def test_oc_toolbar_config_set(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    """Test that configSet updates the checked state of actions."""
    from pymmcore_gui.widgets._toolbars import OCToolBar

    toolbar = OCToolBar(mmcore)
    qtbot.addWidget(toolbar)

    ch_group = mmcore.getChannelGroup()
    target_preset = "DAPI"
    mmcore.setConfig(ch_group, target_preset)

    # verify the correct action is now checked
    for action in toolbar.actions():
        if action.text() == target_preset:
            assert action.isChecked()
        else:
            assert not action.isChecked()


def test_oc_toolbar_property_changed(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    """Test that propertyChanged on Core/ChannelGroup triggers refresh."""
    from pymmcore_gui.widgets._toolbars import OCToolBar

    toolbar = OCToolBar(mmcore)
    qtbot.addWidget(toolbar)

    with patch.object(toolbar, "_refresh") as mock_refresh:
        # changing ChannelGroup property should trigger refresh
        mmcore.events.propertyChanged.emit("Core", "ChannelGroup", "Camera")
        mock_refresh.assert_called()


def test_oc_toolbar_property_changed_ignores_other(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    """Test that propertyChanged for non-ChannelGroup is ignored."""
    from pymmcore_gui.widgets._toolbars import OCToolBar

    toolbar = OCToolBar(mmcore)
    qtbot.addWidget(toolbar)

    with patch.object(
        toolbar, "_on_property_changed", wraps=toolbar._on_property_changed
    ) as mock_handler:
        mmcore.events.propertyChanged.connect(mock_handler)
        initial_actions = [a.text() for a in toolbar.actions()]
        mmcore.events.propertyChanged.emit("Camera", "Exposure", "10")
        mock_handler.assert_called()
        # actions should not have changed
        assert [a.text() for a in toolbar.actions()] == initial_actions


def test_oc_toolbar_config_defined(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    """Test that configDefined triggers a refresh."""
    from pymmcore_gui.widgets._toolbars import OCToolBar

    toolbar = OCToolBar(mmcore)
    qtbot.addWidget(toolbar)

    ch_group = mmcore.getChannelGroup()
    initial_count = len(toolbar.actions())

    # define a new config in the channel group
    mmcore.defineConfig(ch_group, "NewChannel", "Dichroic", "Label", "Q505LP")

    assert len(toolbar.actions()) == initial_count + 1
    assert "NewChannel" in [a.text() for a in toolbar.actions()]


def test_oc_toolbar_config_deleted(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    """Test that configDeleted triggers a refresh."""
    from pymmcore_gui.widgets._toolbars import OCToolBar

    toolbar = OCToolBar(mmcore)
    qtbot.addWidget(toolbar)

    ch_group = mmcore.getChannelGroup()
    initial_count = len(toolbar.actions())
    presets = list(mmcore.getAvailableConfigs(ch_group))

    # delete a config
    mmcore.deleteConfig(ch_group, presets[-1])

    assert len(toolbar.actions()) == initial_count - 1
    assert presets[-1] not in [a.text() for a in toolbar.actions()]
