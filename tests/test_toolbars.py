from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot

    from pymmcore_gui.widgets._toolbars import OCToolBar


@pytest.fixture()
def oc_toolbar(mmcore: CMMCorePlus, qtbot: QtBot) -> OCToolBar:
    from pymmcore_gui.widgets._toolbars import OCToolBar

    toolbar = OCToolBar(mmcore)
    qtbot.addWidget(toolbar)
    return toolbar


def test_oc_toolbar_initial_state(
    mmcore: CMMCorePlus, oc_toolbar: OCToolBar
) -> None:
    """Test that OCToolBar is populated with channel presets on creation."""
    ch_group = mmcore.getChannelGroup()
    assert ch_group == "Channel"

    expected = list(mmcore.getAvailableConfigs(ch_group))
    action_texts = [a.text() for a in oc_toolbar.actions()]
    assert action_texts == expected


def test_oc_toolbar_action_sets_config(
    mmcore: CMMCorePlus, oc_toolbar: OCToolBar
) -> None:
    """Test that clicking an action sets the config on core."""
    current = mmcore.getCurrentConfig(mmcore.getChannelGroup())
    target = next(a for a in oc_toolbar.actions() if a.text() != current)

    target.trigger()
    assert mmcore.getCurrentConfig(mmcore.getChannelGroup()) == target.text()


@pytest.mark.parametrize(
    "signal_attr, emit_args",
    [
        ("systemConfigurationLoaded", ()),
        ("configGroupChanged", ("Channel", "DAPI")),
        ("channelGroupChanged", ("Channel",)),
        ("configDefined", ("Channel", "DAPI", "Dichroic", "Label", "Q505LP")),
        ("configDeleted", ("Channel", "DAPI")),
    ],
    ids=[
        "systemConfigurationLoaded",
        "configGroupChanged",
        "channelGroupChanged",
        "configDefined",
        "configDeleted",
    ],
)
def test_oc_toolbar_refresh_signals(
    mmcore: CMMCorePlus,
    oc_toolbar: OCToolBar,
    signal_attr: str,
    emit_args: tuple[str, ...],
) -> None:
    """Test that signals connected to _refresh repopulate the toolbar."""
    oc_toolbar.clear()
    assert len(oc_toolbar.actions()) == 0

    signal = getattr(mmcore.events, signal_attr)
    signal.emit(*emit_args)

    assert len(oc_toolbar.actions()) > 0


def test_oc_toolbar_config_set(
    mmcore: CMMCorePlus, oc_toolbar: OCToolBar
) -> None:
    """Test that configSet updates the checked state of actions."""
    ch_group = mmcore.getChannelGroup()
    target_preset = "DAPI"
    mmcore.setConfig(ch_group, target_preset)

    for action in oc_toolbar.actions():
        assert action.isChecked() == (action.text() == target_preset)


@pytest.mark.parametrize(
    "device, prop, value, should_refresh",
    [
        ("Core", "ChannelGroup", "Camera", True),
        ("Camera", "Exposure", "10", False),
        ("Core", "Focus", "Z", False),
    ],
    ids=["core-channel-group", "other-device", "core-other-property"],
)
def test_oc_toolbar_property_changed(
    mmcore: CMMCorePlus,
    oc_toolbar: OCToolBar,
    device: str,
    prop: str,
    value: str,
    should_refresh: bool,
) -> None:
    """Test that only Core/ChannelGroup property changes trigger refresh."""
    oc_toolbar.clear()
    assert len(oc_toolbar.actions()) == 0

    mmcore.events.propertyChanged.emit(device, prop, value)

    if should_refresh:
        assert len(oc_toolbar.actions()) > 0
    else:
        assert len(oc_toolbar.actions()) == 0


def test_oc_toolbar_config_defined_adds_preset(
    mmcore: CMMCorePlus, oc_toolbar: OCToolBar
) -> None:
    """Test that defining a new config adds it to the toolbar."""
    ch_group = mmcore.getChannelGroup()
    initial_count = len(oc_toolbar.actions())

    mmcore.defineConfig(ch_group, "NewChannel", "Dichroic", "Label", "Q505LP")

    assert len(oc_toolbar.actions()) == initial_count + 1
    assert "NewChannel" in [a.text() for a in oc_toolbar.actions()]


def test_oc_toolbar_config_deleted_removes_preset(
    mmcore: CMMCorePlus, oc_toolbar: OCToolBar
) -> None:
    """Test that deleting a config removes it from the toolbar."""
    ch_group = mmcore.getChannelGroup()
    presets = list(mmcore.getAvailableConfigs(ch_group))
    initial_count = len(oc_toolbar.actions())

    mmcore.deleteConfig(ch_group, presets[-1])

    assert len(oc_toolbar.actions()) == initial_count - 1
    assert presets[-1] not in [a.text() for a in oc_toolbar.actions()]
