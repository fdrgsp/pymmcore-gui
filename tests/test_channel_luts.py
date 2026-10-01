from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import useq
from cmap import Colormap
from ndv.models import LUTModel

from pymmcore_gui._channel_luts import ChannelLUTMemory
from pymmcore_gui._settings import SettingsV1

if TYPE_CHECKING:
    import ndv


def _viewer(*colors: str) -> ndv.ArrayViewer:
    return cast(
        "ndv.ArrayViewer",
        SimpleNamespace(
            display_model=SimpleNamespace(
                luts={
                    i: LUTModel(cmap=Colormap(color)) for i, color in enumerate(colors)
                }
            )
        ),
    )


def test_live_mda_remembers_luts_by_channel_identity() -> None:
    settings = SettingsV1()
    memory = ChannelLUTMemory(settings)
    sequence = useq.MDASequence(
        channels=(
            useq.Channel(group="Channel", config="DAPI", exposure=10),
            useq.Channel(group="Channel", config="FITC", exposure=10),
        )
    )
    first = _viewer("green", "magenta")
    memory.bind_live_mda(first, sequence)

    first.display_model.luts[0].cmap = Colormap("cyan")

    assert settings.channel_lut("Channel", "DAPI") == "cmap:cyan"
    assert settings.channel_lut("Channel", "FITC") is None

    second = _viewer("green", "magenta")
    memory.bind_live_mda(second, sequence)

    assert second.display_model.luts[0].cmap.name == "cmap:cyan"
    assert second.display_model.luts[1].cmap.name.endswith("magenta")


def test_same_preset_in_different_groups_has_independent_lut() -> None:
    settings = SettingsV1(
        channel_luts={
            "Fluorescence": {"DAPI": "cyan"},
            "Camera mode": {"DAPI": "magenta"},
        }
    )
    memory = ChannelLUTMemory(settings)

    fluorescence = _viewer("green")
    memory.bind_live_mda(
        fluorescence,
        useq.MDASequence(
            channels=(useq.Channel(group="Fluorescence", config="DAPI", exposure=10),)
        ),
    )
    camera_mode = _viewer("green")
    memory.bind_live_mda(
        camera_mode,
        useq.MDASequence(
            channels=(useq.Channel(group="Camera mode", config="DAPI", exposure=10),)
        ),
    )

    assert fluorescence.display_model.luts[0].cmap.name == "cmap:cyan"
    assert camera_mode.display_model.luts[0].cmap.name == "cmap:magenta"


def test_channel_luts_roundtrip_through_settings_json() -> None:
    settings = SettingsV1(channel_luts={"Channel": {"DAPI": "cmap:cyan"}})

    restored = SettingsV1.model_validate_json(settings.model_dump_json())

    assert restored.channel_lut("Channel", "DAPI") == "cmap:cyan"
