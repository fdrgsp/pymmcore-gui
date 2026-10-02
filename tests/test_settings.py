import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic_settings import BaseSettings

from pymmcore_gui import _settings
from pymmcore_gui._settings import MMGuiUserPrefsSource, SettingsV1


def test_settings() -> None:
    settings = SettingsV1()
    assert settings.version == "1.0"
    assert settings.version_tuple == (1, 0, "")

    # settings ignores unrecognized fields
    # (this is important for backwards compatibility)
    v = SettingsV1(random_value="asdf")  # pyright: ignore[reportCallIssue]
    assert not hasattr(v, "random_value")


def test_default_scratch_memory_leaves_room_for_map() -> None:
    with patch("pymmcore_gui._utils.system_memory_gb", return_value=(16.0, 10.0)):
        assert _settings._default_max_memory_gb() == 6.0


def test_user_settings(tmp_path: Path) -> None:
    fake_settings = tmp_path / "settings.json"
    with patch.object(_settings, "SETTINGS_FILE_NAME", fake_settings):
        assert not MMGuiUserPrefsSource.exists()
        assert MMGuiUserPrefsSource.values() == {}

        fake_settings.touch()
        assert MMGuiUserPrefsSource.exists()
        assert MMGuiUserPrefsSource(BaseSettings)() == {}

        # test unrecognized fields are ignored
        fake_settings.write_text('{"a": 1}')
        with pytest.warns(RuntimeWarning, match="not found in model"):
            assert MMGuiUserPrefsSource(BaseSettings)() == {"a": 1}

        # test that the presence of invalid fields don't invalidate
        # the entire settings file
        fake_settings.write_text(
            '{"window": {"geometry": "AAAC", "window_state": [1,2,3] } }'
        )
        with pytest.warns(
            RuntimeWarning, match="Could not validate key 'window_state'"
        ):
            assert MMGuiUserPrefsSource(BaseSettings)() == {
                "window": {"geometry": "AAAC"}
            }

        # test invalid json doesn't ruin everything
        fake_settings.write_text("[]")
        with pytest.warns(RuntimeWarning, match="Failed to read settings"):
            assert MMGuiUserPrefsSource(BaseSettings)() == {}

        obj = SettingsV1.instance()
        assert obj.auto_load_last_config is None
        obj.auto_load_last_config = True
        with patch("pymmcore_gui._settings.TESTING", False):
            obj.flush(timeout=0.2)

        txt = fake_settings.read_text()
        obj2 = SettingsV1.model_validate_json(txt)
        assert obj2.auto_load_last_config is True

        assert fake_settings.exists()
        with patch("pymmcore_gui._settings.TESTING", False):
            _settings.reset_to_defaults()
        assert not fake_settings.exists()


@pytest.mark.parametrize("sections", ["legacy", "modern", "mixed"])
def test_existing_window_sections_round_trip(tmp_path: Path, sections: str) -> None:
    """Read and save existing settings without mixing incompatible dock state."""
    legacy = {
        "geometry": "AAAC",
        "dock_manager_state": "AAAE",
        "open_widgets": ["pymmcore_gui.console"],
    }
    modern = {
        "theme": "light",
        "zoom": 1.25,
        "acquire_panels": ["mda", "presets", "console"],
        "acquire_hidden_panels": ["properties"],
        "acquire_stage_devices": ["XY"],
        "acquire_stage_kind": "per_device",
        "last_layout": "My rig",
    }
    payload: dict[str, object] = {"version": "1.0"}
    path = tmp_path / "settings.json"
    if sections in {"legacy", "mixed"}:
        payload["window"] = legacy
    if sections in {"modern", "mixed"}:
        payload["modern_window"] = modern
    path.write_text(json.dumps(payload))
    with patch.object(_settings, "SETTINGS_FILE_NAME", path):
        loaded = SettingsV1(**MMGuiUserPrefsSource(SettingsV1)())
        path.write_text(loaded.model_dump_json(exclude_defaults=True))
        restored = SettingsV1(**MMGuiUserPrefsSource(SettingsV1)())
    assert restored.model_dump() == loaded.model_dump()
    if sections in {"legacy", "mixed"}:
        assert restored.window.dock_manager_state == b"\x00\x00\x04"
    assert restored.modern_window.acquire_dock_state is None
    if sections == "legacy":
        assert restored.modern_window.theme == "dark"
        assert not restored.modern_window.acquire_panels
    else:
        assert restored.modern_window.theme == "light"
        assert restored.modern_window.acquire_panels == {"mda", "presets", "console"}
        assert restored.modern_window.acquire_hidden_panels == {"properties"}
        assert restored.modern_window.acquire_stage_devices == {"XY"}
