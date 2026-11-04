import json
import os
import stat
import sys
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic_settings import BaseSettings

from pymmcore_gui import _settings
from pymmcore_gui._settings import MMGuiUserPrefsSource, SettingsV1


@pytest.fixture(autouse=True)
def _stable_memory_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep serialization defaults stable as the runner's free RAM changes."""
    monkeypatch.setattr("pymmcore_gui._utils.system_memory_gb", lambda: (16.0, 10.0))


def test_settings() -> None:
    settings = SettingsV1()
    assert settings.version == "1.0"
    assert settings.version_tuple == (1, 0, "")

    # settings ignores unrecognized fields
    # (this is important for backwards compatibility)
    v = SettingsV1(random_value="asdf")  # pyright: ignore[reportCallIssue]
    assert not hasattr(v, "random_value")


@pytest.mark.parametrize("invalid", [None, [], 42, "invalid"])
def test_invalid_nested_settings_do_not_block_valid_preferences(
    tmp_path: Path, invalid: object
) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"window": invalid, "send_error_reports": True}))
    with (
        patch.object(_settings, "SETTINGS_FILE_NAME", path),
        pytest.warns(RuntimeWarning, match="Could not validate key 'window'"),
    ):
        assert MMGuiUserPrefsSource(SettingsV1)() == {"send_error_reports": True}


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
        "acquire_mda_kind": "topbar",
        "last_layout": "My rig",
    }
    payload: dict[str, object] = {"version": "1.0"}
    path = tmp_path / "settings.json"
    if sections in {"legacy", "mixed"}:
        payload["window"] = legacy
    if sections in {"modern", "mixed"}:
        payload["modern_window"] = modern
    path.write_text(json.dumps(payload))
    # the scratch memory default follows the RAM available at the time of each load
    with (
        patch.object(_settings, "SETTINGS_FILE_NAME", path),
        patch("pymmcore_gui._utils.system_memory_gb", return_value=(16.0, 8.0)),
    ):
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
        assert restored.modern_window.acquire_mda_kind == "topbar"


def _flush(settings: SettingsV1, timeout: float = 30) -> None:
    with patch("pymmcore_gui._settings.TESTING", False):
        settings.flush(timeout=timeout)


def test_overlapping_saves_never_leave_a_broken_file(tmp_path: Path) -> None:
    """Saves of different lengths used to interleave, leaving a stray tail behind."""
    path = tmp_path / "settings.json"
    settings = SettingsV1()
    broken: list[str] = []
    done = threading.Event()

    def read_continuously() -> None:
        while not done.is_set():
            if content := path.read_text() if path.exists() else "":
                try:
                    json.loads(content)
                except ValueError:
                    broken.append(content)

    def save_repeatedly(size: int) -> None:
        # each writer has its own length, so an interleaved write can't go unnoticed
        other = SettingsV1(last_config=Path("x" * size))
        for _ in range(100):
            other._write_settings()

    with patch.object(_settings, "SETTINGS_FILE_NAME", path):
        # Windows can't replace a file that's open, so reading this eagerly there
        # uses up the writers' lock retries, which no real reader does (those are
        # covered by the test_windows_save_* tests). The writers still have to
        # leave an intact file below.
        reader = threading.Thread(target=read_continuously)
        if sys.platform != "win32":
            reader.start()
        writers = [
            threading.Thread(target=save_repeatedly, args=(size,))
            for size in (1, 500, 5000, 50000)
        ]
        for writer in writers:
            writer.start()
        for writer in writers:
            writer.join()
        done.set()
        if reader.is_alive():
            reader.join()

        assert not broken, f"read {len(broken)} broken files: {broken[0][-40:]!r}"
        SettingsV1.model_validate_json(path.read_text())
        _flush(settings)
        SettingsV1.model_validate_json(path.read_text())


def test_failed_save_keeps_the_old_file_and_leaves_nothing_behind(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{"send_error_reports": true}')
    with (
        patch.object(_settings, "SETTINGS_FILE_NAME", path),
        patch("pymmcore_gui._settings.os.replace", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        SettingsV1(send_error_reports=False)._write_settings()

    assert path.read_text() == '{"send_error_reports": true}'
    assert [p.name for p in tmp_path.iterdir() if p.is_file()] == ["settings.json"]


def test_save_keeps_the_file_permissions(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{}")
    path.chmod(0o640)
    mode = stat.S_IMODE(path.stat().st_mode)
    with patch.object(_settings, "SETTINGS_FILE_NAME", path):
        SettingsV1(send_error_reports=True)._write_settings()
    assert stat.S_IMODE(path.stat().st_mode) == mode
    assert json.loads(path.read_text()) == {"send_error_reports": True}


def test_save_writes_through_a_symlink(tmp_path: Path) -> None:
    real = tmp_path / "dotfiles" / "settings.json"
    real.parent.mkdir()
    real.write_text("{}")
    link = tmp_path / "settings.json"
    try:
        link.symlink_to(real)
    except OSError:  # pragma: no cover
        pytest.skip("symlinks are not available")
    with patch.object(_settings, "SETTINGS_FILE_NAME", link):
        SettingsV1(send_error_reports=True)._write_settings()
    assert link.is_symlink()
    assert json.loads(real.read_text()) == {"send_error_reports": True}


def test_first_save_creates_the_file(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    with patch.object(_settings, "SETTINGS_FILE_NAME", path):
        SettingsV1(send_error_reports=True)._write_settings()
    assert json.loads(path.read_text()) == {"send_error_reports": True}


@pytest.mark.parametrize(
    "broken",
    ['{"channel_luts": {"Channel": {"DAPI": "cmap:blue"}}} }\n}', "[]"],
    ids=["stray-tail", "not-a-dict"],
)
def test_unreadable_file_is_kept_rather_than_overwritten(
    tmp_path: Path, broken: str
) -> None:
    """The next save used to replace it with defaults, losing everything in it."""
    path = tmp_path / "settings.json"
    path.write_text(broken)
    with (
        patch.object(_settings, "SETTINGS_FILE_NAME", path),
        pytest.warns(RuntimeWarning, match=r"Failed to read.*kept as settings\."),
    ):
        assert MMGuiUserPrefsSource(SettingsV1)() == {}
        SettingsV1(send_error_reports=True)._write_settings()

    [backup] = tmp_path.glob("settings.unreadable-*.json")
    assert backup.read_text() == broken
    assert json.loads(path.read_text()) == {"send_error_reports": True}


def test_file_that_cannot_be_opened_is_left_alone(tmp_path: Path) -> None:
    """Not being able to read it right now says nothing about its contents."""
    path = tmp_path / "settings.json"
    path.write_text('{"send_error_reports": true}')
    with (
        patch.object(_settings, "SETTINGS_FILE_NAME", path),
        patch.object(
            MMGuiUserPrefsSource, "content", side_effect=PermissionError("locked")
        ),
        pytest.warns(RuntimeWarning, match="Failed to read settings"),
    ):
        assert MMGuiUserPrefsSource(SettingsV1)() == {}

    assert [p.name for p in tmp_path.iterdir() if p.is_file()] == ["settings.json"]


def test_warning_still_appears_if_the_file_cannot_be_moved(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("not json")
    with (
        patch.object(_settings, "SETTINGS_FILE_NAME", path),
        patch.object(Path, "replace", side_effect=OSError("read-only")),
        pytest.warns(RuntimeWarning, match="Failed to read settings") as caught,
    ):
        assert MMGuiUserPrefsSource(SettingsV1)() == {}
    assert "kept as" not in str(caught[0].message)


def test_unreadable_backups_within_one_second_are_preserved(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    stamp = datetime(2026, 11, 3, 12, 0, 0)
    with (
        patch.object(_settings, "SETTINGS_FILE_NAME", path),
        patch.object(_settings, "datetime") as clock,
    ):
        clock.now.return_value = stamp
        for content in ("first broken file", "second broken file"):
            path.write_text(content)
            with pytest.warns(RuntimeWarning, match="kept as"):
                assert MMGuiUserPrefsSource(SettingsV1)() == {}
    assert {p.read_text() for p in tmp_path.glob("settings.unreadable-*.json")} == {
        "first broken file",
        "second broken file",
    }


def test_windows_save_retries_a_temporary_file_lock(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{}")
    replace = os.replace
    attempts = 0

    def locked_then_replace(source: str, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            assert path.read_text() == "{}"
            raise PermissionError("sharing violation")
        replace(source, destination)

    with (
        patch.object(_settings, "sys", SimpleNamespace(platform="win32")),
        patch("pymmcore_gui._settings.os.replace", side_effect=locked_then_replace),
        patch("pymmcore_gui._settings.time.sleep") as sleep,
    ):
        _settings._write_atomically(path, '{"send_error_reports": true}')
    assert attempts == 3
    assert sleep.call_count == 2
    assert json.loads(path.read_text()) == {"send_error_reports": True}
    assert [p for p in tmp_path.iterdir() if p.is_file()] == [path]


def test_windows_save_stops_retrying_a_persistent_file_lock(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{}")
    with (
        patch.object(_settings, "sys", SimpleNamespace(platform="win32")),
        patch(
            "pymmcore_gui._settings.os.replace", side_effect=PermissionError("locked")
        ) as replace,
        patch("pymmcore_gui._settings.time.sleep") as sleep,
        pytest.raises(PermissionError, match="locked"),
    ):
        _settings._write_atomically(path, '{"send_error_reports": true}')
    assert replace.call_count == 5
    assert sleep.call_count == 4
    assert path.read_text() == "{}"
    assert [p for p in tmp_path.iterdir() if p.is_file()] == [path]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file sharing semantics")
def test_windows_save_preserves_settings_while_a_reader_holds_the_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{}")
    with path.open() as reader:
        with (
            patch("pymmcore_gui._settings.time.sleep") as sleep,
            pytest.raises(PermissionError),
        ):
            _settings._write_atomically(path, '{"send_error_reports": true}')
        assert sleep.call_count == 4
        assert reader.read() == "{}"
    assert path.read_text() == "{}"
    assert [p for p in tmp_path.iterdir() if p.is_file()] == [path]
