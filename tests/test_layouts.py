"""Named Acquire layouts: the on-disk store, the reserved names, and the resolver.

Layouts are files rather than settings keys (see ``pymmcore_gui._layouts``),
so the autouse ``layouts_dir`` fixture in ``conftest.py`` redirects
``MMGUI_LAYOUTS_DIR`` at a tmp_path for every test here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pymmcore_gui._layouts import (
    DEFAULT_LAYOUT_NAME,
    LAST_SESSION_LAYOUT_NAME,
    AcquireLayout,
    available_layouts,
    delete_layout,
    is_valid_layout_name,
    layout_path,
    list_layouts,
    load_layout,
    resolve_layout,
    save_layout,
    session_layout,
    store_session_layout,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pymmcore_gui._settings import Settings


def _layout() -> AcquireLayout:
    # Deliberately non-utf8 bytes: an ADS state blob is arbitrary binary, and
    # base64 (not str()) is what makes it survive a JSON round trip.
    return AcquireLayout(
        dock_state=b"\x00\x01\xfe\xffads",
        panels=frozenset({"mda", "presets"}),
        hidden_panels=frozenset({"console"}),
    )


def test_layout_round_trips_through_a_file(layouts_dir: Path) -> None:
    original = _layout()
    path = save_layout("My rig", original)

    assert path.parent == layouts_dir
    assert list_layouts() == ["My rig"]
    assert load_layout("My rig") == original


def test_layout_names_survive_filesystem_unsafe_characters() -> None:
    """The display name lives inside the file, so odd names are preserved."""
    save_layout("Rig #2: 60x/oil", _layout())

    assert list_layouts() == ["Rig #2: 60x/oil"]
    assert load_layout("Rig #2: 60x/oil") == _layout()


def test_deleting_a_layout_stops_offering_it() -> None:
    save_layout("Screening", _layout())
    assert "Screening" in available_layouts()

    delete_layout("Screening")

    assert list_layouts() == []
    assert "Screening" not in available_layouts()
    assert load_layout("Screening") is None
    # Deleting something already gone is not an error.
    delete_layout("Screening")


def test_a_layout_deleted_outside_the_app_is_simply_gone() -> None:
    """The combo is a directory listing, so external deletion needs no cleanup."""
    save_layout("Transient", _layout())
    layout_path("Transient").unlink()

    assert list_layouts() == []
    assert resolve_layout("Transient") is None


def test_corrupt_layout_files_are_skipped_not_raised(layouts_dir: Path) -> None:
    """A bad file must never stop the app from launching."""
    save_layout("Good", _layout())
    (layouts_dir / "broken.json").write_text("{ this is not json")
    (layouts_dir / "not_an_object.json").write_text("[1, 2, 3]")

    assert list_layouts() == ["Good"]
    assert load_layout("Good") == _layout()


@pytest.mark.parametrize(
    "name, valid",
    [
        ("My rig", True),
        ("", False),
        ("   ", False),
        (DEFAULT_LAYOUT_NAME, False),
        (LAST_SESSION_LAYOUT_NAME, False),
    ],
)
def test_reserved_and_empty_names_are_rejected(name: str, valid: bool) -> None:
    assert is_valid_layout_name(name) is valid


def test_default_resolves_to_none_meaning_the_built_in_arrangement() -> None:
    """ "Default" is code (``AcquirePage.reset_layout``), not a stored record."""
    assert resolve_layout(DEFAULT_LAYOUT_NAME) is None
    assert layout_path(DEFAULT_LAYOUT_NAME).exists() is False


def test_session_layout_round_trips_through_settings(settings: Settings) -> None:
    """ "Last session" lives in settings, beside the rest of the window state."""
    assert session_layout().is_empty()
    assert LAST_SESSION_LAYOUT_NAME not in available_layouts()

    store_session_layout(_layout())

    prefs = settings.modern_window
    assert prefs.acquire_dock_state == _layout().dock_state
    assert prefs.acquire_panels == {"mda", "presets"}
    assert prefs.acquire_hidden_panels == {"console"}
    assert session_layout() == _layout()
    assert resolve_layout(LAST_SESSION_LAYOUT_NAME) == _layout()


def test_available_layouts_ordering() -> None:
    """Last session first (when there is one), then Default, then saved names."""
    save_layout("zeta", _layout())
    save_layout("Alpha", _layout())

    assert available_layouts() == [DEFAULT_LAYOUT_NAME, "Alpha", "zeta"]

    store_session_layout(_layout())
    assert available_layouts() == [
        LAST_SESSION_LAYOUT_NAME,
        DEFAULT_LAYOUT_NAME,
        "Alpha",
        "zeta",
    ]


def test_an_empty_session_is_not_offered(settings: Settings) -> None:
    """A dock state with no open panels is not a restorable arrangement."""
    store_session_layout(AcquireLayout(dock_state=b"state", panels=frozenset()))
    assert not settings.modern_window.has_last_session_layout
    assert LAST_SESSION_LAYOUT_NAME not in available_layouts()
