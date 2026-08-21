"""Named Acquire-page layouts, stored one JSON file per layout.

Deliberately *not* part of :mod:`pymmcore_gui._settings`: a layout is a
document the user creates, names, and throws away, not a preference. Giving
each one its own file in ``USER_DATA_DIR/layouts`` means the combo box in the
startup dialog is simply a directory listing -- so a layout the user (or a
sync tool, or a fresh install) deletes stops being offered without any
bookkeeping -- and layouts can be copied between machines or shared with a
colleague on their own.

Two names are reserved and never live in this directory:

* ``Default`` -- the built-in arrangement, which is code
  (``AcquirePage.reset_layout``), not data.
* ``Last session`` -- the arrangement auto-saved on close, which lives in
  ``Settings.modern_window`` alongside the other window state.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from base64 import b64decode, b64encode
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from ._settings import USER_DATA_DIR, Settings

if TYPE_CHECKING:
    from collections.abc import Iterable

DEFAULT_LAYOUT_NAME: Final = "Default"
"""The built-in arrangement. Selecting it means "reset", not "restore"."""

LAST_SESSION_LAYOUT_NAME: Final = "Last session"
"""The arrangement auto-saved on close, kept in settings rather than here."""

RESERVED_LAYOUT_NAMES: Final = frozenset(
    {DEFAULT_LAYOUT_NAME, LAST_SESSION_LAYOUT_NAME}
)

LAYOUTS_DIR: Final = USER_DATA_DIR / "layouts"
_SUFFIX: Final = ".json"
_VERSION: Final = 1
_UNSAFE_CHARS = re.compile(r"[^\w.\- ]+")
_DEFAULT_STAGE_KIND: Final = "xyz"
"""Must match ``_modern_gui._panels.StageKind.XYZ``.

See ``AcquireLayout.stage_kind``.
"""


@dataclass(frozen=True)
class AcquireLayout:
    """A saved arrangement of the Acquire page's dock widgets.

    The first three fields are exactly what ``AcquirePage`` needs to
    reproduce an arrangement, and the same triple the modern window has
    always persisted for the (single, anonymous) restore-on-launch layout.
    """

    dock_state: bytes | None = None
    """QtAds ``CDockManager.saveState()`` blob."""
    panels: frozenset[str] = field(default_factory=frozenset)
    """``PanelKey`` values that were open."""
    hidden_panels: frozenset[str] = field(default_factory=frozenset)
    """``PanelKey`` values whose toolbar buttons were hidden."""
    stage_devices: frozenset[str] = field(default_factory=frozenset)
    """Device names open in the Stages panel.

    Stored separately from ``dock_state``: the Stages panel's open devices
    live in its own nested dock manager, which the outer manager's
    ``saveState()`` doesn't capture -- see ``_modern_gui._acquire_stages``.
    """
    stage_kind: str = _DEFAULT_STAGE_KIND
    """Which widget flavor is docked under the Stages button.

    A ``_modern_gui._panels.StageKind`` value, kept as a plain ``str`` here
    the same way ``panels``/``hidden_panels`` keep ``PanelKey`` values as
    plain strings -- this module doesn't otherwise depend on ``_modern_gui``.
    """

    def is_empty(self) -> bool:
        """True if there is nothing here to restore."""
        return not self.dock_state or not self.panels

    def to_dict(self, name: str | None = None) -> dict[str, Any]:
        """Return a JSON-serializable form. ``dock_state`` becomes base64."""
        data: dict[str, Any] = {"version": _VERSION}
        if name is not None:
            data["name"] = name
        state = self.dock_state
        data["dock_state"] = b64encode(state).decode() if state else None
        data["panels"] = sorted(self.panels)
        data["hidden_panels"] = sorted(self.hidden_panels)
        data["stage_devices"] = sorted(self.stage_devices)
        data["stage_kind"] = self.stage_kind
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AcquireLayout:
        """Rebuild from :meth:`to_dict` output, ignoring unknown keys."""
        state = data.get("dock_state")
        return cls(
            dock_state=b64decode(state) if state else None,
            panels=_str_set(data.get("panels")),
            hidden_panels=_str_set(data.get("hidden_panels")),
            stage_devices=_str_set(data.get("stage_devices")),
            stage_kind=str(data.get("stage_kind") or _DEFAULT_STAGE_KIND),
        )


def _str_set(value: object) -> frozenset[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return frozenset()
    return frozenset(str(v) for v in value)


def layouts_dir() -> Path:
    """Return the directory holding saved layouts, creating it if needed.

    ``MMGUI_LAYOUTS_DIR`` overrides the location. This is read on every call
    (not captured at import) so tests -- which would otherwise scribble in the
    developer's real user data directory -- can redirect it with a fixture.
    """
    directory = Path(os.environ.get("MMGUI_LAYOUTS_DIR") or LAYOUTS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _file_stem(name: str) -> str:
    """Return a filesystem-safe stem for *name*.

    The display name is stored *inside* the file, so this only has to be
    safe and stable -- it doesn't have to be reversible.
    """
    stem = _UNSAFE_CHARS.sub("_", name).strip(" .")
    return stem or "layout"


def layout_path(name: str) -> Path:
    """Return the file *name* is (or would be) stored in."""
    return layouts_dir() / f"{_file_stem(name)}{_SUFFIX}"


def is_valid_layout_name(name: str) -> bool:
    """True if *name* is non-empty and doesn't collide with a reserved name."""
    return bool(name.strip()) and name.strip() not in RESERVED_LAYOUT_NAMES


def list_layouts() -> list[str]:
    """Return the names of every saved layout, sorted, case-insensitively.

    Unreadable or malformed files are skipped rather than raising: a corrupt
    layout must never stop the application from launching.
    """
    names: list[str] = []
    try:
        paths = sorted(layouts_dir().glob(f"*{_SUFFIX}"))
    except OSError:  # pragma: no cover -- unreadable user data dir
        return names
    for path in paths:
        if (data := _read(path)) is not None:
            name = data.get("name")
            names.append(str(name) if name else path.stem)
    return sorted(set(names), key=str.casefold)


def load_layout(name: str) -> AcquireLayout | None:
    """Return the layout saved under *name*, or None if it's gone or invalid."""
    data = _read(layout_path(name))
    if data is None:
        return None
    try:
        return AcquireLayout.from_dict(data)
    except Exception as e:  # pragma: no cover -- defensive
        warnings.warn(f"Ignoring invalid layout {name!r}: {e}", RuntimeWarning, 2)
        return None


def save_layout(name: str, layout: AcquireLayout) -> Path:
    """Write *layout* under *name*, overwriting any existing layout of that name."""
    path = layout_path(name)
    path.write_text(json.dumps(layout.to_dict(name), indent=2), errors="ignore")
    return path


def delete_layout(name: str) -> None:
    """Remove *name*'s file if it exists."""
    layout_path(name).unlink(missing_ok=True)


def existing_layouts(names: Iterable[str]) -> list[str]:
    """Filter *names* down to the layouts that still exist on disk."""
    available = set(list_layouts())
    return [n for n in names if n in available]


# ----------------------- the two reserved layouts -----------------------
# These live in Settings rather than in the layouts directory, but callers
# shouldn't have to care which storage a given name maps to -- that's what
# `available_layouts` / `resolve_layout` are for.


def session_layout() -> AcquireLayout:
    """Return the arrangement auto-saved when the app was last closed."""
    prefs = Settings.instance().modern_window
    return AcquireLayout(
        dock_state=prefs.acquire_dock_state,
        panels=frozenset(prefs.acquire_panels),
        hidden_panels=frozenset(prefs.acquire_hidden_panels),
        stage_devices=frozenset(prefs.acquire_stage_devices),
        stage_kind=prefs.acquire_stage_kind,
    )


def store_session_layout(layout: AcquireLayout) -> None:
    """Persist *layout* as the "Last session" arrangement (does not flush)."""
    prefs = Settings.instance().modern_window
    prefs.acquire_dock_state = layout.dock_state
    prefs.acquire_panels = set(layout.panels)
    prefs.acquire_hidden_panels = set(layout.hidden_panels)
    prefs.acquire_stage_devices = set(layout.stage_devices)
    prefs.acquire_stage_kind = layout.stage_kind


def available_layouts() -> list[str]:
    """Every selectable layout name, in the order they should be offered.

    ``Last session`` comes first when there is one to offer, then the
    built-in ``Default``, then the user's saved layouts alphabetically.
    """
    names = [DEFAULT_LAYOUT_NAME, *list_layouts()]
    if Settings.instance().modern_window.has_last_session_layout:
        names.insert(0, LAST_SESSION_LAYOUT_NAME)
    return names


def resolve_layout(name: str) -> AcquireLayout | None:
    """Return the layout *name* refers to, or None to mean "the built-in one".

    None is also returned for a saved layout that has since disappeared, so
    callers land on the working default instead of on nothing.
    """
    if name == LAST_SESSION_LAYOUT_NAME:
        return session_layout()
    if name == DEFAULT_LAYOUT_NAME:
        return None
    return load_layout(name)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(errors="ignore"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
