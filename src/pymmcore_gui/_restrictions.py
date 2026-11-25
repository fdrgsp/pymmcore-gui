"""Lock parts of the GUI that users should not change.

`restrictions.json`, next to the settings, says what a locked GUI disables:

    {"locked": true, "disable": ["installation_tab", "light_sources"]}

A hidden menu (right-click the empty toolbar space) locks and unlocks it. Locking
only disables controls: every setting stays as it is.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

from pymmcore_gui._qt.QtCore import QObject, Signal
from pymmcore_gui._settings import RESTRICTIONS_FILE, TESTING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pymmcore_gui._qt.QtWidgets import QWidget

logger = logging.getLogger(__name__)

NAMES = (
    "installation_tab",
    "hardware_tab",
    "configurations_tab",
    "light_sources",
    "save_channel_settings",
    "preferences.data_and_memory",
)
"""What a lock can disable; a file without a "disable" list disables all of it."""


class Restrictions(QObject):
    """Whether the GUI is locked, and what that disables."""

    changed = Signal()

    def __init__(self, file: Path | None) -> None:
        super().__init__()
        self._file = file
        self._locked = False
        # None: everything in NAMES, including names added by a later version
        self._disable: list[str] | None = None
        if file is not None and file.exists():
            self._read(file)

    def _read(self, file: Path) -> None:
        try:
            data = json.loads(file.read_text())
            disable = data.get("disable")
            if disable is not None and not (
                isinstance(disable, list) and all(isinstance(n, str) for n in disable)
            ):
                raise ValueError('"disable" must be a list of names')
        except (OSError, ValueError, AttributeError) as e:
            logger.warning("Could not read %s, nothing is locked: %s", file, e)
            return
        self._locked = data.get("locked") is True
        self._disable = disable
        if unknown := sorted(set(disable or ()) - set(NAMES)):
            logger.warning("Unknown names in %s: %s", file, ", ".join(unknown))

    @property
    def locked(self) -> bool:
        return self._locked

    def is_disabled(self, name: str) -> bool:
        return self._locked and name in (
            NAMES if self._disable is None else self._disable
        )

    def set_locked(self, locked: bool) -> None:
        """Lock or unlock the GUI, and remember it in the file."""
        self._locked = locked
        if self._file is not None:
            data: dict[str, object] = {"locked": locked}
            if self._disable is not None:
                data["disable"] = self._disable
            try:
                self._file.write_text(json.dumps(data, indent=2))
            except OSError as e:
                logger.warning("Could not save %s: %s", self._file, e)
        self.changed.emit()


def write_template(file: Path) -> None:
    """An unlocked file listing every name: delete those not to lock."""
    file.write_text(json.dumps({"locked": False, "disable": list(NAMES)}, indent=2))


_restrictions: Restrictions | None = None


def restrictions() -> Restrictions:
    """The app's restrictions; without settings (or in tests) there is no file."""
    global _restrictions
    if _restrictions is None:
        no_file = TESTING or os.getenv("PYMM_NO_SETTINGS")
        _restrictions = Restrictions(None if no_file else RESTRICTIONS_FILE)
    return _restrictions


class _Binding(QObject):
    """Enables its widgets as `name` allows; it goes with its parent widget."""

    def __init__(
        self,
        name: str,
        widgets: tuple[QWidget, ...],
        owner: QWidget,
        on_change: Callable[[bool], None] | None,
    ) -> None:
        super().__init__(owner)
        self._name = name
        self._widgets = widgets
        self._on_change = on_change
        restrictions().changed.connect(self.apply)
        self.apply()

    def apply(self) -> None:
        enabled = not restrictions().is_disabled(self._name)
        for widget in self._widgets:
            widget.setEnabled(enabled)
        if self._on_change is not None:
            self._on_change(enabled)


def restrict(
    name: str,
    *widgets: QWidget,
    owner: QWidget | None = None,
    on_change: Callable[[bool], None] | None = None,
) -> None:
    """Disable `widgets` while `name` is locked, from now on.

    `on_change(enabled)` covers what is not a widget's own enabled state. The
    binding lasts as long as `owner`, by default the first widget.
    """
    _Binding(name, widgets, owner or widgets[0], on_change)
