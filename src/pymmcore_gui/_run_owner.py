"""Which page started the acquisition that currently owns the hardware."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from pymmcore_gui._qt.QtCore import QObject, Signal

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus


class RunOwner(str, Enum):
    """A page that can start (and therefore display) an acquisition."""

    ACQUIRE = "acquire"
    SMART = "smart"


class RunOwnership(QObject):
    """Single answer to "who started this run?", shared by the whole window.

    The window needs it to know which page to keep on screen during a run,
    and every viewer manager needs it to know whether to open a viewer for a
    run -- keeping it in one place stops the two from drifting apart.

    A page calls :meth:`claim` immediately before starting its run. A run
    nobody claimed (a script in the console, ``mda.run()`` from anywhere
    else) belongs to Acquire, which is what every run did before other pages
    could start one. The claim is dropped when the run finishes.
    """

    ownerChanged = Signal(object)
    """Emitted with the new owner (a `RunOwner`, or None once released)."""

    def __init__(self, mmcore: CMMCorePlus, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._mmc = mmcore
        self._owner: RunOwner | None = None
        # Bound to a QObject living on the GUI thread, so this is queued from
        # the runner thread and runs after the run has fully finished.
        mmcore.mda.events.sequenceFinished.connect(self._on_sequence_finished)

    @property
    def owner(self) -> RunOwner | None:
        """The page that claimed the current (or pending) run, if any."""
        return self._owner

    def claim(self, owner: RunOwner) -> None:
        """Mark *owner* as the page starting the next run.

        Raises ``RuntimeError`` while another run still owns the hardware.
        """
        if self._mmc.mda.is_running():
            raise RuntimeError("Cannot claim a run while an acquisition is running.")
        self._set_owner(owner)

    def release(self) -> None:
        """Drop the current claim (e.g. a run that failed to launch)."""
        self._set_owner(None)

    def accepts(self, page: RunOwner) -> bool:
        """Whether the current run belongs to *page*; unclaimed runs are Acquire's."""
        if self._owner is None:
            return page is RunOwner.ACQUIRE
        return self._owner is page

    def _set_owner(self, owner: RunOwner | None) -> None:
        if owner is not self._owner:
            self._owner = owner
            self.ownerChanged.emit(owner)

    def _on_sequence_finished(self, *_: object) -> None:
        # Queued delivery can lag: if the next page already claimed and
        # started a new run by the time this arrives, the claim is theirs.
        if not self._mmc.mda.is_running():
            self.release()
