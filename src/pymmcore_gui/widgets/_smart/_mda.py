"""The Smart Microscopy tab's base-acquisition editor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_gui.widgets._mda_widget import MemoryMDAWidget

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import useq
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.mda import SingleOutput

    from pymmcore_gui._qt.QtWidgets import QWidget

    Launcher = Callable[[useq.MDASequence, SingleOutput | None], None]


class SmartMDAWidget(MemoryMDAWidget):
    """The app's MDA editor, whose Run button starts a *smart* run instead.

    Everything else -- the editors, Saving section, Pause/Cancel, the
    acquisition lock and its overlays, the missing-pixel-size guard -- is
    inherited unchanged. Only the final launch is redirected to the
    Smart Microscopy page, which first starts the analysis worker (possibly
    seconds, in process mode) and then the acquisition.
    """

    def __init__(self, mmcore: CMMCorePlus, parent: QWidget | None = None) -> None:
        self._launcher: Launcher | None = None
        super().__init__(mmcore, parent)

    def set_launcher(self, launcher: Launcher) -> None:
        """Set what Run calls with ``(base sequence, output)``."""
        self._launcher = launcher

    def execute_mda(self, output: SingleOutput | Sequence[SingleOutput] | None) -> None:
        """Upstream ``execute_mda`` with ``core.run_mda`` replaced by the launcher.

        Keeps upstream's continuous-focus handling: focus is switched off for
        the run when requested, and restored by upstream's own
        ``sequenceFinished`` handler -- or by `launch_failed` if the run
        never starts.
        """
        if self._launcher is None:
            raise RuntimeError("SmartMDAWidget has no launcher.")
        sequence = self.value()
        if self._disable_af_on_run:
            self._disable_af_on_run = False
            self._disable_continuous_focus()
        try:
            self._launcher(sequence, output)  # type: ignore[arg-type]
        except Exception:
            self.launch_failed()
            raise

    def launch_failed(self) -> None:
        """Undo the pre-launch steps of a run that never started."""
        self._restore_continuous_focus()
        self._progress_overlay.stop()

    def show_busy(self, message: str) -> None:
        """Cover the editor with the progress overlay (e.g. while starting up)."""
        self._progress_overlay.start(message)

    def hide_busy(self) -> None:
        self._progress_overlay.stop()
