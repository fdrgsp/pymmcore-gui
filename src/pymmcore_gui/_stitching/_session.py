"""Acquisitions in, mosaics out: the GUI side of stitching (plan §3.3-§3.5).

`StitchingSession` watches the core's MDA runs. A run whose sequence asks
for stitching (`SequenceStitchingV1` in its metadata) and was saved to disk
becomes a job once the runner has closed the data store, which it does
before emitting `sequenceFinished`. Outcomes are reported as toasts, and
finished mosaics are opened in viewers if the settings ask for it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pymmcore_gui._qt.QtCore import QObject, QUrl, Signal
from pymmcore_gui._qt.QtGui import QDesktopServices
from pymmcore_gui._stitching._controller import (
    StitchingController,
    StitchJob,
    capture_linear_part,
    default_memory_cap,
    own_process_identity,
)
from pymmcore_gui._stitching._inputs import (
    acquisition_stem,
    build_job,
    calibration_from_summary,
)
from pymmcore_gui._stitching._models import StitchSettingsV1, request_from

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    import useq
    from pymmcore_plus import CMMCorePlus

logger = logging.getLogger(__name__)

MAX_AUTO_OPEN = 3
"""Mosaics opened in viewers by themselves; the toast lists the rest."""

OPEN = "Open"
SHOW_IN_FOLDER = "Show in Folder"
SHOW_LOG = "Show Log"


class Notifier(Protocol):
    """Where outcomes are reported: the window's `NotificationManager`."""

    def show_info_message(
        self,
        message: str,
        *actions: str,
        on_action: Callable[[str | None], None] | None = None,
    ) -> None: ...

    def show_warning_message(
        self,
        message: str,
        *actions: str,
        on_action: Callable[[str | None], None] | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class FinishedRun:
    """What a finished run leaves for stitching, captured on the runner thread."""

    sequence: useq.MDASequence
    finish_reason: str
    """"completed", "canceled" or "errored"."""
    source: Path | None = None
    """The closed data store, or None when nothing was saved to disk."""
    tile_shape_yx: tuple[int, int] | None = None
    calibration: tuple[float, np.ndarray] | None = None
    """Pixel size and the pixel -> stage linear part, at the start of the run."""
    stores_events: bool = False
    """The store has a single unbounded axis: its layout is unknown."""


def _closed_store(
    core: CMMCorePlus,
) -> tuple[Path | None, tuple[int, int] | None, bool]:
    """Path, frame shape and layout flag of the run's data sink, if on disk."""
    from ome_writers import ScratchFormat
    from pymmcore_plus.mda._sink import OmeWritersSink

    sink = core.mda.get_sink()
    if not isinstance(sink, OmeWritersSink):
        return None, None, False
    settings = sink.settings
    if isinstance(settings.format, ScratchFormat):
        return None, None, False
    try:
        source = Path(settings.output_path)
    except ValueError:  # the store was never set up
        return None, None, False
    shape: tuple[int, int] | None = None
    if (meta := sink.summary_meta) and (infos := meta.get("image_infos")):
        shape = (int(infos[0]["height"]), int(infos[0]["width"]))
    return source, shape, sink.stores_events


class StitchingSession(QObject):
    """Turns finished, saved grid acquisitions into stitching jobs.

    Parameters
    ----------
    core
        The core whose MDA runs are watched.
    notifications
        Where outcomes are reported.
    open_mosaic
        Opens a finished mosaic in a viewer.
    controller
        The job queue; one is created if not given.
    """

    _runFinished = Signal(object)

    def __init__(
        self,
        core: CMMCorePlus,
        notifications: Notifier,
        open_mosaic: Callable[[Path], Any],
        *,
        controller: StitchingController | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._core = core
        self._notifications = notifications
        self._open_mosaic = open_mosaic
        self._controller = controller or StitchingController(self)
        self._calibration: tuple[float, np.ndarray] | None = None
        self._runFinished.connect(self._submit)
        self._controller.jobFinished.connect(self._on_job_finished)
        self._controller.jobFailed.connect(self._on_job_failed)
        core.mda.events.sequenceStarted.connect(self._on_sequence_started)
        core.mda.events.sequenceFinished.connect(self._on_sequence_finished)

    @property
    def controller(self) -> StitchingController:
        return self._controller

    def shutdown(self, timeout_ms: int = 10_000) -> None:
        """Stop watching runs, cancel every job and wait for the worker to exit."""
        try:
            self._core.mda.events.sequenceStarted.disconnect(self._on_sequence_started)
            self._core.mda.events.sequenceFinished.disconnect(
                self._on_sequence_finished
            )
        except (TypeError, ValueError, RuntimeError):  # pragma: no cover
            pass
        self._controller.cancel_all()
        self._controller.wait_for_idle(timeout_ms)

    # ------------------------ runner thread ------------------------

    def _on_sequence_started(self, sequence: useq.MDASequence, *_: object) -> None:
        # Read now: the pixel configuration may change after the run (§5.1).
        self._calibration = None
        if request_from(sequence.metadata).enabled:
            self._calibration = capture_linear_part(self._core)

    def _on_sequence_finished(self, sequence: useq.MDASequence) -> None:
        # The runner closes the sink before emitting this, and the next run
        # can't start before this returns: the sink here is this run's.
        if not request_from(sequence.metadata).enabled:
            return
        reason = self._core.mda.status.finish_reason
        try:
            source, shape, stores_events = _closed_store(self._core)
        except Exception:  # pragma: no cover - never break the runner
            logger.exception("Could not read the finished run's data store")
            source, shape, stores_events = None, None, False
        self._runFinished.emit(
            FinishedRun(
                sequence=sequence,
                finish_reason=str(reason) if reason is not None else "completed",
                source=source,
                tile_shape_yx=shape,
                calibration=self._calibration,
                stores_events=stores_events,
            )
        )

    # ------------------------ GUI thread ------------------------

    def _submit(self, run: FinishedRun) -> StitchJob | None:
        """Queue a job for `run`, or say why it is not stitched."""
        if run.source is None:
            return None  # not saved to disk: there is nothing to read
        name = acquisition_stem(run.source)
        if run.finish_reason != "completed":
            why = "cancelled" if run.finish_reason == "canceled" else "failed"
            self._notifications.show_info_message(
                f"{name} was not stitched: the acquisition {why}."
            )
            return None
        if run.stores_events:
            self._notifications.show_warning_message(
                f"{name} was not stitched: its data store does not record which "
                "tile each frame belongs to."
            )
            return None
        if run.calibration is None:
            self._notifications.show_warning_message(
                f"{name} was not stitched: no pixel size was set."
            )
            return None
        pixel_size, linear = run.calibration
        return self._queue(
            run.source,
            run.sequence,
            pixel_size,
            linear,
            run.tile_shape_yx,
            request_from(run.sequence.metadata).settings,
        )

    def submit_saved(
        self,
        source: Path,
        sequence: useq.MDASequence,
        summary: Any,
        settings: StitchSettingsV1,
    ) -> StitchJob | None:
        """Stitch an acquisition saved earlier (plan §7, phase 3a).

        The pixel -> stage mapping comes from its summary metadata, as it
        was when the data were acquired.
        """
        name = acquisition_stem(source)
        if (cal := calibration_from_summary(summary)) is None:
            self._notifications.show_warning_message(
                f"{name} can't be stitched: its metadata has no pixel size."
            )
            return None
        return self._queue(
            source,
            sequence,
            cal.pixel_size_um,
            cal.linear_part,
            cal.tile_shape_yx,
            settings,
        )

    def _queue(
        self,
        source: Path,
        sequence: useq.MDASequence,
        pixel_size: float,
        linear: np.ndarray,
        tile_shape_yx: tuple[int, int] | None,
        settings: StitchSettingsV1,
    ) -> StitchJob | None:
        name = acquisition_stem(source)
        pid, create_time = own_process_identity()
        try:
            job = build_job(
                job_id=self._controller.new_job_id(),
                sequence=sequence,
                source=source,
                pixel_size_um=pixel_size,
                linear_part=linear,
                tile_shape_yx=tile_shape_yx,
                settings=settings,
                parent_pid=pid,
                parent_create_time=create_time,
                memory_cap_bytes=default_memory_cap(),
            )
            if job is None:
                return None  # no grid with more than one tile
            return self._controller.submit(job, title=name)
        except Exception as e:
            logger.exception("Could not queue stitching for %s", source)
            self._notifications.show_warning_message(f"{name} was not stitched: {e}")
            return None

    def _on_job_finished(self, handle: StitchJob) -> None:
        result = handle.result
        if result is None:  # pragma: no cover - FINISHED always has one
            return
        outputs = [g.output for g in result.groups]
        auto_open = handle.job.settings.open_when_done
        if auto_open:
            for path in outputs[:MAX_AUTO_OPEN]:
                self._open(path)

        if len(outputs) == 1:
            message = f"Stitched mosaic saved: {outputs[0].name}"
        else:
            message = f"{len(outputs)} stitched mosaics saved in {outputs[0].parent}"
            if auto_open and len(outputs) > MAX_AUTO_OPEN:
                rest = ", ".join(p.name for p in outputs[MAX_AUTO_OPEN:])
                message += f" (not opened: {rest})"
        notes = [
            f"{g.name}: {s}" if g.name else s
            for g in result.groups
            if (s := g.fallback_summary)
        ]
        if notes:
            message += "\n" + "\n".join(notes)

        actions = (SHOW_IN_FOLDER,) if auto_open else (OPEN, SHOW_IN_FOLDER)

        def on_action(choice: str | None) -> None:
            if choice == OPEN:
                for path in outputs[:MAX_AUTO_OPEN]:
                    self._open(path)
            elif choice == SHOW_IN_FOLDER:
                _reveal(outputs[0].parent)

        self._notifications.show_info_message(message, *actions, on_action=on_action)

    def _on_job_failed(self, handle: StitchJob) -> None:
        reason = handle.error.message if handle.error else "unknown error"
        log = handle.dir.log_file

        def on_action(choice: str | None) -> None:
            if choice == SHOW_LOG:
                _reveal(log if log.exists() else log.parent)

        self._notifications.show_warning_message(
            f"Stitching {handle.title} failed: {reason}", SHOW_LOG, on_action=on_action
        )

    def _open(self, path: Path) -> None:
        try:
            self._open_mosaic(path)
        except Exception as e:  # pragma: no cover - reported, never raised
            logger.exception("Could not open %s", path)
            self._notifications.show_warning_message(f"Could not open {path.name}: {e}")


def _reveal(path: Path) -> None:
    """Open a file or folder with the system's default application."""
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
