"""Compact MDA progress and runner state for the main-window status bar."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any

from pymmcore_plus.mda import FinishReason, RunState

from pymmcore_gui._qt.QtCore import QEvent, Qt, QTimer, Signal
from pymmcore_gui._qt.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget

from ._theme import qcolor, theme

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pymmcore_plus import CMMCorePlus
    from useq import MDAEvent, MDASequence


_AXIS_ORDER = ("t", "p", "g", "c", "z")


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _sequence_sizes(sequence: MDASequence) -> dict[str, int]:
    """Return trustworthy axis sizes, or no sizes for irregular sequences."""
    if not sequence.axis_order:
        # GeneratorMDASequence warns when its unknown ``sizes`` is requested.
        return {}

    # Per-position subsequences can override channels, z, grid, and time.  Their
    # emitted indices are not described by the root sequence's ``sizes``.
    with suppress(TypeError):
        if any(
            getattr(pos, "sequence", None) is not None
            for pos in sequence.stage_positions
        ):
            return {}

    return {
        str(axis): int(size) for axis, size in sequence.sizes.items() if int(size) > 0
    }


def _shorten(value: object, limit: int = 24) -> str:
    text = str(value)
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _format_event(event: MDAEvent, sizes: Mapping[str, int]) -> str:
    """Format an event's coordinates using one-based, human-facing indices."""
    index = {str(axis): int(value) for axis, value in event.index.items()}
    parts: list[str] = []
    for axis in _AXIS_ORDER:
        if axis not in index:
            continue
        current = index[axis] + 1
        total = sizes.get(axis)
        value = f"{current}/{total}" if total and current <= total else str(current)

        if axis == "p" and event.pos_name:
            value += f" ({_shorten(event.pos_name)})"
        if axis == "c" and event.channel is not None:
            value += f" {_shorten(event.channel.config)}"
        parts.append(f"{axis.upper()} {value}")

    if event.channel is not None and "c" not in index:
        parts.append(f"C {_shorten(event.channel.config)}")
    return " · ".join(parts)


def _event_tooltip(event: MDAEvent) -> str:
    details: list[str] = []
    if event.x_pos is not None:
        details.append(f"X: {event.x_pos:g} µm")
    if event.y_pos is not None:
        details.append(f"Y: {event.y_pos:g} µm")
    if event.z_pos is not None:
        details.append(f"Z: {event.z_pos:g} µm")
    if event.exposure is not None:
        details.append(f"Exposure: {event.exposure:g} ms")
    return "\n".join(details)


class MDAStatusWidget(QWidget):
    """Display the active MDA runner state and its latest useful coordinates."""

    _sequenceStarted = Signal(object)
    _awaitingEvent = Signal(object)
    _eventStarted = Signal(object)
    _frameObserved = Signal(object)
    _pauseToggled = Signal()
    _sequenceCanceled = Signal()
    _sequenceFinished = Signal(object, object)

    def __init__(self, core: CMMCorePlus, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._runner = core.mda
        self._sequence: MDASequence | None = None
        self._sizes: dict[str, int] = {}
        self._last_event: MDAEvent | None = None
        self._current_event: MDAEvent | None = None
        self._next_event: MDAEvent | None = None
        self._cancel_seen = False
        self._result: str | None = None
        self._result_kind = "green"
        self._idle_visible = False
        # Treat construction as if it followed IDLE so a window created while
        # an externally launched acquisition is active still discovers it.
        self._last_phase = RunState.IDLE.value

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(6)
        self._dot = QLabel("●", self)
        self._dot.setObjectName("mda_status_dot")
        self._state_label = QLabel(self)
        self._state_label.setObjectName("mda_status_state")
        self._details_label = QLabel(self)
        self._details_label.setObjectName("mda_status_details")
        self._details_label.setTextFormat(Qt.TextFormat.PlainText)
        self._details_label.setMinimumWidth(0)
        self._details_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        layout.addWidget(self._dot)
        layout.addWidget(self._state_label)
        layout.addWidget(self._details_label)
        layout.addStretch(1)

        self._result_timer = QTimer(self)
        self._result_timer.setSingleShot(True)
        self._result_timer.setInterval(3000)
        self._result_timer.timeout.connect(self._clear_result)

        # Polling is deliberately lightweight and always active: sequenceStarted is
        # emitted only after engine and data-store preparation, so signals alone
        # cannot reveal the PREPARING phase or an externally requested cancellation.
        # It also coalesces rapid event/frame notifications into at most five visual
        # updates per second.
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(200)
        self._status_timer.timeout.connect(self._poll_status)
        self._status_timer.start()

        self._sequenceStarted.connect(self._on_sequence_started)
        self._awaitingEvent.connect(self._on_awaiting_event)
        self._eventStarted.connect(self._on_event_started)
        self._frameObserved.connect(self._on_frame_observed)
        self._pauseToggled.connect(self._render)
        self._sequenceCanceled.connect(self._on_sequence_canceled)
        self._sequenceFinished.connect(self._on_sequence_finished)

        events = self._runner.events
        self._sequence_started_callback = self._relay_sequence_started
        self._awaiting_event_callback = self._relay_awaiting_event
        self._event_started_callback = self._eventStarted.emit
        self._frame_ready_callback = self._relay_frame_ready
        self._pause_toggled_callback = self._relay_pause_toggled
        self._sequence_canceled_callback = self._relay_sequence_canceled
        self._sequence_finished_callback = self._relay_sequence_finished
        events.sequenceStarted.connect(self._sequence_started_callback)
        events.awaitingEvent.connect(self._awaiting_event_callback)
        events.eventStarted.connect(self._event_started_callback)
        events.frameReady.connect(self._frame_ready_callback)
        events.sequencePauseToggled.connect(self._pause_toggled_callback)
        events.sequenceCanceled.connect(self._sequence_canceled_callback)
        events.sequenceFinished.connect(self._sequence_finished_callback)
        self.destroyed.connect(self._disconnect)
        self.hide()
        self._poll_status()

    def _relay_sequence_started(self, sequence: MDASequence, *_: object) -> None:
        self._sequenceStarted.emit(sequence)

    def _relay_awaiting_event(self, event: MDAEvent, _remaining: float) -> None:
        self._awaitingEvent.emit(event)

    def _relay_frame_ready(
        self, _image: object, event: MDAEvent, _meta: object
    ) -> None:
        # Do not send the potentially large image through a second Qt event queue.
        self._frameObserved.emit(event)

    def _relay_pause_toggled(self, _paused: bool) -> None:
        self._pauseToggled.emit()

    def _relay_sequence_canceled(self, *_: object) -> None:
        self._sequenceCanceled.emit()

    def _relay_sequence_finished(self, sequence: MDASequence) -> None:
        # Capture the reason on the runner thread: it returns to IDLE immediately
        # after this callback, before the queued GUI callback normally executes.
        self._sequenceFinished.emit(sequence, self._runner.status.finish_reason)

    def _start_run(self, sequence: MDASequence | None = None) -> None:
        self._result_timer.stop()
        self._result = None
        self._sequence = sequence
        self._sizes = _sequence_sizes(sequence) if sequence is not None else {}
        self._last_event = None
        self._current_event = None
        self._next_event = None
        self._cancel_seen = False
        self.show()

    def set_idle_visible(self, visible: bool) -> None:
        """Show the idle/result indication only on the Acquire page."""
        self._idle_visible = visible
        self._render()

    def _on_sequence_started(self, sequence: MDASequence) -> None:
        self._start_run(sequence)
        # Prevent the next poll from treating this already-announced run as a
        # fresh transition out of IDLE and wiping the sequence-derived sizes.
        self._last_phase = _enum_value(self._runner.status.phase)
        self._render()

    def _on_awaiting_event(self, event: MDAEvent) -> None:
        # Rendering is coalesced by _status_timer; some acquisitions emit events
        # much faster than it is useful to repaint a status bar.
        self._next_event = event

    def _on_event_started(self, event: MDAEvent) -> None:
        self._current_event = event
        self._next_event = None

    def _on_frame_observed(self, event: MDAEvent) -> None:
        self._last_event = event
        self._next_event = None

    def _on_sequence_canceled(self) -> None:
        self._cancel_seen = True
        self._render()

    def _on_sequence_finished(
        self, sequence: MDASequence, finish_reason: FinishReason | None
    ) -> None:
        if self._sequence is not None and sequence is not self._sequence:
            return
        reason = (
            _enum_value(finish_reason) if finish_reason is not None else "completed"
        )
        if reason == FinishReason.CANCELED.value:
            self._result = "Acquisition canceled"
            self._result_kind = "red"
        elif reason == FinishReason.ERRORED.value:
            self._result = "Acquisition failed"
            self._result_kind = "red"
        else:
            self._result = "Acquisition complete"
            self._result_kind = "green"
        self._sequence = None
        self._result_timer.start()
        self._render()

    def _poll_status(self) -> None:
        phase = _enum_value(self._runner.status.phase)
        if phase != RunState.IDLE.value and self._last_phase == RunState.IDLE.value:
            self._start_run()
        self._last_phase = phase
        self._render()

    def _state(self, status: Any) -> tuple[str, str]:
        phase = _enum_value(status.phase)
        finish_reason = (
            _enum_value(status.finish_reason) if status.finish_reason else ""
        )
        if phase == RunState.IDLE.value:
            if self._result:
                return self._result, self._result_kind
            return "MDA idle", "neutral"
        if (
            status.cancel_requested
            or self._cancel_seen
            or (phase == RunState.FINISHING.value and finish_reason == "canceled")
        ):
            return "Cancelling…", "red"
        if status.pause_requested:
            return "Pausing…", "amber"
        if phase == RunState.PREPARING.value:
            return "Preparing…", "green"
        if phase == RunState.PAUSED.value:
            return "Paused", "amber"
        if phase == RunState.FINISHING.value:
            return "Finishing…", "amber"
        if phase == RunState.ACQUIRING.value:
            return "Acquiring", "green"
        if phase == RunState.WAITING.value:
            if self._next_event is not None:
                return "Waiting", "amber"
            return "Running", "green"
        return "", "green"

    def _render(self) -> None:
        status = self._runner.status
        idle = status.phase == RunState.IDLE
        if idle and not self._idle_visible:
            self.hide()
            return
        state, kind = self._state(status)
        if not state:
            self.hide()
            return

        prefix = ""
        event = None
        if idle and self._result is None:
            pass
        elif self._next_event is not None and not idle:
            prefix, event = "Next: ", self._next_event
        elif self._last_event is not None:
            prefix, event = "Last: ", self._last_event
        elif self._current_event is not None:
            prefix, event = "Current: ", self._current_event

        details = _format_event(event, self._sizes) if event is not None else ""
        if self._state_label.text() != state:
            self._state_label.setText(state)
        text = f"{prefix}{details}" if details else ""
        if self._details_label.text() != text:
            self._details_label.setText(text)
        tooltip = _event_tooltip(event) if event is not None else ""
        if self._details_label.toolTip() != tooltip:
            self._details_label.setToolTip(tooltip)
        self._apply_color(kind)
        self.show()

    def _apply_color(self, kind: str) -> None:
        color = {
            "red": theme().status_red,
            "amber": theme().status_amber,
            "green": theme().status_green,
            "neutral": theme().text_secondary,
        }[kind]
        style = f"color: {qcolor(color).name()}"
        if self._dot.styleSheet() != style:
            self._dot.setStyleSheet(style)

    def _clear_result(self) -> None:
        self._result = None
        self._render()

    def changeEvent(self, a0: QEvent | None) -> None:
        if a0 is not None and a0.type() == QEvent.Type.StyleChange:
            self._render()
        super().changeEvent(a0)

    def _disconnect(self) -> None:
        events = self._runner.events
        callbacks = (
            (events.sequenceStarted, self._sequence_started_callback),
            (events.awaitingEvent, self._awaiting_event_callback),
            (events.eventStarted, self._event_started_callback),
            (events.frameReady, self._frame_ready_callback),
            (events.sequencePauseToggled, self._pause_toggled_callback),
            (events.sequenceCanceled, self._sequence_canceled_callback),
            (events.sequenceFinished, self._sequence_finished_callback),
        )
        for signal, callback in callbacks:
            with suppress(Exception):
                signal.disconnect(callback)


__all__ = ["MDAStatusWidget"]
