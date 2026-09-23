from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
from pymmcore_plus.mda import FinishReason, RunState
from useq import Channel, MDASequence, Position, TIntervalLoops, ZRangeAround

from pymmcore_gui._modern_gui._mda_status import (
    MDAStatusWidget,
    _format_event,
    _sequence_sizes,
)

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot


def test_mda_status_formats_regular_and_irregular_coordinates() -> None:
    sequence = MDASequence(
        time_plan=TIntervalLoops(interval=timedelta(seconds=1), loops=2),
        stage_positions=(
            Position(x=0, y=0, name="A1"),
            Position(x=10, y=10, name="A2"),
        ),
        channels=(
            Channel(config="DAPI", exposure=10),
            Channel(config="FITC", exposure=10),
        ),
        z_plan=ZRangeAround(range=2, step=1),
    )
    event = next(iter(sequence))

    assert _format_event(event, _sequence_sizes(sequence)) == (
        "T 1/2 · P 1/2 (A1) · C 1/2 DAPI · Z 1/3"
    )

    irregular = MDASequence(
        stage_positions=(
            Position(x=0, y=0),
            Position(
                x=10,
                y=10,
                sequence=MDASequence(
                    channels=(
                        Channel(config="DAPI", exposure=10),
                        Channel(config="FITC", exposure=10),
                    ),
                    z_plan=ZRangeAround(range=2, step=1),
                ),
            ),
        )
    )
    assert _sequence_sizes(irregular) == {}


def test_mda_status_tracks_runner_transitions_and_coordinate_meaning(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    widget = MDAStatusWidget(mmcore)
    qtbot.addWidget(widget)
    widget.set_idle_visible(True)
    runner = mmcore.mda
    sequence = MDASequence(
        time_plan=TIntervalLoops(interval=timedelta(seconds=1), loops=2),
        channels=(Channel(config="DAPI", exposure=10),),
        z_plan=ZRangeAround(range=1, step=1),
    )
    event = next(iter(sequence))

    runner._state = RunState.PREPARING
    widget._poll_status()
    assert widget._state_label.text() == "Preparing…"
    assert widget.isVisible()

    runner._state = RunState.WAITING
    widget._on_sequence_started(sequence)
    runner.events.awaitingEvent.emit(event, 10.0)
    qtbot.waitUntil(lambda: widget._next_event is event)
    widget._render()
    assert widget._state_label.text() == "Waiting"
    assert widget._details_label.text().startswith("Next: T 1/2")

    runner._state = RunState.ACQUIRING
    runner.events.eventStarted.emit(event)
    qtbot.waitUntil(lambda: widget._current_event is event)
    widget._render()
    assert widget._state_label.text() == "Acquiring"
    assert widget._details_label.text().startswith("Current: T 1/2")

    runner.events.frameReady.emit(np.zeros((1, 1)), event, {})
    qtbot.waitUntil(lambda: widget._last_event is event)
    widget._render()
    assert widget._details_label.text().startswith("Last: T 1/2")
    assert "image" not in widget._details_label.text().lower()

    runner._pause_requested = True
    widget._render()
    assert widget._state_label.text() == "Pausing…"

    runner._pause_requested = False
    runner._state = RunState.PAUSED
    widget._render()
    assert widget._state_label.text() == "Paused"

    runner._state = RunState.FINISHING
    runner._finish_reason = FinishReason.CANCELED
    widget._render()
    assert widget._state_label.text() == "Cancelling…"

    runner._state = RunState.IDLE
    runner._finish_reason = FinishReason.COMPLETED
    widget._on_sequence_finished(sequence, FinishReason.COMPLETED)
    assert widget._state_label.text() == "Acquisition complete"
    widget._clear_result()
    assert widget._state_label.text() == "MDA idle"
    assert widget._details_label.text() == ""


def test_mda_status_preserves_a_transient_status_bar_message(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    from pymmcore_gui._qt.QtWidgets import QMainWindow

    window = QMainWindow()
    qtbot.addWidget(window)
    status = window.statusBar()
    assert status is not None
    widget = MDAStatusWidget(mmcore, status)
    status.addPermanentWidget(widget)
    status.showMessage("A useful transient message", 5000)

    mmcore.mda._state = RunState.PREPARING
    widget._poll_status()

    assert status.currentMessage() == "A useful transient message"
    assert not widget.isHidden()


def test_main_window_left_status_and_idle_visibility(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    from pymmcore_gui._modern_gui._main_win import MainWindow

    window = MainWindow(mmcore=mmcore)
    qtbot.addWidget(window)
    window.resize(1200, 700)
    window.show()
    status_bar = window.statusBar()
    assert status_bar is not None
    window._stack.setCurrentWidget(window._acquire)
    widget = window._mda_status
    assert not widget.isHidden()
    assert widget._state_label.text() == "MDA idle"
    assert status_bar.currentMessage() == ""

    # Switching the actual stack also covers programmatic navigation, including
    # calibration and externally started runs that bypass the tab-click handler.
    window._stack.setCurrentWidget(window._installation)
    assert widget.isHidden()
    window._stack.setCurrentWidget(window._acquire)
    assert not widget.isHidden()

    mmcore.mda._state = RunState.PREPARING
    widget._poll_status()
    status_bar.showMessage("Saved configuration", 50)
    qtbot.waitUntil(lambda: widget.width() > 0)
    assert window._status_message.text() == "Saved configuration"
    assert widget.isVisible()
    assert widget._state_label.text() == "Preparing…"
    assert widget.mapTo(status_bar, widget.rect().topLeft()).x() < 20
    assert window._bell_button.x() > status_bar.width() / 2
    qtbot.waitUntil(lambda: window._status_message.text() == "")

    # A canceled result must settle back to idle, even after further timer ticks.
    mmcore.mda._state = RunState.IDLE
    widget._cancel_seen = True
    widget._on_sequence_finished(MDASequence(), FinishReason.CANCELED)
    widget._clear_result()
    widget._poll_status()
    assert widget._state_label.text() == "MDA idle"
    assert widget._details_label.text() == ""
