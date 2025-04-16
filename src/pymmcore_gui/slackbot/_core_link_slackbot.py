from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pymmcore_plus import CMMCorePlus
from pymmcore_widgets.useq_widgets._mda_sequence import PYMMCW_METADATA_KEY
from qtpy.QtCore import QObject

if TYPE_CHECKING:
    import numpy as np
    import useq
    from pymmcore_plus.metadata import SummaryMetaV1
    from pymmcore_widgets import MDAWidget

    from . import MMSlackBot


RUN = "run"
CANCEL = "cancel"
PROGRESS = "progress"
MDA = "mda"
IMAGE = "image"

INFO_EMOJI = ":information_source:"
WARNING_EMOJI = ":warning:"
CANCEL_EMOJI = ":x:"
RUN_EMOJI = ":rocket:"
FINISHED_EMOJI = ":checkered_flag:"

NOT_RUNNING_MSG = {"icon_emoji": WARNING_EMOJI, "text": "No MDA Sequence running!"}
RUNNING_MSG = {"icon_emoji": WARNING_EMOJI, "text": "MDA Sequence already running!"}


def _progress_message(event: useq.MDAEvent) -> dict[str, Any]:
    """Return the progress message."""
    if event.sequence is None:
        return {"icon_emoji": WARNING_EMOJI, "text": "No MDASequence found!"}
    try:
        sizes = event.sequence.sizes
        pos_name = event.pos_name or f"p{event.index.get('p', 0)}"
        info = (f"{key}{idx + 1}/{sizes[key]}" for key, idx in event.index.items())
        text = f"Status -> `{pos_name} [{', '.join(info)}]`"
        return {"icon_emoji": INFO_EMOJI, "text": text}
    except Exception as e:
        return {"icon_emoji": WARNING_EMOJI, "text": f"Status -> {e}"}


def _mda_sequence_message(event: useq.MDAEvent) -> dict[str, Any]:
    """Return the MDA message."""
    seq = event.sequence
    if seq is None:
        return {"icon_emoji": WARNING_EMOJI, "text": "No MDASequence found!"}

    # TODO: remove "mm_handler" form metadata

    npos = len(seq.stage_positions)

    text = seq.model_dump_json(
        exclude={"stage_positions"} if npos > 3 else {},
        exclude_none=True,
        exclude_unset=True,
        indent=2,
    )

    # hide the stage_positions if there are too many
    if npos > 3:
        # split the string into lines
        lines = text.strip().split("\n")
        # insert the new line before the last line
        pos_text = f'  "stage_positions": {npos} (hiding because too many)'
        new_lines = lines[:-1] + [pos_text] + [lines[-1]]
        # join the lines back into a single string
        text = "\n".join(new_lines)

    return {"icon_emoji": INFO_EMOJI, "text": f"MDA Sequence:\n```{text}```"}


class MMSlackbotCoreLink(QObject):
    def __init__(
        self,
        *,
        mmcore: CMMCorePlus | None = None,
        slackbot: MMSlackBot | None = None,
        mda: MDAWidget | None = None,
    ):
        super().__init__()
        self._mmc = mmcore or CMMCorePlus.instance()

        self._mda = mda

        self._mda_running: bool = False

        # keep track of the current event
        self._current_data: tuple[useq.MDAEvent | None, np.ndarray | None] = (
            None,
            None,
        )

        self._mmc.mda.events.sequenceStarted.connect(self._on_sequence_started)
        self._mmc.mda.events.sequenceFinished.connect(self._on_sequence_finished)
        self._mmc.mda.events.sequenceCanceled.connect(self._on_sequence_canceled)
        self._mmc.mda.events.frameReady.connect(self._on_frame_ready)

        # handle the slackbot
        self._slackbot = slackbot
        if self._slackbot is None:
            return
        self._slackbot.slackMessage.connect(self._on_slack_bot_signal)

    def _on_slack_bot_signal(self, text: str) -> None:
        """Listen for slack bot signals."""
        if self._slackbot is None:
            return

        text = text.lower()
        if text == PROGRESS:
            if not self._mda_running:
                self._slackbot.send_message(NOT_RUNNING_MSG)
                return
            if (ev := self._current_data[0]) is not None:
                self._slackbot.send_message(_progress_message(ev))

        elif text == RUN:
            if self._mda_running:
                self._slackbot.send_message(RUNNING_MSG)
                return
            if self._mda is None:
                return
            self._mda.run_mda()

        elif text == CANCEL:
            if not self._mda_running:
                self._slackbot.send_message(NOT_RUNNING_MSG)
                return
            self._mmc.mda.cancel()

        elif text == MDA:
            if not self._mda_running:
                self._slackbot.send_message(NOT_RUNNING_MSG)
                return
            if (ev := self._current_data[0]) is not None:
                self._slackbot.send_message(_mda_sequence_message(ev))

        elif text == IMAGE:
            if not self._mda_running:
                self._slackbot.send_message(NOT_RUNNING_MSG)
                return
            if (img := self._current_data[1]) is not None:
                self._slackbot.send_message(img)

    def _on_frame_ready(
        self, img: np.ndarray, event: useq.MDAEvent, metadata: dict
    ) -> None:
        """Called when a frame is ready."""
        self._current_data = (event, img)

    def _on_sequence_canceled(self, sequence: useq.MDASequence) -> None:
        """Called when the MDA sequence is cancelled."""
        # slack bot message
        if self._slackbot is not None:
            self._send_message(sequence, CANCEL_EMOJI, "MDA Sequence Cancelled!")

    def _on_sequence_started(
        self, sequence: useq.MDASequence, meta: SummaryMetaV1
    ) -> None:
        """Called when the MDA sequence is started."""
        self._mda_running = True
        self._current_data = (None, None)
        # slack bot message
        if self._slackbot is not None:
            self._send_message(sequence, RUN_EMOJI, "MDA Sequence Started!")

    def _on_sequence_finished(self, sequence: useq.MDASequence) -> None:
        """Called when the MDA sequence is finished."""
        self._mda_running = False
        self._current_data = (None, None)
        # slack bot message
        if self._slackbot is not None:
            self._send_message(sequence, FINISHED_EMOJI, "MDA Sequence Finished!")

    def _send_message(self, sequence: useq.MDASequence, emoji: str, text: str) -> None:
        """Send a message to the slack channel."""
        if self._slackbot is None:
            return

        meta = cast("dict", sequence.metadata.get(PYMMCW_METADATA_KEY, {}))
        file_name = meta.get("save_name", "")
        if file_name:
            file_name = f" (file: `{file_name}`)"
        self._slackbot.send_message({"icon_emoji": emoji, "text": f"{text}{file_name}"})
