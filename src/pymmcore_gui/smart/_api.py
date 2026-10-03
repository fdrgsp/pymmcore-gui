"""Types that Smart Microscopy analysis scripts code against (API version 1).

Kept free of Qt and of the rest of the GUI: scripts import this module inside
a spawned analysis process, where only the stdlib, numpy and useq are wanted.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, TypedDict, get_args

import useq

API_VERSION: Final = 1
"""Version of this API. A script declares the version it was written for."""

ExecutionMode = Literal["thread", "process"]
SyncMode = Literal["blocking", "async"]
Origin = Literal["base", "analysis"]
LogLevel = Literal["debug", "info", "warning", "error"]
Priority = Literal["next", "end"]
Timing = Literal["relative", "absolute"]

EXECUTION_MODES: Final[tuple[str, ...]] = get_args(ExecutionMode)
SYNC_MODES: Final[tuple[str, ...]] = get_args(SyncMode)
ORIGINS: Final[tuple[str, ...]] = get_args(Origin)
LOG_LEVELS: Final[tuple[str, ...]] = get_args(LogLevel)

RecordValue = float | int | str | bool | None
"""Types `AnalysisContext.record` accepts (numpy scalars are converted)."""


@dataclass(frozen=True, slots=True)
class FrameInfo:
    """Everything known about the frame being analyzed.

    Attributes
    ----------
    frame_id : int
        0-based acquisition order within this run. Also the frame's index along
        the ``t`` axis of the saved data (for a single camera).
    event : useq.MDAEvent
        The event that produced this frame.
    metadata : Mapping[str, Any]
        The frame's metadata as recorded at acquisition time (pymmcore-plus
        ``FrameMetaV1``): ``pixel_size_um``, ``position`` (x/y/z),
        ``exposure_ms``, ``camera_device``, ``runner_time_ms``,
        ``property_values``...
    origin : "base" | "analysis"
        Whether the event came from the base acquisition or was returned by a
        previous analysis.
    parent_frame_id : int | None
        For an analysis-origin frame, the frame whose analysis requested it.
    """

    frame_id: int
    event: useq.MDAEvent
    metadata: Mapping[str, Any]
    origin: Origin = "base"
    parent_frame_id: int | None = None


class AnalysisContext:
    """Per-run services handed to ``setup``, ``analyze`` and ``teardown``.

    Attributes
    ----------
    params : Mapping[str, Any]
        Resolved values of the script's ``PARAMETERS`` (read-only).
    state : dict[str, Any]
        Free-form storage that persists across calls within one run.
    run_dir : Path
        Folder for the script's own outputs (masks, tables...). Always exists.
    execution : "thread" | "process"
        Where the script is running.
    """

    __slots__ = ("_logs", "_records", "execution", "params", "run_dir", "state")

    def __init__(
        self,
        params: Mapping[str, Any],
        run_dir: str | Path,
        execution: ExecutionMode,
    ) -> None:
        self.params: Mapping[str, Any] = MappingProxyType(dict(params))
        self.state: dict[str, Any] = {}
        self.run_dir = Path(run_dir)
        self.execution: ExecutionMode = execution
        self._logs: list[tuple[str, str]] = []
        self._records: dict[str, RecordValue] = {}

    def log(self, message: object, level: LogLevel = "info") -> None:
        """Show *message* in the GUI's run monitor and keep it in the run log."""
        if level not in LOG_LEVELS:
            raise ValueError(f"level must be one of {LOG_LEVELS}, not {level!r}")
        self._logs.append((level, str(message)))

    def record(self, **values: Any) -> None:
        """Attach named scalar results to the current frame.

        They appear in the run monitor and in ``analysis.jsonl``. Values must
        be numbers, strings, booleans or None (numpy scalars are converted).
        """
        for key, value in values.items():
            if getattr(value, "shape", None) == () and callable(
                item := getattr(value, "item", None)
            ):
                value = item()  # numpy scalar (0-d) -> Python scalar
            if not isinstance(value, (float, int, str, bool, type(None))):
                raise TypeError(
                    f"record({key}=...) needs a number, string, bool or None, "
                    f"not {type(value).__name__}"
                )
            self._records[key] = value

    def _drain(self) -> tuple[list[tuple[str, str]], dict[str, RecordValue]]:
        """Return and clear what the last call logged and recorded."""
        logs, records = self._logs, self._records
        self._logs, self._records = [], {}
        return logs, records


@dataclass(frozen=True, slots=True)
class Response:
    """What to do after a frame was analyzed.

    Attributes
    ----------
    events : Sequence[useq.MDAEvent] | useq.MDASequence
        Events to acquire. An `MDASequence` is expanded into its events.
    priority : "next" | "end"
        Acquire them before any remaining base events ("next"), or after them.
    timing : "relative" | "absolute"
        "relative" treats each event's ``min_start_time`` as seconds from now
        (so a returned time-lapse starts its own clock); "absolute" uses the
        values as given, relative to the start of the run.
    stop : bool
        Finish the run after the event currently running; nothing else that is
        queued is acquired.
    drop_base : bool
        Discard the remaining base events (continue purely reactively).
    """

    events: Sequence[useq.MDAEvent] | useq.MDASequence = ()
    priority: Priority = "next"
    timing: Timing = "relative"
    stop: bool = False
    drop_base: bool = False

    def __post_init__(self) -> None:
        if self.priority not in get_args(Priority):
            raise ValueError(f"priority must be 'next' or 'end', not {self.priority!r}")
        if self.timing not in get_args(Timing):
            raise ValueError(
                f"timing must be 'relative' or 'absolute', not {self.timing!r}"
            )


STOP: Final = Response(stop=True)
"""Return this from ``analyze`` to finish the run."""


class ParamSpec(TypedDict, total=False):
    """Documents the dict form of a ``PARAMETERS`` entry (type-hint aid only)."""

    default: Any
    min: float
    max: float
    step: float
    choices: list[Any]
    label: str
    tooltip: str


def normalise_response(value: object, *, max_events: int = 1000) -> Response:
    """Turn whatever ``analyze`` returned into a `Response` with expanded events.

    Accepts None, an `MDAEvent`, an `MDASequence`, any iterable of `MDAEvent`,
    or a `Response`. The returned `Response.events` is always a tuple of
    `MDAEvent`. Raises ``TypeError`` for anything else and ``ValueError`` when
    more than *max_events* events would be produced.
    """
    if value is None:
        return Response()
    if isinstance(value, Response):
        response = value
    elif isinstance(value, useq.MDASequence):
        response = Response(events=value)
    elif isinstance(value, useq.MDAEvent):
        response = Response(events=(value,))
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        response = Response(events=value)  # type: ignore[arg-type]
    else:
        raise TypeError(
            "analyze() must return None, an MDAEvent, an MDASequence, an iterable "
            f"of MDAEvent, or a Response; got {type(value).__name__}"
        )
    events = _expand(response.events, max_events)
    return Response(
        events=events,
        priority=response.priority,
        timing=response.timing,
        stop=response.stop,
        drop_base=response.drop_base,
    )


def _expand(events: object, max_events: int) -> tuple[useq.MDAEvent, ...]:
    if isinstance(events, useq.MDAEvent):
        events = (events,)
    if not isinstance(events, Iterable):
        raise TypeError(f"Response.events must be iterable, not {type(events)}")
    # islice: an endless generator must hit the cap rather than hang.
    expanded = tuple(islice(events, max_events + 1))
    if len(expanded) > max_events:
        raise ValueError(
            f"analyze() requested more than {max_events} events in one response"
        )
    for item in expanded:
        if not isinstance(item, useq.MDAEvent):
            raise TypeError(
                f"expected MDAEvent items, got {type(item).__name__}: {item!r}"
            )
    return expanded
