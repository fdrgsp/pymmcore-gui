from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import useq

from pymmcore_gui.smart import STOP, AnalysisContext, Response
from pymmcore_gui.smart._api import normalise_response

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def test_none_means_no_action() -> None:
    response = normalise_response(None)
    assert response == Response(events=())


def test_single_event_and_iterables_become_tuples() -> None:
    event = useq.MDAEvent(exposure=5)
    assert normalise_response(event).events == (event,)
    assert normalise_response([event, event]).events == (event, event)
    assert normalise_response(e for e in [event]).events == (event,)


def test_sequence_is_expanded() -> None:
    seq = useq.MDASequence(z_plan=useq.ZRangeAround(range=2, step=1))
    response = normalise_response(seq)
    assert isinstance(response.events, tuple)
    assert len(response.events) == 3
    assert all(isinstance(e, useq.MDAEvent) for e in response.events)


def test_response_options_are_kept() -> None:
    event = useq.MDAEvent()
    out = normalise_response(
        Response(events=[event], priority="end", timing="absolute", drop_base=True)
    )
    assert out.events == (event,)
    assert (out.priority, out.timing, out.drop_base) == ("end", "absolute", True)
    assert normalise_response(STOP).stop


@pytest.mark.parametrize("bad", [42, "events", {"a": 1}, [useq.MDAEvent(), 3]])
def test_bad_return_values(bad: object) -> None:
    with pytest.raises(TypeError):
        normalise_response(bad)


def test_event_cap_stops_endless_generators() -> None:
    def endless() -> Iterator[useq.MDAEvent]:
        while True:
            yield useq.MDAEvent()

    with pytest.raises(ValueError, match="more than 10"):
        normalise_response(endless(), max_events=10)


def test_invalid_response_options() -> None:
    with pytest.raises(ValueError):
        Response(priority="soon")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Response(timing="later")  # type: ignore[arg-type]


def test_context_log_and_record(tmp_path: Path) -> None:
    ctx = AnalysisContext({"a": 1}, tmp_path, "thread")
    with pytest.raises(TypeError):
        ctx.params["a"] = 2  # type: ignore[index]
    ctx.log("hello")
    ctx.record(mean=np.float32(1.5), count=np.int64(3), label="x", hit=True)
    with pytest.raises(TypeError):
        ctx.record(arr=np.zeros(3))
    with pytest.raises(ValueError):
        ctx.log("x", level="loud")  # type: ignore[arg-type]
    logs, records = ctx._drain()
    assert logs == [("info", "hello")]
    assert records == {"mean": 1.5, "count": 3, "label": "x", "hit": True}
    assert type(records["count"]) is int
    assert ctx._drain() == ([], {})
