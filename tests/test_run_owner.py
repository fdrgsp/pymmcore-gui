from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import useq

from pymmcore_gui._run_owner import RunOwner, RunOwnership

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot


def test_unclaimed_runs_belong_to_acquire(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    ownership = RunOwnership(mmcore)
    assert ownership.owner is None
    assert ownership.accepts(RunOwner.ACQUIRE)
    assert not ownership.accepts(RunOwner.SMART)


def test_claim_and_release(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    ownership = RunOwnership(mmcore)
    with qtbot.waitSignal(ownership.ownerChanged) as blocker:
        ownership.claim(RunOwner.SMART)
    assert blocker.args == [RunOwner.SMART]
    assert ownership.accepts(RunOwner.SMART)
    assert not ownership.accepts(RunOwner.ACQUIRE)

    with qtbot.waitSignal(ownership.ownerChanged) as blocker:
        ownership.release()
    assert blocker.args == [None]
    assert ownership.owner is None


def test_claim_refused_while_running(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    ownership = RunOwnership(mmcore)
    with patch.object(type(mmcore.mda), "is_running", return_value=True):
        with pytest.raises(RuntimeError, match="acquisition is running"):
            ownership.claim(RunOwner.SMART)
    assert ownership.owner is None


def test_claim_released_when_run_finishes(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    ownership = RunOwnership(mmcore)
    ownership.claim(RunOwner.SMART)
    mmcore.mda.run(useq.MDASequence())
    qtbot.waitUntil(lambda: ownership.owner is None, timeout=2000)
