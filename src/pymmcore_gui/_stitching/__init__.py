"""Stitch grid acquisitions into a mosaic, in a separate worker process.

Layout (see docs/plans/STITCHING.md):

- `_models`: versioned pydantic models shared by the GUI and the worker.
- `_protocol`: the job directory and its files, the only channel between them.
- `_geometry`: pure pixel -> stage transforms, tile placement and anchoring.
- `_inputs`: which tiles form which mosaic, and reading them from a saved store.
- `_engine`: registration and fusion; the only module importing
  `multiview_stitcher`.
- `_worker`: the worker process's entry point.
- `_controller`: the GUI side -- queue, process, progress, cleanup.
- `_session`: acquisitions in, toasts and viewers out (GUI side).

Everything except `_controller` and `_session` stays importable without Qt:
the worker runs it, and loading Qt there would cost startup time and memory
for nothing.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import useq


def is_available() -> bool:
    """Whether the optional `multiview-stitcher` dependency is installed."""
    return importlib.util.find_spec("multiview_stitcher") is not None


def can_stitch(sequence: useq.MDASequence | None) -> bool:
    """Whether stitching is installed and `sequence` would give a mosaic."""
    if sequence is None or not is_available():
        return False
    from pymmcore_gui._stitching._inputs import plan_groups

    try:
        return bool(plan_groups(sequence))
    except Exception:  # pragma: no cover - a sequence we can't interpret
        return False
