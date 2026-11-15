"""Optional grid stitching. Worker modules must remain importable without Qt."""

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
    return is_available() and has_mosaic(sequence)


def has_mosaic(sequence: useq.MDASequence | None) -> bool:
    """Whether a sequence contains a supported grid with multiple tiles."""
    if sequence is None:
        return False
    from pymmcore_gui._stitching._inputs import plan_groups

    try:
        return bool(plan_groups(sequence))
    except (TypeError, ValueError):
        return False
