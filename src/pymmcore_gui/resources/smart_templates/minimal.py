"""Minimal Smart Microscopy script: measure every frame, change nothing.

Use it as a starting point. ``analyze`` runs once per acquired frame (or per
frame matching ``ANALYZE``, if given) and returns what to acquire next --
here ``None``, so the base acquisition runs unchanged.
"""

import numpy as np
from pymmcore_plus.smart import AnalysisContext, FrameInfo

API_VERSION = 1
NAME = "Minimal (measure only)"
DESCRIPTION = "Records the mean and max intensity of every frame."


def analyze(image: np.ndarray, frame: FrameInfo, ctx: AnalysisContext) -> None:
    """Called for every frame; returning None leaves the acquisition unchanged."""
    ctx.record(mean=float(image.mean()), max=float(image.max()))
