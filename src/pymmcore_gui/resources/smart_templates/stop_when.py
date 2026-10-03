"""Stop when: end a long time-lapse as soon as a condition is met.

The base acquisition is a long time-lapse. Each frame's mean intensity is kept
in ``ctx.state``; once the average change over the last ``window`` frames drops
below ``min_change`` (nothing is happening any more), the run stops early.
"""

import numpy as np

from pymmcore_gui.smart import STOP, AnalysisContext, FrameInfo, Response

API_VERSION = 1
NAME = "Stop when idle"
DESCRIPTION = "Stops a time-lapse once the mean intensity stops changing."

PARAMETERS = {
    "window": {"default": 5, "min": 2, "max": 1000, "label": "Frames to compare"},
    "min_change": {
        "default": 1.0,
        "min": 0.0,
        "max": 65535.0,
        "label": "Minimum mean change",
    },
}


def setup(ctx: AnalysisContext) -> None:
    """Called once before acquisition starts: initialize per-run state."""
    ctx.state["means"] = []


def analyze(
    image: np.ndarray, frame: FrameInfo, ctx: AnalysisContext
) -> Response | None:
    """Track the mean intensity and stop once it no longer changes."""
    means = ctx.state["means"]
    means.append(float(image.mean()))
    window = ctx.params["window"]
    if len(means) < window:
        ctx.record(mean=means[-1])
        return None
    change = float(np.mean(np.abs(np.diff(means[-window:]))))
    ctx.record(mean=means[-1], change=change)
    if change < ctx.params["min_change"]:
        ctx.log(f"Mean changed by only {change:.2f} over {window} frames; stopping.")
        return STOP
    return None
