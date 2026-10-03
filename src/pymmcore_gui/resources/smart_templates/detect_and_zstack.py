"""Detect and zoom: acquire a z-stack wherever something interesting is found.

The base acquisition scans positions (e.g. a well plate or a grid). Every frame
is checked for a "hit" -- here simply a bright spot above a threshold; replace
``is_hit`` with your own detection. For each hit a z-stack is queued *at that
position*, ahead of the remaining scan (``priority="next"``).

Async mode lets the scan keep moving while frames are being analyzed.
"""

import numpy as np
import useq

from pymmcore_gui.smart import AnalysisContext, FrameInfo, Response

API_VERSION = 1
NAME = "Detect and z-stack"
DESCRIPTION = "Acquires a z-stack at every position where a bright spot is found."
SYNC = "async"
# Only look at frames from the scan itself, not at the z-stacks we requested.
ANALYZE = {"origins": ["base"]}

PARAMETERS = {
    "threshold": {
        "default": 3000.0,
        "min": 0.0,
        "max": 65535.0,
        "step": 50.0,
        "label": "Hit threshold (max intensity)",
    },
    "z_range_um": {"default": 10.0, "min": 0.1, "max": 500.0, "label": "Z range (µm)"},
    "z_step_um": {"default": 1.0, "min": 0.05, "max": 50.0, "label": "Z step (µm)"},
    "max_stacks": {"default": 10, "min": 1, "max": 10000},
}


def setup(ctx: AnalysisContext) -> None:
    """Called once before acquisition starts: initialize per-run state."""
    ctx.state["stacks"] = 0


def is_hit(image: np.ndarray, threshold: float) -> bool:
    """Replace with your own detection (segmentation, a classifier...)."""
    return float(image.max()) > threshold


def analyze(
    image: np.ndarray, frame: FrameInfo, ctx: AnalysisContext
) -> Response | None:
    """Queue a z-stack at this frame's position when it contains a hit."""
    hit = is_hit(image, ctx.params["threshold"])
    ctx.record(max=float(image.max()), hit=hit)
    if not hit or ctx.state["stacks"] >= ctx.params["max_stacks"]:
        return None

    ctx.state["stacks"] += 1
    pos = frame.metadata.get("position", {})
    x = frame.event.x_pos if frame.event.x_pos is not None else pos.get("x")
    y = frame.event.y_pos if frame.event.y_pos is not None else pos.get("y")
    z = frame.event.z_pos if frame.event.z_pos is not None else pos.get("z")
    ctx.log(f"Hit on frame {frame.frame_id} at x={x}, y={y}: queuing a z-stack")

    # An event's channel is not the Channel type a sequence takes: rebuild it.
    channel = frame.event.channel
    channels = (
        (
            useq.Channel(
                config=channel.config,
                group=channel.group,
                exposure=frame.event.exposure,
            ),
        )
        if channel
        else ()
    )
    stack = useq.MDASequence(
        stage_positions=(useq.Position(x=x, y=y, z=z),),
        channels=channels,
        z_plan=useq.ZRangeAround(
            range=ctx.params["z_range_um"], step=ctx.params["z_step_um"]
        ),
    )
    return Response(events=stack, priority="next")
