"""The parts of an MDA sequence the loaded configuration can't run.

pymmcore-plus doesn't stop a run for these: it logs a warning and acquires
anyway, so the data would be labelled with a channel, position or Z plane the
microscope never reached.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import useq

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus


def _needs_xy_stage(sequence: useq.MDASequence) -> bool:
    if sequence.grid_plan is not None:
        return True
    positions = sequence.stage_positions
    if isinstance(positions, useq.WellPlatePlan):
        return True
    return any(
        pos.x is not None
        or pos.y is not None
        or (pos.sequence is not None and pos.sequence.grid_plan is not None)
        for pos in positions
    )


def drop_unsupported(
    sequence: useq.MDASequence, core: CMMCorePlus
) -> tuple[useq.MDASequence, list[str]]:
    """`sequence` without what `core`'s configuration can't run, and what that was.

    Drops channels whose group or preset doesn't exist, stage positions and
    grids without an XY stage, the Z plan without a focus stage, and an
    autofocus plan without the device it needs.
    """
    update: dict[str, Any] = {}
    dropped: list[str] = []

    groups = set(core.getAvailableConfigGroups())
    channels = []
    for channel in sequence.channels:
        if channel.group not in groups:
            dropped.append(f"channel {channel.config} (no group '{channel.group}')")
        elif channel.config not in core.getAvailableConfigs(channel.group):
            dropped.append(f"channel {channel.config} (not in '{channel.group}')")
        else:
            channels.append(channel)
    if len(channels) != len(sequence.channels):
        update["channels"] = tuple(channels)

    if not core.getXYStageDevice() and _needs_xy_stage(sequence):
        update.update(stage_positions=(), grid_plan=None)
        dropped.append("stage positions and grid (no XY stage)")

    has_focus = bool(core.getFocusDevice())
    if sequence.z_plan is not None and not has_focus:
        update["z_plan"] = None
        dropped.append("Z plan (no focus stage)")

    plan = sequence.autofocus_plan
    if isinstance(plan, useq.AxesBasedAF) and not core.getAutoFocusDevice():
        update["autofocus_plan"] = None
        dropped.append("hardware autofocus (no autofocus device)")
    elif isinstance(plan, useq.SoftwareAxesBasedAF) and not has_focus:
        update["autofocus_plan"] = None
        dropped.append("software autofocus (no focus stage)")

    return (sequence.replace(**update) if update else sequence), dropped
