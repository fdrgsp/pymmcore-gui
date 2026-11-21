"""Filter MDA settings that the loaded hardware configuration cannot run.

Or that the data store cannot hold: OME-TIFF and OME-Zarr data is a regular
array, with every channel at every Z plane and timepoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import useq
from pymmcore_widgets.mda._channel_properties import CHANNEL_PROPERTIES_KEY
from pymmcore_widgets.useq_widgets import PYMMCW_METADATA_KEY

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
    """Return a supported sequence and descriptions of the settings removed."""
    groups = set(core.getAvailableConfigGroups())
    presets: dict[str, set[str]] = {}
    has_xy = bool(core.getXYStageDevice())
    has_focus = bool(core.getFocusDevice())
    has_autofocus = bool(core.getAutoFocusDevice())

    def filter_sequence(seq: useq.MDASequence) -> tuple[useq.MDASequence, list[str]]:
        update: dict[str, Any] = {}
        dropped: list[str] = []
        channels: list[useq.Channel] = []
        channel_indices: dict[int, int] = {}
        for i, channel in enumerate(seq.channels):
            group = channel.group
            if group not in groups:
                dropped.append(f"channel {channel.config} (no group '{group}')")
                continue
            if group not in presets:
                presets[group] = set(core.getAvailableConfigs(group))
            if channel.config not in presets[group]:
                dropped.append(f"channel {channel.config} (not in '{group}')")
                continue
            if seq.z_plan is not None and not channel.do_stack:
                channel = channel.replace(do_stack=True)
                dropped.append(f"channel {channel.config}: Do Stack off")
            if channel.acquire_every != 1:
                dropped.append(
                    f"channel {channel.config}: Acquire Every {channel.acquire_every}"
                )
                channel = channel.replace(acquire_every=1)
            channel_indices[i] = len(channels)
            channels.append(channel)
        if tuple(channels) != seq.channels:
            update["channels"] = tuple(channels)
        if len(channels) != len(seq.channels):
            meta = seq.metadata.get(PYMMCW_METADATA_KEY, {})
            if properties := meta.get(CHANNEL_PROPERTIES_KEY):
                # Saved light-source settings refer to channel row indices.
                properties = [
                    {**entry, "channel_index": channel_indices[entry["channel_index"]]}
                    for entry in properties
                    if entry["channel_index"] in channel_indices
                ]
                update["metadata"] = {
                    **seq.metadata,
                    PYMMCW_METADATA_KEY: {**meta, CHANNEL_PROPERTIES_KEY: properties},
                }

        if not has_xy and _needs_xy_stage(seq):
            update.update(stage_positions=(), grid_plan=None)
            dropped.append("stage positions and grid (no XY stage)")
        elif not isinstance(seq.stage_positions, useq.WellPlatePlan):
            positions = []
            positions_changed = False
            for i, pos in enumerate(seq.stage_positions):
                pos_update: dict[str, Any] = {}
                if not has_focus and pos.z is not None:
                    pos_update["z"] = None
                    dropped.append(f"position {i + 1}: Z coordinate (no focus stage)")
                if pos.sequence is not None:
                    sub_sequence, sub_dropped = filter_sequence(pos.sequence)
                    if sub_sequence is not pos.sequence:
                        pos_update["sequence"] = sub_sequence
                    dropped.extend(f"position {i + 1}: {item}" for item in sub_dropped)
                positions.append(pos.replace(**pos_update) if pos_update else pos)
                positions_changed |= bool(pos_update)
            if positions_changed:
                update["stage_positions"] = tuple(positions)

        if not has_focus:
            for field, label in (
                ("z_plan", "Z plan"),
                ("focus_surface", "focus surface"),
            ):
                if getattr(seq, field) is not None:
                    update[field] = None
                    dropped.append(f"{label} (no focus stage)")

        plan = seq.autofocus_plan
        if isinstance(plan, useq.AxesBasedAF) and not has_autofocus:
            update["autofocus_plan"] = None
            dropped.append("hardware autofocus (no autofocus device)")
        elif isinstance(plan, useq.SoftwareAxesBasedAF) and not has_focus:
            update["autofocus_plan"] = None
            dropped.append("software autofocus (no focus stage)")

        return (seq.replace(**update) if update else seq), dropped

    return filter_sequence(sequence)
