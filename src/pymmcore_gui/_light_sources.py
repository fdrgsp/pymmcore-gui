"""Per-channel settings recorded in a Micro-Manager .cfg.

Each channel preset's exposure, light source and camera properties are stored as a
block of comments at the end of the .cfg, one line per setting::

    #@LightSource,Channel,DAPI,LumencorSola,White_Level,50
           ^ channel group  ^ preset  ^ device   ^ property  ^ default intensity
    #@Exposure,Channel,DAPI,20
    #@CameraProperty,Channel,DAPI,Camera,Gain,2

A light source driving several properties at once has one ``#@LightSource`` line per
property, with the same intensity.

Comments, so that Micro-Manager never sees it: nothing here is applied when the
configuration is loaded, and no new config group shows up in
``getAvailableConfigGroups()``. Storing it as real configuration instead is what
this avoids -- putting the property inside the *channel* preset would break preset
identity, since MMCore compares config settings as strings and
``getCurrentConfig(channel_group)`` then returns ``""`` as soon as the intensity is
edited (or immediately, on a float formatting mismatch).

The cost is that a .cfg written from core state does not carry the block: both
``CMMCorePlus.saveSystemConfiguration()`` and ``pymmcore_plus``'s
``Microscope.save()`` regenerate the file, so every writer has to re-append it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

LIGHT_SOURCE_COMMENT = "#@LightSource"
EXPOSURE_COMMENT = "#@Exposure"
CAMERA_PROPERTY_COMMENT = "#@CameraProperty"
_MARKERS = (LIGHT_SOURCE_COMMENT, EXPOSURE_COMMENT, CAMERA_PROPERTY_COMMENT)

CHANNEL_COMMENT_HEADER = (
    "# ---- pymmcore-gui: per-channel settings ----\n"
    "# Written by the Channels panel. Micro-Manager ignores these lines.\n"
    f"# Format: {LIGHT_SOURCE_COMMENT},<channel group>,<preset>,<device>,"
    "<property>,<intensity>\n"
    f"#         {EXPOSURE_COMMENT},<channel group>,<preset>,<exposure ms>\n"
    f"#         {CAMERA_PROPERTY_COMMENT},<channel group>,<preset>,<camera>,"
    "<property>,<value>\n"
)
# the start of each header line, this one's and the light-source-only one it
# replaced, so rewriting the block drops either
_HEADER_PREFIXES = (
    "# ---- pymmcore-gui:",
    "# Written by the MDA Channels section.",
    "# Written by the Channels panel.",
    "# Format: ",
    "#         #@",
)


@dataclass
class ChannelDeclaration:
    """The settings a .cfg records for one channel preset."""

    exposure: float | None = None
    # [(device, property, intensity), ...]
    light_source: list[tuple[str, str, float]] = field(default_factory=list)
    # [(camera, property, value), ...], the value as written
    camera_properties: list[tuple[str, str, str]] = field(default_factory=list)


# {channel preset: declaration}
ChannelDeclarations: TypeAlias = "dict[str, ChannelDeclaration]"
# {channel preset: [(device, property, intensity), ...]}
Declarations: TypeAlias = "dict[str, list[tuple[str, str, float]]]"


def parse_channel_comments(path: str | Path, channel_group: str) -> ChannelDeclarations:
    """Return the channel settings `path` records for `channel_group`.

    Anything unparsable is skipped rather than raising: this is a hand-editable
    comment block in a file the app does not exclusively own.
    """
    if not path or not channel_group:
        return {}
    try:
        text = Path(path).read_text()
    except OSError:
        return {}

    declarations: ChannelDeclarations = defaultdict(ChannelDeclaration)
    for line in text.splitlines():
        # the header's "Format:" lines mention the markers but do not start with
        # them, so they are not mistaken for declarations
        line = line.strip()
        marker = line.split(",", 1)[0]
        if marker not in _MARKERS:
            continue
        if marker == EXPOSURE_COMMENT:
            parts = line.split(",")
            if len(parts) != 4 or parts[1] != channel_group:
                continue
            try:
                declarations[parts[2]].exposure = float(parts[3])
            except ValueError:
                continue
        elif marker == LIGHT_SOURCE_COMMENT:
            parts = line.split(",")
            if len(parts) != 6 or parts[1] != channel_group:
                continue
            _, _, preset, device, prop, raw_value = parts
            try:
                value = float(raw_value)
            except ValueError:
                continue
            declarations[preset].light_source.append((device, prop, value))
        else:
            # a camera property's value may itself contain commas
            parts = line.split(",", 5)
            if len(parts) != 6 or parts[1] != channel_group:
                continue
            _, _, preset, device, prop, raw_value = parts
            declarations[preset].camera_properties.append((device, prop, raw_value))
    return dict(declarations)


def parse_light_source_comments(path: str | Path, channel_group: str) -> Declarations:
    """Return the light source declarations `path` records for `channel_group`."""
    return {
        preset: declaration.light_source
        for preset, declaration in parse_channel_comments(path, channel_group).items()
        if declaration.light_source
    }


def write_channel_comments(
    path: Path, channel_group: str, declarations: ChannelDeclarations
) -> None:
    """Replace the channel settings comment block at the end of `path`.

    Removing an existing block first means this is idempotent, and that clearing
    every setting leaves the .cfg with no block at all.
    """
    kept = [
        line
        for line in path.read_text().splitlines()
        if line.strip().split(",", 1)[0] not in _MARKERS
        and not line.startswith(_HEADER_PREFIXES)
    ]
    while kept and not kept[-1].strip():
        kept.pop()
    lines = "\n".join(kept)
    block = ""
    for preset, declaration in declarations.items():
        prefix = f"{channel_group},{preset}"
        if declaration.exposure is not None:
            block += f"{EXPOSURE_COMMENT},{prefix},{declaration.exposure}\n"
        for device, prop, value in declaration.light_source:
            block += f"{LIGHT_SOURCE_COMMENT},{prefix},{device},{prop},{value}\n"
        for device, prop, raw in declaration.camera_properties:
            block += f"{CAMERA_PROPERTY_COMMENT},{prefix},{device},{prop},{raw}\n"
    if not block:
        path.write_text(f"{lines}\n")
        return
    path.write_text(f"{lines}\n\n{CHANNEL_COMMENT_HEADER}{block}")
