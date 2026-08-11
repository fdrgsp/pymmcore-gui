"""Per-channel light sources recorded in a Micro-Manager .cfg.

Which device property drives each channel preset's light source is stored as a block
of comments at the end of the .cfg, one line per declaration::

    #@LightSource,Channel,DAPI,LumencorSola,White_Level,50
           ^ channel group  ^ preset  ^ device   ^ property  ^ default intensity

Comments, so that Micro-Manager never sees it: neither the intensity nor anything
else here is applied when the configuration is loaded, and no new config group shows
up in ``getAvailableConfigGroups()``. Storing it as real configuration instead is
what this avoids -- putting the property inside the *channel* preset would break
preset identity, since MMCore compares config settings as strings and
``getCurrentConfig(channel_group)`` then returns ``""`` as soon as the intensity is
edited (or immediately, on a float formatting mismatch).

The cost is that a .cfg written from core state does not carry the block: both
``CMMCorePlus.saveSystemConfiguration()`` and ``pymmcore_plus``'s
``Microscope.save()`` regenerate the file, so every writer has to re-append it.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TypeAlias

# {channel preset: [(device, property, intensity), ...]}
Declarations: TypeAlias = "dict[str, list[tuple[str, str, float]]]"

LIGHT_SOURCE_COMMENT = "#@LightSource"
LIGHT_SOURCE_COMMENT_HEADER = (
    "# ---- pymmcore-gui: per-channel light sources ----\n"
    "# Written by the MDA Channels section. Micro-Manager ignores these lines.\n"
    "# Format: <marker>,<channel group>,<preset>,<device>,<property>,<intensity>\n"
)
_HEADER_PREFIXES = tuple(
    line.split(":")[0] for line in LIGHT_SOURCE_COMMENT_HEADER.splitlines()
)


def parse_light_source_comments(path: str | Path, channel_group: str) -> Declarations:
    """Return the light source declarations `path` records for `channel_group`.

    Anything unparsable is skipped rather than raising: this is a hand-editable
    comment block in a file the app does not exclusively own.
    """
    if not path or not channel_group:
        return {}
    try:
        text = Path(path).read_text()
    except OSError:
        return {}

    declarations: Declarations = defaultdict(list)
    for line in text.splitlines():
        # the header's "Format:" line mentions the marker but does not start with
        # it, so it is not mistaken for a declaration
        if not line.strip().startswith(f"{LIGHT_SOURCE_COMMENT},"):
            continue
        parts = line.strip().split(",")
        if len(parts) != 6 or parts[1] != channel_group:
            continue
        _, _, preset, device, prop, raw_value = parts
        try:
            value = float(raw_value)
        except ValueError:
            continue
        declarations[preset].append((device, prop, value))
    return dict(declarations)


def write_light_source_comments(
    path: Path, channel_group: str, declarations: Declarations
) -> None:
    """Replace the light source comment block at the end of `path`.

    Removing an existing block first means this is idempotent, and that clearing
    every light source leaves the .cfg with no block at all.
    """
    kept = [
        line
        for line in path.read_text().splitlines()
        if not line.strip().startswith(f"{LIGHT_SOURCE_COMMENT},")
        and not line.startswith(_HEADER_PREFIXES)
    ]
    while kept and not kept[-1].strip():
        kept.pop()
    lines = "\n".join(kept)
    if not declarations:
        path.write_text(f"{lines}\n")
        return
    block = "".join(
        f"{LIGHT_SOURCE_COMMENT},{channel_group},{preset},{device},{prop},{value}\n"
        for preset, entries in declarations.items()
        for device, prop, value in entries
    )
    path.write_text(f"{lines}\n\n{LIGHT_SOURCE_COMMENT_HEADER}{block}")
