"""Versioned models exchanged between the GUI and the stitching worker.

Every model the worker reads or writes carries a `schema_version`, and the
worker refuses a job version it does not know. Only pydantic is imported here,
so both processes can use this module.
"""

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION: Literal[1] = 1
"""Version of every model in this module."""

METADATA_NAMESPACE = "pymmcore_gui"
"""Key of the GUI's own entries in `MDASequence.metadata`."""
METADATA_KEY = "stitching"
"""Key of the stitching request under `METADATA_NAMESPACE`."""
PROVENANCE_KEY = "stitching"
"""A mosaic's root attributes record how it was made under
`attrs[METADATA_NAMESPACE][PROVENANCE_KEY]`, where the OME wrappers'
`global_metadata(METADATA_NAMESPACE)` finds it."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


# ----------------------------- user settings -----------------------------


class Alignment(StrEnum):
    REGISTER = "register"
    """Refine the stage positions with registration (phase correlation)."""
    STAGE = "stage"
    """Place tiles at their stage positions only."""


class Blending(StrEnum):
    SMOOTH = "smooth"
    """Weighted average with smooth weights near the tile edges."""
    MAX = "max"
    AVERAGE = "average"


class StitchSettingsV1(_Model):
    """What the user can choose in the stitching settings dialog."""

    schema_version: Literal[1] = SCHEMA_VERSION
    alignment: Alignment = Alignment.REGISTER
    reg_channel: str | None = None
    """Channel used for registration. None means the first channel."""
    reg_binning: Literal[1, 2, 4, 8] | None = None
    """Binning applied for registration. None picks it per tile size."""
    z_max_projection: bool = True
    """Register z-stacks on their max projection (XY only, see §6.2)."""
    register_every_timepoint: bool = False
    """Register each timepoint, instead of reusing the first one's result."""
    reject_poor_matches: bool = True
    quality_threshold: float = Field(default=0.2, ge=0, le=1)
    blending: Blending = Blending.SMOOTH
    blend_width_um: float | None = Field(default=None, gt=0)
    """Width of the blended band at tile edges. None uses the default."""
    open_when_done: bool = True
    worker_threads: int | None = Field(default=None, ge=1)
    """One thread budget for the worker. None means half the CPU cores."""
    low_priority: bool = True


class SequenceStitchingV1(_Model):
    """What a sequence asks for, stored in its metadata (see `request_from`)."""

    schema_version: Literal[1] = SCHEMA_VERSION
    enabled: bool = False
    settings: StitchSettingsV1 = Field(default_factory=StitchSettingsV1)


def request_from(metadata: dict[str, Any] | None) -> SequenceStitchingV1:
    """Read the stitching request from an `MDASequence.metadata` mapping.

    Anything missing or malformed reads as "not enabled", so an MDA file from
    another version never starts a job by accident.
    """
    raw = ((metadata or {}).get(METADATA_NAMESPACE) or {}).get(METADATA_KEY)
    if not isinstance(raw, dict):
        return SequenceStitchingV1()
    try:
        return SequenceStitchingV1.model_validate(raw)
    except ValueError:
        return SequenceStitchingV1()


def stored_request(metadata: dict[str, Any] | None) -> SequenceStitchingV1 | None:
    """The request stored in `metadata`, or None if it has none at all."""
    if ((metadata or {}).get(METADATA_NAMESPACE) or {}).get(METADATA_KEY) is None:
        return None
    return request_from(metadata)


def with_request(
    metadata: dict[str, Any] | None, request: SequenceStitchingV1
) -> dict[str, Any]:
    """Return a copy of `metadata` with `request` stored in it."""
    out = dict(metadata or {})
    ns = dict(out.get(METADATA_NAMESPACE) or {})
    ns[METADATA_KEY] = request.model_dump(mode="json")
    out[METADATA_NAMESPACE] = ns
    return out


# --------------------------------- the job --------------------------------


class TileV1(_Model):
    """One field of view of a mosaic."""

    position_index: int = Field(ge=0)
    """Index of the tile along the saved store's position axis."""
    name: str = ""
    stage_x_um: float
    """Stage X of the field of view's centre."""
    stage_y_um: float
    """Stage Y of the field of view's centre."""
    stage_z_um: float = 0.0
    """Stage Z of the first plane of a z-stack (0 without a stack)."""


class GroupV1(_Model):
    """Tiles fused into one mosaic."""

    name: str = ""
    """Short label (the stage position's name), empty for a single mosaic."""
    tiles: list[TileV1] = Field(min_length=1)
    output: Path
    """Final path of the mosaic; the worker writes `<output>.tmp` first."""

    @property
    def tmp_output(self) -> Path:
        return tmp_path_for(self.output)


def tmp_path_for(output: Path) -> Path:
    """Where a mosaic is written before it is renamed into place."""
    return output.with_name(output.name + ".tmp")


class StitchJobV1(_Model):
    """Everything the worker needs. Written by the GUI as `job.json`."""

    schema_version: Literal[1] = SCHEMA_VERSION
    job_id: str
    source: Path
    """The saved acquisition (OME-Zarr directory or OME-TIFF file/directory)."""
    groups: list[GroupV1] = Field(min_length=1)
    pixel_size_um: float = Field(gt=0)
    linear_part: tuple[tuple[float, float], tuple[float, float]]
    """`A`: camera pixel (column, row) -> stage (x, y) in µm, see §5.1."""
    z_step_um: float | None = Field(default=None, gt=0)
    """Distance between planes, for z-stacks."""
    channel_names: list[str] = Field(default_factory=list)
    """The sequence's channels, used when the saved file doesn't name them
    (OME-TIFF drops a single channel's axis)."""
    settings: StitchSettingsV1 = Field(default_factory=StitchSettingsV1)
    threads: int = Field(default=1, ge=1)
    """Resolved thread budget (see §4.4)."""
    memory_cap_bytes: int | None = Field(default=None, gt=0)
    """Upper bound on the data in flight during fusion."""
    parent_pid: int | None = None
    """The GUI's PID. The worker exits when that process is gone."""
    parent_create_time: float | None = None
    """The GUI's process start time, to detect a reused PID."""

    @property
    def tmp_outputs(self) -> list[Path]:
        return [g.tmp_output for g in self.groups]


# ---------------------------- worker -> GUI ----------------------------


class Phase(StrEnum):
    STARTING = "starting"
    READING = "reading"
    REGISTERING = "registering"
    FUSING = "fusing"
    PYRAMID = "pyramid"
    FINISHING = "finishing"

    @property
    def label(self) -> str:
        return _PHASE_LABELS[self]


_PHASE_LABELS = {
    Phase.STARTING: "Starting",
    Phase.READING: "Reading tiles",
    Phase.REGISTERING: "Registering",
    Phase.FUSING: "Fusing",
    Phase.PYRAMID: "Writing pyramid",
    Phase.FINISHING: "Finishing",
}

# Share of a group's progress bar each phase covers, in order.
_PHASE_WEIGHTS = {
    Phase.STARTING: 0.0,
    Phase.READING: 0.02,
    Phase.REGISTERING: 0.38,
    Phase.FUSING: 0.45,
    Phase.PYRAMID: 0.12,
    Phase.FINISHING: 0.03,
}


class ProgressEvent(_Model):
    """The worker's latest state, rewritten as `progress.json`."""

    schema_version: Literal[1] = SCHEMA_VERSION
    phase: Phase = Phase.STARTING
    group_index: int = 0
    n_groups: int = 1
    done: int = 0
    total: int = 0
    message: str = ""

    @property
    def fraction(self) -> float:
        """Overall progress of the job in [0, 1], for one progress bar."""
        before = 0.0
        for phase, weight in _PHASE_WEIGHTS.items():
            if phase == self.phase:
                break
            before += weight
        step = self.done / self.total if self.total > 0 else 0.0
        in_group = before + _PHASE_WEIGHTS[self.phase] * min(max(step, 0.0), 1.0)
        return min((self.group_index + in_group) / max(self.n_groups, 1), 1.0)

    @property
    def text(self) -> str:
        """E.g. "Registering 14/24 pairs" -- what the popover shows."""
        if self.message:
            return self.message
        if self.total:
            return f"{self.phase.label} {self.done}/{self.total}"
        return self.phase.label


class GroupResultV1(_Model):
    name: str = ""
    output: Path
    n_tiles: int
    registered: bool
    """False when tiles were placed by stage position only."""
    n_pairs: int = 0
    rejected_pairs: int = 0
    isolated_tiles: int = 0
    """Tiles with no accepted pair, left at their stage position."""
    n_components: int = 1
    """Separately aligned groups of tiles (1 when everything is connected)."""

    @property
    def fallback_summary(self) -> str:
        """What the toast says when registration had to fall back, or ""."""
        if not self.registered:
            return ""
        if self.n_pairs and self.rejected_pairs == self.n_pairs:
            return "registration found no usable overlap; placed by stage position"
        parts = []
        if self.isolated_tiles:
            s = "" if self.isolated_tiles == 1 else "s"
            parts.append(f"{self.isolated_tiles} tile{s} placed by stage position")
        if self.n_components - self.isolated_tiles > 1:
            parts.append(
                f"mosaic aligned in {self.n_components - self.isolated_tiles} groups"
            )
        return ", ".join(parts)


class StitchResultV1(_Model):
    """Written as `result.json` when every group was fused."""

    schema_version: Literal[1] = SCHEMA_VERSION
    groups: list[GroupResultV1]
    elapsed_s: float


class StitchErrorV1(_Model):
    """Written as `error.json` when the job failed."""

    schema_version: Literal[1] = SCHEMA_VERSION
    message: str
    traceback: str = ""


class OwnerV1(_Model):
    """Written by the worker as `owner.json` at startup."""

    schema_version: Literal[1] = SCHEMA_VERSION
    pid: int
    create_time: float
