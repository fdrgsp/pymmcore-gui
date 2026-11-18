"""Settings, jobs, progress and provenance shared by the GUI and worker."""

from collections.abc import Mapping
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


class StitchError(Exception):
    """A stitching failure whose message can be shown without a traceback."""


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


class OutputFormat(StrEnum):
    SAME = "same"
    """The acquisition's own format."""
    OME_ZARR = "ome-zarr"
    OME_TIFF = "ome-tiff"


ZARR_EXTENSION = ".ome.zarr"
TIFF_EXTENSION = ".ome.tiff"


def is_tiff_output(path: Path) -> bool:
    """Whether a mosaic at `path` is written as OME-TIFF (else OME-Zarr)."""
    return path.name.lower().endswith((".tif", ".tiff"))


class StitchSettingsV1(_Model):
    """What the user can choose in the stitching settings dialog."""

    schema_version: Literal[1] = SCHEMA_VERSION
    alignment: Alignment = Alignment.REGISTER
    reg_channel: str | None = None
    """Channel used for registration. None means the first channel."""
    reg_binning: Literal[1, 2, 4, 8] | None = None
    """Binning applied for registration. None picks it per tile size."""
    z_max_projection: bool = True
    """Register z-stacks on their max projection (XY only)."""
    register_every_timepoint: bool = False
    """Register each timepoint, instead of reusing the first one's result."""
    reject_poor_matches: bool = True
    quality_threshold: float = Field(default=0.2, ge=0, le=1)
    blending: Blending = Blending.SMOOTH
    blend_width_um: float | None = Field(default=None, gt=0)
    """Width of the blended band at tile edges. None uses the default."""
    open_when_done: bool = True
    output_format: OutputFormat = OutputFormat.SAME
    worker_threads: int | None = Field(default=None, ge=1)
    """One thread budget for the worker. None means half the CPU cores."""
    low_priority: bool = True


REGISTRATION_SETTINGS = (
    "alignment",
    "reg_channel",
    "reg_binning",
    "z_max_projection",
    "register_every_timepoint",
    "reject_poor_matches",
    "quality_threshold",
)
"""The settings that change the registration; the rest only change fusion."""


def same_registration(a: StitchSettingsV1, b: StitchSettingsV1) -> bool:
    """Whether `a` and `b` would register the tiles the same way."""
    return all(getattr(a, f) == getattr(b, f) for f in REGISTRATION_SETTINGS)


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
    raw = _stored_request(metadata)
    if not isinstance(raw, Mapping):
        return SequenceStitchingV1()
    try:
        return SequenceStitchingV1.model_validate(raw)
    except ValueError:
        return SequenceStitchingV1()


def stored_request(metadata: dict[str, Any] | None) -> SequenceStitchingV1 | None:
    """The request stored in `metadata`, or None if it has none at all."""
    if _stored_request(metadata) is None:
        return None
    return request_from(metadata)


def _stored_request(metadata: dict[str, Any] | None) -> Any:
    namespace = (metadata or {}).get(METADATA_NAMESPACE)
    return namespace.get(METADATA_KEY) if isinstance(namespace, Mapping) else None


def with_request(
    metadata: dict[str, Any] | None, request: SequenceStitchingV1
) -> dict[str, Any]:
    """Return a copy of `metadata` with `request` stored in it."""
    out = dict(metadata or {})
    existing = out.get(METADATA_NAMESPACE)
    ns = dict(existing) if isinstance(existing, Mapping) else {}
    ns[METADATA_KEY] = request.model_dump(mode="json")
    out[METADATA_NAMESPACE] = ns
    return out


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


class SavedRegistrationV1(_Model):
    """A previous run's registration, for a re-stitch to reuse.

    The worker reuses it only if the source still matches
    `source_fingerprint` when the job runs; otherwise it registers again.
    """

    transforms: dict[str, list[list[list[float]]]]
    """Per timepoint ("all", or the index), each tile's correction matrix."""
    info: dict[str, Any] = Field(default_factory=dict)
    """What that registration found (pairs, rejections, groups)."""
    source_fingerprint: dict[str, Any]


def same_source(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether two source fingerprints describe the same data.

    The path is left out, so data that were moved (not changed) still match.
    """
    return {k: v for k, v in a.items() if k != "path"} == {
        k: v for k, v in b.items() if k != "path"
    }


class GroupV1(_Model):
    """Tiles fused into one mosaic."""

    name: str = ""
    """Short label (the stage position's name), empty for a single mosaic."""
    tiles: list[TileV1] = Field(min_length=1)
    output: Path
    """Final path of the mosaic; the worker writes `<output>.tmp` first.

    Its extension says the format: `.ome.zarr` or `.ome.tiff`."""
    saved_registration: SavedRegistrationV1 | None = None

    @property
    def tmp_output(self) -> Path:
        return tmp_path_for(self.output)

    @property
    def fused_tmp(self) -> Path:
        return fused_tmp_for(self.output)

    @property
    def tmp_paths(self) -> list[Path]:
        return tmp_paths_for(self.output)


def tmp_path_for(output: Path) -> Path:
    """Where a mosaic is written before it is renamed into place."""
    return output.with_name(output.name + ".tmp")


def fused_tmp_for(output: Path) -> Path:
    """The OME-Zarr that fusion writes; for OME-TIFF, converted afterwards."""
    if is_tiff_output(output):
        return output.with_name(output.name + ".zarr.tmp")
    return tmp_path_for(output)


def tmp_paths_for(output: Path) -> list[Path]:
    """Everything the worker may leave behind while writing `output`."""
    return list(dict.fromkeys((tmp_path_for(output), fused_tmp_for(output))))


class StitchJobV1(_Model):
    """Everything the worker needs. Written by the GUI as `job.json`."""

    schema_version: Literal[1] = SCHEMA_VERSION
    job_id: str
    source: Path
    """The saved acquisition (OME-Zarr directory or OME-TIFF file/directory)."""
    groups: list[GroupV1] = Field(min_length=1)
    pixel_size_um: float = Field(gt=0)
    linear_part: tuple[tuple[float, float], tuple[float, float]]
    """`A`: camera pixel (column, row) -> stage (x, y) in µm"""
    z_step_um: float | None = Field(default=None, allow_inf_nan=False)
    """Signed distance between planes, negative for descending stacks."""
    channel_names: list[str] = Field(default_factory=list)
    """The sequence's channels, used when the saved file doesn't name them
    (OME-TIFF drops a single channel's axis)."""
    sequence: dict[str, Any] | None = None
    """The acquisition's `MDASequence`, kept in the mosaic for "Re-use MDA…"."""
    settings: StitchSettingsV1 = Field(default_factory=StitchSettingsV1)
    threads: int = Field(default=1, ge=1)
    """Resolved worker thread budget."""
    memory_cap_bytes: int | None = Field(default=None, gt=0)
    """Memory budget used to limit concurrent fusion blocks."""
    parent_pid: int | None = None
    """The GUI's PID. The worker exits when that process is gone."""
    parent_create_time: float | None = None
    """The GUI's process start time, to detect a reused PID."""

    @property
    def tmp_outputs(self) -> list[Path]:
        return [p for g in self.groups for p in g.tmp_paths]


class Phase(StrEnum):
    STARTING = "starting"
    READING = "reading"
    REGISTERING = "registering"
    FUSING = "fusing"
    PYRAMID = "pyramid"
    CONVERTING = "converting"
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
    Phase.CONVERTING: "Writing OME-TIFF",
    Phase.FINISHING: "Finishing",
}

# Share of a group's progress bar each phase covers, in order.
_PHASE_WEIGHTS = {
    Phase.STARTING: 0.0,
    Phase.READING: 0.02,
    Phase.REGISTERING: 0.38,
    Phase.FUSING: 0.42,
    Phase.PYRAMID: 0.10,
    Phase.CONVERTING: 0.05,
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
    reused_registration: bool = False
    """The registration came from the mosaic being re-stitched."""

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


class ProvenanceV1(_Model):
    """How a mosaic was made: `attrs[METADATA_NAMESPACE][PROVENANCE_KEY]`.

    Read back to re-stitch it.
    """

    schema_version: Literal[1] = SCHEMA_VERSION
    source: Path
    source_fingerprint: dict[str, Any] = Field(default_factory=dict)
    group: str = ""
    tiles: list[TileV1] = Field(min_length=1)
    settings: StitchSettingsV1 = Field(default_factory=StitchSettingsV1)
    pixel_size_um: float = Field(gt=0)
    linear_part: tuple[tuple[float, float], tuple[float, float]]
    z_step_um: float | None = None
    channel_names: list[str] = Field(default_factory=list)
    sequence: dict[str, Any] | None = None
    origin_xy_um: tuple[float, float] | None = None
    """Stage (x, y) of the mosaic's first pixel; rows and columns follow stage y
    and x."""
    transforms: dict[str, list[list[list[float]]]] | None = None
    registration: dict[str, Any] = Field(default_factory=dict)
    groups: list[list[int]] | None = None
    """Tiles aligned together (registration components)."""
    multiview_stitcher_version: str = "unknown"


class ProvenanceError(ValueError):
    """A mosaic's provenance can't be used; the message says why."""


def read_provenance(metadata: Any) -> ProvenanceV1:
    """Read the provenance from a mosaic's `global_metadata(METADATA_NAMESPACE)`."""
    raw = metadata.get(PROVENANCE_KEY) if isinstance(metadata, Mapping) else None
    if not isinstance(raw, dict):
        raise ProvenanceError("it doesn't record how it was stitched")
    version = raw.get("schema_version")
    if isinstance(version, int) and version > SCHEMA_VERSION:
        raise ProvenanceError("it was made by a newer version of pymmcore-gui")
    try:
        return ProvenanceV1.model_validate(raw)
    except ValueError as e:
        raise ProvenanceError(f"its stitching record is unreadable ({e})") from e


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
