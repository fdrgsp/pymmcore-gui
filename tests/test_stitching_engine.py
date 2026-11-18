"""Registration and fusion on synthetic acquisitions (plan §5, §6, §10).

Runs in-process; skipped without the optional `multiview-stitcher` package.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pytest.importorskip("multiview_stitcher")

import zarr

from _stitch_synth import grid_sequence, write_grid
from pymmcore_gui._acquisition_loader import open_acquisition
from pymmcore_gui._stitching._engine import run_stitch
from pymmcore_gui._stitching._inputs import build_job
from pymmcore_gui._stitching._models import (
    Alignment,
    OutputFormat,
    ProgressEvent,
    StitchError,
    StitchJobV1,
    StitchSettingsV1,
    tmp_paths_for,
)

if TYPE_CHECKING:
    from pathlib import Path

    import useq

    from _stitch_synth import SynthAcquisition

P = 0.65
S = 128


def _job(
    acq: SynthAcquisition,
    linear: np.ndarray | None = None,
    **settings: Any,
) -> StitchJobV1:
    linear = np.diag([P, P]) if linear is None else linear
    job = build_job(
        job_id="test",
        sequence=acq.sequence,
        source=acq.path,
        pixel_size_um=P,
        linear_part=linear,
        tile_shape_yx=(S, S),
        settings=StitchSettingsV1(**settings),
    )
    assert job is not None
    return job.model_copy(update={"threads": 2})


def _run(job: StitchJobV1) -> tuple[Any, list[ProgressEvent]]:
    events: list[ProgressEvent] = []
    return run_stitch(job, events.append), events


def _attrs(path: Path) -> Any:
    """Root attributes of an NGFF 0.4 (zarr v2) mosaic, as plain JSON."""
    return json.loads((path / ".zattrs").read_text())


def _provenance(path: Path) -> Any:
    """The mosaic's provenance, from its zarr attributes or OME-TIFF annotation."""
    if path.is_dir():
        return _attrs(path)["pymmcore_gui"]["stitching"]
    from pymmcore_gui._ome_tiff_wrapper import OMETiffWrapper

    wrapper = OMETiffWrapper(path)
    try:
        meta = wrapper.global_metadata("pymmcore_gui")
        assert meta is not None
        return meta["stitching"]
    finally:
        wrapper.close()


def _shift_errors(acq: SynthAcquisition, prov: dict, key: str = "all") -> np.ndarray:
    """Per tile, |recovered - true| relative placement error in µm."""
    idx = [t["position_index"] for t in prov["tiles"]]
    # (y, x) translation of a 3x3 (y, x) or 4x4 (z, y, x) matrix
    corr_yx = np.array([np.array(m)[-3:-1, -1] for m in prov["transforms"][key]])
    jitter_yx = acq.jitter_xy[idx][:, ::-1]
    diff = corr_yx - jitter_yx
    return np.abs(diff - diff.mean(axis=0)).max(axis=1)


def _rot(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


@pytest.mark.parametrize("channels", [("A", "B"), ("A",)], ids=["2ch", "1ch"])
@pytest.mark.parametrize("fmt", ["ome-zarr", "ome-tiff"])
def test_registration_recovers_jitter(
    tmp_path: Path, fmt: str, channels: tuple[str, ...]
) -> None:
    seq = grid_sequence(3, 3, S, P, channels=channels)
    acq = write_grid(tmp_path / "run", seq, shape=S, pixel_size=P, fmt=fmt)
    result, events = _run(_job(acq))

    (group,) = result.groups
    assert group.registered
    assert group.n_tiles == 9
    assert group.n_pairs == 12
    assert group.rejected_pairs == group.isolated_tiles == 0
    assert group.n_components == 1
    # the mosaic keeps the acquisition's format
    ext = ".ome.zarr" if fmt == "ome-zarr" else ".ome.tiff"
    assert group.output.name == f"run_stitched{ext}"
    assert not any(p.exists() for p in tmp_paths_for(group.output))

    prov = _provenance(group.output)
    assert _shift_errors(acq, prov).max() < 0.5 * P
    assert prov["source"] == str(acq.path)
    assert prov["source_fingerprint"]["size"] > 0
    assert prov["settings"]["alignment"] == "register"

    # the GUI's reopen path reads the mosaic
    loaded = open_acquisition(group.output)
    try:
        dims = loaded.wrapper.dims
        assert dims[-2:] == ("y", "x")
        if "c" in dims:
            assert list(loaded.wrapper.coords["c"]) == list(channels)
        else:  # OME-TIFF readers drop a single channel's axis
            assert len(channels) == 1
    finally:
        loaded.close()

    fractions = [e.fraction for e in events]
    assert fractions == sorted(fractions)
    assert any(e.text.startswith("Registering") for e in events)


def _level0(path: Path) -> tuple[np.ndarray, list[float], list[float]]:
    ds = _attrs(path)["multiscales"][0]["datasets"][0]
    xforms = {t["type"]: t for t in ds["coordinateTransformations"]}
    scale = xforms["scale"]["scale"]
    origin = xforms.get("translation", {}).get("translation", [0.0] * len(scale))
    return np.asarray(zarr.open_array(str(path / ds["path"]), mode="r")), scale, origin


@pytest.mark.parametrize(
    "linear",
    [
        np.diag([P, P]),
        np.diag([-P, P]),
        P * _rot(90),
        P * _rot(90) @ np.diag([1, -1]),
    ],
    ids=["plain", "mirror-x", "rot90", "rot90-mirror-y"],
)
def test_mosaic_matches_the_sample_whatever_the_camera_orientation(
    tmp_path: Path, linear: np.ndarray
) -> None:
    """Stage-frame mosaics show the sample as it is, however the camera sits."""
    seq = grid_sequence(2, 2, S, P)
    acq = write_grid(
        tmp_path / "run.ome.zarr",
        seq,
        shape=S,
        pixel_size=P,
        linear=linear,
        jitter_um=0,
    )
    result, _ = _run(_job(acq, linear, alignment=Alignment.STAGE))
    (group,) = result.groups
    assert not group.registered

    data, scale, origin = _level0(group.output)
    # output spacing is the pixel size, whatever the transform
    assert scale[-2:] == pytest.approx([P, P])
    # the stage (x, y) of the first pixel, kept for moving the stage over it
    prov = _provenance(group.output)
    assert prov["origin_xy_um"] == pytest.approx([origin[-1], origin[-2]])
    assert prov["sequence"] == seq.model_dump(mode="json")
    mosaic = data[0, 0].astype(float)

    # compare with the world image the tiles were cut from, sampled at the
    # mosaic's pixel positions (tile centres fall between pixels)
    from scipy.ndimage import map_coordinates

    rows, cols = np.mgrid[0 : mosaic.shape[0], 0 : mosaic.shape[1]].astype(float)
    wy = rows + (origin[-2] - acq.world_origin_xy[1]) / P
    wx = cols + (origin[-1] - acq.world_origin_xy[0]) / P
    world = map_coordinates(acq.world, [wy, wx], order=1)
    inner = (slice(5, -5), slice(5, -5))
    corr = np.corrcoef(mosaic[inner].ravel(), world[inner].ravel())[0, 1]
    assert corr > 0.99


def test_rejected_pairs_isolated_tiles_and_anchored_groups(tmp_path: Path) -> None:
    """A blank middle tile splits a column into two groups and an isolated tile."""
    seq = grid_sequence(5, 1, S, P)
    acq = write_grid(
        tmp_path / "run.ome.zarr",
        seq,
        shape=S,
        pixel_size=P,
        blank=frozenset({2}),
    )
    result, _ = _run(_job(acq))
    (group,) = result.groups
    assert group.n_pairs == 4
    assert group.rejected_pairs == 2
    assert group.isolated_tiles == 1
    assert group.n_components == 3
    assert "1 tile placed by stage position" in group.fallback_summary
    assert "2 groups" in group.fallback_summary

    prov = _provenance(group.output)
    groups = sorted(prov["groups"])
    assert groups == [[0, 1], [2], [3, 4]]
    corr = np.array([np.array(m)[:2, -1] for m in prov["transforms"]["all"]])
    jitter = acq.jitter_xy[:, ::-1]
    for members in ([0, 1], [3, 4]):
        # aligned inside the group...
        rel = (corr[members] - jitter[members]) - (corr[members] - jitter[members])[0]
        assert np.abs(rel).max() < 0.5 * P
        # ...and anchored: its mean position is its mean stage position
        np.testing.assert_allclose(corr[members].mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(corr[2], 0, atol=1e-9)


def test_every_pair_rejected_is_stage_positions_only(tmp_path: Path) -> None:
    seq = grid_sequence(2, 2, S, P)
    acq = write_grid(tmp_path / "run.ome.zarr", seq, shape=S, pixel_size=P)
    result, _ = _run(_job(acq, quality_threshold=1.0))
    (group,) = result.groups
    assert group.rejected_pairs == group.n_pairs == 4
    assert "no usable overlap" in group.fallback_summary
    prov = _provenance(group.output)
    for m in prov["transforms"]["all"]:
        np.testing.assert_allclose(m, np.eye(3), atol=1e-9)


def test_z_stack_registers_on_max_projection(tmp_path: Path) -> None:
    seq = grid_sequence(2, 2, S, P, z=3)
    acq = write_grid(tmp_path / "run.ome.zarr", seq, shape=S, pixel_size=P)
    job = _job(acq)
    assert job.z_step_um == 1
    result, _ = _run(job)
    (group,) = result.groups
    prov = _provenance(group.output)
    assert prov["registration"]["max_projection"]
    assert _shift_errors(acq, prov).max() < 0.5 * P
    mats = [np.array(m) for m in prov["transforms"]["all"]]
    for m in mats:
        assert m.shape == (4, 4)
        # XY only: z is untouched
        np.testing.assert_allclose(m[0], [1, 0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(m[:, 0], [1, 0, 0, 0], atol=1e-12)
    data, scale, origin = _level0(group.output)
    assert data.shape[2] == 3  # t, c, z, y, x
    assert scale[-3:] == pytest.approx([1.0, P, P])
    # the first plane sits at each tile's own Z origin (stage z 0, range 2)
    assert origin[-3] == pytest.approx(-1.0)


def test_time_lapse_reuses_first_timepoint(tmp_path: Path) -> None:
    seq = grid_sequence(2, 2, S, P, t=2)
    acq = write_grid(tmp_path / "run.ome.zarr", seq, shape=S, pixel_size=P)
    result, _ = _run(_job(acq))
    prov = _provenance(result.groups[0].output)
    assert list(prov["transforms"]) == ["all"]
    data, *_ = _level0(result.groups[0].output)
    assert data.shape[0] == 2


def test_time_lapse_every_timepoint(tmp_path: Path) -> None:
    seq = grid_sequence(2, 2, S, P, t=2)
    acq = write_grid(tmp_path / "run.ome.zarr", seq, shape=S, pixel_size=P)
    result, _ = _run(_job(acq, register_every_timepoint=True))
    prov = _provenance(result.groups[0].output)
    assert sorted(prov["transforms"]) == ["0", "1"]
    for key in ("0", "1"):
        assert _shift_errors(acq, prov, key).max() < 0.5 * P


def test_one_mosaic_per_position(tmp_path: Path) -> None:

    seq: useq.MDASequence = grid_sequence(
        2, 2, S, P, stage_positions=[(0, 0), (5000, 0)]
    )
    acq = write_grid(tmp_path / "run.ome.zarr", seq, shape=S, pixel_size=P)
    result, events = _run(_job(acq))
    assert [g.output.name for g in result.groups] == [
        "run_stitched_p0.ome.zarr",
        "run_stitched_p1.ome.zarr",
    ]
    assert all(g.output.exists() for g in result.groups)
    assert any(e.text.startswith("Mosaic 2/2") for e in events)


def test_output_name_taken_meanwhile_gets_a_new_version(tmp_path: Path) -> None:
    seq = grid_sequence(2, 2, S, P)
    acq = write_grid(tmp_path / "run.ome.zarr", seq, shape=S, pixel_size=P)
    job = _job(acq, alignment=Alignment.STAGE)
    job.groups[0].output.mkdir()  # appeared after the job was built
    result, _ = _run(job)
    assert result.groups[0].output.name == "run_stitched-2.ome.zarr"


def test_missing_positions_is_a_clear_error(tmp_path: Path) -> None:
    seq = grid_sequence(2, 2, S, P)
    acq = write_grid(tmp_path / "run.ome.zarr", seq, shape=S, pixel_size=P)
    bigger = grid_sequence(3, 3, S, P)
    job = build_job(
        job_id="x",
        sequence=bigger,
        source=acq.path,
        pixel_size_um=P,
        linear_part=np.diag([P, P]),
    )
    assert job is not None
    with pytest.raises(StitchError, match="interrupted"):
        run_stitch(job, lambda e: None)


def test_unreadable_source_is_a_clear_error(tmp_path: Path) -> None:
    seq = grid_sequence(2, 2, S, P)
    job = build_job(
        job_id="x",
        sequence=seq,
        source=tmp_path / "missing.ome.zarr",
        pixel_size_um=P,
        linear_part=np.diag([P, P]),
    )
    assert job is not None
    with pytest.raises(StitchError, match="Could not read"):
        run_stitch(job, lambda e: None)


# ------------------------------ OME-TIFF output ------------------------------


def _tiff_levels(path: Path) -> list[np.ndarray]:
    import tifffile

    with tifffile.TiffFile(path) as tf:
        assert tf.is_ome
        return [np.asarray(level.asarray()) for level in tf.series[0].levels]


def _zarr_levels(path: Path) -> list[np.ndarray]:
    root = zarr.open_group(str(path), mode="r")
    paths = [d["path"] for d in _attrs(path)["multiscales"][0]["datasets"]]
    return [np.asarray(root[p]) for p in paths]


@pytest.mark.parametrize("z", [1, 3])
def test_ome_tiff_mosaic_equals_the_ome_zarr_one(
    tmp_path: Path, z: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pymmcore_gui._ome_tiff_wrapper import OMETiffWrapper
    from pymmcore_gui._stitching import _tiff_output

    # small tiles and bands, so tiling and banding are exercised on a small mosaic
    monkeypatch.setattr(_tiff_output, "TILE", 32)
    monkeypatch.setattr(_tiff_output, "BAND_BYTES", 32 * 300 * 2)
    seq = grid_sequence(2, 3, S, P, channels=("A", "B"), z=z)
    acq = write_grid(tmp_path / "run.ome.zarr", seq, shape=S, pixel_size=P)
    as_zarr, _ = _run(_job(acq, output_format=OutputFormat.OME_ZARR))
    as_tiff, events = _run(_job(acq, output_format=OutputFormat.OME_TIFF))
    zpath, tpath = as_zarr.groups[0].output, as_tiff.groups[0].output
    assert tpath.name == "run_stitched.ome.tiff"
    assert tpath.is_file()
    assert not any(p.exists() for p in tmp_paths_for(tpath))

    # same pixels at every level (tifffile drops singleton axes)
    zarr_levels, tiff_levels = _zarr_levels(zpath), _tiff_levels(tpath)
    assert len(tiff_levels) == len(zarr_levels)
    for zl, tl in zip(zarr_levels, tiff_levels, strict=True):
        np.testing.assert_array_equal(np.squeeze(zl), np.squeeze(tl))

    # the same registration recorded, where the GUI's reader finds it
    zprov, tprov = _provenance(zpath), _provenance(tpath)
    assert tprov["transforms"] == zprov["transforms"]
    assert tprov["tiles"] == zprov["tiles"]
    # OME-TIFF has no translation: only the provenance keeps the origin
    assert tprov["origin_xy_um"] == zprov["origin_xy_um"] is not None

    wrapper = OMETiffWrapper(tpath)
    try:
        assert list(wrapper.coords["c"]) == ["A", "B"]
        assert wrapper.sizes()["y"] == zarr_levels[0].shape[-2]
    finally:
        wrapper.close()

    texts = [e.text for e in events]
    assert any(t.startswith("Writing OME-TIFF") for t in texts)
    fractions = [e.fraction for e in events]
    assert fractions == sorted(fractions)


def test_ome_tiff_records_physical_pixel_size(tmp_path: Path) -> None:
    import tifffile

    seq = grid_sequence(2, 2, S, P)
    acq = write_grid(tmp_path / "run.ome.zarr", seq, shape=S, pixel_size=P)
    result, _ = _run(_job(acq, output_format=OutputFormat.OME_TIFF))
    with tifffile.TiffFile(result.groups[0].output) as tf:
        xml = tf.ome_metadata or ""
    import re

    sizes = dict(re.findall(r'PhysicalSize([XYZ])="([^"]+)"', xml))
    assert set(sizes) == {"X", "Y"}  # no z-stack
    assert float(sizes["X"]) == pytest.approx(P)
    assert float(sizes["Y"]) == pytest.approx(P)
    assert 'PhysicalSizeXUnit="&#181;m"' in xml


def test_tiff_conversion_bounds_reads_for_wide_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pymmcore_gui._stitching import _tiff_output

    monkeypatch.setattr(_tiff_output, "TILE", 16)
    data = np.arange(35 * 1027, dtype="uint16").reshape(35, 1027)
    budget = 16 * 32 * data.dtype.itemsize
    reads = []

    class Array:
        shape = data.shape
        dtype = data.dtype
        chunks = (16, 16)

        def __getitem__(self, key: Any) -> np.ndarray:
            chunk = np.asarray(data[key])
            reads.append(chunk.nbytes)
            return chunk

    tiles = list(_tiff_output._plane_tiles(Array(), ["y", "x"], budget, lambda: None))
    expected = [
        data[y : y + 16, x : x + 16]
        for y in range(0, 35, 16)
        for x in range(0, 1027, 16)
    ]
    assert max(reads) <= budget
    for actual, reference in zip(tiles, expected, strict=True):
        np.testing.assert_array_equal(actual, reference)


def test_tiff_name_taken_meanwhile_gets_a_new_version(tmp_path: Path) -> None:
    seq = grid_sequence(2, 2, S, P)
    acq = write_grid(tmp_path / "run.ome.zarr", seq, shape=S, pixel_size=P)
    job = _job(acq, alignment=Alignment.STAGE, output_format=OutputFormat.OME_TIFF)
    job.groups[0].output.write_bytes(b"")  # appeared after the job was built
    result, _ = _run(job)
    assert result.groups[0].output.name == "run_stitched-2.ome.tiff"


# ------------------------------ re-stitching ------------------------------


def _restitch(
    first: Any, source: Path, **settings: Any
) -> tuple[Any, list[ProgressEvent]]:
    from pymmcore_gui._stitching._inputs import build_restitch_job
    from pymmcore_gui._stitching._models import read_provenance

    mosaic = first.groups[0].output
    prov = read_provenance({"stitching": _provenance(mosaic)})
    job = build_restitch_job(
        job_id="again",
        provenance=prov,
        source=source,
        mosaic=mosaic,
        settings=prov.settings.model_copy(update=settings),
    )
    return _run(job.model_copy(update={"threads": 2}))


@pytest.fixture
def stitched(tmp_path: Path) -> tuple[SynthAcquisition, Any]:
    seq = grid_sequence(2, 3, S, P)
    acq = write_grid(tmp_path / "run.ome.zarr", seq, shape=S, pixel_size=P)
    first, _ = _run(_job(acq))
    return acq, first


def test_restitch_reuses_registration_when_only_fusion_changes(
    stitched: tuple[SynthAcquisition, Any],
) -> None:
    from pymmcore_gui._stitching._models import Blending

    acq, first = stitched
    again, events = _restitch(first, acq.path, blending=Blending.MAX)
    (group,) = again.groups
    assert group.reused_registration
    assert group.n_pairs == first.groups[0].n_pairs
    assert group.output.name == "run_stitched-2.ome.zarr"
    assert any(e.text == "Reusing the earlier registration" for e in events)
    assert not any(e.text.startswith("Registering") for e in events)
    old, new = _provenance(first.groups[0].output), _provenance(group.output)
    assert new["transforms"] == old["transforms"]
    assert new["settings"]["blending"] == "max"
    # the first mosaic is untouched
    assert first.groups[0].output.exists()


def test_restitch_registers_again_when_registration_settings_change(
    stitched: tuple[SynthAcquisition, Any],
) -> None:
    acq, first = stitched
    again, events = _restitch(first, acq.path, quality_threshold=0.3)
    assert not again.groups[0].reused_registration
    assert any(e.text.startswith("Registering") for e in events)


def test_restitch_registers_again_when_the_data_changed(
    stitched: tuple[SynthAcquisition, Any],
) -> None:
    acq, first = stitched
    extra = acq.path / "notes.txt"  # the acquisition's directory changed
    extra.write_text("changed")
    again, _ = _restitch(first, acq.path)
    assert not again.groups[0].reused_registration


def test_restitch_reuses_registration_of_moved_data(
    stitched: tuple[SynthAcquisition, Any], tmp_path: Path
) -> None:
    acq, first = stitched
    moved = tmp_path / "elsewhere" / acq.path.name
    moved.parent.mkdir()
    acq.path.rename(moved)
    again, _ = _restitch(first, moved)
    (group,) = again.groups
    assert group.reused_registration
    # still next to the mosaic it re-stitches, named after the acquisition
    assert group.output.parent == first.groups[0].output.parent
    assert _provenance(group.output)["source"] == str(moved)


def test_restitch_into_the_other_format(
    stitched: tuple[SynthAcquisition, Any],
) -> None:
    acq, first = stitched
    again, _ = _restitch(first, acq.path, output_format=OutputFormat.OME_TIFF)
    (group,) = again.groups
    assert group.output.name == "run_stitched.ome.tiff"
    assert group.reused_registration
    old, new = _provenance(first.groups[0].output), _provenance(group.output)
    assert new["sequence"] == old["sequence"] == acq.sequence.model_dump(mode="json")
    assert new["origin_xy_um"] == pytest.approx(old["origin_xy_um"])


@pytest.mark.parametrize(
    "corruption", ["missing-time", "invalid-time", "nan", "ragged"]
)
def test_invalid_saved_registration_is_recomputed(
    tmp_path: Path, corruption: str
) -> None:
    from pymmcore_gui._stitching._inputs import build_restitch_job
    from pymmcore_gui._stitching._models import read_provenance

    seq = grid_sequence(2, 2, S, P, t=2)
    acq = write_grid(tmp_path / "run.ome.zarr", seq, shape=S, pixel_size=P)
    first, _ = _run(_job(acq, register_every_timepoint=True))
    mosaic = first.groups[0].output
    record = _provenance(mosaic)
    transforms = record["transforms"]
    if corruption == "missing-time":
        transforms.pop("1")
    elif corruption == "invalid-time":
        transforms["bad"] = transforms.pop("1")
    elif corruption == "nan":
        transforms["0"][0][0][0] = float("nan")
    else:
        transforms["0"][0][0].pop()
    prov = read_provenance({"stitching": record})
    job = build_restitch_job(
        job_id="again",
        provenance=prov,
        source=acq.path,
        mosaic=mosaic,
        settings=prov.settings,
    )
    again, _ = _run(job.model_copy(update={"threads": 2}))
    assert not again.groups[0].reused_registration
    assert _shift_errors(acq, _provenance(again.groups[0].output), key="0").max() < P


def test_descending_z_stack_keeps_its_physical_orientation(tmp_path: Path) -> None:
    seq = grid_sequence(2, 2, S, P, z=3)
    assert seq.z_plan is not None
    descending = seq.model_copy(
        update={"z_plan": seq.z_plan.model_copy(update={"go_up": False})}
    )
    up = write_grid(tmp_path / "up.ome.zarr", seq, shape=S, pixel_size=P)
    down = write_grid(tmp_path / "down.ome.zarr", descending, shape=S, pixel_size=P)
    up_result, _ = _run(_job(up, alignment=Alignment.STAGE))
    down_result, _ = _run(_job(down, alignment=Alignment.STAGE))
    up_data = np.squeeze(_zarr_levels(up_result.groups[0].output)[0])
    down_data = np.squeeze(_zarr_levels(down_result.groups[0].output)[0])
    np.testing.assert_array_equal(down_data, up_data[::-1])


def _workers(tmp_path: Path, overlap: float, linear: np.ndarray, cap_gb: float) -> int:
    from types import SimpleNamespace

    from pymmcore_gui._stitching._engine import _blocks_in_flight

    job = build_job(
        job_id="test",
        sequence=grid_sequence(10, 10, 2048, 1, overlap=overlap),
        source=tmp_path / "raw.ome.zarr",
        pixel_size_um=1,
        linear_part=linear,
        settings=StitchSettingsV1(worker_threads=8),
        memory_cap_bytes=int(cap_gb * 1024**3),
    )
    assert job is not None
    images: Any = SimpleNamespace(
        linear=linear,
        group=job.groups[0],
        opened=SimpleNamespace(dtype=np.dtype("uint16"), shape_yx=(2048, 2048)),
        has_z=False,
        arrays=[None] * 100,
    )
    return _blocks_in_flight(job, images)


def test_fusion_memory_budget_accounts_for_dense_overlap(tmp_path: Path) -> None:
    sparse = _workers(tmp_path, 10, np.eye(2), cap_gb=2)
    dense = _workers(tmp_path, 90, np.eye(2), cap_gb=2)
    assert sparse > dense == 1


@pytest.mark.parametrize("deg", [0.2, 2, 90])
def test_fusion_memory_budget_ignores_camera_rotation(
    tmp_path: Path, deg: float
) -> None:
    """A calibrated camera is always slightly rotated; that changes nothing."""
    # 10 % overlap: a 2048² block touches 3x3 tiles, ~0.5 GB a block
    assert _workers(tmp_path, 10, _rot(deg), cap_gb=16) == 8
    assert _workers(tmp_path, 10, _rot(deg), cap_gb=2) == _workers(
        tmp_path, 10, np.eye(2), cap_gb=2
    )
