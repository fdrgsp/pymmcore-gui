"""Grouping, job building, models and the job-dir protocol. No stitching package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import useq
from ome_writers._useq import _build_positions

from pymmcore_gui._stitching import _inputs
from pymmcore_gui._stitching._models import (
    Alignment,
    Blending,
    GroupResultV1,
    Phase,
    ProgressEvent,
    SequenceStitchingV1,
    StitchErrorV1,
    StitchSettingsV1,
    request_from,
    with_request,
)
from pymmcore_gui._stitching._protocol import (
    JobDir,
    atomic_write_text,
    process_alive,
    write_model,
)

FOV = 100.0
GRID = useq.GridRowsColumns(
    rows=2, columns=3, overlap=(10, 10), fov_width=FOV, fov_height=FOV
)


def _seq(**kwargs: Any) -> useq.MDASequence:
    return useq.MDASequence.model_validate(kwargs)


def _plate(points: Any) -> useq.WellPlatePlan:
    """Wells A1 and B3 of a 96-well plate, each with `points`."""
    return useq.WellPlatePlan.model_validate(
        {
            "plate": "96-well",
            "a1_center_xy": (0, 0),
            "selected_wells": ([0, 1], [0, 2]),
            "well_points_plan": points,
        }
    )


COVERAGE = useq.GridFromShape.model_validate(
    {
        "width": 250,
        "height": 250,
        "shape": "ellipse",
        "overlap": 10,
        "fov_width": FOV,
        "fov_height": FOV,
    }
)


SEQUENCES = {
    "global grid": _seq(grid_plan=GRID),
    "grid x 2 positions": _seq(
        stage_positions=[(100, 200, 5), (1000, 2000, 7)], grid_plan=GRID
    ),
    "grid-first order": _seq(
        stage_positions=[(100, 200), (1000, 2000)], grid_plan=GRID, axis_order="tgpcz"
    ),
    "per-position grids": _seq(
        stage_positions=[
            useq.AbsolutePosition(
                x=10, y=20, name="left", sequence=_seq(grid_plan=GRID)
            ),
            useq.AbsolutePosition(x=500, y=600),  # a single field of view
            useq.AbsolutePosition(
                x=-10,
                y=-20,
                sequence=_seq(
                    grid_plan=useq.GridRowsColumns(
                        rows=3, columns=1, fov_width=FOV, fov_height=FOV
                    )
                ),
            ),
        ]
    ),
    "absolute grid": _seq(
        grid_plan=useq.GridFromEdges(
            top=200, left=0, bottom=0, right=300, fov_width=FOV, fov_height=FOV
        )
    ),
    "plate grid": _seq(stage_positions=_plate(GRID)),
    "plate coverage": _seq(stage_positions=_plate(COVERAGE)),
}


@pytest.mark.parametrize("name", list(SEQUENCES))
def test_plan_groups_matches_ome_writers_store_order(name: str) -> None:
    """Each tile's index and centre match the position ome-writers saves there."""
    seq = SEQUENCES[name]
    stored = _build_positions(seq)
    groups = _inputs.plan_groups(seq)
    assert groups
    seen = set()
    for group in groups:
        for tile in group.tiles:
            pos = stored[tile.position_index]
            assert (tile.stage_x_um, tile.stage_y_um) == pytest.approx(
                (pos.x_coord, pos.y_coord)
            )
            seen.add(tile.position_index)
    assert len(seen) == sum(len(g.tiles) for g in groups)


def test_plan_groups_shapes() -> None:
    assert [len(g.tiles) for g in _inputs.plan_groups(SEQUENCES["global grid"])] == [6]
    two = _inputs.plan_groups(SEQUENCES["grid x 2 positions"])
    assert [g.name for g in two] == ["p0", "p1"]
    assert [t.stage_z_um for t in two[1].tiles] == [7.0] * 6
    per = _inputs.plan_groups(SEQUENCES["per-position grids"])
    assert [(g.name, len(g.tiles)) for g in per] == [("left", 6), ("p2", 3)]
    # the single field of view between them takes one store index
    assert per[1].tiles[0].position_index == 7


def test_plate_gives_one_mosaic_per_well() -> None:
    groups = _inputs.plan_groups(SEQUENCES["plate grid"])
    assert [(g.name, len(g.tiles)) for g in groups] == [("A1", 6), ("B3", 6)]
    assert [t.name for t in groups[1].tiles][:2] == ["B3_0", "B3_1"]
    assert groups[1].tiles[0].position_index == 6
    # a coverage area is a grid too, whatever its outline
    coverage = _inputs.plan_groups(SEQUENCES["plate coverage"])
    assert [(g.name, len(g.tiles)) for g in coverage] == [("A1", 9), ("B3", 9)]


@pytest.mark.parametrize(
    "seq",
    [
        _seq(channels=["DAPI"]),
        _seq(stage_positions=[(0, 0), (1, 1)]),
        _seq(grid_plan=useq.RandomPoints(num_points=5, max_width=500, max_height=500)),
        _seq(grid_plan=useq.GridRowsColumns(rows=1, columns=1)),
        _seq(
            stage_positions={
                "plate": "96-well",
                "a1_center_xy": (0, 0),
                "selected_wells": ([0], [0]),
            }
        ),
        _seq(
            stage_positions=_plate(
                useq.RandomPoints(num_points=5, max_width=500, max_height=500)
            )
        ),
    ],
    ids=[
        "no grid",
        "positions only",
        "random points",
        "one tile",
        "well plate",
        "plate random points",
    ],
)
def test_plan_groups_nothing_to_stitch(seq: useq.MDASequence) -> None:
    assert _inputs.plan_groups(seq) == []


def test_z_origin_and_step() -> None:
    seq = _seq(
        stage_positions=[(0, 0, 50)],
        grid_plan=GRID,
        z_plan=useq.ZRangeAround(range=4, step=2),
    )
    tiles = _inputs.plan_groups(seq)[0].tiles
    assert {t.stage_z_um for t in tiles} == {48.0}
    assert _inputs.z_step(seq) == 2
    top_down = _seq(grid_plan=GRID, z_plan=useq.ZTopBottom(top=10, bottom=0, step=2))
    assert _inputs.z_step(top_down) in (-2, 2)


@pytest.mark.parametrize("positions", [[0, 1, 3], [0, 0, 1], [0, 1, 0]])
def test_nonuniform_z_planes_are_rejected(positions: list[float]) -> None:
    sequence = _seq(grid_plan=GRID, z_plan={"relative": positions})
    with pytest.raises(ValueError, match="evenly spaced"):
        _inputs.z_step(sequence)


@pytest.mark.parametrize(
    ("name", "stem"),
    [
        ("expt_001.ome.zarr", "expt_001"),
        ("expt_001.ome.tiff", "expt_001"),
        ("expt_001.OME.TIF", "expt_001"),
        ("expt_001", "expt_001"),
    ],
)
def test_acquisition_stem(name: str, stem: str) -> None:
    from pymmcore_gui._stitching._paths import acquisition_stem

    assert acquisition_stem(Path(name)) == stem


def test_unique_output_never_overwrites(tmp_path: Path) -> None:
    first = _inputs.unique_output(tmp_path, "a")
    assert first.name == "a_stitched.ome.zarr"
    first.mkdir()
    assert _inputs.unique_output(tmp_path, "a").name == "a_stitched-2.ome.zarr"
    # a partial mosaic also takes the name
    (tmp_path / "a_stitched-2.ome.zarr.tmp").mkdir()
    assert _inputs.unique_output(tmp_path, "a").name == "a_stitched-3.ome.zarr"
    assert (
        _inputs.unique_output(tmp_path, "a", "pos 1/x").name
        == "a_stitched_pos_1_x.ome.zarr"
    )


def test_build_job(tmp_path: Path) -> None:
    src = tmp_path / "run.ome.zarr"
    seq = SEQUENCES["grid x 2 positions"]
    job = _inputs.build_job(
        job_id="abc",
        sequence=seq,
        source=src,
        pixel_size_um=0.5,
        linear_part=np.diag([0.5, 0.5]),
        tile_shape_yx=(200, 200),
    )
    assert job is not None
    assert [g.output.name for g in job.groups] == [
        "run_stitched_p0.ome.zarr",
        "run_stitched_p1.ome.zarr",
    ]
    assert job.settings.alignment == Alignment.REGISTER
    assert job.linear_part == ((0.5, 0.0), (0.0, 0.5))
    assert job.source.is_absolute()
    assert job.tmp_outputs[0].name.endswith(".ome.zarr.tmp")
    # round trip through job.json
    assert type(job).model_validate_json(job.model_dump_json()) == job


def test_build_job_without_overlap_uses_stage_positions(tmp_path: Path) -> None:
    grid = useq.GridRowsColumns(
        rows=2, columns=2, overlap=(0, 0), fov_width=FOV, fov_height=FOV
    )
    job = _inputs.build_job(
        job_id="x",
        sequence=_seq(grid_plan=grid),
        source=tmp_path / "a.ome.zarr",
        pixel_size_um=0.5,
        linear_part=np.diag([0.5, 0.5]),
        tile_shape_yx=(200, 200),  # 100 µm: tiles just touch
    )
    assert job is not None
    assert job.settings.alignment == Alignment.STAGE


def test_build_job_nothing_to_stitch(tmp_path: Path) -> None:
    assert (
        _inputs.build_job(
            job_id="x",
            sequence=_seq(channels=["DAPI"]),
            source=tmp_path / "a.ome.zarr",
            pixel_size_um=0.5,
            linear_part=np.eye(2),
        )
        is None
    )


# --------------------------------- models ---------------------------------


def test_request_round_trip_through_sequence_metadata() -> None:
    req = SequenceStitchingV1(
        enabled=True, settings=StitchSettingsV1(reg_binning=4, blending=Blending.MAX)
    )
    seq = _seq(grid_plan=GRID, metadata=with_request({"other": 1}, req))
    reloaded = useq.MDASequence.model_validate_json(seq.model_dump_json())
    assert request_from(reloaded.metadata) == req
    assert reloaded.metadata["other"] == 1


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"pymmcore_gui": "invalid"},
        {"pymmcore_gui": [1, 2]},
        {"pymmcore_gui": 42},
        {"pymmcore_gui": {"stitching": "yes"}},
        {"pymmcore_gui": {"stitching": {"enabled": True, "schema_version": 99}}},
        {"pymmcore_gui": {"stitching": {"enabled": "maybe"}}},
    ],
)
def test_malformed_request_reads_as_disabled(metadata: dict | None) -> None:
    assert not request_from(metadata).enabled


def test_progress_fraction_is_monotonic_across_phases() -> None:
    events = [
        ProgressEvent(phase=Phase.STARTING, n_groups=2),
        ProgressEvent(phase=Phase.REGISTERING, done=3, total=10, n_groups=2),
        ProgressEvent(phase=Phase.REGISTERING, done=10, total=10, n_groups=2),
        ProgressEvent(phase=Phase.FUSING, done=0, total=4, n_groups=2),
        ProgressEvent(phase=Phase.PYRAMID, done=900, total=1000, n_groups=2),
        ProgressEvent(phase=Phase.FINISHING, n_groups=2),
        ProgressEvent(phase=Phase.READING, group_index=1, n_groups=2),
        ProgressEvent(
            phase=Phase.FINISHING, done=1, total=1, group_index=1, n_groups=2
        ),
    ]
    fractions = [e.fraction for e in events]
    assert fractions == sorted(fractions)
    assert fractions[-1] == pytest.approx(1.0)
    assert events[1].text == "Registering 3/10"


def test_fallback_summary() -> None:
    def r(**kw: Any) -> GroupResultV1:
        return GroupResultV1(output=Path("x"), n_tiles=9, registered=True, **kw)

    assert r(n_pairs=12).fallback_summary == ""
    assert (
        r(
            n_pairs=12, rejected_pairs=4, isolated_tiles=3, n_components=5
        ).fallback_summary
        == "3 tiles placed by stage position, mosaic aligned in 2 groups"
    )
    assert "no usable overlap" in r(n_pairs=12, rejected_pairs=12).fallback_summary
    assert (
        GroupResultV1(output=Path("x"), n_tiles=4, registered=False).fallback_summary
        == ""
    )


# -------------------------------- protocol --------------------------------


def test_job_dir_files(tmp_path: Path) -> None:
    jd = JobDir(tmp_path / "j1")
    jd.path.mkdir()
    assert jd.job_id == "j1"
    assert jd.read_progress() is None
    write_model(jd.progress_file, ProgressEvent(phase=Phase.FUSING, done=1, total=2))
    progress = jd.read_progress()
    assert progress is not None
    assert progress.phase == Phase.FUSING
    write_model(jd.error_file, StitchErrorV1(message="boom"))
    error = jd.read_error()
    assert error is not None
    assert error.message == "boom"
    # a corrupt file reads as missing rather than raising
    jd.result_file.write_text("{not json")
    assert jd.read_result() is None
    # no temp files left behind
    assert sorted(p.name for p in jd.path.iterdir()) == [
        "error.json",
        "progress.json",
        "result.json",
    ]


def test_log_tail(tmp_path: Path) -> None:
    jd = JobDir(tmp_path)
    assert jd.log_tail() == ""
    jd.log_file.write_text("\n".join(f"line {i}" for i in range(100)))
    assert jd.log_tail(3) == "line 97\nline 98\nline 99"


def test_atomic_write_replaces(tmp_path: Path) -> None:
    target = tmp_path / "f.json"
    atomic_write_text(target, "1")
    atomic_write_text(target, "2")
    assert target.read_text() == "2"
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_process_alive() -> None:
    import os

    import psutil

    me = psutil.Process()
    assert process_alive(os.getpid(), me.create_time())
    assert not process_alive(os.getpid(), me.create_time() - 1000)  # reused PID
    assert not process_alive(2**22 + 12345, None)


# ------------------------------ output format ------------------------------


def test_output_format_follows_the_acquisition(tmp_path: Path) -> None:
    from pymmcore_gui._stitching._models import OutputFormat

    same = OutputFormat.SAME
    zarr_src = tmp_path / "a.ome.zarr"
    zarr_src.mkdir()
    assert _inputs.output_format_for(zarr_src, same) == OutputFormat.OME_ZARR
    assert _inputs.output_format_for(tmp_path / "a.ome.tif", same) == (
        OutputFormat.OME_TIFF
    )
    # ome-writers' one-file-per-position layout is a directory of OME-TIFFs
    per_position = tmp_path / "run"
    per_position.mkdir()
    (per_position / "run_p000.ome.tiff").touch()
    assert _inputs.output_format_for(per_position, same) == OutputFormat.OME_TIFF
    # an explicit choice wins
    assert _inputs.output_format_for(per_position, OutputFormat.OME_ZARR) == (
        OutputFormat.OME_ZARR
    )


def test_build_job_names_the_output_by_format(tmp_path: Path) -> None:
    from pymmcore_gui._stitching._models import OutputFormat, StitchSettingsV1

    seq = useq.MDASequence(grid_plan=GRID)
    for source, choice, name in [
        ("a.ome.tiff", OutputFormat.SAME, "a_stitched.ome.tiff"),
        ("b.ome.zarr", OutputFormat.SAME, "b_stitched.ome.zarr"),
        ("c.ome.zarr", OutputFormat.OME_TIFF, "c_stitched.ome.tiff"),
        ("d.ome.tiff", OutputFormat.OME_ZARR, "d_stitched.ome.zarr"),
    ]:
        job = _inputs.build_job(
            job_id="j",
            sequence=seq,
            source=tmp_path / source,
            pixel_size_um=1.0,
            linear_part=np.eye(2),
            settings=StitchSettingsV1(output_format=choice),
        )
        assert job is not None
        assert job.groups[0].output.name == name


def test_tiff_output_temporaries(tmp_path: Path) -> None:
    from pymmcore_gui._stitching._models import tmp_paths_for

    tiff = tmp_path / "a_stitched.ome.tiff"
    assert tmp_paths_for(tiff) == [
        tmp_path / "a_stitched.ome.tiff.tmp",
        tmp_path / "a_stitched.ome.tiff.zarr.tmp",
    ]
    zarr = tmp_path / "a_stitched.ome.zarr"
    assert tmp_paths_for(zarr) == [tmp_path / "a_stitched.ome.zarr.tmp"]
    # a leftover temporary of either kind keeps the name from being reused
    (tmp_path / "a_stitched.ome.tiff.zarr.tmp").mkdir()
    out = _inputs.unique_output(tmp_path, "a", extension=".ome.tiff")
    assert out.name == "a_stitched-2.ome.tiff"


# ------------------------------ re-stitching ------------------------------


def _record(**kw: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": "/data/run.ome.zarr",
        "tiles": [{"position_index": 0, "stage_x_um": 0, "stage_y_um": 0}],
        "pixel_size_um": 0.5,
        "linear_part": [[0.5, 0], [0, 0.5]],
        **kw,
    }


def test_read_provenance() -> None:
    from pymmcore_gui._stitching._models import ProvenanceError, read_provenance

    prov = read_provenance({"stitching": _record()})
    assert prov.source == Path("/data/run.ome.zarr")
    assert prov.transforms is None
    assert prov.channel_names == []  # older records lack it
    for bad, why in [
        (None, "doesn't record"),
        ({}, "doesn't record"),
        ({"stitching": _record(schema_version=2)}, "newer version"),
        ({"stitching": _record(tiles=[])}, "unreadable"),
    ]:
        with pytest.raises(ProvenanceError, match=why):
            read_provenance(bad)


def test_same_registration_ignores_fusion_settings() -> None:
    from pymmcore_gui._stitching._models import (
        Blending,
        OutputFormat,
        StitchSettingsV1,
        same_registration,
    )

    base = StitchSettingsV1()
    fusion_only = base.model_copy(
        update={
            "blending": Blending.AVERAGE,
            "blend_width_um": 5.0,
            "output_format": OutputFormat.OME_TIFF,
            "worker_threads": 2,
            "open_when_done": False,
        }
    )
    assert same_registration(base, fusion_only)
    for field, value in [("reg_binning", 2), ("quality_threshold", 0.5)]:
        assert not same_registration(base, base.model_copy(update={field: value}))


def test_same_source_ignores_the_path() -> None:
    from pymmcore_gui._stitching._models import same_source

    a = {"path": "/a/run", "kind": "directory", "size": 10, "mtime_ns": 5}
    assert same_source(a, {**a, "path": "/b/run"})
    assert not same_source(a, {**a, "size": 11})


def test_restitch_job_drops_the_registration_for_stage_only(tmp_path: Path) -> None:
    from pymmcore_gui._stitching._models import (
        Alignment,
        StitchSettingsV1,
        read_provenance,
    )

    prov = read_provenance(
        {"stitching": _record(transforms={"all": [np.eye(3).tolist()]})}
    )
    mosaic = tmp_path / "run_stitched.ome.zarr"
    kw: dict[str, Any] = {
        "job_id": "j",
        "provenance": prov,
        "source": tmp_path / "run.ome.zarr",
        "mosaic": mosaic,
    }
    kept = _inputs.build_restitch_job(settings=StitchSettingsV1(), **kw)
    assert kept.groups[0].saved_registration is not None
    stage = StitchSettingsV1(alignment=Alignment.STAGE)
    dropped = _inputs.build_restitch_job(settings=stage, **kw)
    assert dropped.groups[0].saved_registration is None


# ------------------------------ locating tiles ------------------------------


def _tiles(xy: list[tuple[float, float]]) -> list[Any]:
    from pymmcore_gui._stitching._models import TileV1

    return [
        TileV1(position_index=i, stage_x_um=x, stage_y_um=y)
        for i, (x, y) in enumerate(xy)
    ]


SNAKE = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]


def test_locate_tiles_by_recorded_stage_position() -> None:
    # stored row by row (filename order) while acquired as a snake
    stored = [(0.3, -0.2), (100.1, 0.4), (-0.4, 100.2), (99.8, 99.9)]
    assert _inputs.locate_tiles(_tiles(SNAKE), stored) == [0, 1, 3, 2]


def test_locate_tiles_ignores_other_positions_far_away() -> None:
    stored = [(5000.0, 5000.0), *SNAKE]
    assert _inputs.locate_tiles(_tiles(SNAKE), stored) == [1, 2, 3, 4]


@pytest.mark.parametrize(
    "stored",
    [
        None,
        [],
        [(0.0, 0.0), (100.0, 0.0), None, (0.0, 100.0)],  # a position without one
        [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (60.0, 60.0)],  # too far
        [(0.0, 0.0), (0.0, 0.0), (100.0, 100.0), (0.0, 100.0)],  # ambiguous
    ],
)
def test_locate_tiles_keeps_store_order_when_unsure(stored: Any) -> None:
    assert _inputs.locate_tiles(_tiles(SNAKE), stored) == [0, 1, 2, 3]
