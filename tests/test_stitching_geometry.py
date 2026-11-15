"""Pixel -> stage transforms and placement (plan §5, §6.1). No stitching package."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pymmcore_plus import CMMCorePlus

from pymmcore_gui._stitching import _geometry as geo


def _rot(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def _mm(linear: np.ndarray) -> tuple[float, ...]:
    return (linear[0, 0], linear[0, 1], 0.0, linear[1, 0], linear[1, 1], 0.0)


# ------------------------------- §5.1 -------------------------------


@pytest.mark.parametrize("mirror_x", [False, True])
@pytest.mark.parametrize("mirror_y", [False, True])
@pytest.mark.parametrize("binning", [1, 2])
def test_uncalibrated_uses_mirror_flags_and_current_pixel_size(
    mirror_x: bool, mirror_y: bool, binning: int
) -> None:
    p = 0.65 * binning
    # MMCore scales the stored identity by binning: the current affine is not
    # the identity at binning 2, yet the system is still uncalibrated.
    current = (binning, 0, 0, 0, binning, 0)
    a = geo.linear_part(
        p,
        current,
        stored_affine=geo.DEFAULT_MM_AFFINE,
        mirror_x=mirror_x,
        mirror_y=mirror_y,
    )
    np.testing.assert_allclose(
        a, np.diag([-p if mirror_x else p, -p if mirror_y else p])
    )


@pytest.mark.parametrize("deg", [0, 90, 30, -120])
@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("flags", [False, True])
def test_calibrated_affine_used_as_is(deg: float, mirrored: bool, flags: bool) -> None:
    p = 0.325
    # a fitted calibration is never exactly diagonal (see the next tests)
    lin = p * _rot(deg + 0.01)
    if mirrored:
        lin = lin @ np.diag([-1, 1])
    affine = _mm(lin)
    for stored in (affine, None):  # fresh run, reopened data
        a = geo.linear_part(
            p, affine, stored_affine=stored, mirror_x=flags, mirror_y=flags
        )
        np.testing.assert_allclose(a, lin, atol=1e-12)


def test_stale_affine_is_treated_as_uncalibrated() -> None:
    p = 0.5
    stale = _mm(1.0 * _rot(30))  # scale 1.0 vs pixel size 0.5
    for stored in (stale, None):
        a = geo.linear_part(p, stale, stored_affine=stored, mirror_x=True)
        np.testing.assert_allclose(a, np.diag([-p, p]))


def test_metadata_path_reads_pure_diagonal_as_uncalibrated() -> None:
    # pixel size 1.0 with the identity: exactly the demo config
    a = geo.linear_part(1.0, geo.DEFAULT_MM_AFFINE, mirror_y=True)
    np.testing.assert_allclose(a, np.diag([1.0, -1.0]))


def test_exactly_diagonal_calibration_differs_only_with_mirror_flags() -> None:
    """The one case the paths can't agree on, and why it doesn't matter.

    Reopened data doesn't record the stored preset affine, so an exactly
    diagonal calibration is read as "no calibration". Without mirror flags
    both readings give `diag(p, p)`; with them, the calibration was made
    before the flags were turned on, which Micro-Manager also calls stale.
    """
    p = 0.5
    calibrated = (p, 0.0, 0.0, 0.0, p, 0.0)
    fresh = geo.linear_part(p, calibrated, stored_affine=calibrated)
    reopened = geo.linear_part(p, calibrated)
    np.testing.assert_allclose(fresh, reopened)
    flipped = geo.linear_part(p, calibrated, mirror_x=True)
    np.testing.assert_allclose(flipped, np.diag([-p, p]))


def test_no_pixel_size_is_an_error() -> None:
    with pytest.raises(ValueError):
        geo.linear_part(0, geo.DEFAULT_MM_AFFINE)


@pytest.mark.parametrize(
    "setup",
    ["plain", "binning2", "mirror", "calibrated90", "calibrated30m", "stale"],
)
def test_fresh_and_metadata_paths_agree_with_stage_explorer(setup: str) -> None:
    """`AffineState` (fresh runs) and the pure rule (reopened data) agree."""
    from pymmcore_gui._stitching._controller import capture_linear_part

    core = CMMCorePlus()
    core.loadSystemConfiguration()
    cam = core.getCameraDevice()
    preset = core.getCurrentPixelSizeConfig()
    core.setPixelSizeUm(preset, 0.5)
    if setup == "binning2":
        core.setProperty(cam, "Binning", "2")
    if setup in ("mirror", "calibrated30m"):
        core.setProperty(cam, "TransposeMirrorX", "1")
    if setup == "calibrated90":
        core.setPixelSizeAffine(preset, _mm(0.5 * _rot(90)))
    if setup == "calibrated30m":
        core.setPixelSizeAffine(preset, _mm(0.5 * _rot(30) @ np.diag([-1, 1])))
    if setup == "stale":
        core.setPixelSizeAffine(preset, _mm(2.0 * _rot(10)))
    core.setPixelSizeConfig(preset)

    captured = capture_linear_part(core)
    assert captured is not None
    p, fresh = captured
    assert p == core.getPixelSizeUm()

    from_metadata = geo.linear_part(
        core.getPixelSizeUm(),
        core.getPixelSizeAffine(),
        mirror_x=core.getProperty(cam, "TransposeMirrorX") == "1",
        mirror_y=core.getProperty(cam, "TransposeMirrorY") == "1",
    )
    np.testing.assert_allclose(fresh, from_metadata, atol=1e-12)


@pytest.mark.parametrize(
    "setup", ["plain", "binning2", "mirror", "calibrated90", "calibrated30m"]
)
def test_saved_summary_metadata_agrees_with_a_fresh_run(setup: str) -> None:
    """Reopened data (phase 3a) is placed as the run itself would have been."""
    from pymmcore_plus.metadata import summary_metadata

    from pymmcore_gui._stitching._controller import capture_linear_part
    from pymmcore_gui._stitching._inputs import calibration_from_summary

    core = CMMCorePlus()
    core.loadSystemConfiguration()
    cam = core.getCameraDevice()
    preset = core.getCurrentPixelSizeConfig()
    core.setPixelSizeUm(preset, 0.5)
    if setup == "binning2":
        core.setProperty(cam, "Binning", "2")
    if setup in ("mirror", "calibrated30m"):
        core.setProperty(cam, "TransposeMirrorX", "1")
    if setup == "calibrated90":
        core.setPixelSizeAffine(preset, _mm(0.5 * _rot(90)))
    if setup == "calibrated30m":
        core.setPixelSizeAffine(preset, _mm(0.5 * _rot(30) @ np.diag([-1, 1])))
    core.setPixelSizeConfig(preset)

    fresh = capture_linear_part(core)
    assert fresh is not None
    # through JSON, as it comes back from a saved store
    import json

    meta = json.loads(json.dumps(summary_metadata(core, cached=False), default=str))
    saved = calibration_from_summary(meta)
    assert saved is not None
    assert saved.pixel_size_um == pytest.approx(fresh[0])
    np.testing.assert_allclose(saved.linear_part, fresh[1], atol=1e-12)
    assert saved.tile_shape_yx == (core.getImageHeight(), core.getImageWidth())


@pytest.mark.parametrize(
    "summary",
    [None, {}, {"image_infos": []}, {"image_infos": [{"width": 4, "height": 4}]}],
)
def test_summary_without_pixel_size_gives_nothing(summary: object) -> None:
    from pymmcore_gui._stitching._inputs import calibration_from_summary

    assert calibration_from_summary(summary) is None


# ------------------------------- §5.2 -------------------------------


def _apply(m: np.ndarray, pt: tuple[float, ...]) -> np.ndarray:
    out: np.ndarray = (m @ np.array([*pt, 1.0]))[:-1]
    return out


@pytest.mark.parametrize("deg", [0, 90, 30])
@pytest.mark.parametrize("mirrored", [False, True])
def test_tile_affine_on_non_square_image(deg: float, mirrored: bool) -> None:
    p = 0.5
    rows, cols = 100, 160
    lin = p * _rot(deg)
    if mirrored:
        lin = lin @ np.diag([-1, 1])
    center = (1000.0, -250.0)
    m = geo.tile_affine(lin, p, (rows, cols), center)
    assert m.shape == (3, 3)

    def stage_of_pixel(row: float, col: float) -> np.ndarray:
        # physical (y, x) of the pixel, then the affine; back to (x, y)
        y, x = _apply(m, (row * p, col * p))
        return np.array([x, y])

    # the image centre lands on the stage position
    np.testing.assert_allclose(
        stage_of_pixel((rows - 1) / 2, (cols - 1) / 2), center, atol=1e-9
    )
    # one pixel step along a row or a column moves by A's columns
    origin = stage_of_pixel(0, 0)
    np.testing.assert_allclose(stage_of_pixel(0, 1) - origin, lin[:, 0], atol=1e-9)
    np.testing.assert_allclose(stage_of_pixel(1, 0) - origin, lin[:, 1], atol=1e-9)


def test_tile_affine_3d_keeps_z_origin() -> None:
    m = geo.tile_affine(np.diag([0.5, 0.5]), 0.5, (10, 10), (0, 0), z_origin_um=12.5)
    assert m.shape == (4, 4)
    np.testing.assert_allclose(_apply(m, (3.0, 0.0, 0.0))[0], 15.5)
    np.testing.assert_allclose(m[0, :3], [1, 0, 0])


def test_lift_2d_to_3d() -> None:
    m2 = np.array([[1, 0, 2.0], [0, 1, -3.0], [0, 0, 1]])
    m3 = geo.lift_2d_to_3d(m2)
    np.testing.assert_allclose(_apply(m3, (7.0, 1.0, 1.0)), [7.0, 3.0, -2.0])


# ------------------------------- §6.1 -------------------------------


def test_anchor_corrections_removes_group_offset() -> None:
    centers = np.array([[0.0, 0.0], [0.0, 100.0], [500.0, 0.0], [500.0, 100.0]])

    def shift(dy: float, dx: float) -> np.ndarray:
        m = np.eye(3)
        m[:2, 2] = (dy, dx)
        return m

    # group A: true relative offset 1 µm, but resolved around tile 0 carrying +5
    # group B: an isolated tile with the identity
    corrections = [shift(5, 5), shift(6, 5), shift(0, 0), shift(0, 0)]
    out = geo.anchor_corrections(corrections, centers, [[0, 1], [2], [3]])
    np.testing.assert_allclose(out[0][:2, 2], [-0.5, 0])
    np.testing.assert_allclose(out[1][:2, 2], [0.5, 0])
    np.testing.assert_allclose(out[2], np.eye(3))
    # the input is not modified
    np.testing.assert_allclose(corrections[0][:2, 2], [5, 5])


def test_anchor_corrections_handles_rotation() -> None:
    centers = np.array([[0.0, 0.0], [0.0, 100.0]])
    rot = np.eye(3)
    rot[:2, :2] = _rot(1)
    out = geo.anchor_corrections([rot, rot], centers, [[0, 1]])
    moved = [_apply(o, tuple(c)) for o, c in zip(out, centers, strict=True)]
    np.testing.assert_allclose(np.mean(moved, axis=0), centers.mean(axis=0))


# ----------------------------- registration -----------------------------


@pytest.mark.parametrize(
    ("shape", "binning"),
    [((512, 512), 1), ((1024, 1024), 1), ((2048, 2048), 2), ((2048, 4096), 4)],
)
def test_auto_binning(shape: tuple[int, int], binning: int) -> None:
    assert geo.auto_binning(shape) == binning


def test_resolution_method_by_tile_count() -> None:
    assert geo.resolution_method(100) == "global_optimization"
    assert geo.resolution_method(101) == "linear_two_pass"


@pytest.mark.parametrize("deg", [0, 90, 30])
def test_overlap_fraction(deg: float) -> None:
    p = 1.0
    lin = p * _rot(deg)
    shape = (100, 200)
    # neighbours 180 px apart along the camera's x: 10 % overlap
    step = lin @ np.array([180.0, 0.0])
    centers = [(0.0, 0.0), (float(step[0]), float(step[1]))]
    assert geo.overlap_fraction(lin, p, shape, centers) == pytest.approx(0.1)
    far = lin @ np.array([250.0, 0.0])
    assert geo.overlap_fraction(lin, p, shape, [(0, 0), tuple(far)]) == 0
