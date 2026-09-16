"""Tests for the pixel-size calibration routine ported from Java Micro-Manager."""

from __future__ import annotations

from threading import Event
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pymmcore_gui._pixel_calibration import (
    CalibrationCancelled,
    CalibrationCommitError,
    CalibrationOptions,
    PixelCalibrationError,
    affine_to_measurements,
    commit_pixel_calibration,
    cross_correlate,
    deduce_pixel_size,
    fit_affine_with_translation,
    measure_displacement,
    normalize_for_mmcore,
    run_pixel_calibration,
)
from pymmcore_gui._pixel_calibration._routine import (
    _resample_matrix,
    _sub_image,
    _Tracker,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _texture(shape: tuple[int, int] = (256, 320)) -> NDArray[np.float64]:
    rng = np.random.default_rng(12)
    image = rng.normal(size=shape)
    # Repeated local averaging makes a deterministic, nonperiodic microscopy-like field.
    for _ in range(4):
        image = (
            image
            + np.roll(image, 1, 0)
            + np.roll(image, -1, 0)
            + np.roll(image, 1, 1)
            + np.roll(image, -1, 1)
        ) / 5
    return image


def _shift_image(
    image: NDArray[np.float64], shift_rc: tuple[float, float]
) -> NDArray[np.float64]:
    row_frequency = np.fft.fftfreq(image.shape[0])[:, None]
    col_frequency = np.fft.fftfreq(image.shape[1])[None, :]
    phase = np.exp(
        -2j * np.pi * (row_frequency * shift_rc[0] + col_frequency * shift_rc[1])
    )
    return np.fft.ifftn(np.fft.fftn(image) * phase).real


def _fast_options(**kwargs: object) -> CalibrationOptions:
    defaults: dict[str, object] = {"settle_time_s": 0.0}
    defaults.update(kwargs)
    return CalibrationOptions(**defaults)  # type: ignore[arg-type]


def test_default_motion_settings_are_the_java_ones() -> None:
    options = CalibrationOptions()
    assert options.initial_step_um == 0.1
    assert options.settle_time_s == 0.1
    assert options.max_search_steps == 25
    assert options.box_size == 64
    assert options.upsample_factor == 10
    assert options.max_rms_px == 5.0
    # Java's dialog offers 1000 um as its smallest safe travel radius, and the
    # corner geometry needs far more room than a 100 um default would allow.
    assert options.safe_radius_um == 1000


@pytest.mark.parametrize(
    "field",
    ["initial_step_um", "max_search_steps", "box_size", "upsample_factor"],
)
def test_options_reject_nonsense_values(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        CalibrationOptions(**{field: 0})


# --- numerics ----------------------------------------------------------------


def test_resample_matrix_is_center_aligned_and_normalized() -> None:
    matrix = _resample_matrix(64, 10)
    assert matrix.shape == (640, 64)
    # Every destination row is a partition of unity, so a flat patch stays flat.
    assert matrix.sum(axis=1) == pytest.approx(np.ones(640))
    # The destination centre must map exactly onto the source centre, which is
    # what makes a zero displacement read back as exactly zero.
    center = np.zeros(64)
    center[32] = 1.0
    assert float(matrix[320] @ center) == pytest.approx(1.0)


def test_cross_correlate_peaks_at_center_for_identical_images() -> None:
    patch = _texture((64, 64))
    correlation = cross_correlate(patch, patch)
    row, column = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    assert (int(row), int(column)) == (32, 32)


def test_measure_displacement_returns_zero_for_identical_patches() -> None:
    patch = _texture((64, 64))
    assert measure_displacement(patch, patch) == (0.0, 0.0)


@pytest.mark.parametrize(
    "content_shift_rc", [(3.0, -5.0), (-6.0, 2.0), (0.0, 4.0), (1.2, 2.5)]
)
def test_measure_displacement_sign_and_axis_order(
    content_shift_rc: tuple[float, float],
) -> None:
    """The result is the negative of the apparent motion, in geometric x/y."""
    reference = _texture((128, 128))
    moved = _shift_image(reference, content_shift_rc)
    x, y = measure_displacement(reference, moved)
    # content_shift_rc is (row, column) == (y, x)
    assert x == pytest.approx(-content_shift_rc[1], abs=0.15)
    assert y == pytest.approx(-content_shift_rc[0], abs=0.15)


def test_measure_displacement_survives_a_patch_smaller_than_the_box() -> None:
    """A 32 px patch must not be beaten by bicubic overshoot at the zero pad.

    Java crops a fixed 64 px correlation box, which is only safe while the
    patch is at least that big. Below it the box is zero-padded, and since a
    min-subtracted correlation sits on a large positive pedestal the bicubic
    kernel overshoots at the padding edge.
    """
    reference = _texture((32, 32))
    moved = _shift_image(reference, (0.0, -2.0))
    x, y = measure_displacement(reference - reference.min(), moved - moved.min())
    assert x == pytest.approx(2.0, abs=0.2)
    assert y == pytest.approx(0.0, abs=0.2)


def test_sub_image_zero_pads_instead_of_clipping() -> None:
    source = np.ones((10, 10))
    out = _sub_image(source, -3, -2, 6, 6)
    assert out.shape == (6, 6)
    # The requested region starts outside the frame, so those rows/columns are
    # zero rather than being shifted back inside the image.
    assert out[:2].sum() == 0
    assert out[:, :3].sum() == 0
    assert out[2:, 3:] == pytest.approx(np.ones((4, 3)))


def test_fit_affine_with_translation_recovers_a_known_transform() -> None:
    truth = np.asarray([[0.25, -0.03, 12.0], [0.04, 0.26, -7.0]])
    pixels = np.asarray([[0.0, 0.0], [100.0, 0.0], [0.0, 90.0], [70.0, -60.0]])
    stage = np.column_stack([pixels, np.ones(len(pixels))]) @ truth.T
    assert fit_affine_with_translation(pixels, stage) == pytest.approx(truth)


def test_fit_affine_with_translation_rejects_collinear_points() -> None:
    pixels = np.asarray([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    stage = pixels * 0.5
    with pytest.raises(PixelCalibrationError, match="degenerate"):
        fit_affine_with_translation(pixels, stage)


def test_normalize_for_mmcore_accounts_for_binning_and_magnification() -> None:
    matrix = np.asarray([[0.8, 0.0], [0.0, 0.8]])

    raw, size, flat = normalize_for_mmcore(matrix, binning=2, magnification=1.0)

    assert size == pytest.approx(0.4)
    assert raw == pytest.approx(matrix / 2)
    assert flat == pytest.approx((0.4, 0.0, 0.0, 0.0, 0.4, 0.0))

    raw, size, _flat = normalize_for_mmcore(matrix, binning=1, magnification=2.0)
    assert size == pytest.approx(1.6)
    assert raw == pytest.approx(matrix * 2)


# --- the numbers the Java dialog reports -------------------------------------


def test_affine_to_measurements_matches_the_java_dialog() -> None:
    """Reproduce a real 60x result the Java Pixel Calibrator reported.

    XScale=0.1085 YScale=-0.1085 Rotation=-0.71 Shear=-0.0005, built up as
    Java decomposes it: transform = rotation @ scale @ shear.
    """
    x_scale, y_scale, rotation_deg, shear = 0.1085, -0.1085, -0.71, -0.0005
    angle = np.deg2rad(rotation_deg)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    matrix = rotation @ np.diag([x_scale, y_scale]) @ np.asarray([[1, shear], [0, 1]])

    measured = affine_to_measurements(matrix)

    # Java takes each scale as a column norm, and the second column also
    # carries the shear, so a scale is recovered only to a relative
    # 1 + shear**2 / 2. Reproducing that is part of the port.
    assert measured[0] == pytest.approx(x_scale, rel=1e-6)
    assert measured[1] == pytest.approx(y_scale, rel=1e-6)
    assert measured[2] == pytest.approx(rotation_deg, abs=1e-6)
    assert measured[3] == pytest.approx(shear, rel=1e-6)
    # Rounded to the 4 decimals the dialog prints, it is exactly what it showed.
    assert round(measured[0], 4) == 0.1085
    assert round(measured[1], 4) == -0.1085
    assert round(measured[2], 2) == -0.71
    assert round(measured[3], 4) == -0.0005
    # The scalar pixel size the dialog quotes is sqrt(|det|), rounded to 4 dp.
    assert deduce_pixel_size(matrix) == pytest.approx(0.1085)


def test_deduce_pixel_size_rounds_to_four_decimals() -> None:
    assert deduce_pixel_size(np.diag([0.10833333, 0.10833333])) == 0.1083


# --- synthetic hardware ------------------------------------------------------


class _EmptyConfig:
    def size(self) -> int:
        return 0


class _IdleMDA:
    def is_running(self) -> bool:
        return False


class _SyntheticCore:
    """Small MMCore-compatible camera/stage model for end-to-end tests.

    The field is 384 x 448 so the routine's patch is 64 px: it sizes the patch
    as the largest power of two at most a quarter of each axis, and a smaller
    patch carries too little structure for a single unwindowed correlation.
    Real cameras sit far above this (512 px on a 2048 px sensor).
    """

    def __init__(self) -> None:
        angle = np.deg2rad(19)
        self.true_matrix = 0.4 * np.asarray(
            [
                [np.cos(angle), np.sin(angle)],
                [np.sin(angle), -np.cos(angle)],
            ]
        )
        self.base_image = _texture((384, 448))
        self.origin = np.asarray((120.0, -30.0))
        self.position = self.origin.copy()
        self.moves: list[tuple[float, float]] = []
        self.snap_count = 0
        self.fail_snap_at: int | None = None
        self.fail_affine_once = False
        self.mda = _IdleMDA()
        self.stored_size = float(np.sqrt(abs(np.linalg.det(self.true_matrix))))
        self.stored_affine: tuple[float, ...] = (
            float(self.true_matrix[0, 0]),
            float(self.true_matrix[0, 1]),
            0.0,
            float(self.true_matrix[1, 0]),
            float(self.true_matrix[1, 1]),
            0.0,
        )

    def getCameraDevice(self) -> str:
        return "Camera"

    def getXYStageDevice(self) -> str:
        return "XY"

    def getBinning(self, label: str) -> int:
        assert label == "Camera"
        return 1

    def getMagnificationFactor(self) -> float:
        return 1.0

    def getROI(self, label: str) -> tuple[int, int, int, int]:
        assert label == "Camera"
        height, width = self.base_image.shape
        return (0, 0, width, height)

    def getImageWidth(self) -> int:
        return int(self.base_image.shape[1])

    def getImageHeight(self) -> int:
        return int(self.base_image.shape[0])

    def getNumberOfCameraChannels(self) -> int:
        return 1

    def getCurrentPixelSizeConfig(self) -> str:
        return "Resolution"

    def getPixelSizeConfigData(self, config_name: str) -> _EmptyConfig:
        assert config_name == "Resolution"
        return _EmptyConfig()

    def getProperty(self, device: str, prop: str) -> str:
        raise AssertionError(f"Unexpected property query: {device}-{prop}")

    def getXYPosition(self, label: str) -> tuple[float, float]:
        assert label == "XY"
        return (float(self.position[0]), float(self.position[1]))

    def setXYPosition(self, label: str, x: float, y: float) -> None:
        assert label == "XY"
        self.position[:] = (x, y)
        self.moves.append((x, y))

    def waitForDevice(self, label: str) -> None:
        assert label == "XY"

    def isSequenceRunning(self, label: str) -> bool:
        assert label == "Camera"
        return False

    def snapImage(self) -> None:
        self.snap_count += 1
        if self.snap_count == self.fail_snap_at:
            raise RuntimeError("synthetic camera failure")

    def getImage(self) -> np.ndarray:
        stage_delta = self.position - self.origin
        image_shift = np.linalg.solve(self.true_matrix, stage_delta)
        apparent_shift_rc = (-image_shift[1], -image_shift[0])
        return _shift_image(self.base_image, apparent_shift_rc)

    def getPixelSizeUm(self) -> float:
        return self.stored_size

    def getAvailablePixelSizeConfigs(self) -> tuple[str, ...]:
        return ("Resolution",)

    def getPixelSizeUmByID(self, resolution_id: str) -> float:
        assert resolution_id == "Resolution"
        return self.stored_size

    def getPixelSizeAffineByID(self, resolution_id: str) -> tuple[float, ...]:
        assert resolution_id == "Resolution"
        return self.stored_affine

    def setPixelSizeUm(self, resolution_id: str, value: float) -> None:
        assert resolution_id == "Resolution"
        self.stored_size = value

    def setPixelSizeAffine(self, resolution_id: str, value: tuple[float, ...]) -> None:
        assert resolution_id == "Resolution"
        if self.fail_affine_once:
            self.fail_affine_once = False
            raise RuntimeError("synthetic persistence failure")
        self.stored_affine = value

    def getPixelSizeAffine(self) -> tuple[float, ...]:
        return self.stored_affine


# --- end to end --------------------------------------------------------------


def test_calibration_recovers_the_synthetic_matrix() -> None:
    core = _SyntheticCore()
    stored_size, stored_affine = core.stored_size, core.stored_affine
    observations: list[tuple[object, str]] = []

    result = run_pixel_calibration(
        core,
        _fast_options(),
        resolution_id="Resolution",
        observation_callback=lambda observation, kind: observations.append(
            (observation, kind)
        ),
    )

    assert result.fit.matrix == pytest.approx(core.true_matrix, abs=5e-3)
    assert result.fit.determinant < 0  # the synthetic field is mirrored
    assert result.algorithm_version == "v1"
    # Four corners are fitted, and there is no holdout stage at all.
    assert len(result.observations) == 4
    assert [obs.label for obs in result.observations] == ["1", "2", "3", "4"]
    assert all(obs.residual_px is not None for obs in result.observations)
    # Two probe measurements plus the four corners are reported during acquisition.
    assert len(observations) == 6
    assert [kind for _observation, kind in observations] == [
        "probe",
        "probe",
        "corner",
        "corner",
        "corner",
        "corner",
    ]
    # Nothing is persisted by the routine itself.
    assert core.stored_size == stored_size
    assert core.stored_affine == stored_affine


def test_calibration_pixel_size_matches_the_true_field() -> None:
    core = _SyntheticCore()
    expected = float(np.sqrt(abs(np.linalg.det(core.true_matrix))))

    result = run_pixel_calibration(core, _fast_options(), resolution_id="Resolution")

    assert result.fit.pixel_size_um == pytest.approx(expected, rel=2e-2)
    # Binning is 1 and the magnification factor is 1, so storage units match.
    assert result.raw_pixel_size_um == pytest.approx(result.fit.pixel_size_um)


def test_calibration_returns_the_stage_and_reports_it() -> None:
    core = _SyntheticCore()

    result = run_pixel_calibration(core, _fast_options(), resolution_id="Resolution")

    assert core.position == pytest.approx(core.origin)
    assert result.stage_returned
    excursion = max(
        float(np.linalg.norm(np.asarray(move) - core.origin)) for move in core.moves
    )
    assert excursion <= CalibrationOptions().safe_radius_um


def test_calibration_accepts_a_clean_field_well_inside_tolerance() -> None:
    """A noiseless field must land far below Java's 5 px acceptance limit."""
    core = _SyntheticCore()

    result = run_pixel_calibration(core, _fast_options(), resolution_id="Resolution")

    assert result.fit.rms_residual_px < 0.5
    assert result.fit.max_residual_px < 1.0


def test_calibration_rejects_scatter_above_the_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One badly mismeasured corner must fail the 5 px RMS scatter check.

    The four corners form a symmetric rectangle, so the fit's residual space
    has a single degree of freedom per axis: an error on one corner shows up
    spread across all four, and roughly half of it survives as scatter.
    """
    core = _SyntheticCore()
    calls = {"n": 0}
    original = _Tracker.measure

    def measure_with_a_bad_corner(
        self: _Tracker, target: np.ndarray, expected_shift: np.ndarray
    ) -> np.ndarray:
        calls["n"] += 1
        shift = original(self, target, expected_shift)
        # The 4 corner measurements come after the two axis probe searches;
        # corrupt only the last one, as a sample that slipped would.
        return shift + 40.0 if calls["n"] == 6 else shift

    monkeypatch.setattr(_Tracker, "measure", measure_with_a_bad_corner)

    with pytest.raises(PixelCalibrationError, match="scatter exceeds tolerance"):
        run_pixel_calibration(core, _fast_options(), resolution_id="Resolution")
    # The stage still comes home after a rejected fit.
    assert core.position == pytest.approx(core.origin)


def test_failed_fit_still_carries_diagnostics_worth_showing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _SyntheticCore()
    calls = {"n": 0}
    original = _Tracker.measure

    def measure_with_a_bad_corner(
        self: _Tracker, target: np.ndarray, expected_shift: np.ndarray
    ) -> np.ndarray:
        calls["n"] += 1
        shift = original(self, target, expected_shift)
        return shift + 40.0 if calls["n"] == 6 else shift

    monkeypatch.setattr(_Tracker, "measure", measure_with_a_bad_corner)

    with pytest.raises(PixelCalibrationError) as caught:
        run_pixel_calibration(core, _fast_options(), resolution_id="Resolution")

    diagnostics = caught.value.diagnostics
    assert diagnostics is not None
    assert len(diagnostics.observations) == 4
    assert diagnostics.fit.pixel_size_um > 0
    assert not diagnostics.stage_returned


def test_calibration_reports_the_safety_limit_with_advice() -> None:
    core = _SyntheticCore()

    with pytest.raises(PixelCalibrationError, match="safety limit reached"):
        run_pixel_calibration(
            core, _fast_options(safe_radius_um=1.0), resolution_id="Resolution"
        )


def test_calibration_tolerates_a_stage_that_misses_its_return() -> None:
    """An open-loop stage that lands short must not fail the measurement."""

    class DriftingCore(_SyntheticCore):
        def setXYPosition(self, label: str, x: float, y: float) -> None:
            if (x, y) == (float(self.origin[0]), float(self.origin[1])):
                # Land 2 um short of the requested origin.
                super().setXYPosition(label, x + 2.0, y)
                return
            super().setXYPosition(label, x, y)

    core = DriftingCore()
    result = run_pixel_calibration(core, _fast_options(), resolution_id="Resolution")

    assert not result.stage_returned
    assert result.fit.matrix == pytest.approx(core.true_matrix, abs=5e-3)


def test_calibration_retries_a_transient_error_on_the_base_frame() -> None:
    core = _SyntheticCore()
    core.fail_snap_at = 1

    result = run_pixel_calibration(core, _fast_options(), resolution_id="Resolution")

    assert result.fit.matrix == pytest.approx(core.true_matrix, abs=5e-3)
    assert result.stage_returned


def test_calibration_does_not_retry_a_measurement_snap() -> None:
    """Java takes exactly one image per position; only the base frame retries."""
    core = _SyntheticCore()
    core.fail_snap_at = 2

    with pytest.raises(RuntimeError, match="synthetic camera failure"):
        run_pixel_calibration(core, _fast_options(), resolution_id="Resolution")

    assert core.position == pytest.approx(core.origin)


def test_calibration_restores_stage_after_persistent_failure() -> None:
    class FailingCore(_SyntheticCore):
        def snapImage(self) -> None:
            self.snap_count += 1
            if self.snap_count >= 4:
                raise RuntimeError("synthetic camera failure")

    core = FailingCore()

    with pytest.raises(RuntimeError, match="synthetic camera failure"):
        run_pixel_calibration(core, _fast_options(), resolution_id="Resolution")

    assert core.position == pytest.approx(core.origin)


def test_calibration_can_use_an_explicit_nondefault_stage() -> None:
    class ExplicitStageCore(_SyntheticCore):
        def getXYStageDevice(self) -> str:
            return "UnusedDefaultStage"

    core = ExplicitStageCore()

    result = run_pixel_calibration(
        core, _fast_options(), resolution_id="Resolution", xy_stage="XY"
    )

    assert result.fingerprint.xy_stage == "XY"
    assert result.fit.matrix == pytest.approx(core.true_matrix, abs=5e-3)
    assert core.position == pytest.approx(core.origin)


def test_calibration_accepts_an_unsaved_resolution_binding() -> None:
    core = _SyntheticCore()

    result = run_pixel_calibration(
        core,
        _fast_options(),
        resolution_id="New4x",
        config_settings=(),
        require_resolution_match=False,
    )

    assert result.fingerprint.pixel_size_config == "Resolution"
    assert result.fingerprint.config_settings == ()
    assert result.fit.matrix == pytest.approx(core.true_matrix, abs=5e-3)


def test_calibration_aborts_if_acquisition_starts_mid_run() -> None:
    class InterruptedCore(_SyntheticCore):
        def isSequenceRunning(self, label: str) -> bool:
            return self.snap_count >= 3

    core = InterruptedCore()

    with pytest.raises(PixelCalibrationError, match="acquisition started"):
        run_pixel_calibration(core, _fast_options(), resolution_id="Resolution")

    assert core.position == pytest.approx(core.origin)


def test_calibration_honors_preexisting_cancellation() -> None:
    core = _SyntheticCore()
    cancelled = Event()
    cancelled.set()

    with pytest.raises(CalibrationCancelled):
        run_pixel_calibration(
            core, _fast_options(), resolution_id="Resolution", cancel_event=cancelled
        )

    assert core.position == pytest.approx(core.origin)
    assert core.moves == []


def test_calibration_rejects_a_multichannel_camera() -> None:
    class ColorCore(_SyntheticCore):
        def getNumberOfCameraChannels(self) -> int:
            return 3

    with pytest.raises(PixelCalibrationError, match="one camera channel"):
        run_pixel_calibration(ColorCore(), _fast_options(), resolution_id="Resolution")


# --- persistence -------------------------------------------------------------


def test_commit_pixel_calibration_writes_a_measured_result() -> None:
    core = _SyntheticCore()
    result = run_pixel_calibration(core, _fast_options(), resolution_id="Resolution")
    core.stored_size *= 0.99

    commit_pixel_calibration(core, "Resolution", result)

    assert core.stored_size == pytest.approx(result.raw_pixel_size_um)
    assert np.asarray(core.stored_affine).reshape(2, 3)[:, :2] == pytest.approx(
        result.raw_matrix
    )


def test_commit_pixel_calibration_rolls_back_partial_write() -> None:
    core = _SyntheticCore()
    result = run_pixel_calibration(core, _fast_options(), resolution_id="Resolution")
    core.stored_size = 0.31
    core.stored_affine = (0.31, 0.0, 0.0, 0.0, 0.31, 0.0)
    old_size = core.stored_size
    old_affine = core.stored_affine
    core.fail_affine_once = True

    with pytest.raises(CalibrationCommitError, match="synthetic persistence failure"):
        commit_pixel_calibration(
            core, "Resolution", result, allow_large_difference=True
        )

    assert core.stored_size == old_size
    assert core.stored_affine == old_affine


# --- noise and blur ----------------------------------------------------------


class _NoisyFieldCore(_SyntheticCore):
    """Capture a noisy ROI from a larger field, without wrapping the camera edges."""

    def __init__(self, seed: int, blur: float) -> None:
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.field = _texture((768, 896))
        frequencies = (
            np.fft.fftfreq(self.field.shape[0])[:, None] ** 2
            + np.fft.fftfreq(self.field.shape[1])[None, :] ** 2
        )
        self.field = np.fft.ifftn(
            np.fft.fftn(self.field) * np.exp(-2 * np.pi**2 * blur**2 * frequencies)
        ).real
        self.field /= self.field.std()

    def getImage(self) -> np.ndarray:
        shift_xy = np.linalg.solve(self.true_matrix, self.position - self.origin)
        image = _shift_image(self.field, (-shift_xy[1], -shift_xy[0]))[192:576, 224:672]
        return image + self.rng.normal(scale=0.15, size=image.shape)


@pytest.mark.parametrize("blur", [2.0, 4.0, 6.0])
def test_noisy_blurred_calibration_is_repeatable(blur: float) -> None:
    sizes = []
    for seed in range(5):
        core = _NoisyFieldCore(seed, blur)
        # A wrong saved calibration must not change probing or the measurement.
        core.stored_size = (0.0, 0.04, 0.4, 4.0, 40.0)[seed]
        result = run_pixel_calibration(core, _fast_options())
        assert result.fit.matrix == pytest.approx(core.true_matrix, abs=0.005)
        assert result.fit.rms_residual_px < 1.0
        assert core.position == pytest.approx(core.origin)
        sizes.append(result.fit.pixel_size_um)
    assert np.ptp(sizes) / 0.4 < 0.02
