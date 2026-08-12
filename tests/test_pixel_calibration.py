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
    commit_pixel_calibration,
    fit_affine,
    normalize_for_mmcore,
    register_translation,
    run_pixel_calibration,
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


@pytest.mark.parametrize(
    "apparent_shift_rc", [(4.35, -7.2), (-11.55, 3.75), (0.0, 8.4)]
)
def test_register_translation_subpixel_and_xy_order(
    apparent_shift_rc: tuple[float, float],
) -> None:
    reference = _texture()
    moving = _shift_image(reference, apparent_shift_rc)

    result = register_translation(reference, moving, crop_fraction=0.8)

    # The result is the shift to apply to moving (opposite apparent motion), and
    # it is returned as geometric x/y rather than array row/column.
    assert result.shift_xy == pytest.approx(
        (-apparent_shift_rc[1], -apparent_shift_rc[0]), abs=0.1
    )
    assert result.overlap > 0.8
    assert result.psr > 8
    assert result.normalized_error < 0.25


def test_register_translation_handles_gain_offset_and_rgb() -> None:
    reference = _texture((192, 224))
    moving = _shift_image(reference, (5.25, 6.6)) * 2.3 + 17
    rgb_reference = np.stack((reference, reference * 0.8, reference * 1.2), axis=-1)
    rgb_moving = np.stack((moving, moving * 0.8, moving * 1.2), axis=-1)

    result = register_translation(rgb_reference, rgb_moving)

    assert result.shift_xy == pytest.approx((-6.6, -5.25), abs=0.1)


@pytest.mark.parametrize(
    "reference,moving,message",
    [
        (np.ones((64, 64)), np.ones((64, 64)), "insufficient"),
        (np.zeros((64, 64)), np.zeros((63, 64)), "same shape"),
        (np.zeros((64, 64, 2)), np.zeros((64, 64, 2)), "2D grayscale"),
    ],
)
def test_register_translation_rejects_invalid_images(
    reference: np.ndarray, moving: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        register_translation(reference, moving)


def test_fit_affine_recovers_rotated_reflected_matrix_with_outlier() -> None:
    rng = np.random.default_rng(4)
    angle = np.deg2rad(27)
    matrix = np.asarray(
        [
            [0.41 * np.cos(angle), 0.39 * np.sin(angle)],
            [0.41 * np.sin(angle), -0.39 * np.cos(angle)],
        ]
    )
    shifts = np.asarray(
        [
            [-70, 0],
            [70, 0],
            [0, -65],
            [0, 65],
            [-55, -45],
            [55, 45],
            [-55, 45],
            [55, -45],
        ],
        dtype=float,
    )
    deltas = shifts @ matrix.T + rng.normal(scale=0.01, size=shifts.shape)
    deltas[3] += (8, -6)

    result = fit_affine(shifts, deltas, minimum_points=6)

    assert result.matrix == pytest.approx(matrix, abs=5e-4)
    assert np.count_nonzero(~result.inlier_mask) == 1
    assert result.pixel_size_um == pytest.approx(
        np.sqrt(abs(np.linalg.det(matrix))), rel=2e-3
    )
    assert result.determinant < 0
    assert result.rms_residual_px < 0.1


def test_fit_affine_rejects_collinear_design() -> None:
    shifts = np.asarray([[1, 0], [2, 0], [3, 0]], dtype=float)
    with pytest.raises(ValueError, match=r"singular|ill-conditioned"):
        fit_affine(shifts, shifts)


def test_normalize_for_mmcore_accounts_for_binning_and_magnification() -> None:
    current = np.asarray([[0.8, 0.1], [-0.2, 0.7]])

    raw, pixel_size, flattened = normalize_for_mmcore(
        current, binning=2, magnification=1.5
    )

    assert raw == pytest.approx(current * 0.75)
    assert pixel_size == pytest.approx(np.sqrt(abs(np.linalg.det(raw))))
    assert flattened == pytest.approx(
        (raw[0, 0], raw[0, 1], 0, raw[1, 0], raw[1, 1], 0)
    )


class _EmptyConfig:
    def size(self) -> int:
        return 0


class _IdleMDA:
    def is_running(self) -> bool:
        return False


class _SyntheticCore:
    """Small MMCore-compatible camera/stage model for end-to-end tests."""

    def __init__(self) -> None:
        angle = np.deg2rad(19)
        self.true_matrix = 0.4 * np.asarray(
            [
                [np.cos(angle), np.sin(angle)],
                [np.sin(angle), -np.cos(angle)],
            ]
        )
        self.base_image = _texture((192, 224))
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


def _fast_options() -> CalibrationOptions:
    return CalibrationOptions(safe_radius_um=50, settle_time_s=0)


def test_run_pixel_calibration_measures_affine_without_persisting() -> None:
    core = _SyntheticCore()
    old_size = core.stored_size
    old_affine = core.stored_affine

    result = run_pixel_calibration(
        core,
        _fast_options(),
        resolution_id="Resolution",
    )

    assert result.stage_returned
    assert result.fit.matrix == pytest.approx(core.true_matrix, abs=2e-3)
    assert result.fit.determinant < 0
    assert len(result.observations) == 8
    assert len(result.validation_observations) == 3
    assert all(observation.accepted for observation in result.observations)
    assert core.position == pytest.approx(core.origin)
    assert (
        max(np.linalg.norm(np.asarray(move) - core.origin) for move in core.moves) < 50
    )
    assert core.stored_size == old_size
    assert core.stored_affine == old_affine


def test_run_pixel_calibration_restores_stage_after_failure() -> None:
    core = _SyntheticCore()
    core.fail_snap_at = 4

    with pytest.raises(RuntimeError, match="synthetic camera failure"):
        run_pixel_calibration(core, _fast_options(), resolution_id="Resolution")

    assert core.position == pytest.approx(core.origin)


def test_run_pixel_calibration_can_use_explicit_nondefault_stage() -> None:
    class ExplicitStageCore(_SyntheticCore):
        def getXYStageDevice(self) -> str:
            return "UnusedDefaultStage"

    core = ExplicitStageCore()

    result = run_pixel_calibration(
        core,
        _fast_options(),
        resolution_id="Resolution",
        xy_stage="XY",
    )

    assert result.fingerprint.xy_stage == "XY"
    assert result.fit.matrix == pytest.approx(core.true_matrix, abs=2e-3)
    assert core.position == pytest.approx(core.origin)


def test_run_pixel_calibration_aborts_if_acquisition_starts_mid_run() -> None:
    class InterruptedCore(_SyntheticCore):
        def isSequenceRunning(self, label: str) -> bool:
            return self.snap_count >= 3

    core = InterruptedCore()

    with pytest.raises(PixelCalibrationError, match="acquisition started"):
        run_pixel_calibration(core, _fast_options(), resolution_id="Resolution")

    assert core.position == pytest.approx(core.origin)


def test_run_pixel_calibration_honors_preexisting_cancellation() -> None:
    core = _SyntheticCore()
    cancelled = Event()
    cancelled.set()

    with pytest.raises(CalibrationCancelled):
        run_pixel_calibration(
            core,
            _fast_options(),
            resolution_id="Resolution",
            cancel_event=cancelled,
        )

    assert core.position == pytest.approx(core.origin)


def test_commit_pixel_calibration_writes_validated_result() -> None:
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
            core,
            "Resolution",
            result,
            allow_large_difference=True,
        )

    assert core.stored_size == old_size
    assert core.stored_affine == old_affine
