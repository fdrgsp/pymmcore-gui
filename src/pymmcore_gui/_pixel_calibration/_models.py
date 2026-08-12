from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray


@dataclass(frozen=True)
class CalibrationOptions:
    """Numerical and motion limits for an automatic calibration run."""

    safe_radius_um: float = 100.0
    settle_time_s: float = 0.0
    crop_fraction: float = 0.75
    upsample_factor: int = 20
    min_psr: float = 8.0
    min_peak_ratio: float = 1.05
    min_overlap: float = 0.60
    target_shift_fraction: float = 0.16
    min_shift_fraction: float = 0.04
    max_shift_fraction: float = 0.30
    min_shift_px: float = 8.0
    max_probe_steps: int = 16
    initial_probe_um: float = 0.5
    stage_return_tolerance_um: float = 0.5
    max_fit_rms_px: float = 0.5
    max_fit_fraction: float = 0.01
    max_point_residual_px: float = 1.5
    max_point_residual_fraction: float = 0.03

    def __post_init__(self) -> None:
        if self.safe_radius_um <= 0:
            raise ValueError("safe_radius_um must be positive")
        if self.settle_time_s < 0:
            raise ValueError("settle_time_s cannot be negative")
        if not 0.25 <= self.crop_fraction <= 1:
            raise ValueError("crop_fraction must be between 0.25 and 1")
        if self.upsample_factor < 1:
            raise ValueError("upsample_factor must be at least 1")
        if not 0 < self.min_overlap <= 1:
            raise ValueError("min_overlap must be in (0, 1]")
        if not 0 < self.min_shift_fraction < self.target_shift_fraction:
            raise ValueError(
                "min_shift_fraction must be positive and smaller than "
                "target_shift_fraction"
            )
        if not self.target_shift_fraction < self.max_shift_fraction < 0.5:
            raise ValueError(
                "max_shift_fraction must be between target_shift_fraction and 0.5"
            )
        if self.min_shift_px <= 0:
            raise ValueError("min_shift_px must be positive")
        if self.max_probe_steps < 1:
            raise ValueError("max_probe_steps must be at least 1")
        if self.initial_probe_um <= 0:
            raise ValueError("initial_probe_um must be positive")
        if self.stage_return_tolerance_um < 0:
            raise ValueError("stage_return_tolerance_um cannot be negative")


@dataclass(frozen=True)
class RegistrationResult:
    """Subpixel translation and confidence measurements."""

    # Shift applied to moving image to align it to reference, in geometric x/y order.
    shift_xy: tuple[float, float]
    psr: float
    peak_ratio: float
    overlap: float
    normalized_error: float
    method: Literal["phase", "unnormalized"] = "phase"


@dataclass(frozen=True)
class CalibrationWarning:
    """A non-fatal issue attached to a calibration result."""

    code: str
    message: str


class PixelCalibrationError(RuntimeError):
    """Base error for a failed or invalid pixel calibration."""


class CalibrationCancelled(PixelCalibrationError):
    """Raised when cancellation is requested during a calibration."""


class StageRestoreError(PixelCalibrationError):
    """Raised when the stage cannot be returned after calibration."""

    def __init__(
        self,
        restore_error: BaseException,
        calibration_error: BaseException | None = None,
    ) -> None:
        self.restore_error = restore_error
        self.calibration_error = calibration_error
        message = f"Failed to restore the XY stage: {restore_error}"
        if calibration_error is not None:
            message += f" (calibration had already failed: {calibration_error})"
        super().__init__(message)


class CalibrationCommitError(PixelCalibrationError):
    """Raised when committing or rolling back MMCore calibration data fails."""


@dataclass(frozen=True)
class HardwareFingerprint:
    """Hardware and optical state in which a calibration was measured."""

    camera: str
    xy_stage: str
    binning: int
    magnification: float
    roi: tuple[int, int, int, int]
    image_shape: tuple[int, int]
    dtype: str
    channel_count: int
    pixel_size_config: str = ""
    config_settings: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class CalibrationObservation:
    """One stage/image displacement pair used by the calibration."""

    stage_position_um: tuple[float, float]
    stage_delta_um: tuple[float, float]
    registration: RegistrationResult
    corrected_shift_xy: tuple[float, float]
    accepted: bool = True
    rejection_reason: str = ""


@dataclass(frozen=True)
class AffineFitResult:
    """Robust image-pixel to stage-micrometre affine fit."""

    matrix: NDArray[np.float64]
    residuals_um: NDArray[np.float64]
    residuals_px: NDArray[np.float64]
    weights: NDArray[np.float64]
    inlier_mask: NDArray[np.bool_]
    pixel_size_um: float
    pixel_size_x_um: float
    pixel_size_y_um: float
    singular_values: tuple[float, float]
    design_condition: float
    matrix_condition: float
    anisotropy: float
    nonorthogonality_deg: float
    rotation_deg: float
    determinant: float
    rms_residual_px: float
    max_residual_px: float
    warnings: tuple[CalibrationWarning, ...] = ()


@dataclass(frozen=True)
class PixelCalibrationResult:
    """A validated but not necessarily persisted pixel calibration."""

    fit: AffineFitResult
    raw_matrix: NDArray[np.float64]
    raw_pixel_size_um: float
    fingerprint: HardwareFingerprint
    observations: tuple[CalibrationObservation, ...]
    validation_observations: tuple[CalibrationObservation, ...]
    stage_returned: bool
    algorithm_version: str = "1"
    warnings: tuple[CalibrationWarning, ...] = field(default_factory=tuple)


def as_float64_points(
    values: Sequence[Sequence[float]] | NDArray[np.floating], *, name: str
) -> NDArray[np.float64]:
    """Validate and copy an N x 2 point array."""
    import numpy as np

    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result
