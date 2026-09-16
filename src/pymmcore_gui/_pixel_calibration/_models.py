from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass(frozen=True)
class CalibrationWarning:
    """A non-fatal issue attached to a calibration result."""

    code: str
    message: str


class PixelCalibrationError(RuntimeError):
    """Base error for a failed or invalid pixel calibration.

    ``diagnostics``, when set by the routine, carries the fit that existed at
    the point of failure -- a matrix whose four corner measurements scattered
    beyond the acceptance tolerance is still worth plotting and still carries
    a pixel size worth showing. Callers (the GUI panel) use it to show the
    same measured-versus-predicted graph on a failed run that a successful one
    gets, instead of leaving the diagnostics blank just because the run did
    not pass.
    """

    def __init__(
        self, message: str, *, diagnostics: PixelCalibrationResult | None = None
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


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
        # Carry forward diagnostics from the original failure, if it has any,
        # so a scatter failure followed by a restore failure still shows the
        # diagnostics graph instead of losing it to the wrapper.
        super().__init__(
            message, diagnostics=getattr(calibration_error, "diagnostics", None)
        )


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
    """One stage/image displacement pair used by the calibration.

    ``image_shift_xy`` is in geometric x/y order and is the negative of the
    apparent motion of sample features, which is the convention the fitted
    matrix maps to stage micrometres.

    ``residual_px`` is filled in once the final matrix exists, and is the
    translation-aware residual the acceptance check itself uses: the stage
    position mapped back through the inverse affine, minus the measured image
    shift. It stays ``None`` on observations emitted live during acquisition,
    because no matrix exists yet to measure them against.

    ``label`` is the corner number drawn by the diagnostics graph.
    """

    stage_position_um: tuple[float, float]
    stage_delta_um: tuple[float, float]
    image_shift_xy: tuple[float, float]
    residual_px: tuple[float, float] | None = None
    label: str = ""


@dataclass(frozen=True)
class AffineFitResult:
    """Image-pixel to stage-micrometre affine fit and its diagnostics.

    ``residuals_px`` and the RMS and worst values derived from it are the
    translation-aware residuals the acceptance check uses.

    Four corner measurements feeding a six-parameter affine leave only one
    residual degree of freedom per output axis. A bad measurement is therefore
    spread across all four residuals rather than isolated at its source; the
    per-corner values are real, but do not reliably identify which acquisition
    went wrong.
    """

    matrix: NDArray[np.float64]
    residuals_um: NDArray[np.float64]
    residuals_px: NDArray[np.float64]
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
    """A measured but not necessarily persisted pixel calibration."""

    fit: AffineFitResult
    raw_matrix: NDArray[np.float64]
    raw_pixel_size_um: float
    fingerprint: HardwareFingerprint
    observations: tuple[CalibrationObservation, ...]
    stage_returned: bool
    algorithm_version: str = "1"
    warnings: tuple[CalibrationWarning, ...] = field(default_factory=tuple)
    max_rms_px: float = 5.0
