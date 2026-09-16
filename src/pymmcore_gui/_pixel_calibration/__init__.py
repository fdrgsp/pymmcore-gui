"""Headless pixel-size and camera/stage calibration primitives."""

from ._capture import CalibrationCaptureSettings, CaptureStateTransaction
from ._fit import normalize_for_mmcore
from ._models import (
    AffineFitResult,
    CalibrationCancelled,
    CalibrationCommitError,
    CalibrationObservation,
    CalibrationWarning,
    HardwareFingerprint,
    PixelCalibrationError,
    PixelCalibrationResult,
    StageRestoreError,
)
from ._persistence import commit_pixel_calibration
from ._routine import (
    CalibrationOptions,
    affine_to_measurements,
    cross_correlate,
    deduce_pixel_size,
    fit_affine_with_translation,
    measure_displacement,
    run_pixel_calibration,
)

__all__ = [
    "AffineFitResult",
    "CalibrationCancelled",
    "CalibrationCaptureSettings",
    "CalibrationCommitError",
    "CalibrationObservation",
    "CalibrationOptions",
    "CalibrationWarning",
    "CaptureStateTransaction",
    "HardwareFingerprint",
    "PixelCalibrationError",
    "PixelCalibrationResult",
    "StageRestoreError",
    "affine_to_measurements",
    "commit_pixel_calibration",
    "cross_correlate",
    "deduce_pixel_size",
    "fit_affine_with_translation",
    "measure_displacement",
    "normalize_for_mmcore",
    "run_pixel_calibration",
]
