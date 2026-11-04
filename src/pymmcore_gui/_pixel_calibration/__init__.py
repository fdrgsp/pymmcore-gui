"""Headless pixel-size and camera/stage calibration primitives."""

from ._capture import CalibrationCaptureSettings, CaptureStateTransaction
from ._fit import flatten_affine, normalize_for_mmcore
from ._models import (
    AffineFitResult,
    CalibrationCancelled,
    CalibrationObservation,
    CalibrationWarning,
    HardwareFingerprint,
    PixelCalibrationError,
    PixelCalibrationResult,
    StageRestoreError,
)
from ._routine import (
    TEST_POSITION_NAMES,
    CalibrationOptions,
    affine_to_measurements,
    cross_correlate,
    deduce_pixel_size,
    fit_affine_with_translation,
    measure_displacement,
    run_pixel_calibration,
)

__all__ = [
    "TEST_POSITION_NAMES",
    "AffineFitResult",
    "CalibrationCancelled",
    "CalibrationCaptureSettings",
    "CalibrationObservation",
    "CalibrationOptions",
    "CalibrationWarning",
    "CaptureStateTransaction",
    "HardwareFingerprint",
    "PixelCalibrationError",
    "PixelCalibrationResult",
    "StageRestoreError",
    "affine_to_measurements",
    "cross_correlate",
    "deduce_pixel_size",
    "fit_affine_with_translation",
    "flatten_affine",
    "measure_displacement",
    "normalize_for_mmcore",
    "run_pixel_calibration",
]
