"""Headless pixel-size and camera/stage calibration primitives."""

from ._capture import CalibrationCaptureSettings, CaptureStateTransaction
from ._fit import fit_affine, normalize_for_mmcore
from ._models import (
    AffineFitResult,
    CalibrationCancelled,
    CalibrationCommitError,
    CalibrationObservation,
    CalibrationOptions,
    CalibrationWarning,
    HardwareFingerprint,
    PixelCalibrationError,
    PixelCalibrationResult,
    RegistrationResult,
    StageRestoreError,
)
from ._persistence import commit_pixel_calibration
from ._registration import register_translation
from ._routine import run_pixel_calibration

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
    "RegistrationResult",
    "StageRestoreError",
    "commit_pixel_calibration",
    "fit_affine",
    "normalize_for_mmcore",
    "register_translation",
    "run_pixel_calibration",
]
