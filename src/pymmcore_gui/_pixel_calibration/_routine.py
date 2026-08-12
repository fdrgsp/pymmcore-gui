from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from ._fit import fit_affine, normalize_for_mmcore
from ._models import (
    AffineFitResult,
    CalibrationCancelled,
    CalibrationObservation,
    CalibrationOptions,
    CalibrationWarning,
    HardwareFingerprint,
    PixelCalibrationError,
    PixelCalibrationResult,
    RegistrationResult,
    StageRestoreError,
)
from ._registration import register_translation

if TYPE_CHECKING:
    from collections.abc import Sequence
    from threading import Event

    from numpy.typing import NDArray

ProgressCallback = Callable[[str, float], None]


class CalibrationCore(Protocol):
    """The MMCore operations needed by the headless calibration routine."""

    def getCameraDevice(self) -> str: ...

    def getXYStageDevice(self) -> str: ...

    def getBinning(self, label: str) -> int: ...

    def getMagnificationFactor(self) -> float: ...

    def getROI(self, label: str) -> Sequence[int]: ...

    def getImageWidth(self) -> int: ...

    def getImageHeight(self) -> int: ...

    def getNumberOfCameraChannels(self) -> int: ...

    def getCurrentPixelSizeConfig(self) -> str: ...

    def getPixelSizeConfigData(self, config_name: str) -> Any: ...

    def getProperty(self, device: str, prop: str) -> str: ...

    def getXYPosition(self, label: str) -> Sequence[float]: ...

    def setXYPosition(self, label: str, x: float, y: float) -> None: ...

    def waitForDevice(self, label: str) -> None: ...

    def isSequenceRunning(self, label: str) -> bool: ...

    def snapImage(self) -> None: ...

    def getImage(self) -> NDArray[np.generic]: ...

    def getPixelSizeUm(self) -> float: ...


def _notify(progress: ProgressCallback | None, phase: str, fraction: float) -> None:
    if progress is None:
        return
    try:
        progress(phase, min(max(fraction, 0.0), 1.0))
    except Exception:
        # Progress reporting must not be able to corrupt a hardware operation.
        pass


def _check_cancel(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise CalibrationCancelled("Pixel calibration was cancelled")


def _config_settings(
    core: CalibrationCore, resolution_id: str | None
) -> tuple[tuple[str, str, str], ...]:
    if not resolution_id:
        return ()
    config = core.getPixelSizeConfigData(resolution_id)
    settings: list[tuple[str, str, str]] = []
    for index in range(int(config.size())):
        setting = config.getSetting(index)
        item = (
            str(setting.getDeviceLabel()),
            str(setting.getPropertyName()),
            str(setting.getPropertyValue()),
        )
        if str(core.getProperty(item[0], item[1])) != item[2]:
            raise PixelCalibrationError(
                f"Pixel-size configuration {resolution_id!r} does not match the "
                f"current value of {item[0]}-{item[1]}"
            )
        settings.append(item)
    return tuple(sorted(settings))


def _current_config(core: CalibrationCore) -> str:
    try:
        return str(core.getCurrentPixelSizeConfig())
    except Exception:
        return ""


def capture_fingerprint(
    core: CalibrationCore,
    image: NDArray[np.generic],
    *,
    resolution_id: str | None = None,
    xy_stage: str | None = None,
) -> HardwareFingerprint:
    """Capture and validate the optical state relevant to a calibration."""
    camera = str(core.getCameraDevice())
    stage = str(xy_stage or core.getXYStageDevice())
    if not camera:
        raise PixelCalibrationError("No camera device is selected")
    if not stage:
        raise PixelCalibrationError("No XY stage device is selected")
    if image.ndim not in (2, 3):
        raise PixelCalibrationError("The camera did not return a 2D or RGB image")
    shape = (int(image.shape[0]), int(image.shape[1]))
    if min(shape) < 128:
        raise PixelCalibrationError("Calibration images must be at least 128 pixels")
    binning = int(core.getBinning(camera))
    magnification = float(core.getMagnificationFactor())
    if binning < 1 or not np.isfinite(magnification) or magnification <= 0:
        raise PixelCalibrationError("Invalid camera binning or magnification factor")
    roi_values = tuple(int(v) for v in core.getROI(camera))
    if len(roi_values) != 4:
        raise PixelCalibrationError("MMCore returned an invalid camera ROI")
    current_config = _current_config(core)
    if resolution_id and current_config != resolution_id:
        raise PixelCalibrationError(
            f"Pixel-size configuration {resolution_id!r} is not the current match"
        )
    return HardwareFingerprint(
        camera=camera,
        xy_stage=stage,
        binning=binning,
        magnification=magnification,
        roi=roi_values,
        image_shape=shape,
        dtype=str(image.dtype),
        channel_count=int(core.getNumberOfCameraChannels()),
        pixel_size_config=current_config,
        config_settings=_config_settings(core, resolution_id),
    )


def fingerprint_matches(core: CalibrationCore, expected: HardwareFingerprint) -> bool:
    """Return whether the current optical state still matches ``expected``."""
    try:
        if str(core.getCameraDevice()) != expected.camera:
            return False
        if int(core.getBinning(expected.camera)) != expected.binning:
            return False
        if not np.isclose(
            float(core.getMagnificationFactor()), expected.magnification, rtol=1e-12
        ):
            return False
        if tuple(int(v) for v in core.getROI(expected.camera)) != expected.roi:
            return False
        if (
            int(core.getImageHeight()),
            int(core.getImageWidth()),
        ) != expected.image_shape:
            return False
        if int(core.getNumberOfCameraChannels()) != expected.channel_count:
            return False
        if _current_config(core) != expected.pixel_size_config:
            return False
        return all(
            str(core.getProperty(device, prop)) == value
            for device, prop, value in expected.config_settings
        )
    except Exception:
        return False


def _is_acquiring(core: CalibrationCore, camera: str) -> bool:
    try:
        if core.isSequenceRunning(camera):
            return True
    except Exception:
        pass
    runner = getattr(core, "mda", None)
    try:
        return bool(runner is not None and runner.is_running())
    except Exception:
        return False


def _snap(core: CalibrationCore, *, camera: str | None = None) -> NDArray[np.generic]:
    if camera and _is_acquiring(core, camera):
        raise PixelCalibrationError("An acquisition started during calibration")
    core.snapImage()
    return np.asarray(core.getImage()).copy()


def _position(core: CalibrationCore, stage: str) -> NDArray[np.float64]:
    position = np.asarray(core.getXYPosition(stage), dtype=np.float64)
    if position.shape != (2,) or not np.all(np.isfinite(position)):
        raise PixelCalibrationError("XY stage returned an invalid position")
    return position


def _move(
    core: CalibrationCore,
    target: NDArray[np.float64],
    *,
    origin: NDArray[np.float64],
    fingerprint: HardwareFingerprint,
    options: CalibrationOptions,
) -> NDArray[np.float64]:
    if _is_acquiring(core, fingerprint.camera):
        raise PixelCalibrationError("An acquisition started during calibration")
    if not fingerprint_matches(core, fingerprint):
        raise PixelCalibrationError("The camera or optical configuration changed")
    distance = float(np.linalg.norm(target - origin))
    if distance > options.safe_radius_um:
        raise PixelCalibrationError(
            f"Requested stage move ({distance:.3g} µm) exceeds the safe radius"
        )
    core.setXYPosition(fingerprint.xy_stage, float(target[0]), float(target[1]))
    core.waitForDevice(fingerprint.xy_stage)
    if options.settle_time_s:
        time.sleep(options.settle_time_s)
    actual = _position(core, fingerprint.xy_stage)
    actual_distance = float(np.linalg.norm(actual - origin))
    if actual_distance > options.safe_radius_um:
        raise PixelCalibrationError(
            f"Stage readback ({actual_distance:.3g} µm) exceeds the safe radius"
        )
    return actual


def _registration(
    reference: NDArray[np.generic],
    moving: NDArray[np.generic],
    options: CalibrationOptions,
) -> RegistrationResult:
    return register_translation(
        reference,
        moving,
        upsample_factor=options.upsample_factor,
        crop_fraction=options.crop_fraction,
    )


def _registration_is_usable(
    result: RegistrationResult, options: CalibrationOptions
) -> bool:
    return (
        result.psr >= options.min_psr
        and result.peak_ratio >= options.min_peak_ratio
        and result.overlap >= options.min_overlap
        and np.isfinite(result.normalized_error)
    )


def _validate_references(
    first: NDArray[np.generic],
    second: NDArray[np.generic],
    options: CalibrationOptions,
) -> NDArray[np.float64]:
    if first.shape != second.shape or first.dtype != second.dtype:
        raise PixelCalibrationError(
            "Camera image format changed between reference snaps"
        )
    if np.issubdtype(first.dtype, np.integer):
        limits = np.iinfo(first.dtype.name)
        saturated = np.mean((first == limits.min) | (first == limits.max))
        if saturated > 0.10:
            raise PixelCalibrationError("Reference image is more than 10% saturated")
    repeat = _registration(first, second, options)
    if np.linalg.norm(repeat.shift_xy) > 2 or repeat.normalized_error > 0.75:
        raise PixelCalibrationError(
            "Reference images are not stable enough to calibrate"
        )
    return np.asarray(
        (np.asarray(first, dtype=np.float64) + np.asarray(second, dtype=np.float64))
        / 2,
        dtype=np.float64,
    )


def _probe_axis(
    core: CalibrationCore,
    reference: NDArray[np.generic],
    origin: NDArray[np.float64],
    fingerprint: HardwareFingerprint,
    axis: int,
    options: CalibrationOptions,
    cancel_event: Event | None,
) -> CalibrationObservation:
    image_limit = np.asarray(fingerprint.image_shape[::-1], dtype=np.float64)
    minimum = max(options.min_shift_px, options.min_shift_fraction * min(image_limit))
    distance = options.initial_probe_um
    try:
        existing_size = float(core.getPixelSizeUm())
    except Exception:
        existing_size = 0.0
    if existing_size > 0 and np.isfinite(existing_size):
        hinted = existing_size * min(image_limit) * options.min_shift_fraction
        distance = max(distance, min(hinted, options.safe_radius_um / 8))

    for _ in range(options.max_probe_steps):
        _check_cancel(cancel_event)
        if distance > options.safe_radius_um:
            break
        offset = np.zeros(2, dtype=np.float64)
        offset[axis] = distance
        actual = _move(
            core,
            origin + offset,
            origin=origin,
            fingerprint=fingerprint,
            options=options,
        )
        image = _snap(core, camera=fingerprint.camera)
        registration = _registration(reference, image, options)
        _move(
            core,
            origin,
            origin=origin,
            fingerprint=fingerprint,
            options=options,
        )
        shift = np.asarray(registration.shift_xy)
        fraction = float(np.max(np.abs(shift) / image_limit))
        if (
            np.linalg.norm(shift) >= minimum
            and fraction <= options.max_shift_fraction
            and _registration_is_usable(registration, options)
        ):
            return CalibrationObservation(
                stage_position_um=(float(actual[0]), float(actual[1])),
                stage_delta_um=(
                    float(actual[0] - origin[0]),
                    float(actual[1] - origin[1]),
                ),
                registration=registration,
                corrected_shift_xy=registration.shift_xy,
            )
        if fraction > options.max_shift_fraction:
            raise PixelCalibrationError(
                "Adaptive probe moved the image outside its safe overlap"
            )
        distance *= 2
    axis_name = "XY"[axis]
    raise PixelCalibrationError(
        f"Could not measure image motion from stage axis {axis_name} within the "
        "safe radius"
    )


def _acquire_observation(
    core: CalibrationCore,
    reference: NDArray[np.generic],
    origin: NDArray[np.float64],
    target: NDArray[np.float64],
    origin_before_position: NDArray[np.float64],
    origin_before_shift: NDArray[np.float64],
    fingerprint: HardwareFingerprint,
    options: CalibrationOptions,
    cancel_event: Event | None,
) -> tuple[CalibrationObservation, NDArray[np.float64], NDArray[np.float64]]:
    _check_cancel(cancel_event)
    actual = _move(
        core,
        target,
        origin=origin,
        fingerprint=fingerprint,
        options=options,
    )
    target_registration = _registration(
        reference, _snap(core, camera=fingerprint.camera), options
    )
    actual_origin = _move(
        core,
        origin,
        origin=origin,
        fingerprint=fingerprint,
        options=options,
    )
    origin_registration = _registration(
        reference, _snap(core, camera=fingerprint.camera), options
    )
    origin_after_shift = np.asarray(origin_registration.shift_xy)
    drift = 0.5 * (origin_before_shift + origin_after_shift)
    corrected = np.asarray(target_registration.shift_xy) - drift
    local_origin = 0.5 * (origin_before_position + actual_origin)
    delta = actual - local_origin
    usable = _registration_is_usable(
        target_registration, options
    ) and _registration_is_usable(origin_registration, options)
    observation = CalibrationObservation(
        stage_position_um=(float(actual[0]), float(actual[1])),
        stage_delta_um=(float(delta[0]), float(delta[1])),
        registration=target_registration,
        corrected_shift_xy=(float(corrected[0]), float(corrected[1])),
        accepted=usable,
        rejection_reason="" if usable else "registration confidence below threshold",
    )
    return observation, actual_origin, origin_after_shift


def _measurement_targets(
    matrix: NDArray[np.float64],
    fingerprint: HardwareFingerprint,
    options: CalibrationOptions,
    *,
    validation: bool,
) -> list[NDArray[np.float64]]:
    width = fingerprint.image_shape[1] * options.crop_fraction
    height = fingerprint.image_shape[0] * options.crop_fraction
    if validation:
        fraction = options.target_shift_fraction * 0.65
        pixel_targets = [
            (0.87 * width * fraction, 0.50 * height * fraction),
            (-0.87 * width * fraction, 0.50 * height * fraction),
            (0.0, -height * fraction),
        ]
    else:
        x = width * options.target_shift_fraction
        y = height * options.target_shift_fraction
        pixel_targets = [(x, 0), (-x, 0), (0, y), (0, -y)]
        pixel_targets += [(x, y), (-x, -y), (-x, y), (x, -y)]
    offsets = [matrix @ np.asarray(point) for point in pixel_targets]
    maximum = max(float(np.linalg.norm(offset)) for offset in offsets)
    if maximum > options.safe_radius_um:
        scale = 0.9 * options.safe_radius_um / maximum
        offsets = [offset * scale for offset in offsets]
    return offsets


def _confidence_weights(
    observations: list[CalibrationObservation],
) -> NDArray[np.float64]:
    psr = np.asarray([obs.registration.psr for obs in observations])
    return np.clip(psr / max(float(np.median(psr)), np.finfo(float).eps), 0.25, 2.0)


def _diagnostics_snapshot(
    fit: AffineFitResult,
    fingerprint: HardwareFingerprint,
    observations: Sequence[CalibrationObservation],
    validation: Sequence[CalibrationObservation],
) -> PixelCalibrationResult:
    """Package a failed run's fit-so-far for the diagnostics graph only.

    Attached to a validation failure's exception (see the ``except`` blocks
    around ``_validate_fit``/``_validate_holdouts`` in ``_run_calibration``)
    so the panel can still plot measured-vs-predicted arrows for a run that
    didn't pass -- otherwise a failed calibration has nothing to show even
    though the fit and observations that failed validation still exist.
    Never returned as an actual result: it is not applied to core, not
    persisted, and its "stage_returned" is unknown at this point.
    """
    raw_matrix, raw_size, _ = normalize_for_mmcore(
        fit.matrix, binning=fingerprint.binning, magnification=fingerprint.magnification
    )
    return PixelCalibrationResult(
        fit=fit,
        raw_matrix=raw_matrix,
        raw_pixel_size_um=raw_size,
        fingerprint=fingerprint,
        observations=tuple(observations),
        validation_observations=tuple(validation),
        stage_returned=False,
    )


def _validate_fit(
    fit: Any, observations: list[CalibrationObservation], options: CalibrationOptions
) -> None:
    median_shift = float(
        np.median([np.linalg.norm(obs.corrected_shift_xy) for obs in observations])
    )
    rms_limit = max(options.max_fit_rms_px, options.max_fit_fraction * median_shift)
    max_limit = max(
        options.max_point_residual_px,
        options.max_point_residual_fraction * median_shift,
    )
    if fit.rms_residual_px > rms_limit or fit.max_residual_px > max_limit:
        raise PixelCalibrationError(
            "Affine fit residuals exceed the calibration quality threshold: "
            f"RMS {fit.rms_residual_px:.3f} px (limit {rms_limit:.3f}), "
            f"worst {fit.max_residual_px:.3f} px (limit {max_limit:.3f})"
        )
    if fit.anisotropy > 0.10:
        raise PixelCalibrationError("Affine fit has more than 10% pixel anisotropy")
    if fit.nonorthogonality_deg > 5:
        raise PixelCalibrationError(
            "Affine fit camera axes are more than 5° from orthogonal"
        )


def _validate_holdouts(
    matrix: NDArray[np.float64],
    observations: list[CalibrationObservation],
    options: CalibrationOptions,
) -> None:
    if len(observations) < 3 or not all(obs.accepted for obs in observations):
        raise PixelCalibrationError("One or more holdout registrations failed")
    inverse = np.linalg.inv(matrix)
    residuals: list[float] = []
    shifts: list[float] = []
    for observation in observations:
        shift = np.asarray(observation.corrected_shift_xy)
        delta = np.asarray(observation.stage_delta_um)
        residuals.append(float(np.linalg.norm(inverse @ (delta - matrix @ shift))))
        shifts.append(float(np.linalg.norm(shift)))
    median_shift = float(np.median(shifts))
    rms_limit = max(options.max_fit_rms_px, options.max_fit_fraction * median_shift)
    max_limit = max(
        options.max_point_residual_px,
        options.max_point_residual_fraction * median_shift,
    )
    rms = float(np.sqrt(np.mean(np.square(residuals))))
    worst = max(residuals)
    if rms > rms_limit or worst > max_limit:
        raise PixelCalibrationError(
            "Holdout prediction residuals exceed the quality threshold: "
            f"RMS {rms:.3f} px (limit {rms_limit:.3f}), "
            f"worst {worst:.3f} px (limit {max_limit:.3f})"
        )


def _run_calibration(
    core: CalibrationCore,
    options: CalibrationOptions,
    *,
    resolution_id: str | None,
    xy_stage: str,
    cancel_event: Event | None,
    progress: ProgressCallback | None,
) -> tuple[PixelCalibrationResult, NDArray[np.float64], HardwareFingerprint]:
    camera = str(core.getCameraDevice())
    if not camera:
        raise PixelCalibrationError("No camera device is selected")
    if _is_acquiring(core, camera):
        raise PixelCalibrationError("Cannot calibrate while an acquisition is running")
    _check_cancel(cancel_event)
    _notify(progress, "reference", 0.0)
    first = _snap(core, camera=camera)
    fingerprint = capture_fingerprint(
        core,
        first,
        resolution_id=resolution_id,
        xy_stage=xy_stage,
    )
    if fingerprint.channel_count != 1:
        raise PixelCalibrationError("Automatic calibration requires one camera channel")
    origin = _position(core, fingerprint.xy_stage)
    second = _snap(core, camera=fingerprint.camera)
    reference = _validate_references(first, second, options)

    _notify(progress, "probe-x", 0.05)
    probe_x = _probe_axis(
        core,
        reference,
        origin,
        fingerprint,
        0,
        options,
        cancel_event,
    )
    _notify(progress, "probe-y", 0.10)
    probe_y = _probe_axis(
        core,
        reference,
        origin,
        fingerprint,
        1,
        options,
        cancel_event,
    )
    coarse = fit_affine(
        np.asarray([probe_x.corrected_shift_xy, probe_y.corrected_shift_xy]),
        np.asarray([probe_x.stage_delta_um, probe_y.stage_delta_um]),
        minimum_points=2,
    ).matrix

    origin_position = _position(core, fingerprint.xy_stage)
    origin_shift = np.asarray(
        _registration(
            reference,
            _snap(core, camera=fingerprint.camera),
            options,
        ).shift_xy
    )
    observations: list[CalibrationObservation] = []
    targets = _measurement_targets(coarse, fingerprint, options, validation=False)
    for index, offset in enumerate(targets):
        _notify(progress, "measure", 0.15 + 0.55 * index / len(targets))
        observation, origin_position, origin_shift = _acquire_observation(
            core,
            reference,
            origin,
            origin + offset,
            origin_position,
            origin_shift,
            fingerprint,
            options,
            cancel_event,
        )
        observations.append(observation)
    accepted = [obs for obs in observations if obs.accepted]
    if len(accepted) < 6:
        raise PixelCalibrationError(
            "Fewer than six calibration observations were usable"
        )
    fit = fit_affine(
        np.asarray([obs.corrected_shift_xy for obs in accepted]),
        np.asarray([obs.stage_delta_um for obs in accepted]),
        confidence_weights=_confidence_weights(accepted),
        minimum_points=6,
    )
    try:
        _validate_fit(fit, accepted, options)
    except PixelCalibrationError as exc:
        exc.diagnostics = _diagnostics_snapshot(fit, fingerprint, observations, ())
        raise

    validation: list[CalibrationObservation] = []
    validation_targets = _measurement_targets(
        fit.matrix, fingerprint, options, validation=True
    )
    for index, offset in enumerate(validation_targets):
        _notify(progress, "validate", 0.72 + 0.20 * index / len(validation_targets))
        observation, origin_position, origin_shift = _acquire_observation(
            core,
            reference,
            origin,
            origin + offset,
            origin_position,
            origin_shift,
            fingerprint,
            options,
            cancel_event,
        )
        validation.append(observation)
    try:
        _validate_holdouts(fit.matrix, validation, options)
    except PixelCalibrationError as exc:
        exc.diagnostics = _diagnostics_snapshot(
            fit, fingerprint, observations, validation
        )
        raise

    if not fingerprint_matches(core, fingerprint):
        raise PixelCalibrationError(
            "The optical configuration changed during calibration"
        )
    raw_matrix, raw_size, _ = normalize_for_mmcore(
        fit.matrix,
        binning=fingerprint.binning,
        magnification=fingerprint.magnification,
    )
    warnings: list[CalibrationWarning] = list(fit.warnings)
    try:
        existing = float(core.getPixelSizeUm())
    except Exception:
        existing = 0.0
    if existing > 0 and abs(fit.pixel_size_um - existing) / existing > 0.05:
        warnings.append(
            CalibrationWarning(
                "existing_scale_difference",
                "Measured pixel size differs from the current calibration by more "
                "than 5%.",
            )
        )
    result = PixelCalibrationResult(
        fit=fit,
        raw_matrix=raw_matrix,
        raw_pixel_size_um=raw_size,
        fingerprint=fingerprint,
        observations=tuple(observations),
        validation_observations=tuple(validation),
        stage_returned=False,
        warnings=tuple(warnings),
    )
    _notify(progress, "restore", 0.95)
    return result, origin, fingerprint


def run_pixel_calibration(
    core: CalibrationCore,
    options: CalibrationOptions | None = None,
    *,
    resolution_id: str | None = None,
    xy_stage: str | None = None,
    cancel_event: Event | None = None,
    progress: ProgressCallback | None = None,
) -> PixelCalibrationResult:
    """Run an automatic calibration without modifying MMCore calibration data."""
    selected_options = options or CalibrationOptions()
    stage = str(xy_stage or core.getXYStageDevice())
    origin: NDArray[np.float64] | None = None
    result: PixelCalibrationResult | None = None
    failure: BaseException | None = None
    try:
        if stage:
            origin = _position(core, stage)
        result, measured_origin, _ = _run_calibration(
            core,
            selected_options,
            resolution_id=resolution_id,
            xy_stage=stage,
            cancel_event=cancel_event,
            progress=progress,
        )
        origin = measured_origin
    except BaseException as error:
        failure = error

    restore_error: BaseException | None = None
    if stage and origin is not None:
        try:
            core.setXYPosition(stage, float(origin[0]), float(origin[1]))
            core.waitForDevice(stage)
            if selected_options.settle_time_s:
                time.sleep(selected_options.settle_time_s)
            returned = _position(core, stage)
            if (
                np.linalg.norm(returned - origin)
                > selected_options.stage_return_tolerance_um
            ):
                raise PixelCalibrationError(
                    "XY stage did not return within the configured tolerance"
                )
        except BaseException as error:
            restore_error = error

    if restore_error is not None:
        raise StageRestoreError(restore_error, failure) from restore_error
    if failure is not None:
        raise failure.with_traceback(failure.__traceback__)
    if result is None:
        raise PixelCalibrationError("Calibration produced no result")
    _notify(progress, "complete", 1.0)
    return replace(result, stage_returned=True)
