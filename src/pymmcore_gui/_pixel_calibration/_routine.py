"""Automatic pixel-size calibration, ported from Micro-Manager's Java calibrator.

The measurement is a port of ``AutomaticCalibrationThread.java`` plus the
helpers it calls -- ``ImageUtils.crossCorrelate``,
``MathFunctions.generateAffineTransformFromPointPairs``,
``AffineUtils.affineToMeasurements`` and ``AffineUtils.deducePixelSize`` --
under ``mmstudio/src/main/java/org/micromanager/internal/`` in the
Micro-Manager repository (BSD licensed; Arthur Edelstein and Nico Stuurman).

It reproduces the Java routine's strategy and its acceptance criteria: two
axis probes, a three-point first estimate, four corner measurements, and a
single 5 px RMS scatter tolerance. There is deliberately no repeat-capture
agreement, no drift correction, no per-image confidence metric and no
independent holdout stage; a result that passes the scatter check is returned
for the caller to accept or discard, the way the Java dialog asks.

The numerics use NumPy rather than ImageJ and BoofCV. An FFT cross-correlation
replaces ImageJ's FHT ``conjugateMultiply`` (the same operation), and a
separable Catmull-Rom resize replaces ``ImageProcessor.resize`` with
``BICUBIC``, using ImageJ's centre-aligned source mapping so that a zero
displacement maps to exactly zero.

Because the Java routine fits absolute stage positions with a translation
term and then discards that translation, the 2 x 2 linear part it keeps has
the meaning ``AffineFitResult.matrix`` carries here: ``stage_delta_um =
matrix @ image_shift_xy``, where the image shift is the negative of the
apparent motion of sample features.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from ._fit import _diagnostic_warnings, normalize_for_mmcore
from ._models import (
    AffineFitResult,
    CalibrationCancelled,
    CalibrationObservation,
    HardwareFingerprint,
    PixelCalibrationError,
    PixelCalibrationResult,
    StageRestoreError,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from threading import Event

    from numpy.typing import ArrayLike, NDArray

ProgressCallback = Callable[[str, float], None]
ObservationCallback = Callable[[CalibrationObservation, str], None]
FitCallback = Callable[[AffineFitResult], None]


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


def _notify_observation(
    callback: ObservationCallback | None,
    observation: CalibrationObservation,
    kind: str,
) -> None:
    if callback is None:
        return
    try:
        callback(observation, kind)
    except Exception:
        # UI/diagnostic reporting must not be able to interrupt stage restoration.
        pass


def _notify_fit(callback: FitCallback | None, fit: AffineFitResult) -> None:
    if callback is None:
        return
    try:
        callback(fit)
    except Exception:
        # UI/diagnostic reporting must not be able to interrupt stage restoration.
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
    config_settings: Sequence[tuple[str, str, str]] | None = None,
    require_resolution_match: bool = True,
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
    if resolution_id and require_resolution_match and current_config != resolution_id:
        raise PixelCalibrationError(
            f"Pixel-size configuration {resolution_id!r} is not the current match"
        )
    if config_settings is None:
        fingerprint_settings = _config_settings(core, resolution_id)
    else:
        fingerprint_settings = tuple(sorted(config_settings))
        for device, prop, expected in fingerprint_settings:
            actual = str(core.getProperty(device, prop))
            if actual != expected:
                raise PixelCalibrationError(
                    f"Hardware does not match resolution {resolution_id!r}: "
                    f"{device}-{prop} is {actual!r}, expected {expected!r}"
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
        config_settings=fingerprint_settings,
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


def _as_float_image(image: ArrayLike) -> NDArray[np.float32]:
    """Convert a monochrome or RGB(A) image to a finite 2D float image."""
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        # ITU-R BT.709 luminance. Alpha is deliberately ignored.
        array = np.tensordot(
            array[..., :3], np.asarray((0.2126, 0.7152, 0.0722)), axes=([-1], [0])
        )
    if array.ndim != 2:
        raise ValueError("calibration images must be 2D grayscale or RGB(A)")
    if min(array.shape) < 16:
        raise ValueError("calibration images must be at least 16 pixels per axis")
    result = np.asarray(array, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("calibration images must contain only finite values")
    return result


ALGORITHM_VERSION = "v1"


@dataclass(frozen=True)
class CalibrationOptions:
    """Motion and numerical settings for the ported Java routine.

    The defaults are the Java ones: a 0.1 um first probe step, 100 ms settle,
    a 64 px correlation box upsampled tenfold, a 5 px RMS acceptance limit,
    and the Pixel Calibrator dialog's smallest safe travel radius (1000 um).

    ``safe_radius_um`` means what it does in ``CalibrationOptions``: no target
    may sit further than this from the starting position. Java instead rejects
    any single move longer than half its configured radius, measured from the
    current position, which bounds step size but not total excursion. The
    default here is Java's smallest dialog choice, which is ten times the other
    routine's 100 um default because the corner geometry travels much further.
    """

    safe_radius_um: float = 1000.0
    settle_time_s: float = 0.1
    initial_step_um: float = 0.1
    max_search_steps: int = 25
    box_size: int = 64
    upsample_factor: int = 10
    max_rms_px: float = 5.0
    stage_return_tolerance_um: float = 0.5

    def __post_init__(self) -> None:
        for option in fields(self):
            value = getattr(self, option.name)
            if not math.isfinite(value):
                raise ValueError(f"{option.name} must be finite")
        for name in ("max_search_steps", "box_size", "upsample_factor"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
        if self.safe_radius_um <= 0:
            raise ValueError("safe_radius_um must be positive")
        if self.settle_time_s < 0:
            raise ValueError("settle_time_s cannot be negative")
        if self.initial_step_um <= 0:
            raise ValueError("initial_step_um must be positive")
        if self.max_search_steps < 1:
            raise ValueError("max_search_steps must be at least 1")
        if self.box_size < 4 or self.box_size % 2:
            raise ValueError("box_size must be an even number of at least 4")
        if self.upsample_factor < 1:
            raise ValueError("upsample_factor must be at least 1")
        if self.max_rms_px <= 0:
            raise ValueError("max_rms_px must be positive")
        if self.stage_return_tolerance_um < 0:
            raise ValueError("stage_return_tolerance_um cannot be negative")


def _smallest_power_of_two_at_most(value: int) -> int:
    """Return the largest power of two at most ``value``."""
    if value < 1:
        raise PixelCalibrationError(
            "The camera image is too small to calibrate: each axis must be at "
            "least 64 pixels."
        )
    return 1 << math.floor(math.log2(value))


def _sub_image(
    image: NDArray[np.floating], x: int, y: int, width: int, height: int
) -> NDArray[np.float64]:
    """Zero-padded crop, matching Java's ``getSubImage``.

    Java builds a ``FloatProcessor(w, h)`` and inserts the source shifted by
    ``(-x, -y)``, so a requested region reaching past the image edge is padded
    with zeros rather than being clipped back inside the frame. The other
    routine clips the crop position instead; this difference matters for the
    corner measurements, which deliberately sit close to the frame edge.
    """
    source = np.asarray(image, dtype=np.float64)
    out = np.zeros((height, width), dtype=np.float64)
    source_h, source_w = source.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + width, source_w), min(y + height, source_h)
    if x1 > x0 and y1 > y0:
        out[y0 - y : y1 - y, x0 - x : x1 - x] = source[y0:y1, x0:x1]
    return out


def _subtract_minimum(patch: NDArray[np.floating]) -> NDArray[np.float64]:
    """Java's ``subtractMinimum``: remove the patch's own offset."""
    result = np.asarray(patch, dtype=np.float64)
    return result - float(result.min())


def cross_correlate(
    reference: NDArray[np.floating], moving: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Unnormalized Fourier cross-correlation with the zero lag at the centre.

    Equivalent to ImageJ's ``FHT.conjugateMultiply`` followed by an inverse
    transform and ``swapQuadrants``, which is what ``ImageUtils.crossCorrelate``
    does.
    """
    first = np.asarray(reference, dtype=np.float64)
    second = np.asarray(moving, dtype=np.float64)
    if first.shape != second.shape:
        raise PixelCalibrationError(
            "Cross-correlation needs two images of identical shape"
        )
    spectrum = np.fft.fft2(first) * np.conj(np.fft.fft2(second))
    return np.asarray(np.fft.fftshift(np.fft.ifft2(spectrum).real), dtype=np.float64)


def _cubic(distance: NDArray[np.float64]) -> NDArray[np.float64]:
    """ImageJ's ``ImageProcessor.cubic`` kernel with ``a = 0.5``."""
    a = 0.5
    x = np.abs(distance)
    out = np.zeros_like(x)
    near = x < 1
    out[near] = x[near] ** 2 * (x[near] * (2 - a) + (a - 3)) + 1
    far = (x >= 1) & (x < 2)
    out[far] = -a * x[far] ** 3 + 5 * a * x[far] ** 2 - 8 * a * x[far] + 4 * a
    return out


def _resample_matrix(source_size: int, factor: int) -> NDArray[np.float64]:
    """Build the 1-D Catmull-Rom resampling matrix ImageJ's resize implies.

    ImageJ maps a destination index to the source as
    ``(i - dst_size / 2) / scale + source_size / 2``, which for
    ``dst_size == source_size * factor`` and ``scale == factor`` reduces to
    ``i / factor``. That centre alignment is what makes the caller's
    ``peak - dst_size / 2`` arithmetic land on zero for a zero displacement.
    """
    destination_size = source_size * factor
    destination = np.arange(destination_size)
    coordinate = destination / factor
    base = np.floor(coordinate).astype(np.int64)
    fraction = coordinate - base
    offsets = np.asarray([-1, 0, 1, 2])
    weights = _cubic(fraction[:, None] - offsets[None, :])
    # ImageJ clamps sampling at the patch edges; duplicated clamped indices
    # accumulate so each destination row still sums to one.
    indices = np.clip(base[:, None] + offsets[None, :], 0, source_size - 1)
    matrix = np.zeros((destination_size, source_size), dtype=np.float64)
    rows = np.repeat(destination, offsets.size).astype(np.intp)
    columns = indices.ravel().astype(np.intp)
    np.add.at(matrix, (rows, columns), weights.ravel())
    return matrix


def _upsample(patch: NDArray[np.floating], factor: int) -> NDArray[np.float64]:
    """Bicubic enlargement of a correlation patch, as Java does before argmax."""
    values = np.asarray(patch, dtype=np.float64)
    if factor == 1:
        return values
    rows = _resample_matrix(values.shape[0], factor)
    columns = _resample_matrix(values.shape[1], factor)
    return np.asarray(rows @ values @ columns.T, dtype=np.float64)


def measure_displacement(
    reference_patch: NDArray[np.floating],
    found_patch: NDArray[np.floating],
    *,
    box_size: int = 64,
    upsample_factor: int = 10,
) -> tuple[float, float]:
    """Java's static ``measureDisplacement``: correlate, enlarge, take the peak.

    Returns the geometric ``(x, y)`` displacement in original pixels, quantized
    to ``1 / upsample_factor``. The sign convention is Java's: the value is the
    negative of the apparent motion of sample features, which is what the
    fitted matrix maps to stage micrometres.
    """
    correlation = cross_correlate(reference_patch, found_patch)
    height, width = correlation.shape
    # Java always crops a fixed 64 px box, which is safe there because its
    # patch is a power of two of at least 64 for any camera 256 px or larger
    # (512 px on a 2048 px camera). For a smaller patch that box would extend
    # past the correlation and be zero-padded, and because a min-subtracted
    # correlation sits on a large positive pedestal, the bicubic kernel's
    # negative lobes overshoot at the padding edge and can beat the real peak.
    # Clamping keeps Java's behaviour for every realistic image and removes
    # that artefact for small ones.
    box = min(box_size, height, width)
    half = box // 2
    center = _sub_image(correlation, width // 2 - half, height // 2 - half, box, box)
    scaled = _upsample(center, upsample_factor)
    row, column = np.unravel_index(int(np.argmax(scaled)), scaled.shape)
    x = (float(column) - scaled.shape[1] / 2) / upsample_factor
    y = (float(row) - scaled.shape[0] / 2) / upsample_factor
    return x, y


def fit_affine_with_translation(
    image_points: NDArray[np.floating], stage_points: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Least-squares 2 x 3 fit of ``stage = matrix @ [px, py, 1]``.

    This is ``MathFunctions.generateAffineTransformFromPointPairs`` without its
    tolerance check; Java solves the same system by QR decomposition.
    """
    pixels = np.asarray(image_points, dtype=np.float64)
    stage = np.asarray(stage_points, dtype=np.float64)
    if pixels.shape != stage.shape or pixels.ndim != 2 or pixels.shape[1] != 2:
        raise PixelCalibrationError("Point pairs must be two matching N x 2 arrays")
    if len(pixels) < 3:
        raise PixelCalibrationError("At least three point pairs are required")
    design = np.column_stack([pixels, np.ones(len(pixels))])
    solution, _residuals, rank, _singular = np.linalg.lstsq(design, stage, rcond=None)
    if rank < 3:
        raise PixelCalibrationError(
            "Calibration point pairs are degenerate; the stage moves did not "
            "produce independent image displacements"
        )
    matrix = np.asarray(solution.T, dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise PixelCalibrationError("Affine fit produced non-finite coefficients")
    determinant = float(np.linalg.det(matrix[:, :2]))
    if not math.isfinite(determinant) or abs(determinant) <= np.finfo(float).eps:
        raise PixelCalibrationError("Singular matrix encountered")
    return matrix


def _scatter_rms_px(
    matrix: NDArray[np.float64],
    image_points: NDArray[np.float64],
    stage_points: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    """Java's source-domain RMS: inverse-map the stage points back to pixels."""
    linear = matrix[:, :2]
    translation = matrix[:, 2]
    inverse = np.linalg.inv(linear)
    predicted = (stage_points - translation) @ inverse.T
    residuals = predicted - image_points
    norms = np.linalg.norm(residuals, axis=1)
    return float(np.sqrt(np.mean(np.square(norms)))), residuals


def affine_to_measurements(
    matrix: NDArray[np.floating],
) -> tuple[float, float, float, float]:
    """Port of ``AffineUtils.affineToMeasurements``.

    Returns ``(x_scale, y_scale, rotation_deg, shear)`` -- the same four numbers
    the Java Pixel Calibrator shows in its "Calibration succeeded!" dialog.
    """
    linear = np.asarray(matrix, dtype=np.float64)[:2, :2]
    m00, _m01 = float(linear[0, 0]), float(linear[0, 1])
    m10, _m11 = float(linear[1, 0]), float(linear[1, 1])
    if m00 == 0 and m10 == 0:
        return 0.0, 0.0, 0.0, 0.0
    angle = math.atan(m10 / m00) if m00 != 0 else math.copysign(math.pi / 2, m10)
    # Java's quadrant fixups, kept verbatim so reflections report as it does.
    if m10 > 0 and m00 >= 0:
        angle = abs(angle)
    elif m10 > 0 and m00 < 0:
        angle = abs(angle - 2 * (math.pi / 2 + angle))
    elif m10 <= 0 and m00 >= 0:
        pass
    else:
        angle += 2 * (math.pi / 2 - angle)
        angle *= -1
    cos, sin = math.cos(angle), math.sin(angle)
    # Inverse of Java's getRotateInstance(angle), pre-multiplied as
    # at = R(angle)^-1 . transform
    unrotated = np.asarray([[cos, sin], [-sin, cos]], dtype=np.float64) @ linear
    n00, n01 = float(unrotated[0, 0]), float(unrotated[0, 1])
    n10, n11 = float(unrotated[1, 0]), float(unrotated[1, 1])
    x_scale = math.hypot(n00, n10) * (1.0 if n00 > 0 else -1.0)
    y_scale = math.hypot(n01, n11) * (1.0 if n11 > 0 else -1.0)
    if x_scale == 0 or y_scale == 0:
        return x_scale, y_scale, math.degrees(angle), 0.0
    unscaled = np.diag([1.0 / x_scale, 1.0 / y_scale]) @ unrotated
    return x_scale, y_scale, math.degrees(angle), float(unscaled[0, 1])


def deduce_pixel_size(matrix: NDArray[np.floating]) -> float:
    """Port of ``AffineUtils.deducePixelSize``, including its 4-digit rounding."""
    linear = np.asarray(matrix, dtype=np.float64)[:2, :2]
    return round(float(np.sqrt(abs(np.linalg.det(linear)))), 4)


def _describe_fit(
    matrix: NDArray[np.float64],
    image_shifts: NDArray[np.float64],
    residuals_px: NDArray[np.float64],
) -> AffineFitResult:
    """Package the least-squares 2 x 2 and its diagnostics for the GUI.

    ``residuals_px`` must be the residuals the acceptance check computed, so
    that what the GUI reports and plots is what the run was judged on. Deriving
    them here from origin-relative stage deltas instead would substitute the
    origin for the fitted translation, which shifts every residual by a
    constant and, on a noisy run, roughly doubles the apparent worst value
    while inventing a per-corner spread that is not there.

    Every corner contributes equally: the Java fit applies no weighting and
    removes no outliers.
    """
    determinant = float(np.linalg.det(matrix))
    residuals_px = np.asarray(residuals_px, dtype=np.float64)
    residuals_um = residuals_px @ matrix.T
    residual_norms_px = np.linalg.norm(residuals_px, axis=1)
    pixel_size_x = float(np.linalg.norm(matrix[:, 0]))
    pixel_size_y = float(np.linalg.norm(matrix[:, 1]))
    singular = np.linalg.svd(matrix, compute_uv=False)
    anisotropy = abs(pixel_size_x - pixel_size_y) / (
        0.5 * (pixel_size_x + pixel_size_y)
    )
    cosine = float(np.dot(matrix[:, 0], matrix[:, 1]) / (pixel_size_x * pixel_size_y))
    axis_angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    nonorthogonality = abs(90.0 - axis_angle)
    matrix_condition = float(np.linalg.cond(matrix))
    return AffineFitResult(
        matrix=matrix,
        residuals_um=residuals_um,
        residuals_px=residuals_px,
        pixel_size_um=float(np.sqrt(abs(determinant))),
        pixel_size_x_um=pixel_size_x,
        pixel_size_y_um=pixel_size_y,
        singular_values=(float(singular[0]), float(singular[1])),
        design_condition=float(np.linalg.cond(image_shifts)),
        matrix_condition=matrix_condition,
        anisotropy=float(anisotropy),
        nonorthogonality_deg=float(nonorthogonality),
        rotation_deg=float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))),
        determinant=determinant,
        rms_residual_px=float(np.sqrt(np.mean(np.square(residual_norms_px)))),
        max_residual_px=float(np.max(residual_norms_px)),
        warnings=_diagnostic_warnings(
            float(anisotropy), float(nonorthogonality), matrix_condition
        ),
    )


def _observation(
    image_shift: NDArray[np.float64],
    stage_position: NDArray[np.float64],
    origin: NDArray[np.float64],
    label: str = "",
    residual_px: NDArray[np.float64] | None = None,
) -> CalibrationObservation:
    """Wrap one measurement for the diagnostics graph."""
    delta = stage_position - origin
    residual = (
        None if residual_px is None else (float(residual_px[0]), float(residual_px[1]))
    )
    return CalibrationObservation(
        stage_position_um=(float(stage_position[0]), float(stage_position[1])),
        stage_delta_um=(float(delta[0]), float(delta[1])),
        image_shift_xy=(float(image_shift[0]), float(image_shift[1])),
        residual_px=residual,
        label=label,
    )


def _first_snap(
    core: CalibrationCore, camera: str, cancel_event: Event | None
) -> NDArray[np.generic]:
    """Acquire the base frame, retrying a transient camera error twice.

    Java takes a single base image; the bounded retry here only covers a
    camera that throws, which would otherwise abort before any measurement.
    """
    last_error: Exception | None = None
    for _attempt in range(3):
        _check_cancel(cancel_event)
        try:
            return _snap(core, camera=camera)
        except PixelCalibrationError:
            raise
        except Exception as exc:
            last_error = exc
    raise PixelCalibrationError(
        f"Camera failed after repeated snap attempts: {last_error}"
    ) from last_error


class _Tracker:
    """Holds the reference patch and the geometry Java keeps in fields."""

    def __init__(
        self,
        core: CalibrationCore,
        base_image: ArrayLike,
        fingerprint: HardwareFingerprint,
        options: CalibrationOptions,
        cancel_event: Event | None,
        origin: NDArray[np.float64],
    ) -> None:
        self._core = core
        self._fingerprint = fingerprint
        self._options = options
        self._cancel_event = cancel_event
        self._origin = np.asarray(origin, dtype=np.float64)
        frame = _as_float_image(base_image)
        self.height, self.width = frame.shape
        side = min(
            _smallest_power_of_two_at_most(self.width // 4),
            _smallest_power_of_two_at_most(self.height // 4),
        )
        if side < 16:
            raise PixelCalibrationError(
                "The camera image is too small to track a calibration patch: "
                "the tracked region would be under 16 pixels across."
            )
        self.side = side
        self.reference = _subtract_minimum(
            _sub_image(
                frame,
                -side // 2 + self.width // 2,
                -side // 2 + self.height // 2,
                side,
                side,
            )
        )

    def snap_at(self, target: NDArray[np.float64]) -> NDArray[np.float32]:
        """Java's ``snapImageAt``: safety check, move, settle, snap."""
        _check_cancel(self._cancel_event)
        if _is_acquiring(self._core, self._fingerprint.camera):
            raise PixelCalibrationError("An acquisition started during calibration")
        # Java instead rejects a single move longer than half the configured
        # radius, measured from the current position, which never bounds how
        # far the stage gets from where it started. Bounding the excursion from
        # the origin is what actually protects the objective and specimen, and
        # it is what the GUI's "Safe radius" control promises.
        excursion = float(np.linalg.norm(target - self._origin))
        if excursion > self._options.safe_radius_um:
            raise PixelCalibrationError(
                f"XY stage safety limit reached: the next calibration target is "
                f"{excursion:.3g} µm from the starting position, beyond the "
                f"{self._options.safe_radius_um:.3g} µm safe radius. Increase the "
                f"safe radius to calibrate at this magnification."
            )
        self._core.setXYPosition(
            self._fingerprint.xy_stage, float(target[0]), float(target[1])
        )
        self._core.waitForDevice(self._fingerprint.xy_stage)
        if self._options.settle_time_s:
            time.sleep(self._options.settle_time_s)
        return _as_float_image(_snap(self._core, camera=self._fingerprint.camera))

    def measure(
        self, target: NDArray[np.float64], expected_shift: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Java's instance ``measureDisplacement``: track the patch, correlate."""
        frame = self.snap_at(target)
        if frame.shape != (self.height, self.width):
            raise PixelCalibrationError("The camera image size changed mid-calibration")
        found = _subtract_minimum(
            _sub_image(
                frame,
                int((self.width - self.side) // 2 - float(expected_shift[0])),
                int((self.height - self.side) // 2 - float(expected_shift[1])),
                self.side,
                self.side,
            )
        )
        change = measure_displacement(
            self.reference,
            found,
            box_size=self._options.box_size,
            upsample_factor=self._options.upsample_factor,
        )
        return np.asarray(
            [expected_shift[0] + change[0], expected_shift[1] + change[1]],
            dtype=np.float64,
        )

    def run_search(
        self,
        origin: NDArray[np.float64],
        step_x: float,
        step_y: float,
        progress: ProgressCallback | None,
        progress_span: tuple[float, float],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Java's ``runSearch``: double the move until the patch nears the edge."""
        dx, dy = step_x, step_y
        shift = np.zeros(2, dtype=np.float64)
        start, end = progress_span
        for step in range(self._options.max_search_steps):
            if (
                (2.0 * shift[0] + self.side / 2.0) >= self.width / 2.0
                or (2.0 * shift[1] + self.side / 2.0) >= self.height / 2.0
                or (2.0 * shift[0] - self.side / 2.0) < -(self.width / 2.0)
                or (2.0 * shift[1] - self.side / 2.0) < -(self.height / 2.0)
            ):
                break
            dx *= 2
            dy *= 2
            shift = shift * 2
            shift = self.measure(origin + np.asarray([dx, dy]), shift)
            _notify(
                progress,
                "probe",
                start + (end - start) * (step + 1) / self._options.max_search_steps,
            )
        return shift, _position(self._core, self._fingerprint.xy_stage)


def _run_calibration(
    core: CalibrationCore,
    options: CalibrationOptions,
    *,
    resolution_id: str | None,
    xy_stage: str,
    cancel_event: Event | None,
    progress: ProgressCallback | None,
    observation_callback: ObservationCallback | None,
    fit_callback: FitCallback | None,
    config_settings: Sequence[tuple[str, str, str]] | None,
    require_resolution_match: bool,
) -> tuple[PixelCalibrationResult, NDArray[np.float64]]:
    camera = str(core.getCameraDevice())
    if not camera:
        raise PixelCalibrationError("No camera device is selected")
    if _is_acquiring(core, camera):
        raise PixelCalibrationError("Cannot calibrate while an acquisition is running")
    _check_cancel(cancel_event)

    _notify(progress, "reference", 0.0)
    first_frame = _first_snap(core, camera, cancel_event)
    fingerprint = capture_fingerprint(
        core,
        first_frame,
        resolution_id=resolution_id,
        xy_stage=xy_stage,
        config_settings=config_settings,
        require_resolution_match=require_resolution_match,
    )
    if fingerprint.channel_count != 1:
        raise PixelCalibrationError("Automatic calibration requires one camera channel")

    origin = _position(core, fingerprint.xy_stage)
    tracker = _Tracker(
        core, first_frame, fingerprint, options, cancel_event, origin=origin
    )

    # --- first approximation: origin plus one probe per stage axis ----------
    image_points: list[NDArray[np.float64]] = [np.zeros(2, dtype=np.float64)]
    stage_points: list[NDArray[np.float64]] = [origin.copy()]

    _notify(progress, "probe-x", 0.05)
    shift, position = tracker.run_search(
        origin, options.initial_step_um, 0.0, progress, (0.05, 0.35)
    )
    image_points.append(shift)
    stage_points.append(position)
    _notify_observation(
        observation_callback, _observation(shift, position, origin), "probe"
    )

    _notify(progress, "probe-y", 0.35)
    shift, position = tracker.run_search(
        origin, 0.0, options.initial_step_um, progress, (0.35, 0.65)
    )
    image_points.append(shift)
    stage_points.append(position)
    _notify_observation(
        observation_callback, _observation(shift, position, origin), "probe"
    )

    first_approx = fit_affine_with_translation(
        np.asarray(image_points), np.asarray(stage_points)
    )

    # --- second approximation: four corners, well inside the frame ---------
    ax = tracker.width // 2 - tracker.side
    ay = tracker.height // 2 - tracker.side
    if ax <= 0 or ay <= 0:
        raise PixelCalibrationError(
            "The camera image is too small to place the corner measurements "
            "outside the tracked region."
        )
    corners = [(-ax, -ay), (-ax, ay), (ax, ay), (ax, -ay)]

    corner_shifts: list[NDArray[np.float64]] = []
    corner_positions: list[NDArray[np.float64]] = []
    for index, corner in enumerate(corners):
        _notify(progress, "measure", 0.65 + 0.3 * index / len(corners))
        expected = np.asarray(corner, dtype=np.float64)
        predicted_stage = first_approx @ np.asarray([expected[0], expected[1], 1.0])
        measured = tracker.measure(predicted_stage, expected)
        actual = _position(core, fingerprint.xy_stage)
        corner_shifts.append(measured)
        corner_positions.append(actual)
        _notify_observation(
            observation_callback,
            _observation(measured, actual, origin, str(index + 1)),
            "corner",
        )

    corner_image = np.asarray(corner_shifts)
    corner_stage = np.asarray(corner_positions)
    second_approx = fit_affine_with_translation(corner_image, corner_stage)
    rms_px, residuals_px = _scatter_rms_px(second_approx, corner_image, corner_stage)

    # Java zeroes the translation before handing the transform on; what is left
    # maps image displacement to stage displacement.
    matrix = np.asarray(second_approx[:, :2], dtype=np.float64)
    fit = _describe_fit(matrix, corner_image, residuals_px)
    _notify_fit(fit_callback, fit)

    _notify(progress, "finalize", 0.95)
    raw_matrix, raw_size, _flat = normalize_for_mmcore(
        matrix, binning=fingerprint.binning, magnification=fingerprint.magnification
    )
    observations = tuple(
        _observation(shift, position, origin, str(index + 1), residual)
        for index, (shift, position, residual) in enumerate(
            zip(corner_shifts, corner_positions, residuals_px, strict=True)
        )
    )
    result = PixelCalibrationResult(
        fit=fit,
        raw_matrix=raw_matrix,
        raw_pixel_size_um=raw_size,
        fingerprint=fingerprint,
        observations=observations,
        stage_returned=False,
        algorithm_version=ALGORITHM_VERSION,
        warnings=fit.warnings,
        max_rms_px=options.max_rms_px,
    )

    # The scatter check runs last so a rejected run still carries its fit: the
    # pixel size it measured is worth showing even though it is not worth
    # applying, and the corner residuals are what explain the rejection.
    if rms_px > options.max_rms_px:
        raise PixelCalibrationError(
            f"Point mapping scatter exceeds tolerance: RMS {rms_px:.3f} px "
            f"(limit {options.max_rms_px:.3f} px). Improve contrast and focus, "
            f"and make sure the specimen cannot move on the stage.",
            diagnostics=result,
        )
    return result, origin


def run_pixel_calibration(
    core: CalibrationCore,
    options: CalibrationOptions | None = None,
    *,
    resolution_id: str | None = None,
    xy_stage: str | None = None,
    cancel_event: Event | None = None,
    progress: ProgressCallback | None = None,
    observation_callback: ObservationCallback | None = None,
    fit_callback: FitCallback | None = None,
    config_settings: Sequence[tuple[str, str, str]] | None = None,
    require_resolution_match: bool = True,
) -> PixelCalibrationResult:
    """Measure the image-to-stage affine using the ported Java routine.

    A result that passes the 5 px RMS scatter tolerance is returned without
    further validation: there is no holdout stage, no anisotropy or
    orthogonality limit, and no pixel-size comparison against the stored
    calibration. The caller decides whether to keep it, the way the Java
    dialog asks the user.

    The stage is always commanded back to its starting position afterwards. A
    return that lands outside ``stage_return_tolerance_um`` is reported as
    ``stage_returned=False`` rather than raising, because the Java routine does
    not verify the return and an open-loop stage can easily miss by more than
    the tolerance without the measurement being wrong.
    """
    selected = options or CalibrationOptions()
    stage = str(xy_stage or core.getXYStageDevice())
    _check_cancel(cancel_event)
    camera = str(core.getCameraDevice())
    if not camera:
        raise PixelCalibrationError("No camera device is selected")
    if not stage:
        raise PixelCalibrationError("No XY stage device is selected")
    if _is_acquiring(core, camera):
        raise PixelCalibrationError("Cannot calibrate while an acquisition is running")

    origin: NDArray[np.float64] | None = None
    result: PixelCalibrationResult | None = None
    failure: BaseException | None = None
    try:
        origin = _position(core, stage)
        result, origin = _run_calibration(
            core,
            selected,
            resolution_id=resolution_id,
            xy_stage=stage,
            cancel_event=cancel_event,
            progress=progress,
            observation_callback=observation_callback,
            fit_callback=fit_callback,
            config_settings=config_settings,
            require_resolution_match=require_resolution_match,
        )
    except BaseException as error:
        failure = error

    returned = False
    restore_error: BaseException | None = None
    if origin is not None:
        try:
            core.setXYPosition(stage, float(origin[0]), float(origin[1]))
            core.waitForDevice(stage)
            if selected.settle_time_s:
                time.sleep(selected.settle_time_s)
            actual = _position(core, stage)
            returned = bool(
                np.linalg.norm(actual - origin) <= selected.stage_return_tolerance_um
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
    return replace(result, stage_returned=returned)
