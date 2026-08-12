from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from ._models import RegistrationResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray


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


def _crop_center(image: NDArray[np.float32], fraction: float) -> NDArray[np.float32]:
    if not 0.25 <= fraction <= 1:
        raise ValueError("crop_fraction must be between 0.25 and 1")
    height, width = image.shape
    crop_h = max(16, int(np.floor(height * fraction)))
    crop_w = max(16, int(np.floor(width * fraction)))
    y0 = (height - crop_h) // 2
    x0 = (width - crop_w) // 2
    return image[y0 : y0 + crop_h, x0 : x0 + crop_w]


def _prepare_image(
    image: NDArray[np.float32], *, crop_fraction: float
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    cropped = _crop_center(image, crop_fraction)
    low, high = np.percentile(cropped, (0.1, 99.9))
    span = float(high - low)
    scale = max(float(np.max(np.abs(cropped))), 1.0)
    if not np.isfinite(span) or span <= np.finfo(np.float32).eps * scale:
        raise ValueError("calibration image has insufficient intensity variation")
    normalized = np.clip(cropped, low, high)
    normalized = (normalized - np.median(normalized)) / span
    normalized = np.asarray(normalized, dtype=np.float32)
    window = np.outer(np.hanning(normalized.shape[0]), np.hanning(normalized.shape[1]))
    windowed = normalized * np.asarray(window, dtype=np.float32)
    return normalized, windowed


def _upsampled_dft(
    data: NDArray[np.complexfloating],
    upsampled_region_size: int | Sequence[int],
    *,
    upsample_factor: int,
    axis_offsets: Sequence[float],
) -> NDArray[np.complex128]:
    """Evaluate a small upsampled DFT region by matrix multiplication.

    This is the matrix-multiply DFT refinement described by Guizar-Sicairos,
    Thurman, and Fienup (Optics Letters 33, 156-158, 2008).
    """
    ndim = data.ndim
    if isinstance(upsampled_region_size, int):
        region = (upsampled_region_size,) * ndim
    else:
        region = tuple(upsampled_region_size)
    offsets = tuple(axis_offsets)
    if len(region) != ndim or len(offsets) != ndim:
        raise ValueError("upsampled DFT dimensions do not match the input")

    result = np.asarray(data, dtype=np.complex128)
    im2pi = 2j * np.pi
    for size, region_size, offset in zip(
        data.shape[::-1], region[::-1], offsets[::-1], strict=True
    ):
        frequencies = np.fft.fftfreq(size, d=upsample_factor)
        kernel = np.exp(
            -im2pi
            * (np.arange(region_size, dtype=np.float64) - offset)[:, None]
            * frequencies[None, :]
        )
        result = np.tensordot(kernel, result, axes=(1, -1))
    return np.asarray(result, dtype=np.complex128)


def _correlation_metrics(
    magnitude: NDArray[np.float64], peak_index: tuple[int, int]
) -> tuple[float, float]:
    height, width = magnitude.shape
    yy, xx = np.ogrid[:height, :width]
    dy = np.minimum(abs(yy - peak_index[0]), height - abs(yy - peak_index[0]))
    dx = np.minimum(abs(xx - peak_index[1]), width - abs(xx - peak_index[1]))
    exclusion_radius = max(2, min(height, width) // 64)
    sidelobe = magnitude[(dx * dx + dy * dy) > exclusion_radius**2]
    if sidelobe.size < 2:
        return 0.0, 1.0
    peak = float(magnitude[peak_index])
    side_mean = float(np.mean(sidelobe))
    side_std = float(np.std(sidelobe))
    psr = (peak - side_mean) / max(side_std, np.finfo(float).eps)
    second_peak = float(np.max(sidelobe))
    peak_ratio = peak / max(second_peak, np.finfo(float).eps)
    return psr, peak_ratio


def _fourier_shift(
    image: NDArray[np.float32], shift_rc: NDArray[np.float64]
) -> NDArray[np.float64]:
    row_freq = np.fft.fftfreq(image.shape[0])[:, None]
    col_freq = np.fft.fftfreq(image.shape[1])[None, :]
    phase = np.exp(-2j * np.pi * (row_freq * shift_rc[0] + col_freq * shift_rc[1]))
    return np.asarray(np.fft.ifftn(np.fft.fftn(image) * phase).real)


def _normalized_alignment_error(
    reference: NDArray[np.float32],
    moving: NDArray[np.float32],
    shift_rc: NDArray[np.float64],
) -> float:
    aligned = _fourier_shift(moving, shift_rc)
    row_margin = int(np.ceil(abs(shift_rc[0]))) + 2
    col_margin = int(np.ceil(abs(shift_rc[1]))) + 2
    if row_margin * 2 >= reference.shape[0] or col_margin * 2 >= reference.shape[1]:
        return float("inf")
    rows = slice(row_margin, reference.shape[0] - row_margin)
    cols = slice(col_margin, reference.shape[1] - col_margin)
    ref_valid = np.asarray(reference[rows, cols], dtype=np.float64)
    mov_valid = aligned[rows, cols]
    # Remove a residual affine intensity change before scoring geometry.
    design = np.column_stack((mov_valid.ravel(), np.ones(mov_valid.size)))
    gain, offset = np.linalg.lstsq(design, ref_valid.ravel(), rcond=None)[0]
    difference = ref_valid - (gain * mov_valid + offset)
    denominator = np.linalg.norm(ref_valid - np.mean(ref_valid))
    return float(np.linalg.norm(difference) / max(denominator, np.finfo(float).eps))


def register_translation(
    reference: ArrayLike,
    moving: ArrayLike,
    *,
    upsample_factor: int = 20,
    crop_fraction: float = 0.75,
    normalization: str = "phase",
) -> RegistrationResult:
    """Measure the shift required to align ``moving`` to ``reference``.

    The returned shift is in geometric ``(x, y)`` order even though NumPy arrays
    use ``(row, column)`` order.
    """
    if upsample_factor < 1:
        raise ValueError("upsample_factor must be at least 1")
    if normalization not in {"phase", "unnormalized"}:
        raise ValueError("normalization must be 'phase' or 'unnormalized'")

    reference_float = _as_float_image(reference)
    moving_float = _as_float_image(moving)
    if reference_float.shape != moving_float.shape:
        raise ValueError("reference and moving images must have the same shape")
    reference_plain, reference_windowed = _prepare_image(
        reference_float, crop_fraction=crop_fraction
    )
    moving_plain, moving_windowed = _prepare_image(
        moving_float, crop_fraction=crop_fraction
    )

    reference_freq = np.fft.fftn(reference_windowed)
    moving_freq = np.fft.fftn(moving_windowed)
    image_product = reference_freq * moving_freq.conj()
    if normalization == "phase":
        magnitude = np.abs(image_product)
        epsilon = 100 * np.finfo(magnitude.dtype).eps
        image_product /= np.maximum(magnitude, epsilon)

    cross_correlation = np.fft.ifftn(image_product)
    coarse_magnitude = np.asarray(np.abs(cross_correlation), dtype=np.float64)
    peak_coordinates = np.unravel_index(
        np.argmax(coarse_magnitude), coarse_magnitude.shape
    )
    peak_index = (int(peak_coordinates[0]), int(peak_coordinates[1]))
    psr, peak_ratio = _correlation_metrics(coarse_magnitude, peak_index)

    shift_rc = np.asarray(peak_index, dtype=np.float64)
    midpoint = np.asarray([np.trunc(size / 2) for size in reference_plain.shape])
    shape = np.asarray(reference_plain.shape)
    shift_rc[shift_rc > midpoint] -= shape[shift_rc > midpoint]

    if upsample_factor > 1:
        shift_rc = np.round(shift_rc * upsample_factor) / upsample_factor
        region_size = int(np.ceil(upsample_factor * 1.5))
        dft_shift = np.trunc(region_size / 2)
        sample_offset = dft_shift - shift_rc * upsample_factor
        refined = _upsampled_dft(
            image_product.conj(),
            region_size,
            upsample_factor=upsample_factor,
            axis_offsets=sample_offset,
        ).conj()
        refined_peak = np.asarray(
            np.unravel_index(np.argmax(np.abs(refined)), refined.shape),
            dtype=np.float64,
        )
        refined_peak -= dft_shift
        shift_rc += refined_peak / upsample_factor

    overlap_y = max(0.0, 1.0 - abs(float(shift_rc[0])) / reference_plain.shape[0])
    overlap_x = max(0.0, 1.0 - abs(float(shift_rc[1])) / reference_plain.shape[1])
    overlap = overlap_y * overlap_x
    error = _normalized_alignment_error(reference_plain, moving_plain, shift_rc)

    method: Literal["phase", "unnormalized"] = (
        "phase" if normalization == "phase" else "unnormalized"
    )
    return RegistrationResult(
        shift_xy=(float(shift_rc[1]), float(shift_rc[0])),
        psr=psr,
        peak_ratio=peak_ratio,
        overlap=overlap,
        normalized_error=error,
        method=method,
    )
