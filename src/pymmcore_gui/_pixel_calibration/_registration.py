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
) -> NDArray[np.float32]:
    cropped = _crop_center(image, crop_fraction)
    low, high = np.percentile(cropped, (0.1, 99.9))
    span = float(high - low)
    scale = max(float(np.max(np.abs(cropped))), 1.0)
    if not np.isfinite(span) or span <= np.finfo(np.float32).eps * scale:
        raise ValueError("calibration image has insufficient intensity variation")
    normalized = np.clip(cropped, low, high)
    normalized = (normalized - np.median(normalized)) / span
    return np.asarray(normalized, dtype=np.float32)


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
    # Ordinary correlation has a broad main lobe for blurred features. Exclude
    # its full width at half maximum, rather than treating its shoulder as a
    # competing peak. Stop at the first crossing so periodic peaks remain scored.
    peak = float(magnitude[peak_index])
    for profile, center in (
        (magnitude[:, peak_index[1]], peak_index[0]),
        (magnitude[peak_index[0], :], peak_index[1]),
    ):
        for direction in (-1, 1):
            indices = (
                center + direction * np.arange(1, profile.size // 4)
            ) % profile.size
            crossings = np.flatnonzero(profile[indices] < peak / 2)
            if crossings.size:
                exclusion_radius = max(exclusion_radius, 2 * int(crossings[0] + 1))
    sidelobe = magnitude[(dx * dx + dy * dy) > exclusion_radius**2]
    if sidelobe.size < 2:
        return 0.0, 1.0
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
    ref_valid = ref_valid - np.mean(ref_valid)
    mov_valid = mov_valid - np.mean(mov_valid)
    gain = max(0.0, float(np.sum(ref_valid * mov_valid))) / max(
        float(np.sum(mov_valid * mov_valid)), np.finfo(float).eps
    )
    difference = ref_valid - gain * mov_valid
    denominator = np.linalg.norm(ref_valid)
    return float(np.linalg.norm(difference) / max(denominator, np.finfo(float).eps))


class TranslationRegistrar:
    """Register many images against one preprocessed reference image."""

    def __init__(
        self,
        reference: ArrayLike,
        *,
        upsample_factor: int = 20,
        crop_fraction: float = 0.75,
        normalization: str = "phase",
    ) -> None:
        if upsample_factor < 1:
            raise ValueError("upsample_factor must be at least 1")
        if normalization not in {"phase", "unnormalized"}:
            raise ValueError("normalization must be 'phase' or 'unnormalized'")
        reference_float = _as_float_image(reference)
        self._reference_shape = reference_float.shape
        self._reference_plain = _prepare_image(
            reference_float, crop_fraction=crop_fraction
        )
        self._window = np.asarray(
            np.outer(*(np.hanning(size) for size in self._reference_plain.shape)),
            dtype=np.float32,
        )
        self._reference_freq = np.fft.fftn(self._reference_plain * self._window)
        self._upsample_factor = upsample_factor
        self._normalization = normalization

    def register(
        self, moving: ArrayLike, *, expected_shift_xy: ArrayLike = (0.0, 0.0)
    ) -> RegistrationResult:
        """Track the reference patch near its expected location in ``moving``.

        The integer crop offset is measured explicitly; only the residual shift
        is correlated. This follows Micro-Manager's predicted-ROI strategy and
        avoids windowing different sample features at large displacements.
        """
        moving_float = _as_float_image(moving)
        if self._reference_shape != moving_float.shape:
            raise ValueError("reference and moving images must have the same shape")
        expected = np.asarray(expected_shift_xy, dtype=np.float64)
        if expected.shape != (2,) or not np.all(np.isfinite(expected)):
            raise ValueError("expected_shift_xy must contain two finite values")
        shape = np.asarray(self._reference_plain.shape, dtype=np.intp)
        margin = np.asarray(self._reference_shape, dtype=np.intp) - shape
        center = margin // 2
        start = np.clip(np.rint(center - expected[::-1]), 0, margin).astype(int)
        crop_offset = center - start
        moving_plain = _prepare_image(
            moving_float[
                start[0] : start[0] + shape[0], start[1] : start[1] + shape[1]
            ],
            crop_fraction=1.0,
        )

        moving_freq = np.fft.fftn(moving_plain * self._window)
        image_product = self._reference_freq * moving_freq.conj()
        if self._normalization == "phase":
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
        midpoint = np.asarray(
            [np.trunc(size / 2) for size in self._reference_plain.shape]
        )
        shift_rc[shift_rc > midpoint] -= shape[shift_rc > midpoint]

        if self._upsample_factor > 1:
            shift_rc = (
                np.round(shift_rc * self._upsample_factor) / self._upsample_factor
            )
            region_size = int(np.ceil(self._upsample_factor * 1.5))
            dft_shift = np.trunc(region_size / 2)
            sample_offset = dft_shift - shift_rc * self._upsample_factor
            refined = _upsampled_dft(
                image_product.conj(),
                region_size,
                upsample_factor=self._upsample_factor,
                axis_offsets=sample_offset,
            ).conj()
            refined_peak = np.asarray(
                np.unravel_index(np.argmax(np.abs(refined)), refined.shape),
                dtype=np.float64,
            )
            refined_peak -= dft_shift
            shift_rc += refined_peak / self._upsample_factor

        overlap_y = max(
            0.0,
            1.0 - abs(float(shift_rc[0])) / self._reference_plain.shape[0],
        )
        overlap_x = max(
            0.0,
            1.0 - abs(float(shift_rc[1])) / self._reference_plain.shape[1],
        )
        error = _normalized_alignment_error(
            self._reference_plain, moving_plain, shift_rc
        )
        method: Literal["phase", "unnormalized"] = (
            "phase" if self._normalization == "phase" else "unnormalized"
        )
        return RegistrationResult(
            shift_xy=(
                float(shift_rc[1] + crop_offset[1]),
                float(shift_rc[0] + crop_offset[0]),
            ),
            psr=psr,
            peak_ratio=peak_ratio,
            overlap=overlap_y * overlap_x,
            normalized_error=error,
            method=method,
        )


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
    reference_float = _as_float_image(reference)
    moving_float = _as_float_image(moving)
    if reference_float.shape != moving_float.shape:
        raise ValueError("reference and moving images must have the same shape")
    return TranslationRegistrar(
        reference_float,
        upsample_factor=upsample_factor,
        crop_fraction=crop_fraction,
        normalization=normalization,
    ).register(moving_float)
