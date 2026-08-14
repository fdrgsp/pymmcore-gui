from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._models import (
    AffineFitResult,
    CalibrationWarning,
    as_float64_points,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _solve(
    pixel_shifts: NDArray[np.float64],
    stage_deltas: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    scale = np.sqrt(weights)[:, None]
    coefficients, _, rank, _ = np.linalg.lstsq(
        pixel_shifts * scale, stage_deltas * scale, rcond=None
    )
    if rank != 2:
        raise ValueError("pixel shifts do not span two dimensions")
    return np.asarray(coefficients.T, dtype=np.float64)


def _huber_weights(residual_norms: NDArray[np.float64]) -> NDArray[np.float64]:
    median = float(np.median(residual_norms))
    mad = float(np.median(np.abs(residual_norms - median)))
    scale = max(1.4826 * mad, np.finfo(float).eps)
    cutoff = 1.345 * scale
    weights = np.ones_like(residual_norms)
    outside = residual_norms > cutoff
    weights[outside] = cutoff / residual_norms[outside]
    return weights


def _diagnostic_warnings(
    anisotropy: float, nonorthogonality_deg: float, condition: float
) -> tuple[CalibrationWarning, ...]:
    warnings: list[CalibrationWarning] = []
    if anisotropy > 0.02:
        warnings.append(
            CalibrationWarning(
                "anisotropy",
                f"Camera-axis pixel sizes differ by {anisotropy:.1%}.",
            )
        )
    if nonorthogonality_deg > 1:
        warnings.append(
            CalibrationWarning(
                "nonorthogonality",
                f"Camera axes differ from 90 degrees by {nonorthogonality_deg:.2f}°.",
            )
        )
    if condition > 2:
        warnings.append(
            CalibrationWarning(
                "matrix_condition",
                f"Affine matrix condition number is high ({condition:.3g}).",
            )
        )
    return tuple(warnings)


def fit_affine(
    pixel_shifts_xy: NDArray[np.floating],
    stage_deltas_um: NDArray[np.floating],
    *,
    confidence_weights: NDArray[np.floating] | None = None,
    minimum_points: int = 3,
    max_irls_iterations: int = 20,
) -> AffineFitResult:
    """Robustly fit ``stage_delta = matrix @ pixel_shift`` without translation."""
    shifts = as_float64_points(pixel_shifts_xy, name="pixel_shifts_xy")
    deltas = as_float64_points(stage_deltas_um, name="stage_deltas_um")
    if len(shifts) != len(deltas):
        raise ValueError("pixel shifts and stage deltas must have the same length")
    if len(shifts) < minimum_points:
        raise ValueError(f"at least {minimum_points} observations are required")

    design_condition = float(np.linalg.cond(shifts))
    if not np.isfinite(design_condition) or design_condition > 100:
        raise ValueError("pixel-shift design is singular or ill-conditioned")

    if confidence_weights is None:
        base_weights = np.ones(len(shifts), dtype=np.float64)
    else:
        base_weights = np.asarray(confidence_weights, dtype=np.float64)
        if base_weights.shape != (len(shifts),):
            raise ValueError("confidence_weights must have shape (N,)")
        if not np.all(np.isfinite(base_weights)) or np.any(base_weights <= 0):
            raise ValueError("confidence_weights must be finite and positive")
        base_weights /= np.max(base_weights)

    weights = base_weights.copy()
    matrix = _solve(shifts, deltas, weights)

    # Robust (median/MAD-based) outlier rejection needs enough points to
    # spare -- with only two observations there's no way to identify which
    # one is the "outlier" without losing the redundancy needed to fit at
    # all, so skip straight to the plain least-squares result and treat both
    # as inliers.
    if len(shifts) >= 3:
        for _ in range(max_irls_iterations):
            residuals = deltas - shifts @ matrix.T
            robust = _huber_weights(np.linalg.norm(residuals, axis=1))
            new_weights = base_weights * robust
            new_matrix = _solve(shifts, deltas, new_weights)
            if np.allclose(matrix, new_matrix, rtol=1e-10, atol=1e-12):
                matrix = new_matrix
                weights = new_weights
                break
            matrix = new_matrix
            weights = new_weights

        inlier_mask = weights >= 0.25 * base_weights
        outlier_count = int(np.count_nonzero(~inlier_mask))
        if 0 < outlier_count <= 2 and np.count_nonzero(inlier_mask) >= minimum_points:
            matrix = _solve(
                shifts[inlier_mask], deltas[inlier_mask], base_weights[inlier_mask]
            )
            weights = np.where(inlier_mask, base_weights, 0.0)
        elif outlier_count > 2:
            raise ValueError("more than two affine observations are outliers")
    else:
        inlier_mask = np.ones(len(shifts), dtype=bool)

    determinant = float(np.linalg.det(matrix))
    if not np.isfinite(determinant) or abs(determinant) <= np.finfo(float).eps:
        raise ValueError("affine matrix is singular")

    residuals_um = deltas - shifts @ matrix.T
    inverse = np.linalg.inv(matrix)
    residuals_px = residuals_um @ inverse.T
    residual_norms_px = np.linalg.norm(residuals_px, axis=1)
    evaluated = residual_norms_px[inlier_mask]

    pixel_size_x = float(np.linalg.norm(matrix[:, 0]))
    pixel_size_y = float(np.linalg.norm(matrix[:, 1]))
    pixel_size = float(np.sqrt(abs(determinant)))
    singular_values_array = np.linalg.svd(matrix, compute_uv=False)
    singular_values = (
        float(singular_values_array[0]),
        float(singular_values_array[1]),
    )
    matrix_condition = float(np.linalg.cond(matrix))
    anisotropy = abs(pixel_size_x - pixel_size_y) / (
        0.5 * (pixel_size_x + pixel_size_y)
    )
    cosine = float(np.dot(matrix[:, 0], matrix[:, 1]) / (pixel_size_x * pixel_size_y))
    axis_angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    nonorthogonality = abs(90.0 - axis_angle)
    rotation = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))

    return AffineFitResult(
        matrix=matrix,
        residuals_um=residuals_um,
        residuals_px=residuals_px,
        weights=weights,
        inlier_mask=inlier_mask,
        pixel_size_um=pixel_size,
        pixel_size_x_um=pixel_size_x,
        pixel_size_y_um=pixel_size_y,
        singular_values=singular_values,
        design_condition=design_condition,
        matrix_condition=matrix_condition,
        anisotropy=float(anisotropy),
        nonorthogonality_deg=float(nonorthogonality),
        rotation_deg=rotation,
        determinant=determinant,
        rms_residual_px=float(np.sqrt(np.mean(evaluated**2))),
        max_residual_px=float(np.max(evaluated)),
        warnings=_diagnostic_warnings(
            float(anisotropy), float(nonorthogonality), matrix_condition
        ),
    )


def normalize_for_mmcore(
    current_matrix: NDArray[np.floating], *, binning: int, magnification: float
) -> tuple[NDArray[np.float64], float, tuple[float, float, float, float, float, float]]:
    """Convert a measured current-image affine to MMCore's raw storage units."""
    matrix = np.asarray(current_matrix, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError("current_matrix must be a finite 2 x 2 matrix")
    if binning < 1:
        raise ValueError("binning must be at least 1")
    if not np.isfinite(magnification) or magnification <= 0:
        raise ValueError("magnification must be finite and positive")
    raw = matrix * (magnification / binning)
    determinant = float(np.linalg.det(raw))
    if abs(determinant) <= np.finfo(float).eps:
        raise ValueError("current_matrix must be invertible")
    pixel_size = float(np.sqrt(abs(determinant)))
    flattened = (
        float(raw[0, 0]),
        float(raw[0, 1]),
        0.0,
        float(raw[1, 0]),
        float(raw[1, 1]),
        0.0,
    )
    return raw, pixel_size, flattened
