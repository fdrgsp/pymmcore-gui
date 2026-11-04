from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._models import CalibrationWarning

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _diagnostic_warnings(
    anisotropy: float, nonorthogonality_deg: float, condition: float
) -> tuple[CalibrationWarning, ...]:
    """Flag a measured matrix whose geometry looks physically implausible."""
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


def flatten_affine(
    matrix: NDArray[np.float64],
) -> tuple[float, float, float, float, float, float]:
    """Flatten a 2x2 affine into MMCore's six-element row-major form."""
    return (
        float(matrix[0, 0]),
        float(matrix[0, 1]),
        0.0,
        float(matrix[1, 0]),
        float(matrix[1, 1]),
        0.0,
    )


def normalize_for_mmcore(
    current_matrix: NDArray[np.floating], *, binning: int, magnification: float
) -> tuple[NDArray[np.float64], float]:
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
    return raw, float(np.sqrt(abs(determinant)))
