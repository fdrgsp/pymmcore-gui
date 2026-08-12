from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from ._fit import normalize_for_mmcore
from ._models import (
    CalibrationCommitError,
    PixelCalibrationResult,
)
from ._routine import fingerprint_matches

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PersistenceCore(Protocol):
    """MMCore operations used to persist a calibration."""

    def getAvailablePixelSizeConfigs(self) -> tuple[str, ...]: ...

    def getPixelSizeUmByID(self, resolution_id: str) -> float: ...

    def getPixelSizeAffineByID(self, resolution_id: str) -> tuple[float, ...]: ...

    def setPixelSizeUm(self, resolution_id: str, value: float) -> None: ...

    def setPixelSizeAffine(
        self, resolution_id: str, value: tuple[float, ...]
    ) -> None: ...

    def getCurrentPixelSizeConfig(self) -> str: ...

    def getPixelSizeUm(self) -> float: ...

    def getPixelSizeAffine(self) -> tuple[float, ...]: ...


def _flat_raw(matrix: NDArray[np.float64]) -> tuple[float, ...]:
    _, _, flattened = normalize_for_mmcore(matrix, binning=1, magnification=1)
    return flattened


def commit_pixel_calibration(
    core: PersistenceCore,
    resolution_id: str,
    result: PixelCalibrationResult,
    *,
    allow_large_difference: bool = False,
) -> None:
    """Commit a validated result to an existing MMCore pixel-size configuration."""
    if not result.stage_returned:
        raise CalibrationCommitError(
            "Cannot commit before successful stage restoration"
        )
    if resolution_id not in set(core.getAvailablePixelSizeConfigs()):
        raise CalibrationCommitError(
            f"Pixel-size configuration {resolution_id!r} does not exist"
        )
    if result.fingerprint.pixel_size_config not in {"", resolution_id}:
        raise CalibrationCommitError(
            "Result belongs to a different pixel-size configuration"
        )
    # fingerprint_matches needs the broader calibration-core surface. MMCorePlus has it;
    # test fakes implement the same structural API.
    if not fingerprint_matches(core, result.fingerprint):  # type: ignore[arg-type]
        raise CalibrationCommitError(
            "The optical configuration changed after calibration"
        )

    old_size = float(core.getPixelSizeUmByID(resolution_id))
    old_affine = tuple(float(v) for v in core.getPixelSizeAffineByID(resolution_id))
    if (
        old_size > 0
        and abs(result.raw_pixel_size_um - old_size) / old_size > 0.10
        and not allow_large_difference
    ):
        raise CalibrationCommitError(
            "Measured pixel size differs from the stored value by more than 10%"
        )

    new_affine = _flat_raw(result.raw_matrix)
    write_error: BaseException | None = None
    try:
        core.setPixelSizeUm(resolution_id, result.raw_pixel_size_um)
        core.setPixelSizeAffine(resolution_id, new_affine)
        saved_size = float(core.getPixelSizeUmByID(resolution_id))
        saved_affine = tuple(
            float(v) for v in core.getPixelSizeAffineByID(resolution_id)
        )
        if not np.isclose(saved_size, result.raw_pixel_size_um, rtol=1e-10, atol=1e-12):
            raise CalibrationCommitError("MMCore pixel-size readback did not match")
        if not np.allclose(saved_affine, new_affine, rtol=1e-10, atol=1e-12):
            raise CalibrationCommitError("MMCore affine readback did not match")
        if str(core.getCurrentPixelSizeConfig()) == resolution_id:
            if not np.isclose(
                float(core.getPixelSizeUm()), result.fit.pixel_size_um, rtol=1e-9
            ):
                raise CalibrationCommitError(
                    "MMCore current pixel-size correction is wrong"
                )
            current_affine = np.asarray(core.getPixelSizeAffine(), dtype=float).reshape(
                2, 3
            )
            if not np.allclose(
                current_affine[:, :2], result.fit.matrix, rtol=1e-9, atol=1e-12
            ):
                raise CalibrationCommitError(
                    "MMCore current affine correction is wrong"
                )
        return
    except BaseException as error:
        write_error = error

    rollback_errors: list[BaseException] = []
    try:
        core.setPixelSizeUm(resolution_id, old_size)
    except BaseException as error:
        rollback_errors.append(error)
    try:
        core.setPixelSizeAffine(resolution_id, old_affine)
    except BaseException as error:
        rollback_errors.append(error)
    message = f"Failed to commit pixel calibration: {write_error}"
    if rollback_errors:
        message += "; rollback also failed: " + "; ".join(map(str, rollback_errors))
    raise CalibrationCommitError(message) from write_error
