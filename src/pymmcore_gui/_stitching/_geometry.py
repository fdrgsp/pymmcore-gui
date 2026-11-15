"""Pixel -> stage transforms and tile placement. numpy only.

Conventions:

- `A` (the "linear part") is a 2x2 matrix mapping a camera pixel offset
  `(column, row)` to a stage offset `(x, y)` in µm, as Micro-Manager's 2x3
  pixel-size affine `(a0, a1, a2, a3, a4, a5)` does with `[[a0, a1], [a3, a4]]`.
- multiview-stitcher orders spatial axes `(z,) y, x` and describes each tile
  with a physical pixel spacing plus an affine in those physical units. The
  functions returning affines here use that order.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

DEFAULT_MM_AFFINE = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
"""What MMCore stores for a pixel size preset without a calibrated affine."""

STALE_SCALE_TOLERANCE = 0.1
"""An affine whose scale is further than this from the pixel size is stale."""

MAX_TILES_GLOBAL_OPTIMIZATION = 100
"""Above this many tiles, `linear_two_pass` resolves twice as fast."""

REGISTRATION_TARGET_PX = 1024
"""Auto binning brings the longest tile side to at most this many pixels."""


def linear_from_mm_affine(affine: Sequence[float]) -> np.ndarray:
    """The 2x2 linear part of a Micro-Manager 2x3 pixel-size affine."""
    a = [float(v) for v in affine]
    if len(a) != 6:
        raise ValueError(f"expected 6 affine values, got {len(a)}")
    return np.array([[a[0], a[1]], [a[3], a[4]]])


def uncalibrated_linear(
    pixel_size_um: float, mirror_x: bool = False, mirror_y: bool = False
) -> np.ndarray:
    """`diag(±p, ±p)`, the sign from the camera's TransposeMirrorX/Y."""
    p = float(pixel_size_um)
    return np.diag([-p if mirror_x else p, -p if mirror_y else p])


def _scale(linear: np.ndarray) -> float:
    return math.sqrt(abs(float(np.linalg.det(linear))))


def is_calibrated(
    pixel_size_um: float,
    current_affine: Sequence[float],
    stored_affine: Sequence[float] | None = None,
) -> bool:
    """Whether `current_affine` is a usable calibration, as Micro-Manager decides.

    Parameters
    ----------
    pixel_size_um : float
        The current pixel size.
    current_affine : Sequence[float]
        `getPixelSizeAffine()`: the stored affine scaled by binning and
        magnification.
    stored_affine : Sequence[float] | None
        `getPixelSizeAffineByID(<current preset>)`, when known (fresh runs).
        It is MMCore's default identity exactly when there is no calibration.
        Reopened data doesn't record it: then a pure positive diagonal
        `diag(s, s)` is read as "no calibration" -- a fitted calibration is
        never exactly that, and without mirroring both readings agree anyway.
    """
    if not pixel_size_um > 0:
        return False
    linear = linear_from_mm_affine(current_affine)
    if stored_affine is not None:
        if np.allclose(stored_affine, DEFAULT_MM_AFFINE):
            return False
    elif (
        linear[0, 1] == 0
        and linear[1, 0] == 0
        and linear[0, 0] > 0
        and math.isclose(linear[0, 0], linear[1, 1], rel_tol=1e-9)
    ):
        return False
    # Like Micro-Manager, distrust an affine whose scale disagrees with the
    # pixel size, e.g. one left behind after the pixel size was edited.
    tolerance = STALE_SCALE_TOLERANCE * pixel_size_um
    return abs(_scale(linear) - pixel_size_um) <= tolerance


def linear_part(
    pixel_size_um: float,
    current_affine: Sequence[float],
    *,
    stored_affine: Sequence[float] | None = None,
    mirror_x: bool = False,
    mirror_y: bool = False,
) -> np.ndarray:
    """`A` for this system: a calibrated affine as is, else from mirror flags.

    The same rule as pymmcore-widgets' `AffineState` (the Stage Explorer),
    usable without a core: for reopened data the values come from the summary
    metadata.
    """
    if not pixel_size_um > 0:
        raise ValueError("a pixel size is needed to place tiles")
    if is_calibrated(pixel_size_um, current_affine, stored_affine):
        return linear_from_mm_affine(current_affine)
    return uncalibrated_linear(pixel_size_um, mirror_x, mirror_y)


def tile_affine(
    linear: np.ndarray,
    pixel_size_um: float,
    shape_yx: tuple[int, int],
    center_xy_um: tuple[float, float],
    z_origin_um: float | None = None,
) -> np.ndarray:
    """Affine placing one tile in stage coordinates, in multiview-stitcher's order.

    The tile is described with spacing `p` on y and x, so pixel `(row, col)`
    sits at `(row * p, col * p)` before this affine. Its centre maps to the
    stage position (useq grid positions are FOV centres) and `A / p` orients
    it: `T(center) · (A / p) · T(-image centre)`.

    Returns a 3x3 matrix over `(y, x)`, or 4x4 over `(z, y, x)` when
    `z_origin_um` is given (z itself is not rotated).
    """
    p = float(pixel_size_um)
    a = np.asarray(linear, dtype=float) / p
    # A acts on (x, y); swap both its rows and columns for (y, x).
    lin_yx = np.array([[a[1, 1], a[1, 0]], [a[0, 1], a[0, 0]]])
    rows, cols = shape_yx
    image_center = np.array([(rows - 1) / 2 * p, (cols - 1) / 2 * p])
    stage_center = np.array([center_xy_um[1], center_xy_um[0]])
    translation = stage_center - lin_yx @ image_center

    m2 = np.eye(3)
    m2[:2, :2] = lin_yx
    m2[:2, 2] = translation
    if z_origin_um is None:
        return m2
    m3 = np.eye(4)
    m3[1:3, 1:3] = lin_yx
    m3[1:3, 3] = translation
    m3[0, 3] = float(z_origin_um)
    return m3


def lift_2d_to_3d(affine_yx: np.ndarray) -> np.ndarray:
    """A (y, x) affine as a (z, y, x) one with an identity z row and column."""
    m = np.eye(4)
    m[1:, 1:] = np.asarray(affine_yx, dtype=float)
    return m


def anchor_corrections(
    corrections: Sequence[np.ndarray],
    centers: np.ndarray,
    components: Iterable[Iterable[int]],
) -> list[np.ndarray]:
    """Shift each separately aligned group back onto its stage positions.

    Registration resolves each connected group of tiles on its own and keeps
    one arbitrary tile of it fixed, so the whole group carries that tile's
    stage error. The least-squares fix for a translation is to move the group
    so its tiles' mean position matches their mean stage position.

    Parameters
    ----------
    corrections : Sequence[np.ndarray]
        Per tile, the homogeneous correction applied after its stage
        placement (identity = stay at the stage position).
    centers : np.ndarray
        `(n_tiles, ndim)`: each tile's centre at its stage position, in the
        same axis order as the corrections.
    components : Iterable[Iterable[int]]
        Tile indices of each connected group.
    """
    out = [np.asarray(c, dtype=float).copy() for c in corrections]
    centers = np.asarray(centers, dtype=float)
    ndim = centers.shape[1]
    for component in components:
        idx = list(component)
        if not idx:
            continue
        moved = np.array(
            [out[i][:ndim, :ndim] @ centers[i] + out[i][:ndim, ndim] for i in idx]
        )
        shift = (centers[idx] - moved).mean(axis=0)
        for i in idx:
            out[i][:ndim, ndim] += shift
    return out


def auto_binning(
    shape_yx: tuple[int, int], target_px: int = REGISTRATION_TARGET_PX
) -> int:
    """Binning (1, 2, 4 or 8) bringing the longest tile side to <= `target_px`."""
    longest = max(shape_yx)
    binning = 1
    while longest / binning > target_px and binning < 8:
        binning *= 2
    return binning


def resolution_method(n_tiles: int) -> str:
    """multiview-stitcher's groupwise resolution method for this many tiles."""
    if n_tiles > MAX_TILES_GLOBAL_OPTIMIZATION:
        return "linear_two_pass"
    return "global_optimization"


def overlap_fraction(
    linear: np.ndarray,
    shape_yx: tuple[int, int],
    centers_xy_um: Sequence[tuple[float, float]],
) -> float:
    """Largest overlap between two tiles, as a fraction of the tile's size.

    Two tiles overlap by `min(ox, oy)`, `ox`/`oy` being the overlapping share
    of the width and height (a side-by-side neighbour has `oy` = 1). Used to
    decide whether registration can work at all. Tiles are compared in
    the camera frame, where they are axis-aligned rectangles.
    """
    if len(centers_xy_um) < 2:
        return 0.0
    inv = np.linalg.inv(np.asarray(linear, dtype=float))
    # stage offsets -> pixel offsets (column, row)
    pts = np.asarray(centers_xy_um, dtype=float) @ inv.T
    rows, cols = shape_yx
    best = 0.0
    for i in range(len(pts) - 1):
        d = np.abs(pts[i + 1 :] - pts[i])
        ox = 1 - d[:, 0] / cols
        oy = 1 - d[:, 1] / rows
        best = max(best, float(np.minimum(ox, oy).max(initial=0.0)))
    return max(best, 0.0)
