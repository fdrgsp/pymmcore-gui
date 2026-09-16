"""Compare calibration patches using the real pipeline and a simulated camera.

Run from the repository root with PYTEST_RUNNING=1 and uv run python.
This is an opt-in benchmark; pytest does not collect it. Targets and capture
clock are overridden only inside the experiment, never in production code.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import numpy as np

from pymmcore_gui._pixel_calibration import (
    CalibrationOptions,
    PixelCalibrationError,
    _routine,
    run_pixel_calibration,
)
from pymmcore_gui._pixel_calibration._registration import TranslationRegistrar
from test_pixel_calibration import _SyntheticCore

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from pymmcore_gui._pixel_calibration._models import HardwareFingerprint

SCENES = (
    "fine",
    "blurred",
    "noisy",
    "sparse",
    "peripheral",
    "illumination",
    "periodic",
    "blank",
)


def make_field(scene: str, seed: int, size: int = 256) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    shape = (size * 2, size * 2)
    y, x = np.indices(shape)
    frequencies = (
        np.fft.fftfreq(shape[0])[:, None] ** 2 + np.fft.fftfreq(shape[1])[None, :] ** 2
    )
    if scene == "blank":
        return np.zeros(shape)
    if scene == "periodic":
        return np.cos(2 * np.pi * x / 16) + np.cos(2 * np.pi * y / 16)
    if scene == "sparse":
        field = np.zeros(shape)
        # Uniform random spots, approximately twelve per camera field.
        for cy, cx in rng.uniform(0, size * 2, (48, 2)):
            sigma = rng.uniform(2, 6)
            field += rng.uniform(0.5, 1.5) * np.exp(
                -((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma**2)
            )
    else:
        sigma = 6 if scene == "blurred" else 2
        field = np.fft.ifftn(
            np.fft.fftn(rng.normal(size=shape))
            * np.exp(-2 * np.pi**2 * sigma**2 * frequencies)
        ).real
        if scene == "peripheral":
            radius = np.hypot(y - size, x - size)
            field *= np.clip((radius - size * 0.22) / (size * 0.05), 0, 1)
    central = field[size // 2 : 3 * size // 2, size // 2 : 3 * size // 2]
    return np.asarray(field / central.std(), dtype=np.float64)


class FieldCore(_SyntheticCore):
    def __init__(
        self, scene: str, sample_seed: int, noise_seed: int, size: int = 256
    ) -> None:
        super().__init__()
        self.base_image = np.zeros((size, size))
        self.size = size
        self.scene = scene
        self.noise_seed = noise_seed
        self.field = make_field(scene, sample_seed, size)
        self.spectrum = np.fft.fftn(self.field)
        self.row_freq = np.fft.fftfreq(size * 2)[:, None]
        self.col_freq = np.fft.fftfreq(size * 2)[None, :]
        self.cache: dict[tuple[float, float], NDArray[np.float64]] = {}
        yy, xx = np.indices((size, size))
        self.illumination = 0.2 + 0.8 * np.exp(
            -((yy - size / 2) ** 2 + (xx - size / 2) ** 2) / (2 * (size * 0.25) ** 2)
        )

    def getImage(self) -> NDArray[np.float64]:
        key = (float(self.position[0]), float(self.position[1]))
        if key not in self.cache:
            sx, sy = np.linalg.solve(self.true_matrix, self.position - self.origin)
            shifted = np.fft.ifftn(
                self.spectrum
                * np.exp(2j * np.pi * (self.row_freq * sy + self.col_freq * sx))
            ).real
            lo = self.size // 2
            frame = shifted[lo : lo + self.size, lo : lo + self.size]
            if self.scene == "illumination":
                frame = (frame + 3) * self.illumination
            self.cache[key] = frame
        noise = 0.5 if self.scene == "noisy" else 0.15
        if self.scene == "periodic":
            noise = 0.02
        # Frame-indexed noise: matching captures use identical noise across variants.
        rng = np.random.default_rng(
            np.random.SeedSequence([self.noise_seed, self.snap_count])
        )
        return np.asarray(
            self.cache[key] + rng.normal(scale=noise, size=(self.size, self.size)),
            dtype=np.float64,
        )


def target_function(full_fraction: float) -> Callable[..., list[NDArray[np.float64]]]:
    def targets(
        matrix: NDArray[np.float64],
        fingerprint: HardwareFingerprint,
        options: CalibrationOptions,
        *,
        validation: bool,
    ) -> list[NDArray[np.float64]]:
        height, width = fingerprint.image_shape
        if validation:
            q = full_fraction * 0.65
            points = [
                (width * q * 0.87, height * q * 0.50),
                (-width * q * 0.87, height * q * 0.50),
                (0, -height * q),
            ]
        else:
            x, y = width * full_fraction, height * full_fraction
            points = [
                (x, 0),
                (-x, 0),
                (0, y),
                (0, -y),
                (x, y),
                (-x, -y),
                (-x, y),
                (x, -y),
            ]
        offsets = [matrix @ np.asarray(point) for point in points]
        maximum = max(float(np.linalg.norm(offset)) for offset in offsets)
        if maximum > options.safe_radius_um:
            offsets = [
                offset * (0.9 * options.safe_radius_um / maximum) for offset in offsets
            ]
        return offsets

    return targets


def run_case(
    scene: str,
    sample_seed: int,
    noise_seed: int,
    crop: float,
    distance: float,
    size: int,
) -> dict[str, Any]:
    core = FieldCore(scene, sample_seed, noise_seed, size)
    options = CalibrationOptions(crop_fraction=crop, settle_time_s=0)
    clock = itertools.count()
    simulated_time = SimpleNamespace(monotonic=lambda: float(next(clock)))
    row: dict[str, Any] = {
        "scene": scene,
        "sample_seed": sample_seed,
        "noise_seed": noise_seed,
        "crop": crop,
        "distance": distance,
        "size": size,
    }
    started = time.perf_counter()
    with (
        patch.object(_routine, "_measurement_targets", target_function(distance)),
        patch.object(_routine, "time", simulated_time),
    ):
        try:
            result = run_pixel_calibration(core, options)
        except PixelCalibrationError as exc:
            row.update(ok=False, error=str(exc))
        else:
            delta = result.fit.matrix - core.true_matrix
            row.update(
                ok=True,
                pixel_size=result.fit.pixel_size_um,
                size_error_pct=100 * (result.fit.pixel_size_um / 0.4 - 1),
                matrix_error_pct=100
                * float(np.linalg.norm(delta) / np.linalg.norm(core.true_matrix)),
                fit_rms_px=result.fit.rms_residual_px,
                warnings=[w.code for w in result.warnings],
            )
    row.update(
        seconds=time.perf_counter() - started,
        snaps=core.snap_count,
        stage_returned=bool(np.allclose(core.position, core.origin)),
    )
    return row


def timing(crops: Sequence[float]) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    for size in (256, 512, 1024):
        core = FieldCore("fine", 0, 0, size)
        reference = core.getImage()
        shift = np.array([size * 0.12, -size * 0.12])
        core.position = core.origin + core.true_matrix @ shift
        core.snap_count = 1
        moving = core.getImage()
        registrars = {
            crop: TranslationRegistrar(
                reference, crop_fraction=crop, normalization="unnormalized"
            )
            for crop in crops
        }
        samples: dict[float, list[float]] = {crop: [] for crop in registrars}
        for trial in range(13):
            order = list(registrars) if trial % 2 else list(reversed(registrars))
            for crop in order:
                start = time.perf_counter()
                result = registrars[crop].register(
                    moving, expected_shift_xy=(float(shift[0]), float(shift[1]))
                )
                elapsed = (time.perf_counter() - start) * 1000
                if trial >= 3:
                    samples[crop].append(elapsed)
                assert np.linalg.norm(np.asarray(result.shift_xy) - shift) < 0.15
        for crop, values in samples.items():
            results.append(
                {"size": size, "crop": crop, "median_ms": float(np.median(values))}
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("pixel_patch_results.jsonl")
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=SCENES)
    parser.add_argument("--crops", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument(
        "--distances", nargs="+", type=float, default=[0.08, 0.12, 0.25]
    )
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1 or args.size < 64 or args.size % 2:
        parser.error("Use positive counts and an even image size of at least 64")
    if any(not 0 < fraction < 0.5 for fraction in args.distances):
        parser.error("Travel fractions must be between zero and one half")
    if any(not 0.25 <= fraction <= 1 for fraction in args.crops):
        parser.error("Patch fractions must be between one quarter and one")
    if args.timing:
        args.output.write_text(json.dumps(timing(args.crops), indent=2) + "\n")
        return
    with args.output.open("w") as output:
        for scene, crop, distance in itertools.product(
            args.scenes, args.crops, args.distances
        ):
            rows = []
            for sample, noise in itertools.product(
                range(args.samples), range(args.repeats)
            ):
                row = run_case(scene, sample, noise, crop, distance, args.size)
                rows.append(row)
                output.write(json.dumps(row) + "\n")
                output.flush()
            good = [r for r in rows if r["ok"]]
            maximum = max((r["matrix_error_pct"] for r in good), default=float("nan"))
            print(
                f"{scene:12} crop={crop:.2f} travel={distance:.3f}: "
                f"{len(good)}/{len(rows)}, max matrix error {maximum:.3f}%",
                flush=True,
            )


if __name__ == "__main__":
    main()
