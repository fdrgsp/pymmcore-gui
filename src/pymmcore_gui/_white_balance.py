"""Balance red/blue camera gains against green on a neutral field.

MMCore has no generic white-balance API. Adapters exposing manual gains, such
as NikonKs, need host-side balancing, as in Micro-Manager's PVCAM WhiteBalance
plugin.

Saturated pixels are excluded from the channel means. A proportional first
step followed by secant steps accommodates device-specific gain responses.
"""

from __future__ import annotations

import math
import re
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from pymmcore_plus import PropertyType

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from numpy.typing import NDArray
    from pymmcore_plus import CMMCorePlus

Channel = Literal["red", "green", "blue"]
ADJUSTED: tuple[Channel, Channel] = ("red", "blue")

TOLERANCE = 0.02
MAX_ITERATIONS = 8
MAX_SATURATED_FRACTION = 0.2
MIN_MEAN_FRACTION = 0.05
# Bound properties without adapter limits relative to their starting value.
UNLIMITED_FACTOR = 10.0

_WB_NAME = re.compile(
    r"white[\s_-]*bal|\bwb\b|balance[\s_-]*ratio|color - \w+ scale", re.I
)
# "Red", "R gain", or camel case "WhiteBalanceRGain" (AmScope)
_CHANNEL_NAME: dict[Channel, re.Pattern[str]] = {
    "red": re.compile(r"(?i:red|(?<![a-z])r[\s_-]*gain)|R[\s_-]?Gain"),
    "green": re.compile(r"(?i:green|(?<![a-z])g[\s_-]*gain)|G[\s_-]?Gain"),
    "blue": re.compile(r"(?i:blue|(?<![a-z])b[\s_-]*gain)|B[\s_-]?Gain"),
}
_NOT_A_GAIN = re.compile(r"auto|once|one[\s_-]*(push|shot)|target|mode|click", re.I)


class WhiteBalanceError(RuntimeError):
    """White balance cannot proceed with the selected properties or image."""


@dataclass(frozen=True)
class GainProperty:
    """A numeric camera property scaling one color channel."""

    name: str
    lower: float
    upper: float
    integer: bool
    #: False when the adapter defines no limits (``lower``/``upper`` are then
    #: the search bounds around the value the property had when it was read)
    has_limits: bool = True

    def clamp(self, value: float) -> float:
        value = min(max(value, self.lower), self.upper)
        return float(round(value)) if self.integer else value


@dataclass
class WhiteBalanceResult:
    """Outcome of :func:`one_push_white_balance`."""

    values: dict[str, float]
    #: channel mean / green mean, measured with ``values`` applied
    ratios: dict[Channel, float]
    converged: bool
    iterations: int
    #: properties that hit their limit before reaching the target
    limited: list[str] = field(default_factory=list)


def gain_property(core: CMMCorePlus, camera: str, name: str) -> GainProperty | None:
    """Describe `name` as a gain, or None if it isn't a writable number."""
    try:
        if core.isPropertyReadOnly(camera, name):
            return None
        ptype = core.getPropertyType(camera, name)
        if ptype not in (PropertyType.Integer, PropertyType.Float):
            return None
        integer = ptype == PropertyType.Integer
        if core.hasPropertyLimits(camera, name):
            lower = core.getPropertyLowerLimit(camera, name)
            upper = core.getPropertyUpperLimit(camera, name)
            if upper <= lower:
                return None
            return GainProperty(name, lower, upper, integer)
        value = float(core.getProperty(camera, name))
    except Exception:
        return None
    if value > 0:
        lower, upper = value / UNLIMITED_FACTOR, value * UNLIMITED_FACTOR
    else:
        span = UNLIMITED_FACTOR * max(1.0, abs(value))
        lower, upper = value - span, value + span
    return GainProperty(name, lower, upper, integer, has_limits=False)


def candidate_properties(core: CMMCorePlus, camera: str) -> list[GainProperty]:
    """All writable numeric properties of `camera`, in the adapter's order."""
    try:
        names = core.getDevicePropertyNames(camera)
    except Exception:
        return []
    return [p for name in names if (p := gain_property(core, camera, name))]


def guess_gain_properties(
    properties: Iterable[GainProperty],
) -> dict[Channel, GainProperty]:
    """Guess red and blue white-balance gains from candidate property names."""
    found: dict[Channel, GainProperty] = {}
    for prop in properties:
        if not _WB_NAME.search(prop.name) or _NOT_A_GAIN.search(prop.name):
            continue
        channels: list[Channel] = [
            c for c, pat in _CHANNEL_NAME.items() if pat.search(prop.name)
        ]
        if len(channels) == 1 and channels[0] in ADJUSTED:
            found.setdefault(channels[0], prop)
    return found


def channel_means(image: NDArray, max_value: float) -> tuple[float, float, float]:
    """Mean red, green and blue of the unsaturated pixels of an RGB(A) frame."""
    if image.ndim != 3 or image.shape[-1] not in (3, 4) or not image.size:
        raise WhiteBalanceError("White balance needs an RGB image")
    rgb = image[..., :3].reshape(-1, 3)
    saturated = (rgb >= max_value).any(axis=1)
    if saturated.mean() > MAX_SATURATED_FRACTION:
        raise WhiteBalanceError(
            "The image is too bright: reduce the light or the exposure and try again"
        )
    r, g, b = (float(v) for v in rgb[~saturated].mean(axis=0, dtype=np.float64))
    if not all(math.isfinite(v) for v in (r, g, b)):
        raise WhiteBalanceError("The image contains invalid pixel values")
    if g < max_value * MIN_MEAN_FRACTION or r <= 0 or b <= 0:
        raise WhiteBalanceError(
            "The image is too dark: increase the light or the exposure and try again"
        )
    return r, g, b


def _next_value(
    prop: GainProperty,
    value: float,
    ratio: float,
    previous: tuple[float, float] | None,
) -> float:
    """Pick the property value expected to bring ``ratio`` (channel/green) to 1."""
    if previous is not None:
        prev_value, prev_ratio = previous
        if value != prev_value and not math.isclose(ratio, prev_ratio):
            guess = value - (ratio - 1) * (value - prev_value) / (ratio - prev_ratio)
            if math.isfinite(guess):
                return prop.clamp(guess)
    if value > 0:
        # assume the channel mean is proportional to the property value
        return prop.clamp(value / ratio)
    # offset-style property (zero or negative): take a tenth of the range
    step = 0.1 * (prop.upper - prop.lower)
    return prop.clamp(value + (step if ratio < 1 else -step))


def one_push_white_balance(
    core: CMMCorePlus,
    gains: Mapping[Channel, GainProperty],
    *,
    camera: str | None = None,
) -> WhiteBalanceResult:
    """Adjust the red and/or blue `gains` so a neutral field comes out gray.

    Snaps with `camera` (default: the current camera), which is made the
    current camera for the duration. On error the original property values
    are restored; when the target is not reached within ``MAX_ITERATIONS``,
    the best values found are kept.

    For properties without adapter limits, search bounds are centered on the
    current value at the start of each run, including repeated Balance clicks.
    """
    adjust: dict[Channel, GainProperty] = {
        c: p for c, p in gains.items() if c in ADJUSTED
    }
    if not adjust:
        raise WhiteBalanceError("Choose a red and/or a blue gain property")
    if len({p.name for p in adjust.values()}) != len(adjust):
        raise WhiteBalanceError("Choose different properties for red and blue")
    current_camera = core.getCameraDevice()
    camera = camera or current_camera
    if not camera:
        raise WhiteBalanceError("No camera is loaded")
    # Re-read limits: unbounded gains may have changed since the dialog opened.
    for channel, prop in adjust.items():
        if (current := gain_property(core, camera, prop.name)) is None:
            raise WhiteBalanceError(f"{prop.name} is not a writable numeric property")
        adjust[channel] = current
    if camera != current_camera:
        core.setCameraDevice(camera)
    try:
        return _balance(core, camera, adjust)
    finally:
        if camera != current_camera:
            core.setCameraDevice(current_camera)


def _balance(
    core: CMMCorePlus,
    camera: str,
    adjust: Mapping[Channel, GainProperty],
) -> WhiteBalanceResult:
    max_value = float(2 ** core.getImageBitDepth() - 1)
    original = {
        p.name: float(core.getProperty(camera, p.name)) for p in adjust.values()
    }
    values = dict(original)
    previous: dict[Channel, tuple[float, float]] = {}
    best: tuple[float, dict[str, float], dict[Channel, float]] | None = None
    iteration = 0

    try:
        for iteration in range(1, MAX_ITERATIONS + 1):
            core.waitForDevice(camera)
            r, g, b = channel_means(core.snap(), max_value)
            ratios: dict[Channel, float] = {"red": r / g, "blue": b / g}
            error = max(abs(ratios[c] - 1) for c in adjust)
            if best is None or error < best[0]:
                best = (error, dict(values), ratios)
            if error <= TOLERANCE:
                return WhiteBalanceResult(values, ratios, True, iteration)
            if iteration == MAX_ITERATIONS:
                break
            changed = False
            for channel, prop in adjust.items():
                ratio = ratios[channel]
                if abs(ratio - 1) <= TOLERANCE:
                    continue
                value = values[prop.name]
                new = _next_value(prop, value, ratio, previous.get(channel))
                previous[channel] = (value, ratio)
                if new == value:
                    continue
                core.setProperty(camera, prop.name, new)
                values[prop.name] = new
                changed = True
            if not changed:
                break

        assert best is not None
        _, best_values, best_ratios = best
        for name, value in best_values.items():
            if value != values[name]:
                core.setProperty(camera, name, value)
        core.waitForDevice(camera)
        return WhiteBalanceResult(
            best_values,
            best_ratios,
            converged=False,
            iterations=iteration,
            limited=[
                p.name
                for c, p in adjust.items()
                if abs(best_ratios[c] - 1) > TOLERANCE
                and best_values[p.name] in (p.clamp(p.lower), p.clamp(p.upper))
            ],
        )
    except Exception:
        for name, value in original.items():
            with suppress(Exception):
                core.setProperty(camera, name, value)
        with suppress(Exception):
            core.waitForDevice(camera)
        raise
