from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Union


@dataclass(frozen=True, slots=True)
class Color:
    """8-bit RGBA color."""

    red: int = 0
    green: int = 0
    blue: int = 0
    alpha: int = 255


class GradientSpread(enum.IntEnum):
    Pad = 0
    Reflect = 1
    Repeat = 2


@dataclass(frozen=True, slots=True)
class GradientStop:
    position: float  # 0.0 – 1.0
    color: Color


@dataclass(frozen=True, slots=True)
class LinearGradient:
    """Linear gradient in object-bounding-mode coordinates (0–1)."""

    stops: tuple[GradientStop, ...] = ()
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 1.0
    y2: float = 0.0
    spread: GradientSpread = GradientSpread.Pad


#: A brush is either a solid color or a linear gradient.
Brush = Union[Color, LinearGradient]


@dataclass
class ColorGroup:
    """All 21 color roles for a single widget-state group.

    Field names are snake_case versions of `QPalette::ColorRole`.
    `None` means the role is unset and should be inherited or derived.
    """

    # -- backgrounds --------------------------------------------------------
    window: Brush | None = None
    base: Brush | None = None
    alternate_base: Brush | None = None
    button: Brush | None = None
    tooltip_base: Brush | None = None

    # -- foregrounds --------------------------------------------------------
    window_text: Brush | None = None
    text: Brush | None = None
    bright_text: Brush | None = None
    button_text: Brush | None = None
    tooltip_text: Brush | None = None
    placeholder_text: Brush | None = None

    # -- selection ----------------------------------------------------------
    highlight: Brush | None = None
    highlighted_text: Brush | None = None

    # -- links --------------------------------------------------------------
    link: Brush | None = None
    link_visited: Brush | None = None

    # -- 3D bevel / shadow --------------------------------------------------
    light: Brush | None = None
    midlight: Brush | None = None
    dark: Brush | None = None
    mid: Brush | None = None
    shadow: Brush | None = None

    # -- accent (Qt 6.6+) --------------------------------------------------
    accent: Brush | None = None


@dataclass
class Palette:
    """Complete palette: three color groups.

    - `active`   — window that has keyboard focus
    - `inactive` — other (unfocused) windows
    - `disabled` — disabled widgets

    In most Qt styles, `active` and `inactive` are identical.
    """

    active: ColorGroup = field(default_factory=ColorGroup)
    inactive: ColorGroup = field(default_factory=ColorGroup)
    disabled: ColorGroup = field(default_factory=ColorGroup)
