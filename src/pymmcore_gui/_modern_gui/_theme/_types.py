from __future__ import annotations

import enum
from dataclasses import dataclass, field


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
    position: float  # 0.0 - 1.0
    color: Color


@dataclass(frozen=True, slots=True)
class LinearGradient:
    """Linear gradient in object-bounding-mode coordinates (0-1)."""

    stops: tuple[GradientStop, ...] = ()
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 1.0
    y2: float = 0.0
    spread: GradientSpread = GradientSpread.Pad


#: A brush is either a solid color or a linear gradient.
Brush = Color | LinearGradient


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
    """Three color groups: active, inactive, disabled."""

    active: ColorGroup = field(default_factory=ColorGroup)
    inactive: ColorGroup = field(default_factory=ColorGroup)
    disabled: ColorGroup = field(default_factory=ColorGroup)


@dataclass
class Theme:
    """Extended design tokens beyond QPalette.

    All values are base (unscaled). Use `ScaledThemeView` for zoom-aware access.
    """

    palette: Palette = field(default_factory=Palette)

    # -- backgrounds --------------------------------------------------------
    bg_deepest: Color = Color()
    bg_base: Color = Color()
    bg_raised: Color = Color()
    bg_surface: Color = Color()
    bg_hover: Color = Color()
    bg_active: Color = Color()

    # -- text ---------------------------------------------------------------
    text_primary: Color = Color(0xE0, 0xE0, 0xE0)
    text_secondary: Color = Color(0xA0, 0xA0, 0xA0)
    text_disabled: Color = Color(0x70, 0x70, 0x70)

    # -- borders ------------------------------------------------------------
    border_subtle: Color = Color()
    border_default: Color = Color()
    border_focus: Color = Color()

    # -- accent -------------------------------------------------------------
    accent: Color = Color()
    accent_muted: Color = Color()

    # -- semantic status ----------------------------------------------------
    status_green: Color = Color()
    status_red: Color = Color()
    status_amber: Color = Color()

    # -- interaction feedback -----------------------------------------------
    drag_highlight: Color = Color(0xFF, 0xFF, 0xFF, 0x14)  # overlay on hover
    drop_indicator: Color = Color(0xFF, 0xFF, 0xFF)

    # -- spacing (base values, 4px grid) ------------------------------------
    sp_xxs: int = 4
    sp_xs: int = 8
    sp_sm: int = 12
    sp_md: int = 16
    sp_lg: int = 24
    sp_xl: int = 32

    # -- metrics ------------------------------------------------------------
    radius: int = 3
    radius_lg: int = 6
    row_height: int = 36
    sidebar_width: int = 240
