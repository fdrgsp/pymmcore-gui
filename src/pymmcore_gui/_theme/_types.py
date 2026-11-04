from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Color:
    """8-bit RGBA color."""

    red: int = 0
    green: int = 0
    blue: int = 0
    alpha: int = 255


@dataclass
class ColorGroup:
    """All 21 color roles for a single widget-state group.

    Field names are snake_case versions of `QPalette::ColorRole`.
    `None` means the role is unset and should be inherited or derived.
    """

    # -- backgrounds --------------------------------------------------------
    window: Color | None = None
    base: Color | None = None
    alternate_base: Color | None = None
    button: Color | None = None
    tooltip_base: Color | None = None

    # -- foregrounds --------------------------------------------------------
    window_text: Color | None = None
    text: Color | None = None
    bright_text: Color | None = None
    button_text: Color | None = None
    tooltip_text: Color | None = None
    placeholder_text: Color | None = None

    # -- selection ----------------------------------------------------------
    highlight: Color | None = None
    highlighted_text: Color | None = None

    # -- links --------------------------------------------------------------
    link: Color | None = None
    link_visited: Color | None = None

    # -- 3D bevel / shadow --------------------------------------------------
    light: Color | None = None
    midlight: Color | None = None
    dark: Color | None = None
    mid: Color | None = None
    shadow: Color | None = None

    # -- accent (Qt 6.6+) --------------------------------------------------
    accent: Color | None = None


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

    # -- text ---------------------------------------------------------------
    text_primary: Color = Color(0xE0, 0xE0, 0xE0)
    text_secondary: Color = Color(0xA0, 0xA0, 0xA0)
    text_disabled: Color = Color(0x70, 0x70, 0x70)

    # -- borders ------------------------------------------------------------
    border_subtle: Color = Color()
    border_default: Color = Color()

    # -- accent -------------------------------------------------------------
    accent: Color = Color()

    # -- semantic status ----------------------------------------------------
    status_green: Color = Color()
    status_red: Color = Color()
    status_amber: Color = Color()
    status_yellow: Color = Color()

    # -- spacing (base values, 4px grid) ------------------------------------
    sp_xxs: int = 4
    sp_xs: int = 8
    sp_sm: int = 12
    sp_md: int = 16
    sp_lg: int = 24

    # -- metrics ------------------------------------------------------------
    row_height: int = 36
    sidebar_width: int = 260
