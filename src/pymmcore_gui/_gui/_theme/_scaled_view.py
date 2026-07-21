"""ScaledThemeView — live zoom-scaled view of Theme tokens.

Theme stays a pure, unscaled dataclass. This proxy computes scaled values
on every access using the current zoom factor from MicroscopeStyle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._style import MicroscopeStyle
    from ._types import Color, Palette, Theme


class ScaledThemeView:
    """Live view: theme tokens x current zoom factor.

    Every attribute access reads the *current* zoom, so values are never
    stale. This is intentionally not a dataclass — it's a computed view,
    not stored state.
    """

    __slots__ = ("_style", "_theme")

    def __init__(self, theme: Theme, style: MicroscopeStyle) -> None:
        self._theme = theme
        self._style = style

    # ── zoom ──────────────────────────────────────────────────────

    @property
    def zoom_factor(self) -> float:
        """Current zoom factor (public, for font helpers etc.)."""
        return self._style.zoom_factor

    def scaled(self, val: int | float) -> int:
        """Scale an arbitrary pixel value by the current zoom factor."""
        return max(1, round(val * self._style.zoom_factor))

    # ── scaled spacing ────────────────────────────────────────────

    @property
    def sp_xxs(self) -> int:
        return self.scaled(self._theme.sp_xxs)

    @property
    def sp_xs(self) -> int:
        return self.scaled(self._theme.sp_xs)

    @property
    def sp_sm(self) -> int:
        return self.scaled(self._theme.sp_sm)

    @property
    def sp_md(self) -> int:
        return self.scaled(self._theme.sp_md)

    @property
    def sp_lg(self) -> int:
        return self.scaled(self._theme.sp_lg)

    @property
    def sp_xl(self) -> int:
        return self.scaled(self._theme.sp_xl)

    # ── scaled metrics ────────────────────────────────────────────

    @property
    def radius(self) -> int:
        return self.scaled(self._theme.radius)

    @property
    def radius_lg(self) -> int:
        return self.scaled(self._theme.radius_lg)

    @property
    def row_height(self) -> int:
        return self.scaled(self._theme.row_height)

    @property
    def sidebar_width(self) -> int:
        return self.scaled(self._theme.sidebar_width)

    # ── color passthrough (unscaled) ──────────────────────────────

    @property
    def palette(self) -> Palette:
        return self._theme.palette

    @property
    def bg_deepest(self) -> Color:
        return self._theme.bg_deepest

    @property
    def bg_base(self) -> Color:
        return self._theme.bg_base

    @property
    def bg_raised(self) -> Color:
        return self._theme.bg_raised

    @property
    def bg_surface(self) -> Color:
        return self._theme.bg_surface

    @property
    def bg_hover(self) -> Color:
        return self._theme.bg_hover

    @property
    def bg_active(self) -> Color:
        return self._theme.bg_active

    @property
    def text_primary(self) -> Color:
        return self._theme.text_primary

    @property
    def text_secondary(self) -> Color:
        return self._theme.text_secondary

    @property
    def text_disabled(self) -> Color:
        return self._theme.text_disabled

    @property
    def border_subtle(self) -> Color:
        return self._theme.border_subtle

    @property
    def border_default(self) -> Color:
        return self._theme.border_default

    @property
    def border_focus(self) -> Color:
        return self._theme.border_focus

    @property
    def accent(self) -> Color:
        return self._theme.accent

    @property
    def accent_muted(self) -> Color:
        return self._theme.accent_muted

    @property
    def status_green(self) -> Color:
        return self._theme.status_green

    @property
    def status_red(self) -> Color:
        return self._theme.status_red

    @property
    def status_amber(self) -> Color:
        return self._theme.status_amber

    @property
    def drag_highlight(self) -> Color:
        return self._theme.drag_highlight

    @property
    def drop_indicator(self) -> Color:
        return self._theme.drop_indicator
