# Roadmap to Semantic Scaling

## Where we are

The current system has three things working well:

**A clean scaling infrastructure.** Zoom is implemented at the architecturally
correct layer — `QProxyStyle.pixelMetric()` scales Qt-native metrics, app font
scaling handles text, and `ScaledThemeView` provides live-scaled tokens to widget
code. The `proxy()` chain means `sizeFromContents()`, `subElementRect()`, and
Fusion's drawing code all scale without any per-widget work. This is genuinely
hard to get right and it's done.

**A pure data model.** `Theme` is a plain dataclass with no Qt imports, no zoom
awareness, and no behavior. It can be serialized, diffed, and swapped at runtime.
The `ScaledThemeView` proxy keeps the scaling concern separate. This separation
is exactly what professional systems do.

**Color tokens with full coverage.** Every color in the UI traces back to a named
token on `Theme`. There are no stray `QColor(0x...)` literals in widget paint
code (with a few deliberate exceptions for viewport overlays). Swapping to a
light theme or a high-contrast theme would only require a new `Theme` instance.

The gap is on the spatial side. We have a primitive spacing scale (`sp_xxs`
through `sp_xl`) and a handful of metrics (`radius`, `row_height`,
`sidebar_width`), but widget authors are choosing spatial values by intuition
rather than by role. The `scaled(N)` escape hatch accounts for roughly 16
distinct magic numbers across the codebase — each one a design decision made
locally that should have been made centrally.

## What professional token systems add

Systems like Material Design, Carbon (IBM), Spectrum (Adobe), and Fluent
(Microsoft) all use a two-tier spatial architecture:

### Tier 1: Primitive scale

A fixed set of unitless values on a consistent grid. We already have this:

```
sp_xxs=4  sp_xs=8  sp_sm=12  sp_md=16  sp_lg=24  sp_xl=32
```

Primitives are the building blocks. They define the grid, not the usage. Widget
authors should rarely reference them directly — they're the "private API" that
semantic tokens are built from.

### Tier 2: Semantic tokens

Named values that encode *intent*, not magnitude. Instead of "use 12 pixels,"
the system says "use the standard panel inset" — which happens to be 12 pixels
today but could change without touching any widget code.

Semantic tokens fall into a few natural categories:

**Control geometry** — how tall interactive elements are. Most applications have
2-3 tiers: small (icon buttons, inline toggles), standard (buttons, inputs),
and large (panel headers, prominent actions). Our codebase currently expresses
these as `scaled(22)`, `scaled(26)`, `scaled(28)`, `row_height` — four values
that are really two or three tiers chosen inconsistently.

**Gaps** — space between sibling elements. The question isn't "how many pixels"
but "how tightly related are these items?" Tightly related items (buttons in a
button group) get a tight gap. Items in the same toolbar section get an inline
gap. Sections separated by a visual divider get a section gap.

**Insets** — internal padding of containers. A toolbar's vertical padding, a
panel body's margins, a page-level margin. These are distinct from gaps because
they describe the relationship between a container and its content, not between
siblings.

**Control padding** — space between a control's edge and its content. How much
horizontal room does a button give its label? This is distinct from insets
(which are about containers) and gaps (which are between siblings).

### What the two tiers buy you

The primitive scale is stable — you almost never change it. The semantic layer
is where design evolution happens. "Make the UI more compact" means adjusting
semantic tokens, not grepping for pixel values. "Make toolbars breathe more"
means changing `inset_toolbar`, not finding every `setContentsMargins` call in
every toolbar-like widget.

Critically, two developers independently building toolbar-like widgets will make
the **same** spatial choices if the semantic tokens exist, because the token name
tells them what to pick. Without semantic tokens, they both look at the primitive
scale, make slightly different judgment calls, and you discover the inconsistency
months later.

## What to add now (high value, low cost)

These tokens cover roughly 80% of the current `scaled(N)` usage across the
codebase. Each one maps directly to values already in use — this is naming and
consolidating, not inventing.

```python
@dataclass
class Theme:
    # ── Primitive scale (keep, unchanged) ────────────────────────
    sp_xxs: int = 4
    sp_xs: int = 8
    sp_sm: int = 12
    sp_md: int = 16
    sp_lg: int = 24
    sp_xl: int = 32

    # ── Control geometry ─────────────────────────────────────────
    control_height_sm: int = 22    # ChannelButton, small toolbar btns
    control_height: int = 28       # ToolbarButton, standard controls
    control_height_lg: int = 36    # panel headers, prominent rows
    # row_height becomes an alias: row_height = control_height_lg

    # ── Gaps ─────────────────────────────────────────────────────
    gap_tight: int = 2             # ChannelStrip button spacing
    gap_inline: int = 6            # items within a toolbar row
    gap_section: int = 12          # between logical groups

    # ── Insets ───────────────────────────────────────────────────
    inset_toolbar: int = 4         # vertical padding inside toolbars
    inset_panel: int = 12          # inside collapsible panel bodies
```

The implementation cost is small: add the fields to `Theme`, add the `@property`
lines to `ScaledThemeView`, and migrate widget code incrementally. Each migration
replaces a `scaled(N)` call with a named token — no behavioral change, just a
better name.

### What this eliminates

Before:

```python
# Three different widgets, three different "small button height" values
self.setFixedHeight(t.scaled(22))   # ChannelButton
self.setFixedHeight(t.scaled(26))   # ToolbarButton
self.setFixedHeight(t.scaled(28))   # ChannelStrip height
```

After:

```python
# The system defines the tiers; widgets just pick one
def sizeHint(self):
    return QSize(..., theme().control_height_sm)    # ChannelButton

def sizeHint(self):
    return QSize(..., theme().control_height)        # ToolbarButton
```

The 26-vs-28 question is resolved once in `Theme`, not re-debated in every
widget.

## What to add later (real value, but only after the widget count grows)

### Control padding tokens

```python
control_padding_h: int = 12    # horizontal padding inside controls
control_padding_v: int = 4     # vertical padding inside controls
```

Currently some widgets compute padding from `sp_lg`, others from `sp_sm * 2`,
others from `scaled(12)`. These are all expressing the same concept. Worth
adding once you have 10+ custom controls — premature with 5.

### Density modes

Once semantic tokens exist, a compact mode is trivial:

```python
COMPACT = Theme(
    control_height_sm=18,
    control_height=24,
    control_height_lg=30,
    inset_panel=8,
    gap_inline=4,
    # everything else inherited from default
)
```

This is not worth building until users ask for it, but the semantic token
architecture makes it a one-hour task rather than a full audit. Worth knowing
that the investment pays forward.

### Typography scale

Professional systems define font sizes semantically: `caption`, `body`,
`subheading`, `heading`. We currently have raw point sizes (`8`, `10`, `11`)
passed to `ui_font()`. A typography scale would replace:

```python
p.setFont(ui_font(8, QFont.Weight.Medium))   # what is 8pt? a label? a caption?
```

with:

```python
p.setFont(t.font_caption)   # clear intent
```

This is worth adding once you have enough distinct text styles to see the
pattern (probably 4-5 distinct size/weight combinations). Right now you
effectively have three: body (10pt normal), caption (8pt medium), and
heading (11pt demibold). That's borderline — naming them wouldn't hurt but
the current `ui_font(8, Weight.Medium)` calls are clear enough.

## What Qt gives us for free — do not reinvent

Several things that look like gaps in the token system are actually handled by
Qt's style machinery. Duplicating them in `Theme` would create two sources of
truth that can diverge.

### Layout spacing and margins (default case)

When a `QLayout` has no explicit margins or spacing set (the value is -1), it
queries `QStyle::pixelMetric()` for `PM_LayoutHorizontalSpacing`,
`PM_LayoutVerticalSpacing`, and `PM_Layout*Margin`. Our `MicroscopeStyle`
scales all of these by zoom automatically.

**Do not** add theme tokens for default layout spacing. If a layout's spacing
matches the platform default, leave it at -1 and let the style handle it.
Only use theme tokens for layouts that genuinely need non-standard spacing.

### Widget chrome dimensions

Scrollbar width, checkbox indicator size, slider handle dimensions, combobox
arrow size, button frame width, spinbox arrow size, tab bar padding, menubar
item spacing, toolbar handle extent — all of these are `pixelMetric` values
that scale automatically through `MicroscopeStyle.pixelMetric()`.

**Do not** add theme tokens for any standard widget dimension that Qt already
defines as a `PixelMetric`. If you need to override a specific metric (like our
thin scrollbars), use `MicroscopeStyle._BASE_OVERRIDES`. The style is the
correct owner of these values, not `Theme`.

### Icon sizing for standard widget roles

`PM_SmallIconSize`, `PM_ToolBarIconSize`, `PM_ButtonIconSize`,
`PM_TabBarIconSize` — these are all pixel metrics that scale through the style.
Don't add icon size tokens to Theme. If a widget needs a non-standard icon size,
use `theme().scaled(N)` as the escape hatch.

### Focus rectangles, selection highlights, text cursors

All drawn by `QStyle::drawPrimitive()` and scale via the proxy chain. No token
needed.

### Font metrics in layout calculations

`QFontMetrics` / `QFontMetricsF` are derived from the current font, which
scales via Pillar 2. Any `sizeHint()` that measures text via font metrics
automatically produces a zoom-correct value. This is the right way to size
widgets that are primarily text-driven — measure the text, don't hardcode.

### The general principle

If Qt's style system already scales something via `pixelMetric()` or font
metrics, let it. `Theme` should only contain tokens for spatial concepts that
are **ours** — application-level design decisions like "how tall is a panel
header" or "how much padding around a collapsible panel's body." The boundary
is: Qt owns widget chrome, we own application layout.

## What `scaled()` should be for

After adding semantic tokens, `scaled(N)` should be rare. Legitimate uses:

- Stroke widths and dot radii in custom paint code (`scaled(3.5)` for a status
  dot radius)
- One-off visual elements that don't recur (a scale bar thickness, a chevron
  size)
- Prototyping a value before deciding whether to promote it to a token

If you find yourself writing `scaled(N)` with the same N in three or more
places, that's the signal to promote it. If the N appears once, leave it.

## Summary of recommendations

| Priority | Action | Effort |
|----------|--------|--------|
| **Now** | Add `control_height_sm/md/lg`, `gap_tight/inline/section`, `inset_toolbar/panel` | 8 new fields + ScaledThemeView props |
| **Now** | Migrate existing `scaled(22/26/28/2/6/4/12)` to semantic tokens | Find-and-replace per widget |
| **Now** | Deprecate direct use of primitive tokens in widget code | Documentation/review convention |
| Later | Add `control_padding_h/v` | When control count exceeds ~10 |
| Later | Add typography scale (`font_caption`, `font_body`, `font_heading`) | When text styles exceed ~4 |
| Later | Density modes (compact/comfortable) | When users request it |
| Never | Duplicate `pixelMetric` values as theme tokens | Qt owns widget chrome |
| Never | Add layout-default spacing to Theme | Let `PM_Layout*Spacing` handle it |
