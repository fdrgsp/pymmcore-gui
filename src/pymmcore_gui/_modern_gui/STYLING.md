# Styling & Zoom System

This application uses a three-pillar zoom architecture that scales the entire UI
uniformly, like VS Code's `Cmd+/−`. Understanding how it works — and what breaks
it — is essential for anyone writing or modifying widgets.

## How zoom works

Three systems cooperate, all driven by a single `zoom_factor` on `MicroscopeStyle`:

1. **`QProxyStyle.pixelMetric()`** — scales all Qt-standard widget metrics
   (button padding, scrollbar width, checkbox indicators, layout spacing
   defaults). This propagates automatically through `sizeFromContents()`,
   `subElementRect()`, and all Fusion drawing code via Qt's `proxy()` chain.

2. **`QApplication.setFont()`** — scales the app-wide default font. Every widget
   that inherits the app font (i.e. hasn't called `setFont()` explicitly) picks
   this up automatically.

3. **Explicit icon size updates** — widgets like `QToolBar` cache their icon
   dimensions and must be told to update.

When the user presses the zoom shortcut, `set_zoom()` updates all three pillars
and then calls `updateGeometry()` + sends `StyleChange` to every widget in the
app. This means the layout system re-queries `sizeHint()` on every widget, and
every widget repaints.

## The Theme and ScaledThemeView

`Theme` is a pure dataclass holding unscaled design tokens: colors, spacing,
radii, and metrics. It has no awareness of zoom and no Qt imports.

`ScaledThemeView` is a live proxy that wraps `Theme` + the style. Spatial tokens
(spacing, radii, heights) are multiplied by the current zoom factor on every
access. Color tokens pass through unchanged.

```python
t = theme()          # returns ScaledThemeView
t.sp_sm              # → 12 at 1.0x, 18 at 1.5x, 24 at 2.0x
t.accent             # → Color(0x4A, 0x9E, 0xFF) always
t.scaled(42)         # escape hatch for ad-hoc pixel values
```

## Rules for writing widgets

### Sizing: use `sizeHint()`, not `setFixed*()`

`setFixedHeight()` / `setFixedWidth()` / `setFixedSize()` bake a pixel value
into the widget. That value is frozen until someone explicitly calls the setter
again. This means you need a `changeEvent(StyleChange)` override to keep it
correct on zoom — which is boilerplate and easy to forget.

Instead, override `sizeHint()` and `minimumSizeHint()`:

```python
class MyWidget(QWidget):
    _BASE_HEIGHT = 28

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # no setFixedHeight!

    def sizeHint(self):
        return QSize(super().sizeHint().width(), theme().scaled(self._BASE_HEIGHT))

    def minimumSizeHint(self):
        return self.sizeHint()
```

The layout system re-queries `sizeHint()` automatically on zoom change (because
`set_zoom()` calls `updateGeometry()` on every widget). The value is never
cached — it's computed fresh each layout pass.

Store base pixel values as class-level constants (`_BASE_HEIGHT = 28`), not
computed at init time.

### Painting: read `theme()` live, every frame

`paintEvent()` is called after every zoom change. As long as you read from
`theme()` inside `paintEvent()`, all spatial values are automatically correct:

```python
def paintEvent(self, event):
    p = QPainter(self)
    t = theme()
    p.setPen(qcolor(t.border_subtle))
    p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), t.radius, t.radius)
    #                                                      ^^^^^^^^
    #                                        scaled automatically
```

**Do not** cache `theme()` values as instance attributes during `__init__`:

```python
# WRONG — frozen at init-time zoom level
def __init__(self):
    self._radius = theme().radius

def paintEvent(self, event):
    p.drawRoundedRect(..., self._radius, self._radius)  # stale after zoom

# RIGHT — always fresh
def paintEvent(self, event):
    t = theme()
    p.drawRoundedRect(..., t.radius, t.radius)
```

### Fonts: let the app cascade do the work

`QApplication.setFont()` propagates to every widget that hasn't explicitly
called `setFont()`. This means most widgets get zoom-scaled text for free.

Only use `ui_font()` or `mono_font()` when you need a **non-default** font —
different size, weight, or family. These functions scale internally:

```python
# Widget needs demibold 10pt — use ui_font(), which scales by zoom
p.setFont(ui_font(10, QFont.Weight.DemiBold))

# Widget just uses the normal app font — don't call setFont() at all
# The app cascade handles it
```

**Do not** call `widget.setFont(ui_font(10))` with default size and weight.
This sets `WA_SetFont`, opting the widget out of the app font cascade, meaning
it won't track future `QApplication.setFont()` changes from Pillar 2. You'd
then need a `changeEvent` to re-apply the font on zoom — all for no benefit.

If a widget must call `setFont()` (because it needs a custom font), it must
re-apply the font on zoom. The `StyleChange` event fires automatically:

```python
def changeEvent(self, event):
    if event is not None and event.type() == QEvent.Type.StyleChange:
        self._some_label.setFont(mono_font(8))  # re-apply scaled font
    super().changeEvent(event)
```

### Layout margins and spacing

Layouts with default (`-1`) margins and spacing automatically query
`pixelMetric()`, which scales via the style. **This is the best case — layouts
just work with zero effort.**

If you must set explicit margins or spacing, those values are baked in and won't
auto-update on zoom. You have two options:

1. Use theme tokens and add a `changeEvent` to reapply:

```python
def __init__(self):
    t = theme()
    self.layout().setContentsMargins(t.sp_sm, 0, t.sp_sm, 0)

def changeEvent(self, event):
    if event is not None and event.type() == QEvent.Type.StyleChange:
        t = theme()
        self.layout().setContentsMargins(t.sp_sm, 0, t.sp_sm, 0)
    super().changeEvent(event)
```

1. Avoid explicit values entirely — let the style defaults handle it.

Prefer option 2 wherever possible. Option 1 is the correct fallback for layouts
that genuinely need non-standard spacing.

### Raw pixel literals

Every pixel literal in a `paintEvent` is a potential zoom bug. Use `theme()`
tokens or `theme().scaled(N)` for all spatial values:

```python
# WRONG — 4px gap regardless of zoom
p.drawText(x + 4, y, text)

# RIGHT
p.drawText(x + t.sp_xxs, y, text)

# RIGHT (ad-hoc value that isn't a design token)
p.drawText(x + t.scaled(5), y, text)
```

If you find yourself writing `t.scaled(N)` with the same `N` in three or more
places, promote it to a named token on `Theme`.

### Raster assets (pixmaps, icons)

Pixmaps pre-scaled to a fixed size will look wrong after zoom. If a widget
caches a scaled pixmap, it must store the original and rescale on zoom change.
This is one of the few cases where `changeEvent` is genuinely necessary — there
is no lazy equivalent of `sizeHint()` for raster content:

```python
def __init__(self, pixmap):
    self._original = pixmap
    self._scaled = self._rescale()

def _rescale(self):
    s = theme().scaled(self._BASE_SIZE)
    return self._original.scaled(s, s, ...)

def changeEvent(self, event):
    if event is not None and event.type() == QEvent.Type.StyleChange:
        self._scaled = self._rescale()
    super().changeEvent(event)
```

## When you need `changeEvent` — and when you don't

| Situation | `changeEvent` needed? |
|---|---|
| Widget size comes from `sizeHint()` | No |
| `paintEvent` reads `theme()` live | No |
| Widget inherits app font (no `setFont()` call) | No |
| Layout uses default margins/spacing (`-1`) | No |
| Widget calls `setFont()` with custom font | **Yes** — re-apply font |
| Layout has explicit margins or spacing | **Yes** — re-apply values |
| Widget caches a pre-scaled pixmap | **Yes** — rescale from original |
| Widget calls `setFixedHeight()` etc. | **Avoid** — use `sizeHint()` instead |

## Quick reference

| Want to... | Do this | Don't do this |
|---|---|---|
| Set widget size | Override `sizeHint()` with scaled values | `setFixedHeight()` in init |
| Use a color | `qcolor(theme().accent)` | `QColor(0x4A, 0x9E, 0xFF)` |
| Use spacing | `theme().sp_sm` | `12` |
| One-off pixel value | `theme().scaled(5)` | `5` |
| Normal font | Don't call `setFont()` at all | `setFont(ui_font(10))` |
| Custom font | `setFont(ui_font(8, Weight.Bold))` | `setFont(QFont(...))` |
| Monospace font | `mono_font(8)` | `QFont("Consolas", 8)` |
| Draw a radius | `theme().radius` | `3` or `3.0` |
