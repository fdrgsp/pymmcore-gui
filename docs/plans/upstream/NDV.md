# ndv: upstream PRs

`fdrgsp/ndv@cite` against `pyapp-kit/ndv@main`, merge base `f30ec34`
(2026-10-06). See [README.md](README.md) for the conventions.

None of these needs a PR in another repository. pymmcore-gui is the consumer:
the center cross is used by its pixel calibration panel and viewers, and the
ROI and zoom helpers by its camera ROI sync and preview
(`_camera_roi_sync.py`, `_array_viewer.py`, `widgets/image_preview/`).

The fork adds:

- **Qt fixes**: main-thread dispatch, the progress spinner, sliders left over
  after a data change (NDV-1 to NDV-3).
- **Dimension slider usability** (NDV-4).
- **Histogram drawing fixes**: label clipping, collapsed views (NDV-5, NDV-6).
- **Opinionated histogram changes** that need agreement: the shared histogram
  legend and the highlight style (NDV-7, NDV-8).
- **Viewer controls**: the Backspace camera reset, a center cross, ROI and zoom
  helpers on `ArrayViewer` (NDV-9 to NDV-11).

## Summary

<!-- markdownlint-disable MD013 -->
| ID | Type | Title | Size | Needs | Notes |
| --- | --- | --- | --- | --- | --- |
| NDV-1 | fix | Main-thread dispatch survives an invalidated thread wrapper | S | – | – |
| NDV-2 | fix | Progress spinner stays at the canvas' top right | S | – | – |
| NDV-3 | fix | Replacing data drops stale sliders and refits the shared histogram | S | – | – |
| NDV-4a | fix | Dimension slider: click jumps, one step per wheel notch | S | – | – |
| NDV-4b | feat | Dimension slider: 1-based "n / N" and one handle width per index | S | NDV-4a | issue first |
| NDV-5 | fix | Histogram axis labels are not clipped | S | – | – |
| NDV-6 | fix | No singular camera transform in a collapsed histogram | S | – | – |
| NDV-7 | feat | Shared histogram names follow the data, without the in-plot legend | S | NDV-3 | issue first |
| NDV-8 | style | Histogram highlight: white, 2 px | S | – | issue first |
| NDV-9 | fix | Backspace no longer resets the camera to its pre-data range | S | – | – |
| NDV-10 | feat | Center cross overlay | M | – | – |
| NDV-11 | feat | ROI and zoom helpers on `ArrayViewer` | S | NDV-10 | coordinate with #285 |
| NDV-12 | test | No automatic GC in the middle of a paint | S | – | – |
<!-- markdownlint-enable MD013 -->

---

## NDV-1: main-thread dispatch survives an invalidated thread wrapper

*fix · S · `f5b8ce3`, and `_main_thread.py` from `888807d`*

`_main_thread.py` cached `QAPP.thread()` at import. Once PySide6 invalidated
that wrapper, `QThread.currentThread() is not _APP_THREAD` was true even on the
main thread, so calls went through an invoker that never ran, and their futures
never completed. Look the thread up on each call, and raise for a missing
`QApplication` when dispatching rather than at import. The invoker clears its
callable and deletes itself with `deleteLater()` on its own thread.

## NDV-2: progress spinner stays at the canvas' top right

*fix · S · `210d41d`*

The viewer placed the spinner in its own `resizeEvent`, which runs before its
layout resizes the canvas, and not at all when only the canvas resizes (a
splitter or dock move). The spinner now follows its parent's resize events
through an event filter.

## NDV-3: replacing data drops stale sliders and refits the shared histogram

*fix · S · `9acc105`*

Replacing RGB `(y, x, 3)` data with mono `(y, x)` data left the third slider
behind, and the shared histogram kept the first data's range. `create_sliders`
removes axes that no longer exist, and `SharedHistogramCanvas.reset_range()`
makes the next data fit as the first did.

## NDV-4a: dimension slider: click jumps, one step per wheel notch

*fix · S · `0ae83c6` and the wheel part of `3eba584`*

- A click on the groove jumps to that index. Under Fusion-based styles it
  page-stepped toward the cursor.
- A mouse wheel moves one index per notch (120 units, remainders carried over).
  Trackpads, which send pixel deltas, keep Qt's own handling.

## NDV-4b: dimension slider: 1-based "n / N" and one handle width per index

*feat · S · `3eba584`, `8008b51` · needs NDV-4a · open an issue first*

The label shows `n / N` (1-based) instead of `index / last index`, and the
handle gets an equal share of the groove per index, so it shows where it is
among few indices. This changes what users read, so ask first.

## NDV-5: histogram axis labels are not clipped

*fix · S · `8e7205c`, `cc1775a`, `c824431`, `fb7050b`*

- The y-axis width is estimated at about 8 px per glyph (5 px clipped four-digit
  labels on high-DPI displays), and with log ticks it is computed from the
  shown counts, not the log domain. Width steps go up to 74 px.
- Both width bounds are pinned and the grid solver rebuilt, so an existing
  layout grows too.
- The right padding reserves half of the last x tick label (e.g. "65000" on a
  uint16 histogram was cut off, most visibly on Windows).

## NDV-6: no singular camera transform in a collapsed histogram

*fix · S · `888807d`, `8708bb3`*

`PanZoom1DCamera._update_transform` waits until the viewbox spans at least one
pixel, since a layout can collapse it to zero or a sub-pixel residual before the
first resize (e.g. in Jupyter). `pan()` passes plain floats.

## NDV-7: shared histogram names follow the data, without the in-plot legend

*feat · S · `b9773a3`, `9123bc3` · needs NDV-3 · open an issue first*

- `_SharedHistogramLink.set_fallback_name`: a channel's name follows the
  data-derived fallback name as the data coordinates change, unless the LUT has
  an explicit name.
- The vispy and pygfx shared histograms drop the in-plot legend text. Ask
  first: some users may rely on the legend.

## NDV-8: histogram highlight: white, 2 px

*style · S · the highlight part of `b9773a3` · open an issue first*

The highlight line is white at 0.9 alpha and 2 px wide, in both the
per-channel and shared histograms, instead of a thin yellow or channel-colored
line. The TODO it removes asked for something "easier on the eyes".

## NDV-9: Backspace no longer resets the camera to its pre-data range

*fix · S · from `d2aa16a`*

vispy binds Backspace to restoring the camera's initial range, which can make
the image vanish. ndv creates the cameras, so it disconnects that binding from
each new camera.

## NDV-10: center cross overlay

*feat · M · from `d2aa16a`, `fb7050b`*

- `ArrayViewerModel.show_center_cross_button` (default False) and
  `center_cross_visible`.
- `ArrayCanvas.set_center_cross(visible)` on the base class, implemented for
  vispy and pygfx. It draws a yellow cross through the middle of the shown 2D
  image, follows the image transform, and is hidden in 3D.
- A checkable Qt button (`mdi:crosshairs-gps`) kept in sync with the model.

## NDV-11: ROI and zoom helpers on `ArrayViewer`

*feat · S · from `d2aa16a` · needs NDV-10 · coordinate with upstream #285*

`set_center_cross_active`, `center_cross_active`, `reset_zoom`, `clear_roi`,
`set_roi_selection_active`, `roi_selection_active`,
`set_existing_roi_editing_active`, `existing_roi_editing_active`,
`set_roi_visual_selected`, `roi_visual_selected`, `roi_visual_visible`, and
`connect_roi_selection_changed` with its `disconnect_` counterpart. They let an
application drive the ROI and zoom tools without private attributes.

Upstream #285 (kephale) also exposes canvas and camera APIs: agree on names and
scope there first, and drop whatever it already covers.

## NDV-12: no automatic GC in the middle of a paint

*test · S · `23d5a05`*

A 3.10 job segfaulted painting a range slider of a viewer left behind by an
example. `conftest.py` disables automatic GC and collects between tests, as
pymmcore-gui does.
