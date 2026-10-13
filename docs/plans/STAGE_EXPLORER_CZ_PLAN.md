# Stage Explorer: shared MDA data, channels, and Z browsing

Status: proposed implementation plan; no application code changed.
Reviewed: 2026-10-13.
Supersedes the grid-mosaic acquisition-viewer plan.

## Recommendation and feasibility

**Yes, this is more feasible than adding a spatial mosaic mode to ndv.** Keep
Stage Explorer's existing spatial canvas, tile transforms, pan/zoom, stage marker,
and ROIs. Add channel controls and a slider selecting a 2D Z plane. Leave the
ordinary MDA viewer and its flattened position dimension intact.

**Use the same acquisition store as the MDA viewer.** Stage Explorer should be a
second view of that store, not a second collector of every acquired image. The
app already supplies a readable sink for both saved and unsaved MDA runs. This
also lets Stage Explorer open after acquisition has started and display existing
planes without having captured their original frame signals.

The first milestone is deliberately narrower: **latest run only, scratch storage
(RAM or spilled scratch), one displayed channel, a Z slider, per-channel contrast
and colormap, and zoom-aware downsampling**. Composite rendering and live TIFF/Zarr
browsing are follow-ups. There is no need for a new stitched image array, a second
scratch store, a 3D renderer, or changes to acquisition axis order.

Planning estimates for one developer familiar with the repositories:

| Milestone | Estimate |
| --- | --- |
| Plane-source contract, shared ownership, scratch-backed C/Z prototype | 2–3 working days |
| Zoom-aware display, bounded reads, GUI integration and regression tests | 2–4 additional days |

Allow roughly **one working week (4–7 days) for this first milestone**. These are
planning estimates, not measured delivery times. No new upstream writer-readiness
API is required for the scratch milestone. Estimate composite rendering and
TIFF/Zarr support separately after the first version is validated.

## User-facing behavior

- Preserve the stage map: all spatial positions/grid tiles appear at their XY
  coordinates. Different grid sizes need no rectangular padding or new `p` slider.
- Add an ndv-style compact `z` row: slider, current index, and total plane count.
  Hide it for a single plane. This changes displayed pixels only; it never moves
  the physical Z stage.
- Add a channel selector, colormap, contrast limits, and an Auto button, using
  ndv's visual conventions. Display one channel at a time in this milestone;
  multichannel overlays and per-channel visibility checkboxes come later.
- Remember contrast/colormap per channel while switching Z. Reuse the GUI's
  microscope-channel colormap preferences where group/preset identity is known.
  Display controls must not change hardware channel, exposure, or illumination.
- Bind to the latest scratch-backed MDA run, with no older-run selector. Normal
  MDA and Stage Explorer selections remain independent. The run stays browsable
  after completion until replaced or explicitly detached.
- This deliberately changes the acquisition-map behavior: browsing shows **one
  run**, not accumulated images from several runs. Keep the manual snap map as a
  separate mode; never show it through gaps in an acquisition plane. Binding a
  new run replaces the previous source, not its underlying saved data.
- If the newest run uses TIFF/Zarr or an unsupported sink, label C/Z browsing as
  unavailable for that run. Preserve the existing map-only behavior as a clearly
  separate fallback; do not keep showing an older acquisition as though it were
  the new run, and do not allocate a duplicate scratch store.
- Provide follow/lock behavior: follow incoming acquisition indices; browsing
  C/Z locks selection until follow is re-enabled. Keep pan/zoom stable on plane
  changes. Do not repeatedly fit the map while the user is navigating.
- Use one time index across the map, following acquisition unless locked. No
  dedicated time slider in this scope. Show the selected time index when needed;
  never fill missing tiles/channels silently from a different timepoint.
- Show unavailable planes as gaps/placeholders, not a previous Z plane or a
  fabricated black acquisition. A valid acquired all-zero image remains valid.
- Browsing stays available while an MDA runs, even though hardware actions are
  locked. Hiding the panel suspends reads/uploads; reopening reconstructs the
  selected view from the shared store.
- Zoomed-out maps use reduced-resolution previews; zooming in restores detail.
  This is display-only sampling and never changes the stored acquisition.

Z initially means **acquisition slice index**, not one absolute stage height
across the whole map. Positions can have different base Z, autofocus corrections,
and channel Z offsets. Show physical Z per tile/hover when known; show one global
µm label only when it is genuinely common. A shorter local stack has missing
planes at higher indices, not a silently clamped last slice.

## Findings in the current code

| Component | Relevant evidence |
| --- | --- |
| [`AcquireViewersManager`](../../src/pymmcore_gui/_ndv_viewers.py) | Creates `MMArrayViewer` from `core.mda.get_view()`. Holds the run's sink and acquisition record, but cleanup is currently tied to closing its viewer. Its `(p, g)` mapping is currently updated only while following acquisition. |
| [`MemoryMDAWidget`](../../src/pymmcore_gui/widgets/_mda_widget.py) | Supplies scratch-backed acquisition settings when file saving is disabled, so normal GUI MDA runs already have a store. |
| [`AcquisitionRecord`](../../src/pymmcore_gui/_mda_export.py) | Holds resolved settings, summary metadata, a view, and export frame metadata. It is a useful starting point, not yet a shared resource-lifetime manager. |
| [`ThemedStageExplorer`](../../src/pymmcore_gui/widgets/_stage_explorer.py) | Wraps the upstream spatial widget and controls theming, calibration gating, and hardware-action locking during MDA. |
| `pymmcore-widgets/.../_stage_explorer/_stage_explorer.py` | `_LatestFrameRelay`, `_mda_tile_key`, and `_pending_tiles` coalesce by spatial location. C/Z frames overwrite one another. Collection stops when hidden. Simply adding sliders cannot recover discarded planes. |
| `pymmcore-widgets/.../_stage_explorer/_stage_viewer.py` | Already renders affine-transformed image tiles. Contrast/colormap are currently global. Viewport culling directly sets node visibility, so independently hidden channel/Z nodes would otherwise be shown again. |
| `ome-writers/_stream_view.py` | Integer-indexing one position and all non-image axes retrieves one plane. Slicing multiple positions stacks arrays. `coords_changed` reports extent growth, not readiness of every individual frame. |
| `ome-writers/_backends/_scratch.py` | In-memory reads return read-only views. Memmap reads deliberately copy requested slices for safe file resizing/cleanup; those copies cannot simply be removed. |
| `pymmcore-plus/mda/_runner.py`, `ome-writers/_backends/` | The runner calls sink append before emitting `frameReady`. Scratch writes synchronously, including memmap-backed scratch; TensorStore uses futures and TIFF uses writer threads. Submission/coordinate growth is not proof of read completion for those asynchronous backends. |
| `ndv/views/_qt/_array_view.py`, [`ChannelLUTMemory`](../../src/pymmcore_gui/_channel_luts.py) | Provide the control design and colormap preference behavior to match. ndv's dimension-widget implementation is private. |

A small `uv run --no-sync` probe against the installed environment created an
in-memory `p × c × z × y × x` acquisition. Two reads of the same 2D plane returned
read-only arrays for which `numpy.shares_memory` was true. This verifies sharing
for that backend, not zero-copy disk reads or GPU rendering. Sibling source
checkouts were also inspected; align their revisions with the lockfile before
implementation and repeat backend-specific checks.

## Efficient data architecture

```text
MDA writer → one scratch store (RAM or scratch disk, first milestone)
                     ↑                         ↑
                ndv viewer                Stage Explorer
                     └── shared acquisition session ──┘
                         metadata + plane access
```

### 1. One run record with a simple consumer count

Introduce one small run record held by `AcquireViewersManager`, containing the
exact run's sink/view, sequence/settings, geometry, plane availability, storage
locators, and a set of registered consumers. Both views hold the same record;
no general-purpose lease framework is needed. Do not fetch
`core.mda.get_view()` on every slider change: after a new run it refers to another
dataset.

- Create/register the session at sequence start, independently of panel visibility
  and follow mode. Move indexing/metadata ownership out of viewer-only callbacks.
- Release data only when acquisition has finished and the consumer set is empty.
  Count attached views, pending reads, and exports. Closing the MDA tab must not
  release Stage Explorer's data, and vice versa. Remove completed jobs reliably,
  including failed or cancelled jobs.
- Finalize the writer normally at acquisition end. Finalizing writes and releasing
  readable data are different operations; do not keep a writer open just because
  a viewer remains open.
- A new run detaches the old Stage Explorer consumer and cancels its
  reads. Release old scratch resources only after the last consumer/read is done.
  Hide/show keeps the selected session; explicit detach/clear/shutdown releases it.
- Route existing viewer-close and deferred-release paths through this ownership
  policy, including cancellation, export, and application shutdown.

Register the latest supported source even when Stage Explorer is hidden, so
opening it later does not depend on intercepted images. Retain at most one run
on its behalf. Other open MDA viewers retain their own runs independently.

### 2. Lightweight spatial/plane index, not another image collection

Keep tile records mapping acquisition identity to stored-position index, stage
XY, bounds, and pixel-to-stage transform. A plane locator includes session,
stored position, time, channel, Z, and camera where applicable. Build locators
from the writer's resolved layout, not an independent guessed traversal order.

- Preserve `(p, g)` identities where available, but render directly from stored
  spatial locations; there is no need to reverse-flatten the ndv array itself.
- Validate mixed grids, alternate axis orders, skipped frames, repeated XY, and
  subsequences. Two acquisition locations at the same XY must not accidentally
  share one storage identity through the manual-snap proximity deduplication.
- Capture calibration, image shape/ROI, and actual XYZ when available (otherwise
  planned coordinates) at acquisition time. Reading another plane must not
  reinterpret it using the microscope's current calibration or stage position.
- Track submitted/readable/missing/failed states independently of pixel values.
  Keep essential placement/availability metadata even while both views are hidden.
- Keep channel axis slots distinct even if two slots use the same preset; use
  group/preset for preferences, not as the only plane identity.

### 3. Define the widget-facing plane-source contract first

The protocol belongs in `pymmcore-widgets`; the scratch/MDA adapter belongs in
`pymmcore-gui`. It must not expose ndv, a writer backend, or core hardware calls.
An initial contract (illustrative types, not implementation):

```python
class PlaneSource(Protocol):
    changed: Signal  # coalesced layout/availability invalidation, no image payload

    def tiles(self) -> Sequence[TileInfo]: ...
    def channels(self) -> Sequence[ChannelInfo]: ...
    def z_count(self, tile: Hashable) -> int: ...
    def read_plane(
        self, tile: Hashable, *, c: int, z: int, t: int, step: int
    ) -> PlaneResult: ...
```

- `TileInfo`: stable key, original Y/X shape, pixel-to-stage 4×4 affine and bounds.
  `ChannelInfo`: stable axis-slot key and display label, not just preset text.
- `PlaneResult`: explicit ready/missing/pending/error status, optional read-only
  2D array, and placement metadata when that plane differs from the tile default.
  Do not overload zero-valued pixels or `None` to represent every state.
- `step >= 1` requests display subsampling without changing original coordinates.
  The widget adjusts texture placement to preserve the original footprint.
- `read_plane` may block on scratch-disk reads: invoke it through a small bounded
  worker queue. Create/update Qt and VisPy objects only on the GUI thread.
- Connect `changed` through a GUI-safe bridge. Refresh layout/availability even
  while follow is locked; only schedule pixel reads when the panel is visible.
- The adapter fixes the source's camera selection and isolates acquisition
  metadata/indexing. Request generation/cancellation and consumer ownership remain
  explicit host responsibilities, not hidden side effects of reading a plane.

### 4. Pull requested planes; no duplicate MDA frame cache

Stage Explorer asks the session for one stored-position plane at a time, with
integer indices for non-image axes and slices only for Y/X. Never call
`np.asarray(view)`, `view[:]`, or gather all positions into a stitched array.

- For RAM-backed acquisition, reuse read-only array views whenever the backend
  permits. Do not modify acquired arrays to apply LUTs, normalize, or composite.
- For spilled scratch, use bounded background reads and keep only the current
  visible working set. Apply Y/X strides in the source read before the backend's
  required memmap copy. No neighbor-Z prefetch in this milestone.
- Coalesce display invalidations as identifiers, not queued full image payloads.
  The store retains every plane; the UI may discard obsolete render requests.
  **Disable the existing independent MDA-image retention path when first binding
  the source**, before accepting frames, so even the prototype does not maintain
  a competing image map. Restore it only for the explicit legacy/manual mode.
- Tag requests with session/selection generation and discard late results after
  slider changes. Limit outstanding work; do not block the GUI on disk reads.
- Do not add a shared decoded-plane cache or redundant LRU over resident arrays.
  A bounded set of pending display requests and current tile buffers is enough.
- Hidden panels request nothing. Reopening reads from the source, rather than
  depending on missed `frameReady` signals.

For the supported scratch backend, successful `frameReady` follows a synchronous
store write. Use it to register readable locators and invalidate the display;
keep geometry/availability bookkeeping independent of rendering. Skipped frames
remain missing, and valid all-zero frames remain readable. Verify this ordering
against the dependency revision used for implementation.

Reading RAM planes already returns views, so use one source-backed rendering
path initially. Drawing the incoming `frameReady` pixels directly is an optional
later optimization, not a second retained cache. It must obey the selected
session/C/Z/time, request generation, downsampling, and memory budget.

Live TIFF/Zarr browsing is deferred until there is reliable per-plane completion
information. Neither a fixed delay nor "probably written by the next slider move"
is sufficient. A future readiness API must distinguish pending/skipped/failed
planes and trigger retries/final refresh; it is not a prerequisite for scratch.

### 5. Zoom-aware resolution and bounded display memory

Use existing VisPy transforms and tile placement. Maintain image handles only for
the current working set: viewport-intersecting tiles plus a small margin, selected
Z/time, and the selected channel. Fit-to-map uses lightweight bounds for all tiles,
not only currently uploaded images.

- Reuse a tile's image handle when C/Z changes. Keep deterministic spatial overlap
  order across plane reloads, rather than letting asynchronous completion order
  decide which tile appears on top. Composite rendering is not in this milestone.
- Dimension visibility and viewport culling must be combined. The existing culler
  must not re-enable hidden channels or obsolete Z nodes.
- Evict offscreen textures under the display budget, not merely mark them hidden.
  Reuse textures where possible and throttle uploads during rapid slider movement.
- Account separately for acquisition storage, temporary decoded CPU planes, and
  GPU/display buffers. Share existing data/scratch limits; retain a display budget.

**Viewport culling alone is insufficient when the whole map is visible.** Apply
this explicit resolution rule before reading/uploading tiles:

1. Estimate physical screen pixels per original image pixel from the tile affine,
   camera zoom, and device-pixel ratio. Use the largest directional scale for
   rotated/sheared tiles so neither image direction is undersampled unnecessarily.
2. At native magnification or above use `step=1`. Below native magnification choose
   the largest power-of-two step whose sample spacing is at most roughly one
   screen pixel. Read the equivalent of `plane[::step, ::step]` through the source.
3. Preserve the original tile footprint, orientation, and sampling alignment,
   including odd image sizes. Do not let a smaller texture shrink or shift the map.
4. If total display memory still exceeds the budget, increase the step further
   and indicate reduced detail. Bound allocations before uploading, not afterward.
5. Coalesce zoom/slider events and invalidate obsolete results. Refine to `step=1`
   when zooming in; do not retain every resolution level for every plane.

For comparison, 100 full-resolution 2048² uint16 tiles are about 800 MiB for one
channel per Z change (about 1.6 GiB for two). At `step=8`, one channel's sampled
pixels total about 12.5 MiB. This reduces upload size; it is not a latency promise.
Strided RAM views are cheap to construct, but packing/upload still costs work,
and scratch-disk I/O need not decrease in direct proportion to sample count.
Simple subsampling can miss small bright features: label it as an overview and
restore full detail on zoom. Filtered preview generation is a later improvement.
Do not assume ndv provides reusable downsampling code; `_downsample_factors` was
not found in the revisions inspected.

There is **one canonical pixel dataset, not zero extra memory**. Disk decoding,
VisPy's retained plane references, and GPU textures consume memory. Separate
canvases need their own textures; cross-canvas OpenGL texture sharing is not worth
the complexity for this feature. For scale, 100 tiles × 2 channels × 20 Z planes
at 2048² uint16 is about 31.25 GiB before GPU memory. Stage Explorer must never
make a second full-size copy of that dataset.

### 6. UI integration and standalone behavior

Keep generic tile/plane-source hooks in `pymmcore-widgets`; keep MDA session,
store ownership, latest-run binding, and application preferences in
`pymmcore-gui`. Do not add a mandatory ndv/ome-writers dependency to the standalone
Stage Explorer just to reproduce the controls.

Match ndv using the same Qt/superqt building blocks and theme. Prefer a small
public reusable control API if available; avoid importing `_QDimsSliders` or
`_QLUTWidget` directly. Full histogram widgets/playback are not needed initially.
For the initial Auto button, compute min/max over the currently loaded displayed
planes of the selected channel, combining tile ranges into one channel-wide
range. Store it until the user changes it or presses Auto again; avoid per-tile
contrast seams and arrival-order flicker. Label its scope as the displayed data
(possibly downsampled), with no extra full-acquisition scan or histogram system.

Keep manual snaps and no-sink standalone exploration in the existing bounded
map mode. Do not assign them invented C/Z identities or mix autofocus test frames
into an acquisition stack. While browsing an MDA source, manual-map images must
not show through missing planes. Source switching can preserve the manual map
separately under its existing budget; no unlimited history of manual stacks.

Update the GUI wrapper's lock/calibration checks: C/Z/LUT browsing is read-only
and stays enabled during MDA. Retained calibrated data remains viewable even if
the current hardware configuration lacks calibration; hardware movement and the
live stage marker still use the current calibration and existing safety rules.

## First-milestone implementation sequence and checks

1. **Contract and shared ownership:** define `PlaneSource` and the per-run consumer
   set; bind the latest scratch source and disable the old MDA capture path first.
   Move indexing before follow/visibility gates and validate writer locators.
2. **Shared-source proof:** use the same RAM store as ndv; display one channel and
   switch Z across tiles, with no second acquisition array or hardware calls.
3. **Bounded display:** add source-backed reads, zoom-based strides, missing-plane
   placeholders, stale-request rejection, and texture eviction. Exercise RAM and
   spilled scratch only; no new TIFF/Zarr readiness API in this milestone.
4. **Controls and lifecycle:** channel selector, Z slider, remembered per-channel
   contrast/colormap, simple Auto, theme and follow/lock. Test closing either view,
   latest-run replacement, pending reads, cancellation, exports and scratch cleanup.
5. **Regression/performance validation:** run focused tests with `uv`; profile
   memory, read counts, and slider latency with acquisition and both viewers open.

Acceptance cases must include:

- Two channels and several Z planes with distinct synthetic values; plane changes
  show exactly the requested data and do not move Z or change hardware channels.
- Multiple positions with different grid sizes, repeated XY, non-default axis
  order, incomplete/cancelled stacks, channel Z offsets, and skipped/all-zero data.
- Correct affine orientation after hardware calibration/ROI changes; stable
  pan/zoom, ROI coordinates, overlap ordering, and per-channel contrast.
- Opening Stage Explorer after frames were acquired, or after completion while
  another consumer retains the run; all stored planes remain available.
- Slow scratch reads and rapid slider scrubbing: no stale plane flashes, unbounded
  work queue, GUI blocking, premature resource cleanup, or false acquired tiles.
- With a fixed viewport, increasing total Z count does not proportionally increase
  Stage Explorer's retained pixel memory. Zooming out still honors the display
  budget; uploaded pixel count follows screen resolution rather than full tile
  resolution. Verify native detail on zoom-in, affine placement with odd shapes,
  and stable overlap ordering after out-of-order reads.
- Hidden-at-start collection, unsupported-sink fallback, and manual-map separation
  behave as documented, without retaining an extra full acquisition or presenting
  old-run data as current. No TIFF/Zarr integration-test matrix in this milestone.

## Follow-ups, not first-milestone requirements

- Live TIFF/Zarr browsing with explicit completion/retry semantics and backend
  tests; saved-file reopening is a separate extension.
- Multichannel composite rendering. Evaluate additive channels within a tile
  separately from spatial tile-overlap policy. `max` blending is an experiment,
  not the chosen default: component-wise RGB maxima differ from additive color
  mixing when channel colors share components. Test representative LUTs before
  choosing the renderer structure.
- Older-run selector; shared decoded-plane caching and neighbor-Z prefetch only
  if profiling demonstrates value; histogram/continuous autoscale polish and
  filtered overview generation.

Initial scope is regular GUI scratch-backed MDA runs with one camera selection.
Arbitrary/unbounded event streams, multi-camera overlays, historical time
browsing, image registration/stitching, 3D rendering, and a redesign of ndv's
`p/g` representation remain out of scope.
