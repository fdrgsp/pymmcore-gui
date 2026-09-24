# Grid (`g`) axis slider plan

## Recommendation

This is feasible without duplicating the acquisition or changing the writer's
pixel-storage path. The expected additional indexing cost is small, but absence
of acquisition slowdown or GUI lag must be measured after implementation. This
plan does not promise zero overhead.

Keep the canonical data exactly as it is today: `ome-writers` flattens the useq
`p` and `g` axes into its single position dimension, and the sink continues to
write that representation. Add a small, read-only `ndv.DataWrapper` that presents
the flattened position dimension to the viewer as logical `p` and `g` dimensions.
The wrapper translates a selected `(p, g)` pair back to the existing flattened
position slot only when ndv requests the displayed frame.

The result is:

- no second acquisition-sized image array;
- no change to the acquisition or writer path;
- no change to the saved OME-TIFF/OME-Zarr representation;
- an independent `g` slider only for datasets that can be proven to contain a
  grid axis;
- the existing flattened `p` slider for ordinary multi-position acquisitions;
- a safe fallback to the flattened `p` slider for a ragged grid that cannot be
  represented truthfully by two independent rectangular sliders.

Do not implement this by reshaping or copying the complete `StreamView`, and do
not add a fake `g` dimension to `AcquisitionSettings`. Both would mix a display
concern into the storage contract and would make saving/exporting less reliable.

## Current behavior and why it happens

The live path is currently:

```text
useq MDA event with p/g indices
  -> pymmcore-plus OmeWritersSink
  -> ome-writers StreamView with flattened p dimension
  -> MMArrayViewer / ndv
```

`ome-writers.useq_to_acquisition_settings()` intentionally turns position and
grid locations into one OME position dimension. Its `Position` coordinates retain
grid row/column and stage information, but the `StreamView.dims` tuple contains
`p`, not `g`.

Both viewer managers compensate while following a live acquisition:

- `src/pymmcore_gui/_modern_gui/_acquire_viewers.py`
- `src/pymmcore_gui/_ndv_viewers.py`

Their `_shot_indices` maps each arriving `(p, g)` event identity to a monotonically
increasing flattened `p` slider slot. This works when every planned location is
observed in order. It is not a reliable source of storage identity: the runner can
advance the sink with `skip()` without emitting `frameReady`, and both managers
currently stop this bookkeeping while follow-lock is enabled. The new layout must
derive storage indices independently of arriving frames and the follow-lock.

`ndv` creates its slider rows from `DataWrapper.dims`/`coords`, and already reacts
to `DataWrapper.dims_changed`. Therefore the GUI does not need a custom Qt slider.
It needs a wrapper that reports the logical dimensions and translates data
requests.

## Proposed design

### 1. Describe the logical grid layout

Add an internal immutable model, for example `GridAxisLayout`, in a new module:

```text
src/pymmcore_gui/_grid_axis.py
```

It should contain:

- whether the logical data has `p`, `g`, or both;
- the order of `p` and `g` from the useq `axis_order`;
- an arithmetic mapping, or immutable lookup table when needed, from each valid
  logical `(p | None, g | None)` identity to the planned writer position slot;
- logical coordinate labels for each exposed axis;
- whether the layout is rectangular and therefore safe to expose as independent
  sliders.

Use one shared builder for live and reopened data. It must not iterate all MDA
frames, because time/channel/z loops could make that unnecessarily expensive.
Inspect only the position and grid plans (including position subsequences) and
validate the resulting number/order of flattened locations against the resolved
writer positions. Require a canonical position dimension and a proven mapping;
fall back for custom/generator sequences, missing metadata, or a sink that has
fallen back to generic `(t, y, x)` storage. Do not regenerate random grid points to
infer identities or infer them from stage-coordinate equality.

Recognize these cases:

| Sequence | Viewer axes |
|---|---|
| No grid | Existing axes; no adapter |
| Grid only | `g` replaces the writer's flattened `p` display axis |
| Stage positions plus one uniform global grid | Separate `p` and `g` |
| Uniform grid subsequence on every position | Separate `p` and `g` |
| Differently sized grids, or a mixture of gridded and plain positions | Keep flattened `p` in the first version |

The last case is ragged. A rectangular `p x g` view would contain invalid
combinations, and ndv sliders do not currently support a `g` range that changes
with the selected `p`. Showing empty frames or silently repeating another tile
would be misleading. Falling back preserves access to every stored location
through flattened `p`, using the planned mapping described below. A future
dependent-slider UI can add ragged support separately.

For validated regular layouts with `P` positions and `G` tiles per position, use:

- grid only: `flat = g`;
- position first: `flat = p * G + g`;
- grid first: `flat = g * P + p`.

Here `g` is the useq tile index in visit order, including snake traversal; it is
not necessarily `row * columns + column`. Uniform position subsequences must be
checked against their actual writer ordering. For other supported layouts, build
an immutable table once. Arriving events may validate the layout but must never
assign or renumber slots. Skipped frames retain their original storage slots.

### 2. Define live coordinates and display modes

Expose the planned logical `p`/`g` extents from viewer construction; retain the
source's dynamic extents for other dimensions. This avoids changing grid layout
as successive locations arrive. A partial acquisition is not a smaller rectangular
grid: the last position may contain only some tiles. Selecting an unwritten slot
must preserve its identity and use the source's missing-data behavior, never
redirect to an acquired tile. Explain in the viewer that planned slots can be
empty. Coordinate high-water marks alone cannot distinguish every unwritten or
skipped frame from a legitimately black image.

Handle the empty initial source and delayed writes without out-of-bounds reads.
The adapter must not represent planned extents as evidence that frames were
acquired, or use those extents to determine export completeness.

Keep ndv's native singleton policy: a one-tile `g` axis exists logically but its
slider is hidden, as are other singleton axes. A navigable `g` slider appears when
the grid has at least two tiles. For a grid-only run, no logical `p` is exposed.

Declare channel and spatial semantics explicitly. Never let ndv infer `p` or `g`
as a channel or as Z. Override the wrapper's channel/Z guessing as needed, and
guard ndv's 3D toggle fallback as well: without a genuine Z axis, disable the
spatial 3D toggle for these viewers. Composite mode may select multiple channels;
3D and orthogonal views may select Z stacks. All must keep `p`/`g` fixed to one
location and preserve the logical result axis order.

### 3. Add an ndv wrapper with no additional pixel copy

Implement a `GridAxisDataWrapper` subclass of `ndv.models.DataWrapper`. Wrap the
original `StreamView` (or a lazy view opened from disk) and delegate dtype and all
non-position coordinates to its ordinary ndv wrapper.

The wrapper should:

- report logical `dims` and `coords`, replacing or expanding the flattened `p`
  display dimension with `g` or `p, g`;
- translate the one-element `p`/`g` slices generated by ndv into a single flattened
  **integer** position index;
- delegate the actual read to the original wrapper/view;
- restore each logically retained `p`/`g` axis with `numpy.expand_dims`, in the
  correct order; preserve the remaining axes and their requested slices;
- support integer logical indices too, collapsing their axes for callers such as
  ROI extraction;
- deliver dimension and data notifications through the GUI-thread update policy
  below;
- reject inconsistent mappings and fall back before viewer construction rather
  than returning the wrong frame.

This integer-index detail is necessary: the installed `StreamView` calls
`np.stack` for a position slice even when it contains only one position. A local
check confirmed that `view[p:p+1]` does not share source memory, while integer
indexing followed by `expand_dims` does for a NumPy-backed view. Test memory
sharing against the source result, not just between two intermediate arrays.

The grayscale XY fast path reads one plane. Composite, orthogonal and 3D requests
read the channel/volume subset needed for that view, as they already do. Pixel
reading, decompression and rendering can still allocate buffers; the adapter must
add no pixel copy to the single-location request. O(1) refers only to logical
position translation, not the complete data-read/render operation.

Keep multi-value `p`/`g` requests outside the initial supported viewer modes.
Reject them explicitly if requested programmatically; never silently read or
stack the entire acquisition. The wrapper must preserve ndv's singleton-slice
shape contract even though it uses integer indexing internally.

Do not make `viewer.data` or the export layer materialize the logical view. The
adapter is a display boundary, not a replacement acquisition datastore.

### 4. Use the wrapper in both live viewer managers

At sequence start:

1. Obtain the raw writer-backed view as today.
2. Build and validate `GridAxisLayout` from the `MDASequence` and the sink's
   resolved `AcquisitionSettings`.
3. Pass `GridAxisDataWrapper(raw_view, layout)` to `MMArrayViewer` only when the
   layout is a supported grid layout; otherwise pass `raw_view` unchanged.
4. Keep the raw view in `AcquisitionRecord`.

On each `frameReady`, before checking follow-lock:

1. Preserve the event's real indices and record the existing per-frame metadata.
2. Validate the event against the precomputed mapping without modifying it.
3. Mark relevant display data as dirty. If following is enabled, replace the
   pending follow target with the latest event's logical indices.
4. Schedule one GUI refresh if none is pending; do not start one timer per frame.

After this is established, remove the duplicated `_shot_indices` bookkeeping from
`AcquireViewersManager` and `NDVViewersManager`. The no-grid and ragged fallback
paths should share the same validated planned-slot mapping where applicable,
while keeping their flattened display axes. Do not preserve the arrival-counter
bug in a fallback path.

The modern manager already attaches an `AcquisitionRecord`. Make the same raw-data
ownership explicit for every adapted MDA viewer path so `MMArrayViewer._save_data()`
always exports the canonical flattened view/settings, never the synthetic display
dimensions. Keep layout state per viewer so a new run cannot change older viewers.

### 5. Prevent redundant reads and GUI update backlog

Use a single pending refresh per viewer, with a bounded display cadence (start
with approximately 30 Hz and validate responsiveness). Coalesce display targets
and dirty notifications only; every acquired image must still reach the sink and
every required metadata record must be captured. Do not retain image payloads in
pending display updates.

Apply `p`, `g`, `t`, `c`, and `z` changes atomically using a verified ndv/psygnal
batching mechanism, or a narrow compatibility helper. The current ndv model can
resolve and request data on individual index changes; a subsequent unconditional
`data_changed` can request it again. Ensure each refresh submits at most one data
request for its final state. Refresh the same displayed index when its pixels
change, even if the index itself did not change.

Emit `dims_changed` only when exposed coordinates actually change. Cache layout
and coordinate descriptions rather than rebuilding position tables on each frame.
ndv may itself traverse coordinates while resolving display state, so benchmark
the complete refresh, not just the O(1) lookup.

Keep Qt mutations on the GUI thread and image reads on ndv's worker path. Worker
requests must use immutable layout/request snapshots. Ignore late completions
after viewer closure or replacement, disconnect signals and stop refresh timers,
and preserve the existing sink-release lifecycle.

Coalescing redraws does not by itself bound incoming Qt signal queues. Measure
the queued `frameReady` handling cost and backlog at the target acquisition rate.
If it cannot keep up, resolve that before claiming acceptable performance; any
further metadata/notification batching must remain lossless and thread-safe.

The existing 10 ms single-shot timer is a heuristic, not a write-completion
guarantee. Use a verified read-readiness mechanism when available, or a bounded,
nonblocking retry policy. Include a final refresh after source finalization so
the last frame cannot remain stale. Never wait for disk I/O on the GUI or
acquisition thread merely to refresh the display.

### 6. Route ROI reads through the logical wrapper

`MMArrayViewer._get_roi_data()` currently takes logical indices from
`self._resolved` and applies them directly to `self.data`. With a wrapper whose
data remains the flattened source, the indices refer to different dimensions.
A local reproduction returned a 1D slice from the wrong position instead of the
requested 2D ROI.

Change ROI extraction to call `self.data_wrapper.isel(nd_index)` using the resolved
logical axis keys and ROI slices. Implement and test the adapter's integer-index
behavior, channel selections, and orthogonal views. For reads that can involve
disk I/O or substantial volume data, submit the ROI request to a worker and
deliver its result asynchronously; update the stats caller accordingly rather
than moving a blocking read into a GUI callback. Audit other consumers of
`viewer.data` for the same assumption, including RGB save behavior; storage/export
consumers must use the canonical acquisition record, while display-coordinate
consumers must use the wrapper.

### 7. Preserve saving and acquisition efficiency

Maintain two deliberate references:

```text
viewer.data_wrapper -> logical grid adapter -> raw flattened view
AcquisitionRecord.view ---------------------> raw flattened view
```

They point to the same underlying arrays. Regular layouts need only arithmetic
mapping parameters plus labels; any necessary lookup table is proportional to the
number of physical position/tile locations, not the number or size of images.

Saving from the viewer must continue through `AcquisitionRecord`, whose
`AcquisitionSettings` and `view` remain flattened exactly as `ome-writers` expects.
This avoids:

- a second acquisition-sized allocation;
- replaying or transforming data during acquisition;
- incompatible `g` dimensions in OME output;
- changes to frame metadata order;
- changes to the writer's append and finalization logic.

An unchanged writer path alone does not guarantee unchanged throughput: GUI CPU,
memory bandwidth, and concurrent disk reads can compete with acquisition. Measure
those effects using the performance checks below.

Add an assertion/test seam so an adapted viewer cannot silently fall through to
`_synthesize_record()` with logical dimensions paired with the raw flattened view.

## Drag-and-drop/reopened acquisitions

The modern GUI currently has no acquisition-file drag/drop opening path. This is a
separate prerequisite, not something the slider wrapper itself provides.

Implement file opening as a second phase with a small normalized result object, for
example:

```text
OpenedAcquisition
  raw_view: lazy, canonical flattened array view
  settings: AcquisitionSettings-compatible dimensions/positions
  sequence: MDASequence | None
  summary_meta: SummaryMetaV1 | None
```

The opener should:

- accept a dropped OME-Zarr directory, a single OME-TIFF, or an app-created
  multi-position OME-TIFF directory;
- open pixels lazily (TensorStore/Zarr or page-backed TIFF), never via
  `numpy.asarray()` on the whole acquisition;
- enumerate files, parse metadata, resolve TIFF series/IFD mappings, and construct
  readers in a background worker; lazy pixels alone do not prevent these steps
  from blocking the GUI;
- read the `pymmcore_plus` global metadata written by this application and rebuild
  the serialized `MDASequence` when present;
- build the same `GridAxisLayout` used by live acquisition;
- cross-check the reconstructed layout against stored position count and position
  metadata before showing `g`;
- create a normal docked `MMArrayViewer` through a public
  `AcquireViewersManager.open_acquisition(...)` method;
- attach an `AcquisitionRecord` containing the raw lazy view and canonical settings,
  so Save remains a streaming export of the source rather than the display adapter;
- show only the flattened stored `p` axis when metadata is absent, malformed, or
  insufficient to prove the original `p`/`g` mapping.

Use OME metadata to resolve position order and acquisition-versus-storage axis
order, including linked TIFF files; do not assume filename sorting is sufficient.
A single file from a multi-position set must resolve its companions or be treated
as a subset, without inventing absent positions. Keep open readers alive for the
viewer's lifetime and close them on failure, cancellation, or viewer closure.
Publish completed readers and create widgets on the GUI thread. Bound concurrent
opening jobs and support cancellation; when a live run is active, defer expensive
opening work until it ends to avoid additional I/O contention.

Add Qt drag handling at the narrowest owner of the viewer workspace (prefer
`AcquirePage` or its viewer workspace over application-wide interception). Accept
the drag only when all URLs are local and at least one supported acquisition target
is recognized. Switch to the Acquire page and report parse/open failures through the
existing notification path.

Do not infer a `g` axis only from a filename, position count, or a square number of
positions. Those heuristics can relabel a genuine multi-position acquisition
incorrectly. App-written sequence/grid metadata is the authoritative signal.

## Implementation sequence

1. Add planned `GridAxisLayout` construction and focused tests for grid-only,
   both axis orders, snake traversal, position subsequences, skipped locations,
   repeated timepoints, and unsupported/fallback layouts.
2. Add wrapper tests using a tiny fake array and the real `StreamView`. Verify
   pixels, result-axis order, integer/singleton-slice semantics, and source memory
   sharing. Cover grayscale, composite, Z volumes and orthogonal slices.
3. Integrate the wrapper and coalesced refresh policy into `AcquireViewersManager`;
   replace arrival-based flattening with the planned mapping.
4. Integrate the same helper into the classic `NDVViewersManager` to avoid divergent
   semantics between the two GUIs.
5. Fix ROI reads and axis guessing/3D controls. Add GUI tests for singleton grids,
   empty/partial acquisitions, lock/unlock across unseen locations, final-frame
   refresh, multiple runs, and closing during pending reads. Assert bounded pending
   refresh work and no duplicate data requests per refresh.
6. Add export regression tests proving adapted and unadapted viewers export the
   same canonical data and metadata. Test partial/skipped acquisitions separately:
   existing high-water-based export logic is not proof of exact frame completeness,
   and synthetic grid extents must never make that problem worse.
7. Implement the lazy acquisition opener, background metadata work and drag/drop
   plumbing, including cancellation and reader cleanup.
8. Reuse the same layout/wrapper tests for reopened app-created OME-Zarr and
   OME-TIFF data, plus a metadata-free file that must not show `g`.
9. Run the performance comparison below before claiming negligible impact.

## Performance validation

Compare the existing flattened viewer against the adapted viewer under identical
settings, after warm-up, with repeated runs. Test small fast frames to expose event
overhead and large frames to expose copies, plus large grids and composite/Z
display. Cover scratch-memory, scratch-spill, OME-TIFF and OME-Zarr as supported;
include follow-lock and rapid manual slider movement. Use `uv` for checks.

Measure acquisition frames/second and duration, callback cost, Qt event-loop delay,
slider-to-image latency, data requests per refresh, pending refresh count, peak
RSS, and metadata/frame integrity. Test delayed reads and dropped-file opening
independently. Compare decoded pixels and relevant metadata rather than binary
file checksums, which can differ because of timestamps or identifiers.

Suggested initial acceptance targets are no reproducible throughput regression
greater than 5%, at most one pending display refresh per viewer, and no growing
event backlog over a sustained high-rate run. For the same source and display
mode, target no more than 10 ms additional p95 event-loop delay or 20 ms additional
p95 slider-to-image latency. Record absolute latency too: a laggy baseline is not
an acceptable result merely because the adapter matches it. These are validation
targets, not measured results or guarantees; investigate failures and report any
remaining limits before shipping.

The memory gate is structural: no allocation proportional to the full pixel
dataset, and no adapter-induced copy of a single-location source result. Allow
normal reader/decompression and rendering buffers and bounded in-flight reads.

## Acceptance criteria

- A grid-only MDA exposes `g` and no synthetic `p`; the slider is visible when
  there are at least two tiles, following ndv's singleton policy.
- A regular position-plus-grid MDA exposes independent `p` and `g` axes in useq
  order, with singleton sliders hidden.
- An ordinary multi-position MDA displays only `p`, exactly as it does now.
- Selecting every valid `(p, g)` pair displays the same pixels as indexing the
  corresponding flattened position in the raw view.
- Live following lands on the event's actual `p` and `g` values across timepoints,
  skipped locations and lock/unlock. Slot identities never depend on observed
  frames. The last completed frame is refreshed after finalization when following;
  locked viewers retain their selected location.
- The `g` slider is not shown when grid provenance cannot be established.
- Ragged grids retain the existing flattened `p` behavior until dependent slider
  ranges are supported.
- Partial acquisitions preserve planned slot identities; missing slots never show
  a different tile. Planned extents are not treated as acquisition completeness.
- ROI pixels, composite channels, Z volumes and orthogonal views are correct;
  neither `p` nor `g` is inferred as channel or Z.
- Viewer Save/export uses canonical raw settings/data and round-trips the same frame
  count, order, pixels, and position/grid metadata.
- No full-array conversion or acquisition-sized copy occurs.
- Position translation is O(1); full display work is measured separately. Mapping
  storage is constant-size for regular arithmetic layouts, or at most proportional
  to the number of position/grid locations when a table is required.
- Refreshes are batched and bounded, metadata capture remains lossless, and the
  performance targets above are verified on representative acquisition rates.
- File enumeration, metadata parsing and pixel reads do not block the GUI thread.
- Acquisition writing and finalization semantics remain unchanged.

## Likely files

| File | Change |
|---|---|
| `src/pymmcore_gui/_grid_axis.py` | New layout model, validation, and ndv wrapper |
| `src/pymmcore_gui/_modern_gui/_acquire_viewers.py` | Wrap supported live/file views and follow real `p`/`g` indices |
| `src/pymmcore_gui/_ndv_viewers.py` | Reuse the same live adapter in the classic GUI |
| `src/pymmcore_gui/_array_viewer.py` | Logical ROI reads, axis/3D controls, and canonical export guards |
| `src/pymmcore_gui/_modern_gui/_acquire.py` | Viewer-workspace drop acceptance and routing |
| `src/pymmcore_gui/_modern_gui/_main_win.py` | Activate Acquire on a successfully opened drop, if needed |
| `src/pymmcore_gui/_modern_gui/_acquisition_open.py` | New lazy OME opener and metadata normalization |
| `tests/test_grid_axis.py` | Mapping/wrapper unit tests |
| `tests/test_new_gui.py` | Live sliders, bounded refreshes, lock/skip, drop, and lifecycle tests |
| `tests/test_array_viewer.py` | Logical ROI extraction and display/export integration |
| `tests/test_ndv_viewers.py` | Classic manager integration tests |
| `tests/test_mda_export.py` | Canonical export regression tests |

## Main risk to avoid

The risky version of this feature is to advertise a regular `p x g` array when the
acquisition is actually ragged, or to let the synthetic display axes leak into the
writer/export settings. The proposed validation, fallback, and separate raw
`AcquisitionRecord` address both. Planned slot identities, logical ROI reads,
integer source indexing, and bounded GUI refreshes are also required. With these
corrections the design should add little overhead; acquisition throughput and
responsiveness remain implementation acceptance checks, not assumptions.
