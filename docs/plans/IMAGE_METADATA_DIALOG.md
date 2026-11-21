# Image metadata dialog

Press **`I`** over a viewer, or choose **Show Metadata…** from its context menu,
to open a non-modal dialog with the metadata of the acquisition it displays.
The dialog has five tabs: **MDA Sequence**, **Timestamps**, **Frame**,
**Acquisition** and **Raw**. It works for live, finished and reopened
OME-TIFF/OME-Zarr acquisitions. The snap/live Preview and snap-album viewers
are out of scope.

The guiding rule: **show what ome-writers writes**. Each viewer already holds
an `AcquisitionRecord`, which is what its Save button exports. The dialog
reads that same record, so what you inspect is what gets saved. To make the
record trustworthy, this plan first fixes the existing frame-metadata bugs.

---

## Phase 1: one aligned frame-metadata list per record

`AcquisitionRecord.frame_meta` becomes `list[dict[str, Any] | None]`. Entry
*n* belongs to ordinal *n* over the record's **original** dimensions
(`settings.dimensions` before any clamping, without y/x), including skipped
slots. A slot without metadata is `None`, and missing trailing entries are
implicitly `None`.

- Live records use the sink's resolved acquisition dimensions.
- Reopened records use the wrapper's dimension order, which matches
  `_WrapperView`.

### 1a. Live runs: take the list from the sink (pymmcore-plus `cite`)

Today `_on_frame_ready` builds its own list, which has two bugs:

- The runner calls `sink.skip()` for failed frames without emitting
  `frameReady`. Every later entry then shifts by one, and the export writes
  each frame with the wrong frame's timestamps and positions.
- It calls `frame_meta_to_ome(meta)` without `include_event`, so runs that
  store events (`OmeWritersSink.stores_events`) lose `mda_event` on re-export.

The fix is in pymmcore-plus, where every append and every skip is visible:

```python
# pymmcore_plus/mda/_sink.py, OmeWritersSink
self._frame_metadata: list[dict[str, Any] | None] = []

def append(self, img, event, meta):
    frame_meta = frame_meta_to_ome(meta, include_event=self._store_events)
    self._stream.append(img, frame_metadata=frame_meta)
    self._frame_metadata.append(frame_meta)

def skip(self, *, frames: int = 1):
    self._stream.skip(frames=frames)
    self._frame_metadata.extend([None] * frames)

@property
def frame_metadata(self) -> list[dict[str, Any] | None]:
    """Per-frame metadata as written, in acquisition order (None = skipped)."""
    return self._frame_metadata
```

Each `setup()` assigns a new list, so a list held by an older viewer is never
cleared. Entries are added only after a successful stream append or skip.
The sink owns the list: consumers must not mutate it or its dictionaries.

Then, in the GUI:

- `_on_sequence_started` builds
  `AcquisitionRecord(..., frame_meta=sink.frame_metadata)`. This is the same
  list object; it grows while the run acquires and outlives the sink when the
  next run replaces it.
- Remove the `frame_meta.append(...)` block from `_on_frame_ready`, which then
  only does follow-viewer work.
- Bump the pymmcore-plus revision in `uv.lock` and confirm that
  the `memory`/`scratch` path also goes through `OmeWritersSink.append`.

Tests:

- pymmcore-plus:
  - a sink run with a skip in the middle has `None` at that ordinal;
  - `mda_event` is present when `stores_events` is true;
  - reusing a sink for a second `setup()` leaves the old list intact;
  - a failed append or skip adds no entry.

### 1b. Export looks metadata up by original ordinal

`export_acquisition` walks the **clamped** dimensions for pixels but does
`record.frame_meta[n]`, where *n* counts frames over the clamped dimensions.
When an axis that is not outermost is clamped, the strides change and *n* no
longer matches the original ordinal. For example, with `c=2, z=11` clamped to
`z=5`, frame `(c1, z0)` is original ordinal 11 but clamped ordinal 5, so the
old code writes it with the metadata of `(c0, z5)`. A normal run rarely ends
in this state, but the fix is small and makes the alignment explicit.

Add two small helpers to `_mda_export.py`, built from the original dimension
counts:

```python
def frame_ordinal(dims: Sequence[Dimension], index: Sequence[int]) -> int:
    """Ordinal of `index` over the original frame-index dims."""

def frame_index(dims: Sequence[Dimension], ordinal: int) -> tuple[int, ...]:
    """Inverse of `frame_ordinal`."""
```

They use plain strides from the inner dimension counts, so an unbounded outer
axis (`count=None`) needs no special case. `export_acquisition` keeps
iterating the clamped dimensions for pixels, and looks up metadata with
`frame_meta[frame_ordinal(settings.dimensions[:-2], idx)]` when that ordinal
is in range. The dialog uses `frame_index` for the opposite direction, so the
dialog and the saved file always agree. (Phase 1 ships `frame_ordinal` only;
`frame_index` is added with the dialog, its first user.)

Tests (`tests/test_mda_export.py`):

- exporting a record with a `None` gap writes the later frames' `DeltaT`
  and positions on the right planes;
- clamped strides: a hand-built `AcquisitionRecord` with original dims
  `c=2, z=11`, a view whose `coords` clamp z to 5, and distinct metadata at
  original ordinals 0–4 and 11–15 (`c1, z0–4`). The exported `c1` planes
  must carry the ordinal 11–15 metadata. This test fails on the old
  `frame_meta[n]` lookup. (Cancelling during the first z-stack is not
  enough: only ordinals 0–4 have metadata there, so both lookups agree.)
- `frame_ordinal` / `frame_index` with known expected values, including an
  unbounded outer axis.

### 1c. Reopened files: fill `frame_meta` from disk

`open_acquisition` currently builds the record with an empty `frame_meta`,
so re-saving a reopened file drops all per-frame metadata. Add a reader to
each wrapper, next to `global_metadata()`:

```python
def frame_metadata(self) -> dict[tuple[int, ...], dict[str, Any]]:
    """ome-writers-style frame dicts keyed by their index over the wrapper's
    frame-index axes (all except y/x, including z and any exposed p)."""
```

Convert each validated index with `frame_ordinal`. Allocate the list only up
to the highest ordinal, fill internal gaps with `None`, and use `[]` when
there are no entries. Do not walk the full Cartesian product: a partial
Zarr array's declared shape can be much larger than the acquired data. GUI
opens already run on the viewer manager's background `QThreadPool`.

**OME-TIFF** (`xml.etree.ElementTree`, like the existing helpers):

- Match `Image` elements to the wrapper's series and position order. Include
  `p` only when the wrapper exposes it.
  - Linked, master and redundant layouts: read `self._tf.ome_metadata` once.
    The images are all there; do not read them again per sibling file.
  - Independent files (`self._position_files`): read each file's
    `ome_metadata` in wrapper order, closing every temporary `TiffFile`.
- Each `Plane` maps to a frame dict:
  - `TheT/TheC/TheZ` give the index, but only for axes present in
    `self._dims`.
  - `DeltaT` → `delta_t`, `ExposureTime` → `exposure_time` (both converted
    to seconds; OME's default unit is `s`), and `PositionX/Y/Z` →
    `position_x/y/z` (converted to µm). Honor the OME unit attributes and
    defaults. A value in an unsupported unit stays visible in Raw, and is not
    shown as seconds or µm.
  - A plane's `AnnotationRef` → `MapAnnotation` entries go into the dict
    as strings, the same form the sink writes, e.g. `mda_event` JSON.

**OME-Zarr**: for each image group (the root group, or `self._group[pos]` for
each of `self._positions`), read
`attrs.get("ome_writers", {}).get("frame_metadata", [])`. Each entry's
`storage_index` is its index over that group's stored axes excluding y/x.
Validate its length and bounds, and prefix the position index only when the
wrapper exposes `p`. Remove `storage_index` from a copied dict, leaving the
group's attributes unchanged.

Skip malformed entries with a `logger.warning` instead of failing the open.
Missing entries are normal: ome-writers writes no `Plane` / entry for a frame
without metadata, so "no metadata" never means "not acquired".

Tests (`tests/test_ome_wrappers.py`, `tests/test_acquisition_loader.py`,
`FORMATS`-parametrized, written through real ome-writers streams with
`frame_metadata=`):

- `frame_metadata()` round-trips values and indices for single-position,
  multi-position and snake-grid data, and for all three TIFF layouts. Assert
  each entry against the pixels at the same index. Include singleton axes,
  z stacks, interleaved positions, and differing storage/acquisition orders.
- Unit defaults and conversions work, and extra keys (`mda_event`) survive.
- A partial or cancelled file returns only the written entries.
- A large declared Zarr shape with a short acquired prefix produces a short
  metadata list; files without entries leave it empty.
- Extend `test_record_roundtrips_through_export_acquisition`: reopen →
  export → reopen keeps `delta_t`, exposure and positions.

---

## Phase 2: the dialog

### Data: `src/pymmcore_gui/_image_metadata.py` (new, no Qt)

```python
@dataclass(frozen=True, slots=True)
class FrameInfo:
    index: Mapping[str, int]          # {"p": 0, "t": 3, "c": 1, "z": 0}
    delta_t: float | None             # s
    exposure: float | None            # s
    position: tuple[float | None, float | None, float | None]  # µm
    extra: Mapping[str, Any]

def frame_infos(
    settings: AcquisitionSettings,
    frame_meta: Sequence[dict[str, Any] | None],
    stop: int,
) -> list[FrameInfo]:
    """Normalize the non-None entries in frame_meta[:stop]."""
```

Entry *n*'s index is `frame_index(settings.dimensions[:-2], n)` (see 1b).
Do not use `view.coords` or `_clamp_dimensions`: they update asynchronously
and can lag behind the metadata list. Axis labels come from the record's
dimension `coords`.

Also in this module:

- `format_elapsed(seconds)`, giving `HH:MM:SS.mmm`.
- `timepoint_stats(frames, sequence)`: for each position, the start of each
  timepoint, the actual intervals (mean, population std, min, max), and the
  requested interval when `sequence.time_plan` has a single comparable one.
  For multiphase, zero-interval, or otherwise incomparable plans the drift is
  omitted. Compare gaps between observed timepoints with the corresponding
  scheduled gap, not a single interval. Exclude non-finite timestamps.
- `frames_to_csv(frames, labels, path)`, using `csv` with UTF-8,
  `newline=""` and units in the headers.

### Snapshots and performance

- With the dialog closed, the GUI does no metadata work. The sink already
  converts each frame's metadata once, and the record only holds a reference
  to its list.
- Opening or refreshing captures `len(frame_meta)` and processes only that
  prefix; later appends show up on the next Refresh. Nothing copies or locks
  the list.
- Build each tab on the GUI thread the first time it is shown. Add a worker
  only if a real large acquisition turns out to be slow.
- The tables use models, not per-cell widgets, and never autosize every
  column.
- Batch follow-viewer and Frame updates with a short single-shot timer, and
  apply only the latest index, only while the tab is visible.

### Raw documents

Only reopened files have a raw document. Add `raw_metadata() -> dict[str,
Callable[[], str]]` to each wrapper:

- **TIFF:** `"OME-XML"` (one entry per file for independent files).
- **Zarr:** the root and per-image-group `zarr.json` documents, keeping
  nested group paths in their names. ome-writers writes Zarr v3 only.

Loaders return the original text. The view may pretty-print it, but Copy
and Save always use the original. `_viewer_for_loaded` sets
`viewer._raw_metadata = loaded.wrapper.raw_metadata`. Live viewers leave it
`None`, and the Raw tab says that this viewer shows the captured record and
that a finished file can be reopened to see its raw metadata. A large
document may get a truncated preview, but Copy and Save keep the full text.

### Viewer: `src/pymmcore_gui/_array_viewer.py`

- `metadata_available`: a raw source is present, or the viewer has an MDA
  sequence or a record with summary/metadata entries. Raw-only reopened files
  remain inspectable. Preview and snaps have none of these sources. Do not
  scan the metadata list or read documents just to enable the action.
- Shortcut:
  `QShortcut(QKeySequence(Qt.Key.Key_I), widget, self.show_metadata,
  context=Qt.ShortcutContext.WidgetWithChildrenShortcut)`.
  Only the focused viewer reacts, and ndv's line edits keep typing "i".
  Check that clicking the canvas gives the viewer focus, and set
  `ClickFocus` on the canvas only if it doesn't.
- `_show_context_menu` builds the menu from the actions that apply:
  - **Show Metadata…**, with `I` shown as a hint only;
  - **Re-use MDA…**, as today.

  It returns `False` only when the menu would be empty.
- `show_metadata()` creates a `MetadataDialog(self, parent=self.widget())`,
  or raises the existing one. The dialog uses `WA_DeleteOnClose`; clear the
  stored reference on `destroyed` so a reopen never touches a deleted object.
  `_on_viewer_closed` closes it explicitly before releasing the wrapper. On
  Close, Escape or viewer destruction, disconnect the viewer signals and
  stop the update timer.

### Dialog: `src/pymmcore_gui/widgets/_metadata_dialog.py` (new)

- Title `Metadata: <viewer.source_title>`, read on open and on Refresh,
  because Save renames the tab.
- Header: the source path (selectable) and a format label (`OME-TIFF`,
  `OME-Zarr`, `In memory`). Take these from `LoadedAcquisition` or the live
  sink when wiring the viewer, because a reopened record's settings don't
  keep the original path or format. Save As changes the title, not the
  source.
- `QTabWidget`. Each tab is built the first time it is shown. A tab with no
  data is disabled, with a tooltip saying why. A tab that fails to build
  shows the error instead of breaking the dialog.
- Footer: **Refresh** (captures a new prefix and rebuilds the current tab;
  other tabs rebuild when next shown) and **Close**.
- Geometry and the last tab are saved through the existing `Settings`
  model. If the saved tab is disabled, the first available tab is used.
- Palette, style and the system fixed font only: no stylesheets.

#### MDA Sequence

```text
axis_order: tpgcz
channels:
- config: DAPI
  exposure: 10.0
time_plan:
  interval: 0:00:05
  loops: 50
…
────────────────────────────────────────────────────────────
Planned: 3 positions · 50 timepoints · 2 channels · 11 z
                         [Copy] [Save as YAML…] [Re-use MDA…]
```

- `sequence.yaml(exclude_unset=True, exclude_defaults=True)`, the same
  serialization as the MDA widget's Save.
- The sizes are labelled as planned, not acquired.
- **Re-use MDA…** has the same enablement as the context-menu action.

#### Timestamps

```text
Started 2026-11-20 14:03:12 · 3300 frames with metadata · last at 00:04:12.381
Interval: requested 5.000 s · actual 5.012 s ± 0.008 (min 5.001, max 5.044)

Show: (•) Frames ( ) Timepoints    Position: [All ▾]    [✓] Follow viewer
p │t │c    │z │Elapsed      │Δt s  │Δprev ms │Exp ms │X µm  │Y µm  │Z µm
0 │0 │DAPI │0 │00:00:00.000 │0.000 │—        │10.0   │120.4 │-88.1 │12.0
0 │0 │DAPI │1 │00:00:00.031 │0.031 │31.2     │10.0   │120.4 │-88.1 │13.0
                                                         [Copy] [Export CSV…]
```

- Built from `QTableView`, `QAbstractTableModel` and `QSortFilterProxyModel`,
  because there can be 10⁵ or more frames. Sorting uses the numeric values.
  A column whose values are all missing is hidden.
- **Δprev** is the time since the previous timestamped frame **at the same
  position**, in timestamp order. It is computed once, so sorting doesn't
  change it.
- **Timepoints** mode shows one row per `(p, t)`: its start, the interval
  since the previous timepoint, and the drift from the requested interval.
  Highlight deviations over 10 % only for comparable positive requested
  intervals. Use the earliest valid timestamp as the timepoint start.
- **Follow viewer** listens to `display_model.current_index`,
  `visible_axes` and `channel_mode`, and selects **every** displayed row.
  It uses an index lookup built once per snapshot and maps source rows to
  proxy rows. Axes are matched by name, with the same rules ndv uses
  (`ndv.models._resolve`):
  - the channel axis matches all values when
    `channel_mode.is_multichannel()` (composite/color), even if
    `current_index` still holds a channel;
  - a visible axis (e.g. z in an orthogonal view) matches all values;
  - a slice value matches its range;
  - any other axis missing from `current_index` is 0, as ndv fills it, and
    an integer is clamped to the axis range.

  Composite mode therefore selects one row per channel. Double-clicking a
  row updates the existing `current_index` mapping with that row's valid
  viewer axes, without replacing the mapping or changing the display mode.
- **Copy** gives the selected rows as TSV. **Export CSV…** calls
  `frames_to_csv`.

#### Frame

A key/value tree for the frame(s) currently on screen, with one top-level
item per matching frame (one per channel in composite mode). Each shows the
index, channel, elapsed time, exposure, position and every extra key, with
`mda_event` decoded into a subtree (invalid JSON is shown as text). If no
frame matches, it shows "No metadata for this frame". It uses the same
batched index updates as Timestamps.

#### Acquisition

The serialized summary (`_serialize_summary_meta`, as the export uses) in a
filterable key/value tree: date, system info, devices → properties, config
groups, pixel size configs, image info, stage position and extra. Unknown
keys are kept. It has **Expand/Collapse all** and **Copy as JSON**. The
filter matches keys and values and keeps their parents visible. The Frame
tab uses the same tree renderer.

#### Raw

A document combo. The selected document is loaded when first chosen and
shown in a read-only fixed-font `QPlainTextEdit`, with a `Ctrl+F` find bar,
**Copy** and **Save as…**.

### Tests

- `tests/test_image_metadata.py`:
  - `frame_infos` indices match hand-written expected values for skips,
    partial runs, unbounded axes, and metadata ahead of `view.coords`;
  - processing stops at the captured length despite later appends;
  - `timepoint_stats` with single, multiphase and zero-interval plans, and
    with missing timepoints;
  - `format_elapsed`;
  - CSV contents.
- `tests/test_array_viewer.py`:
  - `I` opens one dialog, and pressing it again raises the same one;
  - reopening after a close works;
  - nothing happens for snaps or the Preview;
  - typing "i" in an ndv line edit doesn't open it.
- `tests/test_acquisition_reopen.py` (with the `shown_menus` fixture): the
  menu lists only the actions that apply, including metadata for raw-only
  files. Don't patch `QMenu.exec`, which doesn't work on PySide6.
- `tests/test_metadata_dialog.py`:
  - disabled-tab states;
  - YAML text;
  - row count and hidden columns;
  - numeric sorting through the proxy;
  - Follow viewer selects the displayed rows: all channels in composite
    mode even when `current_index` keeps a `c` value, one channel in
    grayscale mode, row 0 for an axis missing from `current_index`, all
    values for a visible axis, and a slice's range;
  - double-clicking a row moves the viewer;
  - the Frame tab shows one item per channel in composite mode;
  - Refresh picks up newly appended entries;
  - raw documents load lazily, and Copy/Save keep the original text;
  - closing the viewer or pressing Escape disconnects signals and stops the
    timer.
- `tests/test_ndv_viewers.py`:
  - a live record shares the sink's list;
  - a second run leaves the first viewer's record intact.

Use the autouse theme and settings fixtures in `tests/conftest.py`.

---

## Delivery order

1. pymmcore-plus `cite`: `OmeWritersSink.frame_metadata` with skip/event
   tests. Then bump the pin in the GUI.
2. GUI: the record uses the sink's list; remove the `_on_frame_ready`
   capture; add `frame_ordinal` / `frame_index` and the export lookup by
   original ordinal, with the skip and clamped-stride regression tests.
3. Wrapper `frame_metadata()`; `open_acquisition` fills `record.frame_meta`;
   add the re-export round-trip tests.
4. `_image_metadata.py`, wrapper `raw_metadata()`.
5. Viewer and manager wiring, then the dialog (MDA Sequence and Timestamps
   first, then Frame, Acquisition and Raw).
6. A short "Inspecting metadata" section in
   `docs/architecture/DATA_SAVING.md`.

Steps 1–3 fix bugs that exist today, and can ship as their own PR before the
dialog.

Possible later additions: a Δt plot in Timestamps, and snap metadata (from
`SnappedImage`) in the Frame tab.
