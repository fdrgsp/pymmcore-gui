# Reliable ndv OME export plan

This is the implementation checklist for making the save action in
`MMArrayViewer` reliable, metadata-complete, and recoverable. The user must be
able to export a completed viewer dataset as either OME-TIFF or OME-Zarr, see
honest progress and errors, and retry to another drive without losing the
source data.

Complete the phases in order. Do not describe this feature as reliable until
all release-gate items at the end are checked.

## Required user outcomes

- [ ] The save action offers **OME-TIFF** and **OME-Zarr** explicitly.
- [ ] The selected format and final path always agree. Normalize the suffix or
  ask the user to resolve a mismatch; never infer a different format silently.
- [ ] Large datasets are copied one frame or bounded chunk at a time. Saving
  must never call `np.asarray()` on the complete viewer dataset.
- [ ] Pixel data, dimensions, coordinates, physical units, channel information,
  positions, acquisition settings, summary metadata, and per-frame metadata are
  preserved whenever the source contains them.
- [ ] Missing source metadata is reported as unavailable; it is never invented.
- [ ] Success is shown only after the writer has closed, deferred writes have
  completed, metadata has been flushed, and the result has passed validation.
- [ ] A failed export never damages or consumes the source dataset.
- [ ] A failed export never replaces an existing valid destination.
- [ ] Disk-full, disconnected-drive, read-only-filesystem, permission, source
  read, target write, finalization, and validation failures are visible to the
  user and retained in the log with their original exception and path.
- [ ] After any target failure, the user can choose another drive and retry
  from the unchanged source.
- [ ] Partial outputs are unmistakably marked as incomplete and are never
  presented as successful OME datasets.
- [ ] Export work runs outside the GUI thread and provides progress,
  cancellation, and a deterministic completion result.

## Scope and explicit limits

- This plan covers the save/export button added to ndv by
  `src/pymmcore_gui/_array_viewer.py`.
- Acquisition-time saving in `MemoryMDAWidget` already offers `ome-tiff` and
  `ome-zarr` through `pymmcore-widgets`. Reuse the same format vocabulary, but
  do not create a second acquisition-output implementation.
- The first reliable version exports only a **finished or canceled** acquisition.
  Disable export while its source acquisition is still writing. Live snapshot
  export requires a separate consistency design.
- A retry starts a fresh transaction at the new destination. Do not attempt to
  resume or append to a partial TIFF/Zarr unless `ome-writers` later provides a
  validated public resume API.
- Full read-back verification can detect many storage failures, but software
  cannot guarantee that failing hardware will not corrupt data after the
  application reports success. State this limit in user documentation.
- Do not add GUI imports to the export core. The export transaction must be
  testable without Qt.
- Do not import private `ome_writers` or `pymmcore_plus` modules from this
  project. Any missing public capability must be fixed upstream and versioned.

## Audited baseline

Audit date: 2026-07-24.

Installed versions:

- `ome-writers==0.3.2`
- `pymmcore-plus==0.18.1`
- `ndv==0.5.1.dev2`
- `tifffile==2026.7.14`
- `tensorstore==0.1.84`

Current behavior and risks:

- `MMArrayViewer._save_data()` supports only OME-TIFF.
- The viewer save path bypasses `OmeWritersSink`/`ome-writers` and writes with
  `tifffile.imwrite()` directly.
- The complete source is materialized with `np.asarray(data)`. This can exhaust
  RAM and is explicitly discouraged for an `ome-writers.StreamView`.
- The direct TIFF path preserves only inferred axes and X/Y/Z scale values.
  It drops the MDA sequence, summary metadata, channel details, position
  semantics, per-frame timing/exposure/stage metadata, hardware state, and
  provenance.
- Multi-position data is split manually into files. It does not use
  `ome-writers` position/plate metadata or its supported multi-file OME
  structures.
- Dimension lookup, save-button creation, and several viewer integrations are
  wrapped in broad exception suppression. Some failures can leave no button or
  no useful error.
- Saving is synchronous on the GUI thread and has no progress, cancellation,
  capacity check, result model, retry flow, or post-write validation.
- `_extract_scales()` passes physical scale to ndv, but the viewer managers
  discard the source `MDASequence` and `SummaryMetaV1` after construction.
- `OmeWritersSink._frame_meta_to_ome()` currently forwards only elapsed time,
  exposure, and active X/Y/Z positions. Its own TODO notes that the rest of
  `FrameMetaV1` is omitted.
- `OmeWritersSink` catches summary-metadata failures and logs warnings rather
  than failing the write.
- `MDARunner._finish_run()` catches sink-close failures, logs them, and still
  emits `sequenceFinished`.
- The `ome-writers` TIFF worker catches background write errors, emits a
  warning, and does not propagate the failure through `append()` or `close()`.
  Its finalizer joins with a timeout and does not check a stored worker error.
- `ome-writers.create_stream()` currently finalizes a partially prepared backend
  on `FileExistsError`, but not on every preparation exception.
- The pure in-memory scratch backend does not retain per-frame metadata; its
  JSONL metadata journal exists only in disk-backed scratch mode.

Baseline commands and evidence:

- [ ] Record the current save tests: `pytest -q tests/test_array_viewer.py`.
- [ ] Record peak memory while exporting a representative dataset larger than
  available RAM.
- [ ] Record current OME metadata for single-position, multi-position, and
  multichannel exports.
- [ ] Reproduce and record a target write failure and a target close/finalize
  failure before changing the code.
- [ ] Link the upstream `pymmcore-plus` and `ome-writers` issues/PRs created from
  Phase 1:
  - `ome-writers`: `________________`
  - `pymmcore-plus`: `________________`

## Target architecture

Keep the ndv integration thin:

```text
MMArrayViewer save action
        |
        v
Export dialog -> ExportRequest
        |
        v
Qt worker adapter -> DatasetExporter (no Qt)
                         |
              +----------+-----------+
              |                      |
       ViewerDatasetSource     ExportTransaction
       pixels + metadata       stage/flush/verify/commit
              |                      |
              +----------+-----------+
                         |
                         v
             public ome_writers API
       AcquisitionSettings + create_stream()
```

Suggested package boundary:

```text
src/pymmcore_gui/_saving/
    __init__.py
    _errors.py        # stable error categories and original exception
    _models.py        # request, source descriptor, progress, result
    _metadata.py      # strict serialization and metadata mapping
    _frames.py        # bounded acquisition-order frame iteration
    _filesystem.py    # injectable storage checks and commit operations
    _transaction.py   # partial target, validation, commit, rollback
    _export.py        # ome-writers orchestration
```

Use fewer modules if the implementation remains small. Keep these boundaries,
not the filenames, as the design constraint.

## Phase 0 — Freeze the contract

### 0.1 Define source states

- [ ] Define whether a viewer source is:
  - a completed MDA sink view;
  - a canceled/incomplete MDA sink view;
  - a live MDA sink view;
  - a plain ndarray or other non-MDA array-like object;
  - already backed by a persistent OME-TIFF or OME-Zarr dataset.
- [ ] Permit export for completed, canceled, and static sources.
- [ ] Disable export for a live source and explain: “Finish or cancel the
  acquisition before exporting.”
- [ ] Preserve an explicit `complete`, `canceled`, or `incomplete` status in
  export provenance.
- [ ] Define source lifetime: closing an ndv tab during export must either keep
  a strong, read-only source handle alive or ask the user to cancel the export.

### 0.2 Define success

- [ ] An export is successful only when all of these are true:
  1. target preparation succeeded;
  2. every expected written or skipped coordinate was processed;
  3. all writer tasks/futures/threads completed without error;
  4. the writer closed successfully;
  5. global and per-frame metadata flushed successfully;
  6. the staged output reopened successfully;
  7. structural, metadata, and pixel-integrity validation passed;
  8. the staged output was committed to the requested final path.
- [ ] A warning about lost metadata is not success. If promised metadata cannot
  be serialized or written, fail before commit.
- [ ] Preflight free-space checks are advisory. Runtime `ENOSPC` and other I/O
  errors remain mandatory failure paths.

### 0.3 Define format behavior

- [ ] Use `AcquisitionSettings(format="ome-tiff", ...)` for OME-TIFF and
  `AcquisitionSettings(format="ome-zarr", ...)` for OME-Zarr. Do not rely only
  on suffix inference.
- [ ] Use `.ome.tif` or `.ome.tiff` for TIFF and `.ome.zarr` for Zarr.
- [ ] Display `AcquisitionSettings.output_path` before starting so the user
  understands the actual output.
- [ ] Document that `ome-writers` defaults to a single TIFF for one position
  and a directory containing one TIFF per position for multiple positions.
- [ ] Select and document the multi-position TIFF metadata policy. Prefer
  `redundant` initially so each position file remains self-describing.
- [ ] Never use the legacy direct `tifffile.imwrite()` path after migration.

## Phase 1 — Fix upstream error contracts first

The viewer cannot honestly report success until writer failures propagate.
Complete and release these fixes upstream, then pin minimum working versions.

### 1.1 Harden `ome-writers`

- [ ] Give the TIFF writer thread a stored terminal exception state.
- [ ] On the first TIFF worker failure:
  - stop accepting/queueing more frames;
  - unblock producers and finalization;
  - close the writer best-effort;
  - preserve the original exception and traceback.
- [ ] Make TIFF `write()`, subsequent `append()`, and `finalize()/close()` raise
  the stored worker exception. A warning alone is not sufficient.
- [ ] Replace the unchecked five-second thread join with deterministic shutdown
  and an explicit timeout error if shutdown cannot complete.
- [ ] Use a bounded TIFF queue or backpressure so a dead writer cannot cause
  unbounded memory growth.
- [ ] Ensure TensorStore/Zarr async write errors propagate from `close()`.
- [ ] Ensure metadata-flush errors propagate from `close()` for every backend.
- [ ] Make `create_stream()` clean up a partially prepared backend for every
  preparation exception, not only `FileExistsError`.
- [ ] Make cleanup idempotent without masking the primary write exception.
- [ ] Retain per-frame metadata for an in-memory scratch stream or expose a
  public metadata journal/iterator suitable for export.
- [ ] Expose the immutable effective `AcquisitionSettings` and metadata through
  a public API. Do not require GUI code to inspect `stream._settings` or
  backend-private state.
- [ ] Add upstream fault-injection tests for `ENOSPC`, `EIO`, finalization
  failure, metadata failure, and a writer thread that does not stop.

### 1.2 Harden `pymmcore-plus`

- [ ] Make `OmeWritersSink` serialize the complete `FrameMetaV1`, while mapping
  known values to structured OME fields.
- [ ] Preserve these structured per-plane values when available:
  `delta_t`, `exposure_time`, `position_x`, `position_y`, and `position_z`.
- [ ] Store the complete versioned frame metadata as unstructured,
  JSON-compatible metadata so camera metadata, changed properties, the MDA
  event, hardware-trigger state, buffer state, all-stage positions, and `extra`
  are not discarded.
- [ ] Replace backend-specific summary `get_metadata()/update_metadata()` logic
  with public `OMEStream.set_global_metadata("pymmcore_plus", ...)` where
  possible.
- [ ] Stop swallowing summary-metadata errors.
- [ ] If sink close or metadata finalization fails, set the runner finish reason
  to errored and emit an error result/event that the GUI can consume.
- [ ] Add the documented public `get_sink()` or an equivalent public snapshot
  API so consumers do not import `pymmcore_plus.mda._sink`.
- [ ] Add round-trip tests for full `SummaryMetaV1` and `FrameMetaV1` through
  both OME-TIFF and OME-Zarr.

### 1.3 Version gate

- [ ] Update minimum dependency versions only after the upstream fixes are
  released.
- [ ] Add a startup/export capability check with an actionable message for
  unsupported versions.
- [ ] Remove any temporary local compatibility adapter once the fixed minimum
  versions are required.

### Phase 1 done when

- [ ] A forced disk-full error in a TIFF worker reaches the caller of
  `close()` and cannot be reported as success.
- [ ] A forced Zarr future/finalize error reaches the caller.
- [ ] A metadata-write failure reaches the caller.
- [ ] The MDA runner reports a sink finalization failure as an error.
- [ ] No GUI code imports an upstream private module.

## Phase 2 — Preserve export context with each viewer

### 2.1 Add a source descriptor

- [ ] Define a small immutable `ViewerDatasetSource`/`ViewerExportContext` that
  contains only export-relevant state:
  - array-like source handle;
  - dtype;
  - ordered dimensions and coordinates;
  - source acquisition settings when available;
  - MDA sequence when available;
  - `SummaryMetaV1` when available;
  - per-frame metadata access/journal when available;
  - written/skipped coordinate information;
  - source UID, original path/format, and completion status.
- [ ] Prefer the original effective `AcquisitionSettings` from
  `OmeWritersSink`; reconstruct settings only when the source does not provide
  them.
- [ ] Keep the source descriptor independent from ndv private models.
- [ ] Attach the descriptor to `MMArrayViewer` through an explicit constructor
  argument or setter, not an undocumented dynamic attribute.

### 2.2 Populate it at acquisition start

- [ ] In both `AcquireViewers` and `NDVViewersManager`, retain the
  `MDASequence` and `SummaryMetaV1` already received by
  `sequenceStarted`.
- [ ] Obtain the sink view and public sink/export metadata as one coherent
  snapshot. Do not call unrelated getters that could refer to different runs.
- [ ] Track the exact acquisition dimension order and position semantics.
- [ ] Track which frames were written versus skipped; high-water shape alone
  cannot distinguish a skipped zero-filled frame from a real zero image.
- [ ] Finalize the source descriptor when `sequenceFinished` or the new sink
  error event arrives.
- [ ] Preserve the descriptor after the active acquisition reference is
  cleared.

### 2.3 Static/non-MDA viewers

- [ ] Derive minimal dimensions from the array wrapper’s public dimension API.
- [ ] Preserve known scales and units.
- [ ] Mark acquisition summary and per-frame metadata as unavailable.
- [ ] Show a pre-export summary stating exactly which metadata classes are
  unavailable.
- [ ] Never query the current microscope state and pretend it describes an old
  or unrelated ndarray.

## Phase 3 — Build one metadata model

### 3.1 Strict serialization

- [ ] Implement one strict JSON serializer for both global and per-frame
  metadata.
- [ ] Support Pydantic models, enums, datetimes, paths, tuples, NumPy scalars,
  and other known metadata value types explicitly.
- [ ] Reject unsupported values with a field path such as
  `frame[12].camera_metadata.vendor_value`.
- [ ] Do not use `default=str` as a catch-all; it destroys type meaning.
- [ ] Include schema names and versions for every unstructured metadata block.

### 3.2 Structured OME metadata

- [ ] Preserve exact dimension names, types, counts, order, scale, unit,
  chunking intent, and coordinates where supported.
- [ ] Preserve dtype and Y/X frame shape without coercion.
- [ ] Preserve channel names. Add color, fluorophore, excitation wavelength,
  and emission wavelength only when the source provides reliable values.
- [ ] Preserve physical pixel sizes and Z spacing/coordinates with explicit
  units.
- [ ] Preserve time interval or per-plane elapsed time.
- [ ] Preserve position names, X/Y/Z coordinates, grid coordinates, well/plate
  layout, and field indices when present.
- [ ] Preserve acquisition datetime and meaningful image/position names.
- [ ] Map exposure and active stage coordinates to OME plane fields.
- [ ] Do not flatten the position axis manually; let `ome-writers` create its
  supported TIFF/Zarr position or plate structure.

### 3.3 Complete application metadata

- [ ] Store the complete `SummaryMetaV1` under the `pymmcore_plus` namespace,
  including:
  - metadata format/version and datetime;
  - device and property descriptions;
  - system and version information;
  - image/camera information, ROI, affine transform, and pixel-size config;
  - configuration groups and presets;
  - pixel-size presets;
  - starting stage position;
  - complete MDA sequence;
  - user/application `extra`.
- [ ] Store complete `FrameMetaV1` per frame in addition to structured OME
  fields.
- [ ] Add export provenance:
  - pymmcore-gui, pymmcore-plus, ome-writers, and ndv versions;
  - export timestamp and transaction UUID;
  - source dataset/sequence UID;
  - source path and format when known;
  - source completion/cancellation status;
  - expected, written, and skipped frame counts;
  - target format and writer backend.
- [ ] Avoid credentials, secrets, and unrelated log contents in provenance.

### 3.4 Metadata manifest test

- [ ] Create one canonical test fixture containing every supported
  `SummaryMetaV1` and `FrameMetaV1` field, Unicode text, multiple channels,
  multiple positions, Z, time, and a plate/grid case.
- [ ] Define a format-neutral expected metadata manifest.
- [ ] Round-trip both target formats and compare semantic values and units, not
  raw XML/JSON formatting.

## Phase 4 — Stream pixels without materializing the dataset

### 4.1 Acquisition-order iterator

- [ ] Implement a frame iterator over the source’s ordered non-spatial
  dimensions.
- [ ] Yield one record containing source index, 2D frame, frame metadata, and
  whether the coordinate was written or skipped.
- [ ] Read at most one frame or one explicitly bounded chunk into memory.
- [ ] Feed frames to `OMEStream.append()` in the acquisition order described by
  the target `AcquisitionSettings`.
- [ ] Call `OMEStream.skip()` for known missing/skipped frames.
- [ ] Handle position as an `ome-writers` meta-dimension rather than assuming
  it is an ordinary stored axis.
- [ ] Handle arrays with only Y/X and arrays with T/C/Z/P combinations.
- [ ] Refuse unsupported ragged or multi-camera shapes with a clear error
  before creating the final target.
- [ ] Do not silently fall back to a generic flattened dimension layout unless
  the user explicitly chooses a documented “raw frames” export.

### 4.2 Source consistency

- [ ] Freeze the source descriptor before export.
- [ ] Verify source shape/dtype/dimensions have not changed between preflight
  and the first write.
- [ ] Detect a source read error separately from a target write error.
- [ ] If the source itself is on a failing drive, stop and report that choosing
  a different destination cannot repair unreadable source frames.
- [ ] Reject exporting onto the source path or anywhere inside a source
  `.ome.zarr` hierarchy.

### 4.3 Memory and performance checks

- [ ] Add a source object whose `__array__` raises, proving full-array
  conversion is never used.
- [ ] Assert peak additional memory remains bounded by a documented number of
  frames/chunks.
- [ ] Benchmark TIFF and Zarr exports separately; correctness remains the gate,
  not maximum throughput.

## Phase 5 — Make each export a filesystem transaction

### 5.1 Preflight

- [ ] Resolve the target and nearest existing parent without mutating the
  requested final path.
- [ ] Check that the selected volume is present, writable, and not read-only.
- [ ] Create an exclusive probe/staging entry on that exact volume. Do not rely
  only on `os.access()`.
- [ ] Estimate uncompressed pixel bytes plus format metadata and safety
  overhead.
- [ ] Compare the conservative estimate with available bytes on the target
  volume and show both values.
- [ ] Include space needed to retain an existing target until replacement
  commits.
- [ ] Treat the estimate as advisory because compression and concurrent disk
  usage can change the result.
- [ ] Detect path conflicts, case-folded conflicts, symlinks, and a target
  nested within its own source.

### 5.2 Stage beside the destination

- [ ] Write to a unique hidden sibling such as
  `.<target>.partial-<transaction-id>`.
- [ ] Store a small transaction manifest beside/inside the partial output with
  requested target, source UID, format, state, timestamps, application
  versions, and last completed phase.
- [ ] Confirm the staging and final paths have the same filesystem/device so
  commit does not degrade into a cross-device copy.
- [ ] Pass `overwrite=False` to the staged `AcquisitionSettings`; a UUID
  collision is an error.
- [ ] Keep an existing final target untouched throughout writing and
  validation.

### 5.3 Flush and durability

- [ ] Always call `OMEStream.close()` in a controlled `try/finally`.
- [ ] Preserve a primary write error if cleanup also fails; attach cleanup
  errors as secondary diagnostics.
- [ ] Ensure backend futures, buffers, queues, writers, and metadata mirrors
  are closed before validation.
- [ ] Add a filesystem durability adapter that can flush files and parent
  directories where the platform supports it.
- [ ] Document weaker guarantees for filesystems that do not provide reliable
  flush/rename semantics, especially some network and removable filesystems.

### 5.4 Commit and replace

- [ ] For a new destination, atomically rename the validated sibling staging
  path to the final path.
- [ ] Require explicit confirmation before replacing an existing target.
- [ ] Never ask `ome-writers` to overwrite the existing final target directly.
- [ ] For replacement:
  1. rename the existing target to a unique sibling backup;
  2. rename the validated staging output to the final name;
  3. flush the parent directory;
  4. remove the backup only after commit succeeds.
- [ ] If commit fails, roll the backup back to the original name.
- [ ] Persist enough transaction state to recover after a process crash between
  the two renames.
- [ ] On startup or before a new export, detect stale partial/backup
  transactions and offer recovery or cleanup; never delete unknown paths.

## Phase 6 — Validate before reporting success

### 6.1 Structural validation

- [ ] Reopen the staged output using an independent public reader appropriate
  for the format.
- [ ] Verify format identity, all expected files/groups, shape, dtype,
  dimension order, position count, channel count, and frame count.
- [ ] Parse and validate OME-XML for TIFF.
- [ ] Parse and validate OME-Zarr/NGFF metadata for Zarr.
- [ ] Verify incomplete/canceled dimensions and skipped coordinates match the
  source descriptor.

### 6.2 Metadata validation

- [ ] Read back and compare the canonical metadata manifest.
- [ ] Verify the `pymmcore_plus` summary namespace exists and is complete.
- [ ] Verify per-plane timing, exposure, position, and full frame metadata.
- [ ] Verify physical sizes, units, channel names, positions, and plate/grid
  information.
- [ ] Treat a missing promised metadata field as validation failure.

### 6.3 Pixel-integrity validation

- [ ] Compute a canonical content hash while reading each source frame.
- [ ] Read every target frame in logical acquisition order and compute the same
  hash.
- [ ] Compare hashes before commit. Do not hash compressed container bytes,
  because the two formats encode them differently.
- [ ] Include dtype, logical index, and frame shape in the hash stream so
  reorderings cannot pass.
- [ ] Make full verification the default for the “super solid” mode.
- [ ] If a faster sampled mode is ever offered, label it clearly and never call
  it full verification.

## Phase 7 — Build the user experience

### 7.1 Export dialog

- [ ] Replace the TIFF-only file dialog with one export dialog containing:
  - format: OME-TIFF or OME-Zarr;
  - destination;
  - normalized resulting path;
  - estimated uncompressed size and available target space;
  - source shape, dtype, dimensions, and completion status;
  - metadata availability summary;
  - verification mode, defaulting to full.
- [ ] Remember the last successful directory and format, not a failed partial
  path.
- [ ] Keep the source format as a suggested default when known.
- [ ] Explain multi-position TIFF output before writing.
- [ ] Disable Start until path, format, source state, and metadata are valid.

### 7.2 Progress and cancellation

- [ ] Run export and verification on a worker owned by the viewer/export job.
- [ ] Emit typed progress phases:
  `preflight`, `writing`, `finalizing`, `verifying`, `committing`, `done`.
- [ ] Show frames and bytes processed when known.
- [ ] Check cancellation between frames/chunks and during verification.
- [ ] Cancellation must close the writer, mark the transaction canceled, keep
  the source, and avoid commit.
- [ ] Prevent two concurrent exports from the same viewer unless they have
  independent immutable source handles.
- [ ] Do not let closing the progress dialog orphan a running writer.

### 7.3 Failure and retry

- [ ] Map failures to stable user-facing categories while preserving the
  original exception:
  - insufficient space (`ENOSPC`, quota exceeded);
  - volume disconnected/unavailable (`ENODEV`, `ESTALE`);
  - I/O/device corruption (`EIO`);
  - read-only filesystem (`EROFS`);
  - permission denied;
  - source read failure;
  - target preparation/write/finalize failure;
  - metadata serialization/write failure;
  - validation mismatch;
  - destination conflict;
  - commit/rollback failure;
  - canceled;
  - unknown.
- [ ] Error text must name the failed phase and affected path without dumping a
  traceback into the dialog.
- [ ] Log the exception chain, errno, writer/backend, transaction ID, source
  UID, target volume, and partial path.
- [ ] Offer:
  - **Choose another location and retry**;
  - **Show partial output** when it is safe to inspect;
  - **Delete this partial output** with exact target confirmation;
  - **Cancel**.
- [ ] Retry creates a new transaction and rereads the unchanged source.
- [ ] Never retry indefinitely or switch destinations automatically.
- [ ] If deleting a partial fails, keep it and report its exact path.
- [ ] On success, show the committed path and a reveal/open-folder action.

## Phase 8 — Replace the old save path

- [ ] Move save orchestration out of `MMArrayViewer._save_data()`.
- [ ] Keep `_array_viewer.py` responsible only for attaching the button,
  gathering the viewer’s export context, opening the dialog, and displaying
  job results.
- [ ] Change the tooltip from “Save as OME-TIFF” to “Export OME dataset…”.
- [ ] Remove `_save_as_tiff()` and `_save_multiposition()` after equivalent
  ome-writers coverage is proven.
- [ ] Remove the direct runtime `tifffile` dependency from this path. Keep the
  dependency only if it is still required as an `ome-writers` backend or
  independent validation reader.
- [ ] Replace broad exception suppression around save-button construction with
  a logged, tested compatibility adapter or a visible disabled action.
- [ ] Avoid direct access to ndv’s private `_btn_layout` if ndv provides or can
  add a public toolbar-extension API.

## Phase 9 — Test matrix

### 9.1 Functional matrix

- [ ] OME-TIFF and OME-Zarr.
- [ ] YX, CYX, ZYX, TCZYX, reordered acquisition axes, and position axes.
- [ ] Single position, multiple positions, grid, and well plate.
- [ ] `uint8`, `uint16`, and `float32`.
- [ ] Completed and canceled/incomplete acquisitions.
- [ ] Plain ndarray with minimal metadata.
- [ ] Source scratch memory, scratch memmap, TIFF-backed view, and Zarr-backed
  view.
- [ ] Empty source, zero-length live dimension, one frame, and a dataset larger
  than RAM.
- [ ] Unicode names and metadata.
- [ ] Existing destination: cancel replacement, successful replacement, and
  rollback.

### 9.2 Failure injection

- [ ] Failure creating staging output.
- [ ] Disk full before the first frame.
- [ ] Disk full halfway through pixel writes.
- [ ] Disk full during writer close.
- [ ] Disk full while writing metadata.
- [ ] Drive disconnect during write, verify, and commit.
- [ ] `EIO`, `EROFS`, `EACCES`, quota exceeded, and stale network handle.
- [ ] Source read failure at a known frame.
- [ ] TIFF worker failure and non-terminating worker.
- [ ] Zarr/TensorStore deferred future failure.
- [ ] Metadata serializer and metadata backend failure.
- [ ] Validation detects wrong shape, reordered frame, changed pixel, missing
  metadata, malformed OME-XML, and malformed Zarr JSON.
- [ ] Atomic rename failure, backup rename failure, and rollback failure.
- [ ] User cancellation in every phase where cancellation is safe.
- [ ] Application restart with stale staging and backup states.
- [ ] Retry to a different mounted volume succeeds after the first target
  fails.

### 9.3 Assertions for every failed export

- [ ] The source remains readable and unchanged.
- [ ] No success signal/message is emitted.
- [ ] The original final target remains unchanged or is restored.
- [ ] The partial path is uniquely owned by the transaction.
- [ ] All worker threads/futures/file handles terminate or produce a specific
  shutdown error.
- [ ] The Save action becomes usable again.
- [ ] A retry can start without restarting the application.

### 9.4 Cross-platform and filesystem coverage

- [ ] Native macOS.
- [ ] Linux.
- [ ] Windows, including open-file rename/delete restrictions.
- [ ] A removable volume test where CI hardware permits it.
- [ ] A network filesystem test where CI infrastructure permits it.
- [ ] Document which durability guarantees could not be tested automatically.

## Phase 10 — Documentation and rollout

- [ ] Add user documentation for choosing formats, multi-position TIFF layout,
  estimated space, progress, verification, partial files, and retry.
- [ ] Document that OME-Zarr is a directory even though it has a
  `.ome.zarr` suffix.
- [ ] Document the difference between acquisition-time saving and exporting a
  completed viewer.
- [ ] Document how canceled acquisitions and skipped frames are represented.
- [ ] Add a troubleshooting table for space, permission, disconnected drive,
  source read, writer finalization, validation, and rollback errors.
- [ ] Include the transaction ID in logs and support reports.
- [ ] Add telemetry only if the project already has an approved, privacy-safe
  mechanism; never include paths or metadata content by default.
- [ ] Roll out behind a feature flag until both formats pass the failure matrix.
- [ ] Remove the old TIFF path and feature flag only after a release has tested
  real datasets on all supported platforms.

## Release gate

- [ ] The selected OME-TIFF or OME-Zarr format is honored explicitly.
- [ ] No complete-dataset `np.asarray()` call exists in the export path.
- [ ] The full metadata fixture round-trips through both formats.
- [ ] A TIFF background write error cannot become a successful result.
- [ ] A Zarr deferred write/finalize error cannot become a successful result.
- [ ] Metadata failure cannot become a successful result.
- [ ] Full read-back pixel verification passes for both formats.
- [ ] Existing destinations survive every injected pre-commit failure.
- [ ] Retry to another drive works without reacquiring or restarting.
- [ ] Peak export memory is bounded and documented.
- [ ] The GUI stays responsive throughout write, finalize, and verification.
- [ ] Ruff, mypy, Pyright, and the full test suite pass.
- [ ] The supported `ome-writers` and `pymmcore-plus` minimum versions contain
  all required upstream error-propagation fixes.

## Reference material

- Current viewer save implementation:
  [`src/pymmcore_gui/_array_viewer.py`](src/pymmcore_gui/_array_viewer.py)
- Current legacy and new-GUI viewer integration:
  [`src/pymmcore_gui/_ndv_viewers.py`](src/pymmcore_gui/_ndv_viewers.py) and
  [`src/pymmcore_gui/_gui/_acquire_viewers.py`](src/pymmcore_gui/_gui/_acquire_viewers.py)
- Current MDA output integration:
  [`src/pymmcore_gui/widgets/_mda_widget.py`](src/pymmcore_gui/widgets/_mda_widget.py)
- Current tests:
  [`tests/test_array_viewer.py`](tests/test_array_viewer.py) and
  [`tests/test_ndv_viewers.py`](tests/test_ndv_viewers.py)
- Official `ome-writers` usage:
  <https://pymmcore-plus.github.io/ome-writers/usage/>
- Official `ome-writers` API:
  <https://pymmcore-plus.github.io/ome-writers/reference/>
- Official OME-TIFF format notes:
  <https://pymmcore-plus.github.io/ome-writers/formats/tiff/>
- Official OME-Zarr format notes:
  <https://pymmcore-plus.github.io/ome-writers/formats/zarr/>
- Official `pymmcore-plus` metadata schema:
  <https://pymmcore-plus.github.io/pymmcore-plus/metadata/>
