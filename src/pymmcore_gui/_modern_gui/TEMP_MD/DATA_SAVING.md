# How image data gets saved

There are two independent ways data ends up on disk:

1. **During acquisition** — the MDA widget's Saving section streams
   straight to disk as frames arrive.
2. **From a viewer's Save button** — after the fact, exporting whatever
   a viewer is currently displaying (including a `"memory"`-only run
   that was never saved to disk).

Both paths ultimately go through the same underlying writer
(`ome_writers`), so they produce byte-for-byte equivalent
OME-TIFF/OME-Zarr output with full metadata.

---

## 1. Saving during MDA

```text
MemoryMDAWidget.run_mda()
  └─ prepare_mda()                      widgets/_mda_widget.py
       ├─ Saving checked   → save path (e.g. "/data/expt.ome.zarr")
       └─ Saving unchecked → "memory" (fallback, see §1b)
  └─ execute_mda(output) → core.mda.run(sequence, output=output)
       └─ pymmcore_plus.mda._sink.OmeWritersSink
            └─ ome_writers.create_stream(AcquisitionSettings)
                 └─ writes frames as they arrive, via
                    stream.append(frame, frame_metadata=...)
```

- **Format** is chosen in the MDA widget's Saving section combo box:
  `ome-tiff` or `ome-zarr` (`tiff-sequence` is removed — see
  `MemoryMDAWidget.__init__`).
- **Metadata written automatically, per format:**
  - *Summary metadata* (devices, system info, config groups, the
    `MDASequence` itself) is attached once via
    `OMEStream.set_global_metadata("pymmcore_plus", ...)` — lands as
    a Zarr root attribute, or an OME-XML `MapAnnotation` for TIFF.
  - *Per-frame metadata* (`delta_t`, `exposure_time`, stage
    `position_x/y/z`) is attached on every `stream.append()` call,
    via `pymmcore_plus.mda.frame_meta_to_ome()`.
  - Channel names, physical pixel size (µm), Z-step, and
    multi-position layout come from `AcquisitionSettings.dimensions`,
    derived from the `MDASequence` by
    `ome_writers.useq_to_acquisition_settings()`.
- **Multi-position OME-TIFF** resolves to a *directory* of
  per-position files (`expt/expt_p000.ome.tiff`, `expt_p001.ome.tiff`,
  ...), not a single file.

### 1b. The `"memory"` fallback

If the Saving section is unchecked, `prepare_mda()` still passes
`output="memory"` instead of `None` — this makes `CMMCorePlus.run_mda`
create an `OmeWritersSink` backed by `ome_writers`' **scratch**
backend: an in-RAM (or, if it would exceed a memory cap,
memmap-spilled) numpy array per position, *not* a file. This exists
purely so the acquisition still has a live, viewable array — nothing
is written to disk. This is the case the viewer's Save button (§2)
exists to cover.

### Files involved in the MDA-time path

<!-- markdownlint-disable MD013 -->
| File | Role |
|---|---|
| `src/pymmcore_gui/widgets/_mda_widget.py` | `MemoryMDAWidget.prepare_mda()` — picks save path vs `"memory"` |
| `pymmcore_plus/mda/_sink.py` (`OmeWritersSink`) | Bridges `MDARunner` events → `ome_writers` stream |
| `pymmcore_plus/mda/_runner.py` | `MDARunner.run()`, `.get_view()`, `.get_sink()` |
| `ome_writers` (`create_stream`, `OMEStream`) | Actual OME-TIFF/OME-Zarr writer |
<!-- markdownlint-enable MD013 -->

---

## 2. Saving from a viewer's Save button

This is what covers a `"memory"` run (§1b) — or re-exporting any open
viewer's data to a different format/location.

```text
MMArrayViewer._save_data()                  _array_viewer.py
  ├─ RGBA data (Preview snapshot only)
  │    → _save_rgb_snapshot()  [direct tifffile write, no metadata]
  └─ everything else:
       ├─ _prompt_save_path()
       │    → one file dialog, OME-TIFF / OME-Zarr filters
       ├─ record = self._acquisition_record
       │            or _synthesize_record(self)  (fallback)
       └─ _export_with_overwrite_prompt(record, path, fmt)
            └─ export_acquisition()            _mda_export.py
                 └─ ome_writers.create_stream(new AcquisitionSettings)
                      └─ replays record.view frame-by-frame into the
                         new stream
```

### Where the data/metadata come from (`AcquisitionRecord`)

For a viewer created from an MDA run, `AcquireViewersManager`
(`src/pymmcore_gui/_modern_gui/_acquire_viewers.py`) builds and
maintains an `AcquisitionRecord`:

- **At `sequenceStarted`**: snapshots the sink's resolved
  `AcquisitionSettings` and `SummaryMetaV1` (`core.mda.get_sink()`),
  and keeps a reference to the live view (`core.mda.get_view()`) — a
  `StreamView` indexable in acquisition order.
- **On every `frameReady`**: appends that frame's metadata
  (`frame_meta_to_ome(meta)`) to the record — this happens
  *unconditionally*, even if the viewer's follow/lock button is
  toggled off, so locking the slider mid-run never truncates what
  gets exported later.

If no record was captured (e.g. the snap/live Preview, which isn't
backed by an MDA at all), `_synthesize_record()` builds a minimal one
on the fly straight from what the viewer is currently displaying —
real axis names/scale where available, generic `"other"` axes
otherwise, best-effort summary metadata from the live core. No
per-frame metadata is available in this fallback case.

### `export_acquisition()` mechanics

- **Never materializes the whole array.** It reads one 2D frame at a
  time from `record.view` and writes it straight into the new stream
  — same principle as a live acquisition, just replayed after the
  fact.
- **Clamps to what was actually acquired** — a cancelled run or an
  unbounded (`GeneratorMDASequence`) acquisition only exports the
  frames that exist, not the originally-planned count.
- **Multi-position OME-TIFF** → directory of per-position files, same
  as a live acquisition.
- **Overwrite handling**: the first attempt always runs with
  `overwrite=False`; only if `ome_writers` reports the resolved
  output path already exists does the user get asked to confirm,
  then it retries with `overwrite=True`. (The literal path typed in
  the dialog and the *resolved* output path can differ — e.g.
  multi-position TIFF turns a file path into a directory — so
  `ome_writers` itself, not a pre-check on the typed path, is the
  authority.)
- **Cancellable** via a progress dialog; whatever was written before
  Cancel stays on disk (same as a cancelled live MDA).

### The one gap: RGB/RGBA

`ome_writers` models a written frame as a plain 2D (Y, X) plane — it
has no concept of a color/sample axis. The only place an RGB frame
can appear is the snap/live Preview (a 1-frame ring buffer), so that
one case bypasses `ome_writers` entirely and writes a plain TIFF via
`tifffile` directly (no OME metadata, no format choice).

### Files involved in the Save-button path

<!-- markdownlint-disable MD013 -->
| File | Role |
|---|---|
| `src/pymmcore_gui/_array_viewer.py` | `MMArrayViewer._save_data`, save-path prompt, overwrite prompt, record synthesis fallback |
| `src/pymmcore_gui/_mda_export.py` | `AcquisitionRecord`, `export_acquisition()` — the actual replay-to-new-stream logic |
| `src/pymmcore_gui/_modern_gui/_acquire_viewers.py` | `AcquireViewersManager` — captures/updates the live `AcquisitionRecord` per MDA viewer |
<!-- markdownlint-enable MD013 -->

---

## Data duplication?

None until you actually click Save:

- `AcquisitionRecord.view` is a **reference** to the same live array
  the viewer is already displaying — not a copy.
- `export_acquisition()` streams one frame at a time into the new
  writer, so peak extra memory during a save is ~one frame, not a
  second copy of the whole acquisition.
- Once a file is actually written, you do have two copies (the
  original — in RAM for a `"memory"` run, or on disk if re-exporting
  an already-saved run — plus the new file). That's inherent to what
  "save" means, and only happens when you explicitly trigger it.
