# Plan: optional grid stitching with multiview-stitcher

Status: proposal, not started. Measurements taken 2026-11-14 against
`multiview-stitcher` 0.1.63 (PyPI) on an Apple M2 Pro (12 cores), Python 3.13.
Revised after review: process launch, orientation, output spacing,
registration fallbacks, resource claims and the worker protocol.

## TL;DR

- **Feasible, and fast enough for the common case.** On synthetic data, a
  2D grid of up to about 100 tiles registers and fuses in **5–55 s**. The
  upper end is a 10×10 grid of 2048² tiles written to disk; binning brings
  it closer to 40 s. That is usually shorter than acquiring the grid.
  Registration put tiles back within **0.1–0.4 µm** of where they belonged
  on every test case. These are feasibility numbers, not guarantees: real
  compressed data read from disk still has to be measured (§2).
- **Stitching runs in a separate worker process**, never inside the GUI
  process. The acquisition runs as a Python thread in the GUI process, so
  in-process stitching would compete with it for the GIL. A separate
  process also isolates crashes, makes Cancel immediate, and returns all
  memory when it exits. It does **not** guarantee the acquisition
  priority over disk bandwidth or memory, so that has to be measured (§4.4).
- **Stitching requires Saving to be checked.** The worker reads the tiles
  from the saved acquisition and writes the mosaic next to it. The option
  therefore lives **in the Saving section**, shown only when the sequence
  contains a grid.
- **Progress lives in the status bar** (a small item plus a popover with
  Cancel), and a toast announces when a mosaic is done. **The app refuses
  to close** while stitching runs unless the user cancels it.
- **Tile orientation follows Micro-Manager's convention** (§5). A
  calibrated pixel affine is used as is; camera mirror settings only apply
  when there is no calibration. While checking this, three bugs turned up
  in the Stage Explorer's transform: mirroring was ignored for rotated
  calibrations, calibrated affines were flipped twice, and uncalibrated
  scale was wrong at binning ≠ 1. **They are fixed** in pymmcore-widgets
  (`cite`, uncommitted), with tests.
- **Z-stacks:** full 3D registration is very slow (93 s for 3×3 tiles of
  1024², 20 planes). Registering on the max projection takes **1.6 s**.
  That is the default, with its limits stated in §6.
- **Dependency cost is real but has no conflicts.** The package adds 34
  packages and about 325 MB. None of them conflict with the GUI's current
  pins. It goes in an optional extra,
  `pymmcore-gui[stitching]`, pinned to `multiview-stitcher>=0.1.63,<0.2`.
  Only the worker process ever imports it.
- **Reprocessing:** "Stitch…" on a reopened raw acquisition comes first.
  The re-stitch button on a mosaic comes after it and writes new versions
  instead of overwriting (§7).

---

## 1. What multiview-stitcher gives us

<!-- markdownlint-disable MD013 -->
| Need | API | Notes |
| --- | --- | --- |
| Describe tiles | `si_utils.get_sim_from_array(arr, dims, scale, translation, affine, transform_key, c_coords, t_coords)` → `msi_utils.get_msim_from_sim(sim, scale_factors=[])` | Accepts numpy/dask/zarr arrays, so tiles can stay lazy. A full affine per tile is supported. |
| Register | `registration.register(msims, reg_channel=…, transform_key="stage", new_transform_key="registered", registration_binning=…, pre_registration_pruning_method="keep_axis_aligned", groupwise_resolution_method=…, post_registration_do_quality_filter=…, pairwise_executor=…)` | Phase correlation by default. Registers every timepoint separately. |
| Fuse | `fusion.fuse(msims, transform_key="registered", output_spacing=…, output_zarr_url=…, zarr_options={"ome_zarr": True}, batch_options={"batch_func": …, "n_batch": …}, blending_widths=…)` | Streams chunk by chunk into a pyramidal OME-Zarr (NGFF 0.4). `output_spacing` defaults to the input spacing, so we always pass it (§5). |
| Progress | `pairwise_executor(msims, edges, kwargs)` and `batch_func(fuse_chunk, block_ids, …)` | Both are user-supplied callables. The worker counts completed pairs and blocks. |
<!-- markdownlint-enable MD013 -->

**Verified:** the GUI's own `open_acquisition()` opens multiview-stitcher's
OME-Zarr output directly (`OMEZarrWrapper`, dims `t,c,y,x`, pyramid levels
ignored), so the result viewer reuses the existing reopen path.

## 2. Measured speed (feasibility evidence)

Method: textured synthetic uint16 tiles **held in memory**, cut from one
large image at 10 % overlap. The declared stage positions were jittered by
±3 µm (pixel size 0.65 µm) to give registration a real error to remove.
"err" is the largest remaining error after registration, compared with the
known jitter.

<!-- markdownlint-disable MD013 -->
| Case | Input | Register | Fuse | err |
| --- | ---: | ---: | ---: | ---: |
| 3×3, 2048², 1 ch | 75 MB | 2.8 s | 1.3 s | 0.16 µm |
| 3×3, 2048², 1 ch, binning 2 | 75 MB | 2.2 s | 1.1 s | 0.42 µm |
| 3×3, 2048², 3 ch | 226 MB | 2.8 s | 2.1 s | 0.11 µm |
| 5×5, 2048², 1 ch | 210 MB | 7.4 s | 2.3 s | 0.19 µm |
| 5×5, 2048², 1 ch, binning 4 | 210 MB | 4.3 s | 2.4 s | 0.32 µm |
| 10×10, 1024², 1 ch | 210 MB | 13.0 s | 4.0 s | 0.33 µm |
| 10×10, 2048², 1 ch | 840 MB | 25.6 s (binning 4: 11.2 s) | 8.7 s in memory | — |
| 20×20, 512² (400 tiles) | 210 MB | 61 s (`global_optimization`) / **29 s** (`linear_two_pass`) | — | — |
| **3D** 3×3, 1024², z=20 | 377 MB | **92.8 s** full 3D · 30.8 s (4 pairs in parallel) · 4.4 s (binning xy 4) · **1.6 s (max projection)** | 3–5 s | 0.80 µm (full 3D) |
<!-- markdownlint-enable MD013 -->

Fusing straight to OME-Zarr (10×10, 2048², 840 MB in, 695 MB fused, 421 MB
on disk):

| Mode | Time |
| --- | ---: |
| `output_zarr_url`, default (one chunk at a time) | 79.9 s |
| `output_zarr_url`, `batch_func` running 8 threads, `n_batch=32` | **27.8 s** |
| No zarr: fused in memory via dask (no write) | 8.7 s |

Startup: the `registration` and `fusion` submodules import in **0.9 s
warm** (27 s the very first time after installing, while bytecode is
compiled; a bundle ships it compiled). Worker startup in the frozen app has
not been measured yet.

**Takeaways**

- In 2D, registration costs more than fusion. Binning cuts its time by
  20–55 % and still recovers tiles within 0.3–0.4 µm. Default to
  **automatic binning** that brings each tile to about 512–1024 px per
  side for registration.
- Above ~100 tiles, `linear_two_pass` is twice as fast as the default global
  optimizer. Choose the method automatically by tile count.
- Always pass a threaded `batch_func` when fusing: it is 3× faster than the
  default. Chunk size is worth tuning later.
- **Extrapolation, not a measurement:** at ~30 MB/s, a 10×10 grid of 2048²
  tiles with 30 planes (~25 GB) would take roughly 15 min. This has to be
  measured on compressed OME-TIFF and OME-Zarr data read from disk before
  it is quoted anywhere.
- **Peak memory was not measured reliably**: the synthetic data itself
  needed several GB. Chunked output limits the size of the fused array in
  memory, but it does not bound the worker's peak by itself; the number of
  chunks in flight, chunk size and the per-pair registration buffers all
  add up. Phase 1 measures the peak (§4.4). The GUI checks free disk space
  for the estimated output before starting.

## 3. User experience

### 3.1 Enabling it (MDA widget → Saving section)

Only when `importlib.util.find_spec("multiview_stitcher")` succeeds, add a
row to the **Saving section**, below Tiff Mode:

```text
☑ Save
  Directory   [ /data/expt ]                 [📁]
  Name        [ expt_001 ]
  Format      [ ome-tiff ▾ ]
  Tiff Mode   [ Independent file per position ▾ ]
  Stitching   [✓] Also save a stitched mosaic    [⚙]   ← only when a grid is used
```

Why the Saving section and not the Grid section:

- **The dependency on Saving is built in.** Inside the Saving group box,
  the option is off whenever Saving is off, like every other Saving
  option, and a run with Saving off never stitches.
- **One setting covers every grid source:** the global grid, per-position
  grids from the positions table, and later well-plate grids.
- **It describes an output**, like Format and Tiff Mode.
- **pymmcore-widgets doesn't change.** The GUI already adds its own row to
  the Saving layout (`_install_ome_tiff_options`); this is a fourth row
  added the same way.

Behaviour:

- **Shown only when the sequence uses a grid**, global or per position,
  using the existing `_sequence_uses_grid()`, re-checked whenever the
  sequence changes. Showing or hiding the row recomputes the Saving
  section's fixed height, as `_sync_ome_tiff_options` already does.
- The setting is stored in
  `MDASequence.metadata["pymmcore_gui"]["stitching"]` (enabled plus the
  settings), so saved/loaded MDA files restore it and the dataset records
  how it was stitched.
- When every grid has no overlap (≤ 0 %), registration can't work: the
  checkbox stays enabled but "Use stage positions only" is forced, and the
  tooltip says so.
- **Discoverability (optional, later):** a read-only note on the grid's
  summary line, e.g. `… · stitched after acquisition`.

### 3.2 The ⚙ settings dialog

<!-- markdownlint-disable MD013 -->
| Group | Setting | Default |
| --- | --- | --- |
| Alignment | **Refine with registration** / Use stage positions only | Refine |
| | Registration channel (from the sequence's channels) | first channel |
| | Registration binning: Auto / 1 / 2 / 4 / 8 | Auto (~1024 px per tile side) |
| | Z-stacks: register on max projection | on |
| | Time-lapse: register first timepoint only / every timepoint | first only |
| | Reject poor matches (threshold) | on, 0.2 |
| Blending | Smooth (weighted average) / Max / Average | Smooth |
| | Blend width (µm) | Auto |
| Output | Open the result in a viewer when done | on |
| Performance | Worker threads (one budget for everything, §4.4) | half the CPU cores |
| | Run at low CPU priority | on |
<!-- markdownlint-enable MD013 -->

"Reject poor matches" drops those pairs from the registration graph. §6.1
says exactly what happens to the tiles involved. Defaults persist in a new
`StitchingSettingsV1` in `_settings.py`; the dialog edits the copy stored
on the sequence.

### 3.3 While it runs: status bar, popover, toast

`BusyOverlay` covers and blocks the widget under it, and the MDA widget must
stay usable, so it is the wrong tool here. Instead:

- **Status bar (right side):** `⟳ Stitching 42 %`, shown only while jobs are
  queued or running; with several jobs, `⟳ Stitching 1/3 · 42 %`.
- **Clicking it opens a popover** with one row per job: the name, the phase
  ("Reading tiles", "Registering 14/24 pairs", "Fusing 120/400 blocks",
  "Writing pyramid"), a determinate `QProgressBar`, and **Cancel**. Queued
  jobs show "Waiting" and can be removed.
- **When a job finishes:** an info toast with an **Open** button (or the
  viewer opens by itself if that setting is on). If registration had to
  fall back, the toast says so, e.g. "3 tiles placed by stage position,
  mosaic aligned in 2 groups" (§6.1).
- **On failure:** a warning toast with the reason and a pointer to the
  worker log. A failed or cancelled job never leaves a half-written
  mosaic that looks finished.
- Jobs start at `sequenceFinished`, for completed runs only; a cancelled or
  failed acquisition is skipped, with an info toast saying why.

### 3.4 Closing the app while stitching

`closeEvent` already refuses to close during an acquisition
(`_ready_to_close_during_acquisition()`). Add the same check for
stitching, after the acquisition check:

```text
┌ Stitching in progress ────────────────────────────────┐
│ expt_001_stitched is 42 % done (1 more job queued).   │
│ Quitting now cancels it and deletes the partial file. │
│                       [Keep working]  [Cancel & quit] │
└───────────────────────────────────────────────────────┘
```

- **Keep working** (default) ignores the close event.
- **Cancel & quit** kills the worker, **waits for it to exit**, schedules
  deletion of the partial output (§4.3), clears the queue, then continues
  the normal close sequence.
- A later option only a separate process allows: **"Finish in background &
  quit"**. Not in v1.

### 3.5 Where the result goes

<!-- markdownlint-disable MD013 -->
| Saving section | Result |
| --- | --- |
| Checked | `<dir>/<stem>_stitched.ome.zarr` next to the acquisition, then (by default) opened in a viewer. |
| Unchecked | No stitching; the option sits inside the unchecked Saving section (§3.1). |
<!-- markdownlint-enable MD013 -->

- **Format: always OME-Zarr in v1**, even when the acquisition was OME-TIFF:
  it streams to disk, has a pyramid, and the GUI already opens it.
- **Never overwrite.** If the name exists, write `_stitched-2`, `_stitched-3`, …
- **Provenance:** a versioned zarr root attribute
  `pymmcore_gui.stitching = {schema_version, source, source_fingerprint,
  settings, transforms, groups, multiview_stitcher_version}`.
  `source_fingerprint` (paths, sizes, modification times) lets reprocessing
  detect moved or changed raw data (§7).

### 3.6 What gets stitched

<!-- markdownlint-disable MD013 -->
| Sequence shape | Groups → outputs |
| --- | --- |
| Global `grid_plan`, no stage positions | 1 mosaic |
| Global relative `grid_plan` × N stage positions | 1 mosaic per position |
| Per-position sub-sequence grids | 1 mosaic per position that has a grid; positions without a grid are skipped |
| `WellPlatePlan` with a grid `well_points_plan` | 1 mosaic per well (later phase) |
<!-- markdownlint-enable MD013 -->

Several groups produce `<stem>_stitched_<position>.ome.zarr` files from one
job. Open at most a few viewers automatically and list the rest in the
toast.

## 4. Architecture

### 4.1 Overview

```text
GUI process                                         Worker process
───────────                                         ──────────────
MDA finishes (completed, Saving on, grid, stitching enabled)
  └─ StitchingController (QObject)  _stitching/_controller.py
       ├─ waits until the sink is closed and flushed
       ├─ creates a job dir in the registry (§4.3):
       │    job.json  = StitchJobV1 (validated, versioned)
       ├─ queue: one worker at a time
       └─ QProcess(<worker command>, job dir)  (§4.2)
                                 ───────────────►  pymmcore_gui.__main__:
                                                   dispatch before any Qt import
                                                   └─ _stitching/_worker.py
                                                      resource limits (§4.4)
                                                      run_stitch(job, report)
                                                       (_stitching/_engine.py,
                                                        the only module that
                                                        imports multiview_stitcher)
                                                       1. open saved store,
                                                          lazy tile per position
                                                       2. per-tile affine (§5)
                                                       3. register (§6)
                                                       4. fuse → <name>.tmp
                                                          (threaded batch_func)
                                                       5. provenance, rename
       ◄── polls progress.json (QTimer, 4 Hz) ────  atomic rewrites, ≤ 5 Hz
       ◄── QProcess.finished + result.json ───────  result / error + exit code
       ├─ status bar item and popover (§3.3)
       └─ on success: toast + open_acquisition_async(path)
          on failure or cancel: cleanup (§4.3), toast
```

### 4.2 Launching the worker

- **The worker module must not import Qt.** Today it would: the package's
  [`__init__.py`](../../src/pymmcore_gui/__init__.py) imports `_app` and
  `_main_window` eagerly, and `import pymmcore_gui` pulls in PyQt6. Python
  runs that file before `python -m pymmcore_gui` and before any
  `pymmcore_gui._stitching` import. Fix in phase 1: make `__init__.py` lazy
  (module `__getattr__` for `MicroManagerGUI` and `create_mmgui`, real
  imports under `TYPE_CHECKING`). This also speeds up every
  `import pymmcore_gui`. A test runs the worker module in a fresh
  interpreter and asserts `"PyQt6" not in sys.modules`.
- **The command differs between development and the frozen app.** In
  development `sys.executable` is the Python interpreter, so the command is
  `[sys.executable, "-m", "pymmcore_gui", "_stitch-worker", <job dir>]`.
  Frozen (`getattr(sys, "frozen", False)`), `sys.executable` is the `mmgui`
  binary: `[sys.executable, "_stitch-worker", <job dir>]`. No
  `multiprocessing`, no `freeze_support()`, no pickling.
- **Dispatch first.** The typer callback in `_cli.py` always launches the
  GUI, so the worker argument is caught on the first line of
  `__main__.main()`. It must import only `_stitching._worker`.

### 4.3 Protocol, cancellation and cleanup

Everything goes through files in the job directory. The protocol does not
depend on the worker's stdio: PyInstaller's windowed Windows build
(`console=False`) can leave `sys.stdout`, `sys.stderr` and `sys.stdin` as
`None`.

- **Job registry:** `<user data dir>/stitch-jobs/<uuid>/` holds `job.json`
  (input), `progress.json`, `result.json`, `worker.log` and `owner.json`
  (the worker's PID and process start time, written by the worker at
  startup).
- **Typed, versioned models** (pydantic): `StitchJobV1`,
  `StitchSettingsV1`, `ProgressEvent` (phase enum, done, total, message),
  `StitchResultV1` (paths, fallback report) and `StitchErrorV1`. Each has a
  `schema_version`; the worker refuses a job version it does not know.
- **Progress:** the worker rewrites `progress.json` atomically (write to a
  temp file, then `os.replace`) at most 5 times a second. The GUI polls it
  with a `QTimer` every 250 ms, only while a job runs.
- **Result:** `result.json` or `error.json` is written last, then the worker
  exits 0 or 1. An abnormal exit without either file (segfault, killed for
  running out of memory) is reported with the tail of `worker.log`.
- **Library output:** at startup the worker points `sys.stdout` and
  `sys.stderr` at `worker.log`, whether they existed or not, and sets
  `TQDM_DISABLE=1`.
- **GUI gone:** `job.json` carries the GUI's PID and process start time. A
  watchdog thread in the worker checks both every 2 s with `psutil`; if the
  GUI process is gone (or the PID was reused), the worker exits.
- **Cancel:** `QProcess.kill()`, then wait for `QProcess.finished` before
  touching any output. **Deleting** a large `.tmp` zarr directory runs on
  a `QThreadPool` task, never on the GUI thread.
- **Output is never half-visible:** the worker writes
  `<name>.ome.zarr.tmp` and renames it at the end.
- **Startup cleanup only removes what the app owns:** for each registry
  entry whose `owner.json` process is no longer alive, delete the `.tmp`
  output paths listed in its `job.json`, then the entry. No filename
  globbing in user directories. Failed jobs keep their log for a few days.
- **Controller state model:** `Queued → Starting → Running →
  (Finished | Failed | Cancelling → Cancelled)`, one enum per job. Every
  transition goes through one method that also updates the status item.

### 4.4 Resource budget

Process isolation does not reserve memory, disk bandwidth or GPU for the
acquisition, and the GUI can still suffer under system-wide memory
pressure. So there is one coordinated budget instead of independent knobs:

- **N = "Worker threads"** (default: half the cores).
- Math libraries single-threaded in the worker: `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`,
  `NUMEXPR_NUM_THREADS` = 1, set in the worker's environment by the
  controller before launch.
- Registration: `n_parallel_pairwise_regs = N`.
- Fusion: the threaded `batch_func` runs N blocks at a time, and each block
  uses dask's **synchronous** scheduler, so there is no second pool nested
  inside. `n_batch` and chunk size are set so that
  N × (output chunk + the input regions it needs) stays below a memory cap
  derived from available RAM.
- Low CPU priority via `psutil` (`nice(10)` / `BELOW_NORMAL_PRIORITY_CLASS`).

**Acceptance measurements (phase 1, before any promise in the UI):**
peak worker memory on the 10×10 2048² and 3D cases from real compressed
files; acquisition frame-interval jitter during a fast time-lapse with and
without a concurrent large stitch; GUI responsiveness during it.

## 5. Coordinates

### 5.1 Pixel → stage transform (Micro-Manager's convention)

Java Micro-Manager (`XYNavigator`) uses a calibrated pixel affine **as
is**, because it is measured on the images as the camera delivers them,
including mirroring. Mirror and swap flags are applied only when there is
no calibration. The GUI's own calibration
(`_pixel_calibration`) also measures on delivered images. So:

<!-- markdownlint-disable MD013 -->
| Situation | Linear part `A` |
| --- | --- |
| Stored preset affine is MMCore's default identity (no calibration) | `diag(±p, ±p)`, the sign from `TransposeMirrorX/Y`, with `p` the current pixel size |
| Calibrated affine, scale within 10 % of `p` | the current `getPixelSizeAffine()` as is, no flags |
| Calibrated affine whose scale disagrees with `p` (stale) | treated as uncalibrated, like Micro-Manager |
<!-- markdownlint-enable MD013 -->

The **stored** affine (`getPixelSizeAffineByID(current preset)`) is what
identifies "no calibration". MMCore scales the default identity by binning
and magnification, so the *current* affine at binning 2 is
`(2,0,0,0,2,0)`.

**Stage Explorer bugs found and fixed** (pymmcore-widgets `cite`,
`AffineState._compute_system_affine`, uncommitted):

1. Mirroring was applied by negating the diagonal of the calibrated
   affine. For a 90° calibration both diagonal entries are 0, so mirroring
   had no effect; for other angles it produced a distorted matrix.
2. Mirror flags were applied **on top of** calibrated affines, flipping a
   system twice when it was calibrated with mirroring on.
3. "No calibration" was detected by comparing the *current* affine with
   the identity. At binning 2 that comparison fails, so tiles were scaled
   by 2.0 µm/px instead of the real pixel size.

The new tests cover the uncalibrated case at binning 2, calibrated
affines (axis-aligned, 90°, 30°, mirrored) with mirror on and off, and a
stale affine. The HCS automated calibration uses the same transform and
gets the fix too.

Known limitation (unchanged): `TransposeSwapXY` is not handled in the
uncalibrated case. Toggling a mirror setting after calibrating makes the
affine stale; a warning for that would be a separate improvement.

For stitching, **the GUI computes `A` and passes it in `job.json`**; the
worker never derives it:

- **Fresh runs:** `AffineState(core).system_affine` captured at
  `sequenceStarted`, the same transform the Stage Explorer uses.
- **Reopened data:** a pure function over the summary metadata
  (`image_infos[].pixel_size_affine`, pixel size, camera mirror
  properties). The stored preset affine isn't in the metadata, so
  "calibrated" is decided by the scale check alone, which is
  Micro-Manager's own rule. A shared test asserts that both paths agree.

### 5.2 Per-tile placement and output spacing

- useq grid positions are the **FOV center** in stage µm, stored as-is by
  ome-writers; multiview-stitcher wants the image origin.
- Each tile gets `scale = p` (isotropic pixel size) and
  `affine = T(stage_center) · (A / p) · T(−W·p/2, −H·p/2)`. For the usual
  calibrations `A / p` is a rotation or mirror. With shear or unequal X/Y
  scale it is still a valid affine and is passed unchanged. The
  calibration routine already warns above 2 % anisotropy.
- **Output spacing is always explicit:** `output_spacing = {"y": p, "x": p}`
  (plus the z step for stacks). This fixes the mosaic's resolution, size
  and the time estimates, whatever representation the transform uses.
- Planned positions by default; measured positions (per-frame
  `position_x/y`) as an option.
- A rotated camera gives a mosaic that is not axis-aligned; fusion's
  `"union"` bounding box then adds blank corners. Accepted and documented.

## 6. Registration behaviour

### 6.1 Rejected pairs, isolated tiles, separate groups

Upstream behaviour (verified in `param_resolution/__init__.py`): pairs
below the quality threshold are **removed from the graph**, then every
connected group of tiles is resolved on its own. A group with no edges gets
the identity transform. What that means for the user, and what we add:

<!-- markdownlint-disable MD013 -->
| Case | Behaviour |
| --- | --- |
| One rejected pair, tiles still connected through other pairs | Aligned through the other pairs. Nothing else to do. |
| A tile with no accepted pairs | Stays at its stage position (identity). Counted in the toast. |
| Several separate groups | Each group is aligned internally. **Anchoring (ours):** after resolution, each group is shifted so its tiles' mean position matches their mean stage position (least-squares). Otherwise, the group would carry its arbitrarily chosen reference tile's stage error. |
| Every pair rejected | Equivalent to "stage positions only"; the toast says registration found no usable overlap. |
<!-- markdownlint-enable MD013 -->

The result records the number of rejected pairs, isolated tiles and
groups, in both the toast and the provenance.

### 6.2 Max-projection registration (z-stacks)

- It estimates **XY translation only** and cannot recover relative Z
  displacement. Z placement comes from each tile's own Z origin (stage Z
  plus the z-plan offset) and z step, which are kept unchanged.
- Lifting to 3D: the 2D correction `[[R, t]]` becomes a 3D affine with an
  identity Z row and column, composed onto each tile's own 3D placement.
- Supported: all tiles in a group have the same number of planes and the
  same step; per-tile Z origins may differ (e.g. absolute positions), and
  fusion places each tile at its own Z.
- Rejected with a clear message: different plane counts or steps within a
  group, and missing planes (an interrupted run). Per-tile autofocus that
  changes Z between tiles is fine for placement, but max projection can't
  correct a wrong Z. The settings dialog explains that.
- Full 3D registration stays available as an advanced option.

### 6.3 Time-lapse

Registering the first timepoint and reusing the result assumes the
relative tile placement is stable through the run, which holds for
mechanical stage repeatability. Sample drift between tiles over time is not
corrected. "Every timepoint" is the setting for that, at T times the cost.

## 7. Reprocessing: "Stitch…" first, re-stitch later

Reprocessing needs more than "run it again": validating provenance,
detecting moved or changed raw data, invalidating saved registrations, and
handling a viewer that holds the file open. v1 therefore:

- **Writes new versions** (`_stitched-2`, …) and never overwrites, so no
  file is swapped under an open viewer.
- **Phase 3a — "Stitch…" on a reopened raw acquisition** (viewer button
  when the recovered sequence has a grid, and File menu). It covers "I
  forgot to tick the box" and old data. It needs only the summary
  metadata, with no saved registration to reuse.
- **Phase 3b — Re-stitch button on a mosaic viewer.** It reads the
  provenance and checks `schema_version` and `source_fingerprint`. If the
  source moved, it asks the user to locate it. It reuses the saved
  registration only if the source fingerprint **and** the registration
  settings are unchanged; otherwise it registers again.

A "Show tile boundaries" overlay with the rejected pairs would help judge a
result. Later.

## 8. Packaging

- `pyproject.toml`:
  `[project.optional-dependencies] stitching = ["multiview-stitcher>=0.1.63,<0.2"]`.
  The package is pre-alpha, so the range is tested in CI and widened
  deliberately. A dry run against the current `.venv` shows only
  additions, no up- or downgrades (34 packages, ~325 MB: scipy 70, pandas
  44, itkwasm-downsample-wasi 33, scikit-image 25, wasmtime 23,
  matplotlib 23 MB, …).
- **PyInstaller bundle: open question.** ~325 MB more, and probably new
  hooks (wasmtime and itkwasm ship native libraries and `.wasm` files). The
  worker itself needs nothing special in the bundle.
- Every multiview-stitcher call stays in `_stitching/_engine.py`. A CI job
  installs the extra; all other tests skip when it is missing.

## 9. Phased implementation

<!-- markdownlint-disable MD013 -->
| Phase | Scope | Main files |
| --- | --- | --- |
| **0. Spike** | Headless script: saved GUI grid acquisition (OME-TIFF and OME-Zarr) → stitched OME-Zarr. **Orientation on a real microscope** with the fixed transform, including mirrored and rotated cameras. Worker startup time in the frozen app. | scratch |
| **0b. Explorer fix** | Commit and push the `AffineState` fix in pymmcore-widgets `cite`; relock the GUI. | pymmcore-widgets `_stage_explorer.py`, tests; GUI `uv.lock` |
| **1. Core** | Lazy `__init__.py`; models; tile grouping and placement; `run_stitch` (§5–§6); worker entry point, file protocol, watchdog; controller with state model, registry and cleanup; resource budget; **acceptance measurements** (§4.4). | `__init__.py`, `__main__.py`, `_stitching/{_models,_geometry,_inputs,_engine,_worker,_controller}.py`, `_settings.py` |
| **2. UI** | Saving-section row and ⚙ dialog (shown only with a grid); settings in `sequence.metadata`; status bar item and popover; toasts; close guard; extra in `pyproject.toml`. | `widgets/_mda_widget.py`, `widgets/_stitching_dialog.py`, `widgets/_stitching_status.py`, `_main_window.py`, `_ndv_viewers.py` |
| **3a. "Stitch…"** | Reopened raw grid acquisitions; affine from metadata. | `_ndv_viewers.py`, `_acquisition_loader.py` |
| **3b. Re-stitch** | Provenance validation, saved-registration reuse, versioned outputs. | `_ndv_viewers.py`, `_array_viewer.py` |
| **4. Later** | Per-well HCS mosaics; OME-TIFF output; "Finish in background & quit"; pairwise registration during acquisition; upstream PRs (`progress_callback`, optional matplotlib). | |
<!-- markdownlint-enable MD013 -->

## 10. Testing

- **Geometry (pure, no package needed):** combined rotation and reflection
  (90°, an arbitrary angle such as 30°, mirrored), on a **non-square**
  image; uncalibrated with mirror flags at binning 1 and 2; stale affine;
  the fresh-run and metadata paths agree.
- **Engine (in-process, skipped if the package is missing):** jitter
  recovered within 0.5 px; mirrored or rotated placements give the same
  mosaic as the plain reference; **output spacing equals `p`** for every
  representation; **disconnected graphs** (one rejected pair with an
  alternate path, an isolated tile, two groups, everything rejected) with
  the anchoring of §6.1; **2D→3D transfer** keeps each tile's Z origin and
  step; unequal stacks are rejected; **timepoint reuse** applies the first
  timepoint's transforms to all; one test per row of §3.6.
- **Worker:** importing the worker module leaves `PyQt6` out of
  `sys.modules`; it handles `sys.stdout`/`sys.stderr`/`sys.stdin` being
  `None`; progress, result and error files are valid models; the watchdog
  exits when the parent PID disappears.
- **Controller (fake worker, no package needed):** state transitions;
  cancel waits for exit before deleting; deletion runs off the GUI thread;
  startup cleanup removes only registry-listed paths; an abnormal exit
  becomes a warning toast; the sink not closing in time becomes a clear
  error.
- **UI:** the Saving row appears for a global grid and for per-position-only
  grids, disappears without a grid, and the section height follows; a run
  with Saving off never starts a job; the close guard behaves as in §3.4.
- **Integration (one real worker):** a demo-config MDA with a 2×2 grid saved
  to `tmp_path` produces `_stitched.ome.zarr` and opens a viewer. The demo
  camera's images don't depend on stage position, so it uses "stage
  positions only".
- **Windows windowed bundle:** a CI smoke test in the bundle workflow that
  launches the frozen worker on a tiny job and checks `result.json`.

## 11. Risks

<!-- markdownlint-disable MD013 -->
| Risk | Mitigation |
| --- | --- |
| Wrong orientation | Micro-Manager's convention, Explorer bugs fixed, geometry test matrix, phase 0 on real hardware. |
| Calibrated affine made stale by a later mirror change | Documented; a warning on mirror changes is a possible follow-up. |
| Slow 3D registration | Max projection by default; limits stated in §6.2. |
| Sparse or low-texture samples | §6.1 behaviour with group anchoring; reported to the user. |
| Worker memory or crash | Separate process; measured peak; memory cap in the budget; log tail on failure. |
| Acquisition disturbed by stitching | One resource budget, low priority, one job at a time; measured jitter before promising anything. |
| Worker accidentally importing Qt | Lazy `__init__.py`; `sys.modules` test. |
| Windowed bundle stdio | File-based protocol; stdio redirected; bundle smoke test. |
| Orphaned workers or files | PID + start-time watchdog; `.tmp` then rename; registry-based cleanup. |
| Upstream API churn | One adapter module, `<0.2` cap, CI job with the extra. |
| Reading the store before the writer finishes | Start only after the sink has closed. |
<!-- markdownlint-enable MD013 -->

## 12. Open questions

1. Output format: always OME-Zarr (my recommendation), or match the
   acquisition's format?
2. v1 scope: global grid only, or also per-position grids, multiple
   positions and well plates?
3. Should the bundled app include stitching (+~325 MB)?
4. Add "Finish in background & quit" later, or always block closing?
