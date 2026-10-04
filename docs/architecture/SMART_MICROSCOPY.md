# Smart Microscopy (event-driven acquisition)

<!-- markdownlint-disable MD013 -->
<!-- Long lines are kept in tables and code. -->

The **Smart Microscopy** tab runs acquisitions that react to their own
images. A Python script analyzes frames as they arrive and decides what to
acquire next: change the exposure, add a z-stack where something interesting
was found, image a region at a higher magnification, or stop.

The engine lives in **pymmcore-plus** (`pymmcore_plus.smart`), so the same
scripts also run headless from Python. This tab is a front end for it.

| For | Read |
|---|---|
| Writing scripts: the contract, return values, rules, threads vs processes, records | pymmcore-plus guide, `docs/guides/smart_microscopy.md` |
| How the engine works inside | `pymmcore_plus/smart/README.md` |
| Using the tab | this page, §1–3 |
| The GUI's own pieces | this page, §4 |

---

## 1. Quick start

1. Open **Smart Microscopy** and choose **New from template**. Save the copy
   somewhere, and it opens in your text editor.
2. Set up the base acquisition in the left-hand MDA editor. It needs at least
   one channel or position.
3. Adjust the script's parameters in the middle column.
4. Optional: **Snap** on the Acquire tab, then **Test on last image** to see
   what the script would request. Nothing is acquired.
5. Press **Run**. The live viewer opens on this tab, and the monitor below it
   lists every frame, what analysis made of it, and the script's messages.

## 2. Templates

| Template | Experiment |
|---|---|
| *Minimal* | Measures every frame and changes nothing; a starting point. |
| *Adaptive exposure* | A reactive loop: each frame sets the next exposure. |
| *Detect and act* | At each hit during a scan: a z-stack, an image at higher magnification, or both. The objective is switched without imaging and restored after each follow-up. |
| *Survey and target* | A two-phase experiment. Scan at low magnification, stitch and segment the mosaic (`after_base`), then image a grid around each object, optionally at the zoom objective. |
| *Stop when idle* | Ends a time-lapse once nothing changes any more. |

The *Adaptive exposure*, *Detect and act* and *Survey and target* templates
are identical to pymmcore-plus's `examples/smart_microscopy/` scripts.

A script may hold its hooks as plain functions (as the templates do) or as
the methods of one class, optionally subclassing
`pymmcore_plus.smart.SmartAnalyzer`; the tab handles both. See the
pymmcore-plus guide for which to choose.

## 3. The tab

- **Script column:** the script's name and status (*Ready*, an error with
  its line number, or *Modified*: changes apply at the next run). It also
  holds the parameter form (from `PARAMETERS`) and the run settings:
  thread or process, blocking or async, hardware sequencing, what to do on
  an error, the analysis timeout, the max events, and which frames to
  analyze. Settings are remembered **per script**. The script's own
  `EXECUTION`, `SYNC`, `SEQUENCING` and `ANALYZE` apply only the first time
  it is loaded.
- **Hardware sequencing** pre-triggers the camera so a run of events is
  acquired at full speed. Events a script returns together are always
  sequenced; base events are sequenced with *Async* timing, and with
  *Blocking* only if you choose "Also base events when blocking" (blocking
  otherwise means each frame's analysis gates the next acquisition). See
  the pymmcore-plus guide for the full rules.
- **Viewer:** the run's live viewer opens on this tab. A smart run is
  stored along one `t` axis, so the viewer shows a single `t` slider in
  grayscale. **Re-use MDA…** loads the run's *base* sequence back into the
  editor.
- **Monitor:** a row per frame (origin, channel, position, analysis result,
  requested events, recorded values), the script's log, totals, and the
  **run folder**. The folder is always written: next to the data when
  saving (`<name>_smart/`), otherwise a temporary one, so use **Open run
  folder** to keep it.
- **While a run is going**, the window stays on this tab and the other tabs
  are disabled. An MDA started from the console or the Acquire tab still
  switches to Acquire and opens its viewer there. Closing the window
  cancels the run and stops the analysis worker.
- **Pixel sizes:** a script that returns a grid needs the pixel size where
  that grid runs. The grid is sized the moment it is about to be acquired,
  so an objective switched earlier in the run is taken into account. If the
  pixel size there isn't calibrated, the run stops with an explanation
  (or skips that grid, with *On error: skip*). Calibrate it under
  *Configurations → Pixel Configuration*.

- **Steering a run by hand:** from the console panel,
  `window.smart.controller.runner.request(...)` adds events to the run in
  progress (see the pymmcore-plus guide). Such frames show up in the
  monitor with origin *external*.

## 4. For developers: the GUI side

```text
SmartMDAWidget.run_mda()                          widgets/_smart/_mda.py
  └─ execute_mda(output) → SmartMicroscopyPage._start_run(seq, output)
       ├─ reload + validate script                pymmcore_plus.smart.inspect_script
       ├─ SmartController.prepare(seq, config, output) on a thread
       │     └─ SmartRunner.prepare(..., run_dir="auto", packages=("pymmcore-gui",))
       └─ RunOwnership.claim(SMART); SmartController.start()

SmartController (widgets/_smart/_bridge.py): owns a SmartRunner and
re-emits its psygnal signals as Qt signals → slots run on the GUI thread.
```

| Piece | Where | Role |
|---|---|---|
| `SmartMicroscopyPage` | `widgets/_smart/_page.py` | The tab: toolbar, layout, run start, "Test on last image" (`pymmcore_plus.smart.dry_run`), file watching, settings. |
| `SmartMDAWidget` | `widgets/_smart/_mda.py` | The MDA editor; its Run calls the page instead of `core.run_mda`, keeping upstream's autofocus handling. |
| `SmartController` | `widgets/_smart/_bridge.py` | Qt front for `SmartRunner`. |
| `ScriptPanel`, `ParamsForm`, `SmartMonitor` | `widgets/_smart/` | Script and settings column, parameter form, monitor. |
| `RunOwnership` | `_run_owner.py` | Which page started the run: the window keeps that tab on screen, and only that page's viewers display the run. Unclaimed runs belong to Acquire. |
| Viewer changes | `_ndv_viewers.py` | `accepts_run` gate, a title prefix, and iterator runs followed by frame count, in grayscale. The viewer's Save keeps per-frame events. |
| Templates | `resources/smart_templates/` | Bundled (collected explicitly in `app/mmgui.spec`). |

Tests: `tests/test_smart_page.py`, `tests/test_run_owner.py`, and the
iterator-run cases in `tests/test_ndv_viewers.py`. The engine's tests live
in pymmcore-plus, under `tests/smart/`.

## 5. Known gaps and future work

- **Not verified:** a process-mode run from the frozen (PyInstaller) app. The
  entry point calls `multiprocessing.freeze_support()` and the spec collects
  the templates, but the bundle has not been built and tried.
- **Upstream ome-writers:** the OME-TIFF backend raises (failing the
  acquisition) when extra per-frame metadata holds a non-string value, although
  its docs promise a warning. pymmcore-plus works around it by storing each
  frame's event as a JSON string. Worth reporting.
- **Reopening a smart run** with channel, z and position regrouped from the
  per-frame events (`mda_event` in the data file), by integrating with
  `_acquisition_loader.py`. Today it reopens as a flat `t` series.
- **Live plots** of `ctx.record` values in the monitor.
- **Safety envelope:** optionally reject requested stage moves outside a
  user-set region, or outside the base positions plus a margin. Today stage
  limits are the script's responsibility.
- **Process mode:** shared-memory frame transfer for high frame rates; an
  external interpreter for scripts needing libraries the app does not bundle.
- **Throughput:** parallel workers for stateless scripts, and GPU-friendly
  batching.
