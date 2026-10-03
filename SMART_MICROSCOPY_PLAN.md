# Smart Microscopy tab: implementation plan

<!-- markdownlint-disable MD013 -->
<!-- Long lines are kept in code signatures and tables. -->

Branch: `smart-microscopy`, created from `modern-gui-only` at `2f2c705`.

## Implementation status (2026-10-03)

Phases 0–5 are implemented on this branch. User and developer documentation
is in `docs/architecture/SMART_MICROSCOPY.md`.

| Commit | Content |
|---|---|
| `3af45e9` | Phase 0: lazy package exports, `freeze_support`, `RunOwnership`, viewer gating and iterator-run following |
| `e60edca` | Phase 1: headless engine (`smart/`, `_smart/`), four templates |
| `c421bdd` | Phases 2–4: the tab, window integration, monitor, templates menu, editor, file watcher, "Test on last image", per-script settings |
| `024c2bf` | Layout and viewer fixes found by running the tab with the demo config |

Verified: the full suite (`pytest -n auto`), mypy, pyright, and every
pre-commit hook pass. The tab was run with the demo configuration and its
screenshots were checked.

**Not verified:** a frozen (PyInstaller) build. It was not built. The spec
now collects the templates, which was checked with `collect_data_files`,
and the entry point calls `freeze_support()`. A process-mode run from a
built bundle still needs to be tried.

### Deviations from the plan below (deliberate)

1. **Relative timing is rebased when an event is handed out, per segment,
   not when it is injected** (§3.5, §5.4). useq marks every time block with
   `reset_event_timer`, so a returned multi-position time-lapse would be
   mistimed by a single injection-time offset. Passing the flag through
   would reset the clock the base events depend on. The iterator consumes
   the flag and re-anchors only that response's events.
2. **The analysis timeout always stops the run** (§5.5). One worker runs
   analyses in order, so "skip" could not help: everything queued behind a
   hung call would also time out.
3. **Process workers are terminated by the PIDs the pool reports**, read
   from `ProcessPoolExecutor._processes` because there is no public API.
   They are not killed by a PID the child reports. A child hung while
   importing the script could never report one, and the leftover process
   blocked interpreter exit.
4. **Scripts are compiled from the inspected source text** rather than
   imported with the normal loader (§5.2). The bytecode cache keys on
   whole-second mtimes plus size, so a same-length edit saved within a
   second ran the old code. This also guarantees that `script.py` is the
   code that ran.
5. **An empty base sequence is refused**, in both the page and the
   controller. `useq.MDASequence()` with no axes yields zero events.
6. **The run status also reads the runner's own `finish_reason`.** A cancel
   that arrives while an event is being acquired ends the run at the event
   boundary without asking the iterator again. With the G3 fix that value
   is reliable.
7. **Viewers of iterator runs open in grayscale.** ndv took the single `t`
   axis for a channel axis and drew one LUT per frame.
8. **Settings are per script only** (§6.4): `last_script`,
   `recent_scripts`, and `script_settings[path]`. There are no global
   execution/sync defaults. A script's own `EXECUTION`/`SYNC` apply the
   first time it is loaded.
9. **Tests live at `tests/test_smart_*.py`** rather than `tests/smart/`,
   matching the existing `tests/*.py` lint configuration.
10. **The plan's Phase 3 and Phase 4 items were folded into the Phase 2
    commit**, apart from the later layout fixes.
11. **`detect_and_zstack.py` became `detect_and_act.py`** (§7). It is one
    detect → follow-up template whose action is a parameter: z-stack, an
    image at higher magnification, or both. It is built from reusable blocks
    (`center_on`, `switch_objective`, `snap`, `z_stack`). An objective
    switch is an image-less `CustomAction` event, and the scan objective is
    restored after each follow-up, because device settings persist for
    later events. `frames.jsonl` also records `pixel_size_um`, which
    changes with the objective.

## Part 2: move the engine to `pymmcore_plus.smart` (decided 2026-10-03)

Status: **in progress.** Decided with the user:

| Topic | Decision |
|---|---|
| What moves | The whole GUI-free engine. pymmcore-gui keeps the tab, a thin Qt bridge, `RunOwnership`, the viewer changes, per-script settings, and its templates. |
| Entry point | **A + C:** `SmartRunner(core).run(...)`, plus a thin `core.run_smart(...)`. `MDARunner` stays untouched. |
| Run folder | The writer lives in pymmcore-plus, **opt-in** (`run_dir=None` writes nothing). The GUI always passes a folder. |
| Quality bar | Lives on the `cite` fork for now, but is written as if for upstream `main`: minimal public surface, private `_modules`, docs and tests in pymmcore-plus's style. |
| Name | `pymmcore_plus.smart`: the established term, and pymmcore-plus's own guide already calls this "smart-microscopy". |
| Examples / templates | Headless examples in pymmcore-plus `examples/smart_microscopy/`. GUI templates stay in pymmcore-gui. |
| Stitching (survey template) | Simple placement by stage position, no registration. |

### Layout in pymmcore-plus

```text
src/pymmcore_plus/smart/
  __init__.py    public: FrameInfo, AnalysisContext, Response, STOP, ParamSpec,
                 SystemInfo, PixelConfig, SmartRunner, SmartRunConfig,
                 ScriptSpec, ScriptError, inspect_script
  _api.py        script contract (+ SystemInfo, PixelConfig)
  _loader.py     ast inspection
  _worker.py     _ScriptHost + process entry points (Qt-free, cheap import)
  _executors.py  thread / process; no psutil (pymmcore-plus does not depend
                 on it): workers are terminated through multiprocessing
  _scheduler.py  SmartEventIterator (+ after_base trigger)
  _log.py        run folder writer (opt-in)
  _runner.py     SmartRunner: psygnal SignalGroup `SmartSignaler`
examples/smart_microscopy/   adaptive_exposure.py, survey_and_target.py,
                             run_smart.py (headless runner usage)
docs/guides/smart_microscopy.md, docs/api/smart.md
tests/smart/ (or tests/test_smart_*.py, following pymmcore-plus's layout)
```

`SmartRunner` holds today's controller logic with psygnal signals:
`frameAcquired`, `analysisQueued`, `analysisFinished`, `logMessage`,
`analysisError`, `runStarted`, `runFinished`. The frame handler stays on the
runner thread. When `core.mda.events` is a `QObject`, it connects with
`DirectConnection`, importing Qt lazily only in that case. The GUI's
`SmartController` becomes a thin `QObject` that re-emits these signals as Qt
signals on the GUI thread.

```python
runner = SmartRunner(core)
summary = runner.run(base_sequence, "script.py", execution="process",
                     output="data.ome.zarr", run_dir="data_smart/")  # blocks
thread = core.run_smart(base_sequence, "script.py")             # non-blocking
```

### API additions (script API v1, additive)

1. **`after_base(ctx) -> Response | None`.** This optional hook is called once
   all base events have been acquired and every analysis has finished. Its
   response is queued. If it returns `None` and nothing else is pending, the
   run ends. It supports the *survey → targeted acquisition* pattern.
2. **`ctx.base_sequence`.** The `MDASequence` being run.
3. **`ctx.system: SystemInfo`.** A read-only, picklable snapshot taken at run
   start. It holds the image width and height, the pixel size and pixel
   configuration at start, every pixel configuration (name, `pixel_size_um`,
   and the properties that select it), and the start values of those
   properties. Helpers: `pixel_config_for(properties)` and
   `fov_um(pixel_config=None)`. Scripts still never touch the core.
4. **`Response.events` may mix `MDAEvent`s and `MDASequence`s.** They are
   expanded in order, so `[switch_objective, subgrid_sequence, switch_back]`
   works.

### Grid field of view and pixel-size guard

Verified: a returned grid with no `fov_width`/`fov_height` is expanded by
useq with tiles **1 µm apart**. The engine fills in the FOV only for the
base sequence, in `setup_sequence`. Rules, applied while a response is
expanded in the worker:

- The pixel state starts from `ctx.system` (the state at run start) and
  follows the `properties` of earlier events *in the same response*. A
  response is assumed to start from the run-start state. Responses that
  leave the objective switched break that assumption, and the docs say so.
- A grid with no FOV gets one from the resolved pixel configuration
  (`image size × pixel size`). If no pixel size is known (no matching
  configuration, or its size is 0), the response is **refused** with a clear
  error. That counts as an analysis error, handled by the on-error policy,
  which stops by default.
- An event that switches into a state with no pixel size, without any grid,
  produces a **warning** (log and notification) but is not refused.
- Runtime backstop: the first frame of a run whose `pixel_size_um` is 0 or
  missing produces a warning.

### Per-frame event in the data file

ome-writers already stores arbitrary JSON per frame. In
`OmeWritersSink.append`, for iterator-driven runs only (detected at
`setup()`), add `"mda_event": event.model_dump(mode="json",
exclude={"sequence"}, exclude_none=True)` to the frame metadata. Smart
provenance (`event.metadata["pymmcore_gui_smart"]`, renamed
`"pymmcore_plus_smart"`) travels with it. Normal sequence-shaped runs are
unchanged, because their axes already encode this, and an OME-TIFF would
grow by about 0.6 KB of OME-XML per frame. `frame_meta_to_ome()` gets an
`include_event` flag, so the GUI's viewer export reproduces it.
`frames.jsonl` remains as a convenience copy.

### Order of work

1. pymmcore-plus: move the engine, add `SmartRunner`/`core.run_smart`,
   remove `psutil`, port the engine tests (no qtbot; both signal backends).
2. pymmcore-plus: `after_base`, `ctx.system`, `ctx.base_sequence`, mixed
   responses, the FOV fill and pixel-size guard, per-frame event metadata,
   examples, guide and API docs.
3. pymmcore-gui: replace `_smart/` and `smart/` with imports from
   `pymmcore_plus.smart`, make the Qt bridge, have templates import
   `pymmcore_plus.smart`, add a *Survey and target* template, pass
   `ctx.system` to "Test on last image", and update the docs.
4. The user commits and pushes the fork. Then `uv lock --upgrade-package
   pymmcore-plus` in the GUI. Until then the GUI is developed against an
   editable install (`uv run --no-sync`).

The original plan follows, unchanged, for reference.

This document is a self-contained handoff. An implementer should not need
the conversation that produced it. Each claim about existing behaviour cites
the file it was verified in. Section 2 lists facts that were **verified by
running code**. They are not assumptions. Do not re-litigate them without new
evidence.

---

## 0. Goal

Add a fifth top-level mode tab, **"Smart Microscopy"**, after "Acquire". It
runs *event-driven* (feedback, "smart") acquisitions:

1. The user defines a **base acquisition** with the app's existing MDA editor.
   The base acquisition covers channels, positions, z, time, and grid.
2. The user loads a **Python analysis script** that follows a small, versioned
   API, defined here and exposed as `pymmcore_gui.smart`.
3. During the run, each acquired frame that passes a filter is sent to the
   script's `analyze()` function. The function returns **what to do next**:
   nothing, one or more `useq.MDAEvent`s, a `useq.MDASequence`, or "stop".
   The engine executes the returned events.
4. The user chooses whether analysis runs in a **background thread** or in a
   **separate process** (an explicit requirement). The same script must run
   unchanged in either mode.
5. Every run is reproducible and inspectable. The image data, a per-frame
   event log, an analysis log, the parameters, and a verbatim copy of the
   script are written together.

Reference for the underlying mechanism:
`/Users/fdrgsp/Documents/git/pymmcore-plus/examples/event_driven_acquisition.py`.
That example passes `iter(queue.get, STOP)` to `core.run_mda()` and pushes
`MDAEvent`s into the queue from a `frameReady` callback.

### Non-goals (v1)

- Hardware-sequenced (triggered) bursts for smart runs. See G1.
- Running user scripts in an external or different Python interpreter.
- Parallel analysis workers (more than one in flight). v1 uses one worker so
  ordering and script state stay simple.
- Editing scripts inside the GUI. The tab opens them in the system editor.
- Live plotting of recorded values. The values are logged so this can be
  added later (Section 11).

---

## 1. How it works (one-paragraph architecture)

`SmartMDAWidget` is a subclass of the app's `MemoryMDAWidget`. Its Run button
hands `(base_sequence, output)` to a `SmartController` instead of calling
`core.run_mda(sequence)`. The controller does the following:

- It starts an `AnalysisExecutor`, either thread or process. The executor
  loads the script and calls `setup()` **before** acquisition starts.
- It builds a `SmartEventIterator`. This is an `Iterator[MDAEvent]` that merges
  the base events with events injected by analysis, and honours
  blocking/async semantics and cancellation.
- It calls `core.run_mda(iterator, output=output)`.
- It handles `frameReady` with a Qt `DirectConnection`, so the handler runs on
  the runner thread. The handler numbers the frame, logs it, and submits it to
  the executor if the filter passes.
- When a result arrives, it normalises the return value to a `Response`,
  rebases timing, tags metadata, and injects the events into the iterator.

A small `RunOwnership` object tells the main window and the viewer managers
which page started the run. The Smart page then gets its own viewers, and the
window stops forcing the user back to Acquire.

```text
SmartMDAWidget ──Run──▶ SmartMicroscopyPage.start_run(seq, output)
                              │ claim RunOwnership("smart")
                              ▼
                        SmartController ──start()──▶ AnalysisExecutor (thread|process)
                              │                          ▲   │ Future[_WorkerResult]
                              │ core.run_mda(iterator)   │   ▼
   runner thread ──next()──▶ SmartEventIterator ◀─inject─ _on_result (normalise, rebase, tag)
   runner thread ──frameReady (DirectConnection)──▶ _on_frame_ready ──submit──┘
                              │
                              ├─▶ SmartRunLog (run.json, frames.jsonl, analysis.jsonl, script.py)
                              └─▶ Qt signals ─▶ SmartMonitor (table + log) / notifications
```

---

## 2. Verified facts and gotchas (read before coding)

Paths are relative to the repo root unless they are absolute.
`site-packages` means `.venv/lib/python3.13/site-packages`.

**G1. Passing an iterator bypasses the engine's event iterator and the
sequence-shaped sink.**
`site-packages/pymmcore_plus/mda/_runner.py` contains `MDARunner.run`:
`sequence = events if isinstance(events, MDASequence) else GeneratorMDASequence()`.
Its `_run` method uses plain `iter` when `isinstance(events, Iterator)`.
Consequences:

- `engine.event_iterator` (hardware sequencing) is not applied. One event
  produces one frame. This is desirable for feedback semantics. Keep it.
- `sequenceStarted` is emitted with an empty `GeneratorMDASequence`, not the
  base sequence.
- `OmeWritersSink.setup` (`mda/_sink.py`) uses `_unbounded_3d_settings` for a
  `GeneratorMDASequence`. The data is **one unbounded `t` axis** of
  `(t, y, x)`, and channels and positions are flattened into `t`.
- **Spike result (verified):** running `core.mda.run(iter([...3 events...]),
  output=X)` succeeded for every output the GUI can produce:
  `_memory_output_settings()` (ScratchFormat), `"memory"`, `*.ome.zarr`, and
  `*.ome.tiff`. Each gave a view of shape `(3, 512, 512)` with dims
  `('t','y','x')`. So the MDA widget's Saving section can be reused as is.
  The **frame index → event mapping must be recorded by us** (see
  `frames.jsonl`, Section 5.6) because the data store alone loses channel and
  position identity.
- **This is the intended design (confirmed with the user).** Data is saved the
  way ome-writers handles an unbounded acquisition. In exchange, *any* jagged
  or irregular smart run can be stored and shown live, whatever the analysis
  injects: a different channel set per position, z-stacks only where there is
  a hit, mixed exposures, and so on. The live viewer shows the run as a single
  `t` slider in acquisition order (Section 6.3). Regrouping by channel or
  position from `frames.jsonl` is future work (Section 11).

**G2. Inside the GUI, a plain callable connected to `frameReady` runs on the
GUI main thread, not the runner thread.**
The GUI sets `PYMM_SIGNALS_BACKEND=qt` (`src/pymmcore_gui/__init__.py`), so
`core.mda.events` is a `QMDASignaler` QObject. A probe script in a
`QApplication` printed:
`lambda on: MainThread`, `plain method on: MainThread`, and
`direct lambda on: Thread-1 (run)`. The last one used
`Qt.ConnectionType.DirectConnection`.
Rule: connect the controller's frame handler with `DirectConnection` when
`isinstance(core.mda.events, QObject)`, and with a plain `connect` otherwise
(psygnal is synchronous in the emitting thread). Put this in one helper,
`_connect_on_runner_thread(signal, cb)`. The handler must be fast: no Qt
widget calls, and no blocking. Otherwise analysis latency is tied to GUI
repaint, and blocking mode can deadlock against the GUI thread.

**G3. Cancelling while the iterator is blocked in `__next__` reported the run
as COMPLETED (upstream bug, now FIXED in the fork).**
In `MDARunner.cancel()`, a cancel in WAITING state sets
`_finish_reason = CANCELED` and `_state = FINISHING` at once. Our iterator
then raises `StopIteration`, and the `for ... else:` in `_run` used to run
unconditionally `self._finish_reason = FinishReason.COMPLETED`. The effect was
that `sequenceCanceled` was never emitted.

- **Status: fixed, committed, and pushed** to the fork's `cite` branch as
  `5ff067a` ("fix: ensure finish reason is not overwritten when canceling
  during event iteration"). In `src/pymmcore_plus/mda/_runner.py`, the
  for-else now sets COMPLETED only `if self._finish_reason is None`, the same
  guard `_finish_run` already uses. The regression test is
  `tests/test_mda_status.py::test_cancel_while_event_iterator_blocks`: the
  iterator cancels while the runner waits for its next event, then ends. It
  failed before the fix and passes after it on both the qt and psygnal
  backends.
  - `tests/test_mda_status.py`, `test_mda.py`, `test_mda_output.py`, and
    `test_events.py` give 247 passed and 2 failed.
  - The 2 failures are `test_run_with_tiff_output_multiposition[qt|psygnal]`.
    They fail the same way **without** the fix: the installed `ome-writers`
    fork names files `multi_A1_p001.ome.tiff` where the test expects
    `multi_p000.ome.tiff`. They are unrelated.
- `pyproject.toml` pins the fork as `rev = "cite"`, but `uv.lock` records a
  commit. To pick up the fix, run
  `uv lock --upgrade-package pymmcore-plus && uv sync`. Before relying on the
  fix, check that the installed version string contains `5ff067a` or later
  (`python -c "import pymmcore_plus; print(pymmcore_plus.__version__)"`).
  For further fork changes during development, use the editable workflow:
  `uv pip install -e /Users/fdrgsp/Documents/git/pymmcore-plus --no-deps`,
  then run the GUI tests with `uv run --no-sync`. Run the fork's own tests
  with `PYTHONPATH=src python -m pytest`.
- Do not commit or push to the fork without asking the user. As a
  belt-and-braces measure, the controller still records
  `user_cancelled` itself and treats "FINISHING without our own stop reason"
  as a cancel (Section 6.6). It stays correct against an older pin.

**G4. Any MDA run currently locks the Acquire page and forces the window onto
Acquire.**
Every `MemoryMDAWidgetBase` relays *every* `sequenceStarted` to its lock
(`widgets/_mda_widget.py`, `_on_sequence_started_in_gui` → `_set_mda_lock`).
`AcquirePage.set_mda_lock` → `mdaRunningChanged` →
`MicroManagerGUI._on_mda_running` (`_main_window.py:792`). That handler
disables the other tabs and **selects Acquire**. If nothing changes, starting
a smart run yanks the user to Acquire. The lock itself is correct, because
the hardware is busy. The *navigation* must follow the run owner (Section 6.2).

Required behaviour (confirmed with the user):

- An MDA started from the **console**, a script, or the Acquire page's Run
  button still switches to **Acquire**, and its viewer opens there, exactly
  as today. Nobody claims such a run, and an unclaimed run belongs to
  Acquire.
- A **smart run** keeps the window on **Smart Microscopy**. Its own viewer
  **pops up inside the Smart Microscopy tab at run start**, in the tab's
  viewer workspace, at `sequenceStarted`. The Acquire page creates no viewer
  for it.

**G5. One viewer manager follows `event.index` keys.**
`AcquireViewersManager` (`_ndv_viewers.py:121`) connects to every
`sequenceStarted` and `frameReady`. It builds a viewer from
`core.mda.get_view()` and, in `_on_frame_ready`, moves the slider using
`event.index` keys (`c`, `p`, `g`, ...). For a smart run the view dims are
`('t','y','x')`, so those keys do not exist. The update fails silently inside
a bare `try` and the viewer never follows. `_on_sequence_started` also passes
the `GeneratorMDASequence` to `_extract_scales(...)` and
`ChannelLUTMemory.bind_live_mda(...)`. `GeneratorMDASequence.sizes` warns and
returns `{}`. Check that both tolerate this, and guard them otherwise.

**G6. `min_start_time` is relative to the runner's event clock.**
`_wait_until_event` uses `event.min_start_time + paused_time` against
`event_seconds_elapsed()`, which counts from the start or from the last
`reset_event_timer`. An `MDASequence` with a time plan returned from analysis
mid-run has `min_start_time` values measured from 0. Those times are already
in the past, so the events would fire at once. Injected events must be
**rebased** by adding `core.mda.event_seconds_elapsed()` at injection time
(Section 5.4). Do not use `reset_event_timer` for this, because it would
shift the timing of the remaining base events too.

**G7. The runner thread blocks inside our `__next__`, so cancel and pause must
be polled there.**
While `__next__` waits (for analysis in blocking mode, or for a
held-back base event), the runner cannot reach its own cancel or pause
checks. `__next__` must poll `runner.status.phase` (FINISHING means stop) at
least every 50 ms, and must wait on a `threading.Condition` with a timeout.
It must never wait without a timeout.

**G8. Importing `pymmcore_gui` imports Qt and the whole window.**
`src/pymmcore_gui/__init__.py` eagerly imports `create_mmgui`,
`MicroManagerGUI`, and the actions. Python always runs a package's
`__init__.py` before any submodule. So `import pymmcore_gui.smart` (the first
line of every user script) would load the whole GUI even if `smart/` itself
is Qt-free.

Measured: `import pymmcore_gui._settings`, a module with no Qt in it, took
**0.56 s** with a warm cache and loaded **1211 modules**, including `PyQt6`,
`qtpy`, `ndv`, `vispy`, and `pymmcore_widgets`. `import useq` took 0.10 s and
loaded 348 modules. A frozen bundle on a cold start is slower still.

- **Thread mode** does not care, because the GUI process has already loaded
  everything.
- **Process mode** pays this on every run's worker start, and each analysis
  process carries the GUI's libraries in memory for nothing.
**Fix:** make those exports lazy with a module `__getattr__`, typed with an
`if TYPE_CHECKING:` import block. Keep the env-var and Windows DLL preload
code. Add a test that `import pymmcore_gui.smart` and
`import pymmcore_gui._smart._worker` load no `PyQt6`, `qtpy`,
`pymmcore_widgets`, or `ndv` modules.

**G9. The frozen app (PyInstaller) needs `freeze_support()` for spawn.**
The bundle entry is `src/pymmcore_gui/__main__.py` (`app/mmgui.spec:91`).
Change it to the following, so a spawned child bootstraps before importing
the CLI or GUI:

```python
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    from pymmcore_gui._cli import main
    main()
```

Keep `main` importable, for example with a lazy `__getattr__` or by
importing it inside the guard and also exposing it. Check that
`tests/test_bundle.py` and `tests/test_cli.py` still pass. In the bundle,
user scripts can only import packages bundled in the app, in both thread and
process mode. State this in the user docs.

**G10. `MDAEvent.sequence` drags the whole parent sequence along.**
A pickled event from a sequence was 1180 bytes, against 611 bytes with
`sequence=None` (verified). Strip it before sending an event to a process
(`event.model_copy(update={"sequence": None})`) and before logging
(`model_dump(mode="json", exclude={"sequence"}, exclude_none=True)`).
`MDASequence` and `MDAEvent` pickle round-trip correctly (verified).

**G11. The upstream `execute_mda` contains autofocus handling.**
See `site-packages/pymmcore_widgets/mda/_core_mda.py:431`:
`_disable_af_on_run`, `_disable_continuous_focus`, and
`_restore_continuous_focus` on a failed launch. The override in
`SmartMDAWidget` must keep this exactly and replace only the launch call.

---

## 3. The user-facing API (`pymmcore_gui.smart`, API version 1)

This is the contract script authors code against. Keep it **small, pure
Python, and Qt-free**. It may depend only on stdlib, numpy, and useq. It must
work in a spawned child process.

### 3.1 Script contract

A smart-microscopy script is a single `.py` file. The GUI reads its metadata
**statically, with `ast`, without executing it** (Section 5.1). User code
therefore runs only inside the chosen executor.

```python
# my_script.py
from pymmcore_gui.smart import AnalysisContext, FrameInfo, Response, STOP
import numpy as np
import useq

API_VERSION = 1                    # required; loader refuses unknown versions
NAME = "Adaptive exposure"         # optional, shown in the GUI
DESCRIPTION = "Keeps mean intensity near a target by adjusting exposure."

# Optional defaults; the user can override both in the GUI.
EXECUTION = "thread"               # "thread" | "process"
SYNC = "blocking"                  # "blocking" | "async"  (see 3.4)

# Optional: which frames reach analyze(). GUI can override. All keys optional.
ANALYZE = {"channels": ["FITC"], "every_nth": 1, "origins": ["base", "analysis"]}

# Optional: user-tunable parameters -> auto-generated form in the GUI.
# Values must be literals (ast.literal_eval). Plain value or a spec dict.
PARAMETERS = {
    "target_mean": {"default": 2000.0, "min": 0, "max": 65535, "step": 100,
                    "label": "Target mean intensity"},
    "max_exposure_ms": 500.0,
    "mode": {"default": "fast", "choices": ["fast", "accurate"]},
    "save_masks": False,
}

def setup(ctx: AnalysisContext) -> None:            # optional, once per run
    ctx.state["history"] = []

def analyze(image: np.ndarray, frame: FrameInfo, ctx: AnalysisContext):  # REQUIRED
    mean = float(image.mean())
    ctx.record(mean=mean)                            # goes to analysis.jsonl + monitor
    if mean == 0:
        return STOP
    exp = frame.event.exposure or 10
    new = min(exp * ctx.params["target_mean"] / mean, ctx.params["max_exposure_ms"])
    return frame.event.model_copy(update={"exposure": new})

def teardown(ctx: AnalysisContext) -> None:          # optional, once per run (always called)
    ...
```

Rules:

- `analyze` is required, with exactly three positional parameters.
  `setup` and `teardown` are optional and take one parameter. The loader
  checks this with `ast`.
- Module-level constants must be literal-evaluable, or the loader reports an
  error with the line number. Anything else at module level (imports, helper
  functions, globals) is allowed, and executes only in the executor.
- Sibling modules next to the script can be imported, because the script's
  directory is put first on `sys.path` inside the executor.
- **No access to the microscope core from scripts.** This applies to both
  modes, so the semantics are identical and the process mode is possible at
  all. Everything a script needs about the hardware state at acquisition
  time is in `frame.metadata` (the `FrameMetaV1` dict: pixel size, stage
  position, exposure, timing, camera) and `frame.event`. Scripts *act* only by
  returning events. `MDAEvent.properties`, `x_pos`/`y_pos`/`z_pos`,
  `exposure`, `channel`, `roi`, and `slm_image` cover device changes.

### 3.2 Public types (`src/pymmcore_gui/smart/_api.py`)

```python
API_VERSION: Final = 1
ExecutionMode = Literal["thread", "process"]
SyncMode = Literal["blocking", "async"]
Origin = Literal["base", "analysis"]

@dataclass(frozen=True, slots=True)
class FrameInfo:
    frame_id: int                 # 0-based acquisition order in this run == `t` index in the data store
    event: useq.MDAEvent          # event that produced the frame (sequence stripped)
    metadata: Mapping[str, Any]   # FrameMetaV1 as a plain dict (picklable)
    origin: Origin                # "base" or "analysis"
    parent_frame_id: int | None   # frame whose analysis injected this event (None for base)

class AnalysisContext:
    params: Mapping[str, Any]     # resolved PARAMETERS (read-only MappingProxy)
    state: dict[str, Any]         # free-form, persists across calls within one run/worker
    run_dir: Path                 # folder for the script's own outputs (always exists)
    execution: ExecutionMode
    def log(self, message: str, level: Literal["debug","info","warning","error"] = "info") -> None
    def record(self, **values: float | int | str | bool | None) -> None   # per-frame scalars

@dataclass(frozen=True, slots=True)
class Response:
    events: Sequence[useq.MDAEvent] | useq.MDASequence = ()
    priority: Literal["next", "end"] = "next"     # front of the queue, or after remaining base events
    timing: Literal["relative", "absolute"] = "relative"   # see 3.5
    stop: bool = False             # finish after already-running event; queued events are dropped
    drop_base: bool = False        # discard remaining base events (switch to purely reactive)

STOP: Final = Response(stop=True)

class ParamSpec(TypedDict, total=False):   # documentation/type-hint aid only
    default: Any; min: float; max: float; step: float; choices: list[Any]; label: str; tooltip: str
```

`ctx.log` and `ctx.record` buffer entries into the current call's result,
which is returned alongside it (Section 5.2). They are therefore identical in
both modes, and nothing needs to cross processes asynchronously. Messages
logged in `setup` and `teardown` are returned the same way.

`src/pymmcore_gui/smart/__init__.py` re-exports the following names, and
nothing else is public: `API_VERSION`, `AnalysisContext`, `FrameInfo`,
`Response`, `STOP`, `ParamSpec`, `ExecutionMode`, `SyncMode`.

### 3.3 Accepted return values of `analyze` (normalisation table)

| Returned | Normalised to |
|---|---|
| `None` | `Response()` (no action) |
| `MDAEvent` | `Response(events=[e])` |
| `MDASequence` | `Response(events=seq)`, expanded with `list(seq)` in the main process |
| iterable of `MDAEvent` (list, tuple, generator) | `Response(events=list(it))` |
| `Response` | as is (`events` expanded if it is an `MDASequence`) |
| anything else, or a non-`MDAEvent` item | `TypeError` → error policy (5.5) |

Expansion is capped by `max_events_per_response` (default 1000). Going over
the cap is an error. In process mode the return value is pickled back to the
main process. `MDAEvent` and `MDASequence` are pydantic models and pickle
fine. A generator cannot be pickled, so the worker calls `list()` on
iterables **inside the worker** before returning.

### 3.4 Blocking versus async

- **`blocking`** (strict feedback): while *any* submitted analysis is
  pending, the iterator releases **no** event. The next acquisition always
  sees the latest decision. This is the right mode for decisions like "adjust
  exposure, then acquire". Frames excluded by `ANALYZE` never block.
- **`async`** (opportunistic): base events keep flowing. Events returned from
  analysis are inserted when they arrive, at the front of the queue for
  `priority="next"` or after the remaining base events for `priority="end"`.
  This is the right mode for "scan a plate and zoom in on hits".

In both modes the run ends when the base events are exhausted, the injected
queue is empty, **and** no analysis is pending. A pending analysis may still
inject events. A `Response(stop=True)` also ends the run, and so does the
user. Purely reactive runs, like the reference example, work naturally: the
base sequence holds a single event, and each analysis returns the next one.

### 3.5 Timing of injected events

With `timing="relative"`, the default, the controller rebases each injected
event's `min_start_time` (G6). If the event has a `min_start_time`, it gets
`min_start_time + core.mda.event_seconds_elapsed()` at injection. If it has
none, it stays `None`, meaning as soon as possible. A returned
`MDASequence` with a time plan therefore starts its own clock at the moment
it is returned. With `timing="absolute"`, values are used as given, relative
to the run's event clock.

Document this caveat: in `blocking` mode, analysis time delays base events.
Base events whose `min_start_time` has already passed fire immediately.
This is the runner's normal behaviour.

### 3.6 Metadata tagging (done by the controller, not the script)

Every event the iterator yields gets
`event.metadata["pymmcore_gui_smart"] = {"origin": ..., "parent_frame_id": ..., "response_id": ...}`.
Use `model_copy(update=...)` and never mutate. `_on_frame_ready` reads this
tag back to build `FrameInfo`. It also travels into the data store's
per-frame metadata through `FrameMetaV1`, if the sink records event
metadata. Check this, but do not depend on it, because `frames.jsonl` is the
source of truth.

---

## 4. Files to add

```text
src/pymmcore_gui/smart/__init__.py          public API re-exports (Qt-free)
src/pymmcore_gui/smart/_api.py              types from §3.2 + normalise_response()
src/pymmcore_gui/_smart/__init__.py
src/pymmcore_gui/_smart/_loader.py          static script inspection -> ScriptSpec
src/pymmcore_gui/_smart/_worker.py          _ScriptHost + module-level worker fns (Qt-free, picklable)
src/pymmcore_gui/_smart/_executors.py       AnalysisExecutor protocol, ThreadAnalysisExecutor, ProcessAnalysisExecutor
src/pymmcore_gui/_smart/_scheduler.py       SmartEventIterator
src/pymmcore_gui/_smart/_log.py             SmartRunLog (run.json / frames.jsonl / analysis.jsonl / script.py)
src/pymmcore_gui/_smart/_controller.py      SmartController(QObject)
src/pymmcore_gui/_run_owner.py              RunOwner enum + RunOwnership(QObject)
src/pymmcore_gui/widgets/_smart/__init__.py
src/pymmcore_gui/widgets/_smart/_page.py          SmartMicroscopyPage(TabPage)
src/pymmcore_gui/widgets/_smart/_mda.py           SmartMDAWidget(MemoryMDAWidget)
src/pymmcore_gui/widgets/_smart/_script_panel.py  load/validate/params/execution settings
src/pymmcore_gui/widgets/_smart/_params_form.py   PARAMETERS -> form widgets
src/pymmcore_gui/widgets/_smart/_monitor.py       frames/analysis table + log console + counters
src/pymmcore_gui/resources/smart_templates/minimal.py
src/pymmcore_gui/resources/smart_templates/adaptive_exposure.py
src/pymmcore_gui/resources/smart_templates/detect_and_zstack.py
src/pymmcore_gui/resources/smart_templates/stop_when.py
docs/architecture/SMART_MICROSCOPY.md       user + developer guide
tests/smart/... (see §9)
```

`smart/` (public) and `_smart/` (internal engine) must not import Qt, except
`_smart/_controller.py`, which is the Qt bridge. `widgets/_smart/` holds the
UI. Match existing conventions: `from __future__ import annotations`, Qt
imported via `pymmcore_gui._qt.*`, the theme via `pymmcore_gui._theme`,
docstrings that explain *why*, and `Final` keys.

Confirm before adding: `pyproject.toml` must package `resources/` (it already
ships `resources/` icons) and the PyInstaller spec must collect the new
template files.

---

## 5. Component specifications

### 5.1 `_smart/_loader.py`: static inspection

```python
@dataclass(frozen=True)
class ParamDef: name: str; default: Any; kind: Literal["float","int","bool","str","choice"]
                min: float|None; max: float|None; step: float|None; choices: tuple|None; label: str; tooltip: str
@dataclass(frozen=True)
class AnalyzeFilter: channels: tuple[str,...]|None; every_nth: int; origins: frozenset[Origin]
                     def accepts(self, frame_id: int, event: MDAEvent, origin: Origin) -> bool
@dataclass(frozen=True)
class ScriptSpec: path: Path; sha256: str; source: str; api_version: int; name: str; description: str
                  execution: ExecutionMode; sync: SyncMode; filter: AnalyzeFilter; params: tuple[ParamDef,...]
                  has_setup: bool; has_teardown: bool
class ScriptError(Exception): line: int|None
def inspect_script(path: Path) -> ScriptSpec
```

- Read the source and `compile()` it, reporting `SyntaxError` line and
  column. Walk the top-level `ast.Assign` and `ast.AnnAssign` nodes for the
  known names, apply `ast.literal_eval` to each value, and validate types.
  Find the top-level `def analyze/setup/teardown` and check the positional
  parameter count. `async def` is an error.
- `every_nth` must be at least 1. `kind` is inferred from the default's type.
  A `bool` must be checked before `int`.
- No import or execution of the script happens here.

### 5.2 `_smart/_worker.py`: script host (shared by both executors)

Qt-free and importable cheaply in a spawned child (G8).

```python
@dataclass
class WorkerResult:                 # picklable
    frame_id: int | None            # None for setup/teardown/ping
    ok: bool
    response: Response | None       # already list()-expanded iterables; MDASequence left as is
    logs: list[tuple[str, str]]     # (level, message)
    records: dict[str, Any]
    duration_ms: float
    error: str | None               # formatted traceback

class _ScriptHost:
    def __init__(self, path: str, params: dict, run_dir: str, execution: ExecutionMode) -> None
        # sys.path.insert(0, script dir); import via importlib.util.spec_from_file_location
        # under a unique module name f"_pymmgui_smart_{sha[:8]}_{counter}"; build AnalysisContext
    def setup(self) -> WorkerResult
    def analyze(self, image: np.ndarray, frame: FrameInfo) -> WorkerResult   # never raises
    def teardown(self) -> WorkerResult
    def close(self) -> None   # remove module from sys.modules, restore sys.path

# process-mode entry points (module-level => picklable by reference)
_HOST: _ScriptHost | None = None
def _proc_init(path, params, run_dir) -> None          # creates _HOST (does NOT call setup)
def _proc_setup() -> WorkerResult
def _proc_analyze(image, frame) -> WorkerResult
def _proc_teardown() -> WorkerResult
```

- `analyze` wraps the user call in `try/except BaseException` (excluding
  `KeyboardInterrupt`/`SystemExit`). It times the call, normalises the
  return value with `smart._api.normalise_response` (expanding iterables),
  and validates the item types. Errors go into `WorkerResult.error`.
- In thread mode `image` is the same array the sink stores. Pass a
  read-only view (`v = image.view(); v.flags.writeable = False`) so a script
  cannot corrupt saved data. In process mode the array is a pickled copy.

### 5.3 `_smart/_executors.py`

```python
class AnalysisExecutor(Protocol):
    mode: ExecutionMode
    def start(self, spec: ScriptSpec, params: dict, run_dir: Path, timeout: float) -> WorkerResult  # runs setup
    def submit(self, image: np.ndarray, frame: FrameInfo) -> Future[WorkerResult]
    def stop(self, timeout: float) -> WorkerResult | None   # teardown (best effort) + shutdown; idempotent
    @property
    def broken(self) -> bool

def create_executor(mode: ExecutionMode) -> AnalysisExecutor
```

**ThreadAnalysisExecutor**: uses
`ThreadPoolExecutor(max_workers=1, thread_name_prefix="smart-analysis")`. In
`start`, `_ScriptHost(...)` and `host.setup()` run *on the worker thread*,
through `submit(...).result(timeout)`. Module import side effects therefore
happen off the GUI thread. `stop` submits `teardown` and then calls
`shutdown(wait=True, cancel_futures=True)`. Document the GIL caveat: pure
Python analysis competes with the GUI, while numpy, scipy, and skimage mostly
release the GIL.

**ProcessAnalysisExecutor**:

- Uses `ProcessPoolExecutor(max_workers=1,
  mp_context=multiprocessing.get_context("spawn"), initializer=_proc_init,
  initargs=(...))`. **Always spawn**, never fork: forking a process with Qt
  and the core's threads is unsafe on macOS and Linux.
- `start` submits `_proc_setup` and waits with a timeout, default 60 s
  (configurable). The child's interpreter start, its imports, and the user's
  `setup` all happen **before acquisition starts**. The first frame does not
  pay that cost, and setup errors abort before any hardware moves. The page
  shows a `BusyOverlay` ("Starting analysis process…") while it waits. Use
  the existing `widgets/_busy.py`, and run the wait without freezing the GUI,
  for example on a `QThreadPool` task or by polling the future with a
  `QTimer`.
- `submit` strips `event.sequence` (G10) before pickling `FrameInfo`.
- `broken` becomes true on `BrokenProcessPool`, which means the child
  crashed, for example with a segfault in a C extension. That is the one
  big advantage of process mode: a crashing analysis cannot take the GUI or
  the microscope down. Report it as a fatal analysis error (5.5).
- `stop` submits `_proc_teardown` with a timeout, then calls
  `shutdown(wait=False, cancel_futures=True)`. If the child is still alive
  after `timeout`, terminate it. `ProcessPoolExecutor` does not expose its
  processes publicly. Either keep your own handle by passing an
  `initializer` that reports `os.getpid()` and then use
  `psutil.Process(pid).terminate()` (`psutil` is already a dependency), or
  use `multiprocessing.Process` plus `Pipe` directly instead of
  `ProcessPoolExecutor`. Pick whichever reads cleaner. A single dedicated
  process with request and response queues is a perfectly good
  implementation of the same protocol.
- Image transfer goes through pickling over a pipe. That is fine for v1, at
  a few ms for 2048×2048 uint16. Leave a `TODO(perf)` pointing to
  `multiprocessing.shared_memory` (Section 11).

Both executors must run the **same** `_ScriptHost` code. That is how "same
script, either mode" is guaranteed. Test it (Section 9).

### 5.4 `_smart/_scheduler.py`: `SmartEventIterator`

```python
class SmartEventIterator(Iterator[MDAEvent]):
    def __init__(self, base: Iterable[MDAEvent], runner: MDARunner, *, sync: SyncMode,
                 lead_time_s: float = 0.25, max_total_events: int = 10_000,
                 poll_s: float = 0.05) -> None
    # called from executor callback threads:
    def analysis_submitted(self) -> None           # pending += 1
    def analysis_finished(self) -> None            # pending -= 1; notify
    def inject(self, events: Sequence[MDAEvent], *, priority: Literal["next","end"],
               parent_frame_id: int, response_id: int) -> None
    def drop_base(self) -> None
    def stop(self, reason: str) -> None            # graceful: StopIteration at next __next__
    @property
    def stop_reason(self) -> str | None
    def __next__(self) -> MDAEvent
```

The `__next__` algorithm runs entirely under one `threading.Condition`:

```text
loop:
  if stopped or runner.status.phase is RunState.FINISHING: raise StopIteration   # G7
  if yielded >= max_total_events: stop("max events reached"); raise StopIteration
  if sync == "blocking" and pending > 0: cond.wait(poll_s); continue
  if injected_next: return tag(injected_next.popleft())
  if base_head is None and not base_exhausted: base_head = next(base, None) (mark exhausted on None)
  if base_head is not None:
      remaining = (base_head.min_start_time or 0) - runner.event_seconds_elapsed()
      if base_head.min_start_time is None or remaining <= lead_time_s:
          return tag(pop base_head)
      cond.wait(min(remaining - lead_time_s, poll_s)); continue
  if injected_end: return tag(injected_end.popleft())
  if pending > 0: cond.wait(poll_s); continue
  raise StopIteration                                                             # natural end
```

- Holding back base events until they are nearly due is what lets an
  injected `priority="next"` event pre-empt a base event scheduled far in the
  future. Once an event is yielded, the runner sleeps until its time and can
  no longer be overtaken. Keep `lead_time_s` small.
- `tag()` adds the metadata from 3.6 and increments `yielded`.
- Pause needs no special handling. Nothing is released while the runner is
  paused, because the runner does not call `next()` until it resumes, and
  the runner applies its own paused-time offset after we yield.
- Keep this class free of Qt and of the core, so it is unit-testable with a
  fake runner that only has `status` and `event_seconds_elapsed()`.

### 5.5 `_smart/_controller.py`: `SmartController(QObject)`

```python
@dataclass(frozen=True)
class SmartRunConfig:
    spec: ScriptSpec; params: dict[str, Any]; execution: ExecutionMode; sync: SyncMode
    filter: AnalyzeFilter; on_error: Literal["stop", "skip"] = "stop"
    analysis_timeout_s: float | None = None; max_total_events: int = 10_000
    max_events_per_response: int = 1_000; setup_timeout_s: float = 60.0

class SmartController(QObject):
    runStarted = Signal(object)          # SmartRunLog (run_dir etc.)
    frameAcquired = Signal(object)       # FrameInfo
    analysisQueued = Signal(int)         # frame_id
    analysisFinished = Signal(object)    # WorkerResult (+ normalised summary)
    eventsInjected = Signal(int, int)    # parent_frame_id, n_events
    logMessage = Signal(str, str)        # level, message
    analysisError = Signal(str, bool)    # message, fatal
    runFinished = Signal(object)         # summary dict (counts, reason, run_dir)

    def __init__(self, mmcore: CMMCorePlus, parent: QObject | None = None) -> None
    def is_active(self) -> bool
    def prepare(self, config: SmartRunConfig, run_dir: Path) -> None   # start executor (blocking-ish; see 5.3)
    def start(self, base: MDASequence, output: SingleOutput | None) -> None
    def request_stop(self) -> None       # graceful stop-after-current (iterator.stop)
    def cancel(self) -> None             # user_cancelled = True; core.mda.cancel()
    def shutdown(self) -> None           # stop executor, kill worker, disconnect; idempotent
```

`start`:

1. Write `run.json` (with `"status": "running"`) and `script.py`.
2. Build the iterator from `iter(base)`.
3. Connect `frameReady` with `_connect_on_runner_thread` (G2), and connect
   `sequenceFinished`. A queued connection to the GUI thread is fine there.
4. Call `core.run_mda(iterator, output=output)`. If the launch raises, undo
   steps 1–3 and stop the executor.

`_on_frame_ready(img, event, meta)` runs on the **runner thread** and must be
quick:

1. `frame_id = next(counter)`, then read the origin and parent from the
   metadata tag.
2. Build `FrameInfo(event=event.model_copy(update={"sequence": None}), metadata=dict(meta), ...)`.
3. `log.write_frame(...)`, then `frameAcquired.emit(frame)`. Signals emitted
   from a foreign thread are queued to the receivers in the GUI thread.
4. If `filter.accepts(...)` and the executor is not broken, call
   `iterator.analysis_submitted()`, then `fut = executor.submit(img, frame)`
   and `fut.add_done_callback(partial(self._on_result, frame))`. Then emit
   `analysisQueued`.
5. With multiple cameras, `frameReady` fires once per camera image. Each
   frame gets its own `frame_id`. Document this, and leave the filtering to
   the script (`frame.metadata` names the camera).

`_on_result(frame, fut)` runs on an executor callback thread:

1. Get `res = fut.result()` (a `BrokenProcessPool` or a cancelled future is
   a fatal error).
2. Write `log.write_analysis(res)`. Relay `res.logs` through `logMessage`.
3. If `res.ok`, normalise. If the response stops, call
   `iterator.stop("script")`. If it drops the base, call
   `iterator.drop_base()`. Expand `MDASequence` events with `list()`, check
   the cap, rebase the timing (3.5), and call
   `iterator.inject(..., response_id=next(resp_counter))`. Emit
   `eventsInjected`.
4. If it is not ok, apply `on_error`. `"stop"` calls
   `iterator.stop("analysis error")` and emits
   `analysisError(msg, fatal=False)`. `"skip"` logs the error and continues.
   A broken executor is always treated as `stop`.
5. Always call `iterator.analysis_finished()` in a `finally` block.
6. Results that arrive after the run has finished are logged as `dropped`
   and never injected.

`analysis_timeout_s` is enforced only in blocking mode, where it matters
most. The iterator records the submit time of the oldest pending analysis.
When it exceeds the timeout, the controller applies `on_error`. A hung
*thread* cannot be killed, so `stop` ends the run but the thread lingers
until it returns. Document this, and recommend process mode for untrusted
or long-running code. A hung *process* is terminated by `stop()`.

On `sequenceFinished` (GUI thread): call `executor.stop()`, relay the
teardown logs, and finalise `run.json` with the end time, the status
(`completed`, `stopped_by_script`, `cancelled`, `error`, or
`max_events_reached`), and the counts. Close the log, disconnect `frameReady`,
and emit `runFinished`. Determine the status from the controller's own flags
first (`user_cancelled`, `iterator.stop_reason`) and from
`core.mda.status.finish_reason` second (G3).

### 5.6 `_smart/_log.py`: `SmartRunLog` (on-disk record)

`run_dir` is the folder for the run's records:

- **Saving enabled:** the output is `AcquisitionSettings` or a path with a
  `root_path`. Use `<root_path without .ome.zarr/.ome.tiff suffix>_smart/`,
  next to the data. Use the same suffix-stripping helper the MDA widget uses
  (`ome_writers._schema._ome_stem_suffix`, already imported in
  `widgets/_mda_widget.py`).
- **Saving disabled (memory run):** use
  `tempfile.mkdtemp(prefix="pymmgui-smart-")`. Show the path in the monitor
  with an "Open folder" button (`QDesktopServices.openUrl`).

Files:

- `run.json` records the following: the API version, `pymmcore_gui`,
  `pymmcore_plus`, and `useq` versions, the script path and sha256, the
  script name, the resolved parameters, the execution, sync, filter,
  on-error, and limit settings, `base_sequence`
  (`base.model_dump(mode="json")`), the data output path or `null`, the
  start and end times in ISO 8601, the status, and the counts (frames,
  analyses, errors, injected events, dropped results).
- `script.py` is a byte-for-byte copy of the script as loaded at run start.
- `frames.jsonl` has one line per acquired frame:
  `{"frame_id", "t_index", "origin", "parent_frame_id", "response_id",
  "event": <model_dump json exclude sequence, exclude_none>,
  "runner_time_ms", "camera", "position": {x, y, z}}`.
  `t_index == frame_id` for a single camera. **This is the only way to map
  frames in the flattened store back to channels and positions (G1).**
- `analysis.jsonl` has one line per analysis call, plus the `setup` and
  `teardown` calls: `{"frame_id", "call", "ok", "duration_ms", "records",
  "logs", "response": {"n_events", "priority", "stop", "drop_base"},
  "error", "dropped"}`.

Writes come from the runner thread and the executor callback threads. Guard
them with a single `threading.Lock`, append one line, and `flush()` after
each line, so a crash leaves a valid prefix.

### 5.7 `_run_owner.py`

```python
class RunOwner(str, Enum): ACQUIRE = "acquire"; SMART = "smart"

class RunOwnership(QObject):
    ownerChanged = Signal(object)   # RunOwner | None
    def __init__(self, mmcore: CMMCorePlus, parent: QObject | None = None)  # connects sequenceFinished -> release
    @property
    def owner(self) -> RunOwner | None
    def claim(self, owner: RunOwner) -> None   # RuntimeError if core.mda.is_running()
    def release(self) -> None
    def accepts(self, page: RunOwner) -> bool   # owner == page, or (owner is None and page is ACQUIRE)
```

A run that nobody claimed counts as an Acquire run. Examples are a script in
the console and the Acquire MDA widget's own Run button. The Acquire button
can call `claim(ACQUIRE)` explicitly for clarity, but it does not have to.

### 5.8 UI: `widgets/_smart/`

**`SmartMDAWidget(MemoryMDAWidget)`** in `_mda.py`:

```python
class SmartMDAWidget(MemoryMDAWidget):
    def set_launcher(self, fn: Callable[[MDASequence, SingleOutput | None], None]) -> None
    def execute_mda(self, output): ...   # upstream body (G11) with run_mda replaced by self._launcher(...)
```

Always use the collapsible flavour. The Acquire page's Collapsible/Topbar
switch is not needed here. Everything else is inherited: the Run, Pause, and
Cancel buttons, the lock, the overlays, Saving, and the pixel-size guard.

**`SmartMicroscopyPage(TabPage)`** in `_page.py`. Signals:
`mdaRunningChanged = Signal(bool)` and `analysisError = Signal(str)`.
Layout, using `TabPage` regions:

```text
toolbar:  [Load script…] [Reload] [New from template ▾] [Open in editor] │ [Stop after current]
┌────────────── left (Sidebar) ──────────────┬─────────────── content ───────────────────────┐
│ Base acquisition  (SmartMDAWidget)          │ ┌ Script panel ───────────────────────────────┐ │
│                                             │ │ name · path · status badge (valid/error/    │ │
│                                             │ │ modified) · description · error w/ line no. │ │
│                                             │ │ Parameters (auto form)                      │ │
│                                             │ │ Execution: (•) Thread ( ) Process           │ │
│                                             │ │ Sync: (•) Blocking ( ) Async                │ │
│                                             │ │ Analyze: channels [..] every Nth [1] origins│ │
│                                             │ │ On error: [Stop|Skip]  Limits: max events…  │ │
│                                             │ │ [Test on last image]                        │ │
│                                             │ └─────────────────────────────────────────────┘ │
│                                             │ ┌ splitter: Viewers (own CDockManager) │ Monitor┐│
│                                             │ │  live viewer of the smart run         │ table ││
│                                             │ │                                       │ log   ││
│                                             │ └───────────────────────────────────────┴───────┘│
└─────────────────────────────────────────────┴──────────────────────────────────────────────────┘
```

Page behaviour:

- `start_run(seq, output)` is the launcher. It refuses and shows a message
  when no valid script is loaded. Otherwise it calls
  `ownership.claim(SMART)`, creates `run_dir`, runs
  `controller.prepare(config, run_dir)` behind the BusyOverlay, then
  `controller.start(seq, output)`. On any failure it releases ownership and
  shows the error.
- Its `set_mda_lock(locked)` disables the script panel, the toolbar's
  load/template buttons, and the parameter form, but keeps "Stop after
  current" enabled while running. It emits `mdaRunningChanged`. This
  mirrors `AcquirePage.set_mda_lock` and is driven by
  `SmartMDAWidget.mdaLockChanged`.
- `cancel_acquisition()` calls `controller.cancel()` and the MDA widget's
  `cancel_acquisition()` for the overlay. `shutdown()` calls
  `controller.shutdown()`, closes the viewers, and stops the watchers.
- A `QFileSystemWatcher` watches the loaded script. When it changes on disk,
  the page re-runs `inspect_script` and shows "modified — changes apply at
  next run". Changes never apply mid-run. Thread mode imports a fresh module
  under a unique name each run, and process mode spawns a fresh child each
  run, so edits always take effect on the next run.
- "Test on last image" runs a single `analyze` through a temporary executor
  of the selected mode. The executor goes through `setup`, `analyze`, and
  `teardown`, then is discarded. The image is the latest preview or snap
  (`core.getImage()` after a snap, or the preview's last frame). The
  `FrameInfo` is synthetic: it uses the current core state, `frame_id=0`,
  and `origin="base"`. The page shows the returned `Response`, the
  `records`, the `logs`, and any traceback in a dialog. **Nothing is
  executed on the hardware.** This is the main development loop for
  script authors, so give it care.
- "New from template ▾" copies a file from
  `resources/smart_templates/` to a location the user picks, loads it, and
  opens it in the system editor.

**`_params_form.py`** turns `ParamDef`s into a `QFormLayout` of
`QDoubleSpinBox`, `QSpinBox`, `QCheckBox`, `QComboBox`, and `QLineEdit`
widgets, honouring `min`, `max`, `step`, `label`, and `tooltip`. Values
persist per script path (Section 6.4). It offers "Reset to defaults".

**`_monitor.py`** is a `QTableView` over a `QAbstractTableModel` that the
controller signals append to. Its columns are #, time (s), origin, channel,
position, analysis status (queued, done, skipped, error, dropped), duration
in ms, action ("+3 → next", "stop", "—"), and records (`k=v`). It is capped
at 10 000 rows in memory, because the full record is on disk. Below the
table sit a read-only log console (`QPlainTextEdit`) with level colouring
from the theme, and a row of counters: frames, analyses, pending, injected,
queue length, and errors. Theme-aware colours must come from
`pymmcore_gui._theme` tokens, as everywhere else in the app.

**Viewer:** the page owns a `CDockManager` and an `AcquireViewersManager`
built with `accepts_run=lambda: ownership.accepts(RunOwner.SMART)`
(Section 6.3). It has no snap or live preview.

- At each smart run's `sequenceStarted`, the manager creates a new viewer
  tab in this workspace and makes it current. The page is already the
  visible tab, because the window follows the run owner, so the viewer
  visibly pops up as the run begins.
- Earlier smart-run viewers stay open as tabs beside it, the same as on
  Acquire.
- The viewer shows one `t` slider in acquisition order and follows the
  newest frame. The follow-lock button works as it does on Acquire.
- Its tab title is the data file name when saving, otherwise
  `Smart <script name> <sha>`.

---

## 6. Changes to existing code

### 6.1 `src/pymmcore_gui/__init__.py` and `__main__.py`

Apply G8 (lazy exports) and G9 (`freeze_support`).

### 6.2 `src/pymmcore_gui/_main_window.py`

- `TAB_LABELS = ("Installation", "Hardware Setup", "Configurations", "Acquire", "Smart Microscopy")`.
- Create `self._run_ownership = RunOwnership(self._mmc, self)` before the
  pages, and pass it to `AcquirePage` (new optional kwarg, default `None`
  meaning today's behaviour) and to `SmartMicroscopyPage`. Add the smart
  page to `self._stack` after Acquire.
- Connect `self._smart.mdaRunningChanged` to `self._on_mda_running`, as is
  already done for Acquire. Both pages emit for every run, so the handler
  must stay idempotent. Connect `self._smart.analysisError` to
  `self._notification_manager.show_error_message`.
- In `_on_mda_running(running)`, replace "select Acquire" with "select the
  owner page". `owner_page = self._smart if ownership.owner is RunOwner.SMART else self._acquire`.
  Disable every other tab, and keep the existing `QSignalBlocker` pattern
  and the comments that explain it.
- `_on_pixel_calibration_running` adds `self._smart` to the pages it
  disables.
- `_update_mda_status_visibility` makes the status visible on Acquire
  **or** the smart page.
- `_ready_to_close_during_acquisition` calls `cancel_acquisition()` on the
  owner page instead of always on `self._acquire`.
- `closeEvent` calls `self._smart.shutdown()` next to
  `self._acquire.shutdown()`. It must terminate any analysis child process.
- Add a `smart` property next to `acquire`, for tests.

### 6.3 `src/pymmcore_gui/_ndv_viewers.py`, `AcquireViewersManager`

- Add the constructor kwarg `accepts_run: Callable[[], bool] | None = None`.
  `_on_sequence_started` returns early when `accepts_run` is set and returns
  `False`. The Acquire page passes
  `lambda: ownership.accepts(RunOwner.ACQUIRE)` when it has an ownership
  object.
- Generator runs (G5): in `_on_sequence_started`, detect
  `isinstance(sequence, GeneratorMDASequence)` (import from
  `pymmcore_plus.mda._generator_sequence`, or check
  `type(sequence).__name__` if the private import is unacceptable) and set
  `self._generator_frames = 0`. In `_on_frame_ready`, for such runs, use
  `index = {"t": self._generator_frames}` and increment it, instead of the
  `event.index` mapping. Make sure `_extract_scales` and
  `bind_live_mda` do not warn or crash for a `GeneratorMDASequence`, and
  guard them if they do. The "Re-use MDA…" action of a smart viewer should
  load the **base** sequence back into the Smart page's MDA editor. Let the
  page set `viewer._reuse_mda_callback` after `mdaViewerCreated`. Otherwise
  set it to `None`.

### 6.4 `src/pymmcore_gui/_settings.py`

Add `SmartMicroscopySettingsV1(BaseMMSettings)`, following the pattern of
`ModernWindowSettingsV1`, with these fields: `recent_scripts: list[str]`
(max 10, most recent first), `last_script: str | None`,
`execution: ExecutionMode = "thread"`, `sync: SyncMode = "blocking"`,
`on_error: Literal["stop","skip"] = "stop"`, `max_total_events: int = 10_000`,
and `params_by_script: dict[str, dict[str, Any]]`. Add it to `SettingsV1` as
the field `smart_microscopy`. Existing settings files without the section
must load, and `tests/test_settings.py` has round-trip tests to extend. The
script's own `EXECUTION` and `SYNC` are only defaults the *first* time a
script is loaded. After that, the user's last choice for that script wins.
Store it in `params_by_script[path]["__execution__"]` or in a sibling dict.

### 6.5 pymmcore-plus fork (`cite`)

The G3 fix and its regression test are pushed to the fork as `5ff067a`.
What remains is to make sure `uv.lock` points at `5ff067a` or later (see G3).

### 6.6 Not changed

`AcquirePage` keeps locking itself during smart runs, which is correct
because the hardware is busy. The Acquire MDA widget's Cancel button still
cancels a smart run, through `mda.cancel()`. The iterator then sees
FINISHING, and the controller logs the run as `cancelled` because the
runner reports a FINISHING phase without a script stop. Treat "FINISHING
without our own stop reason" as a cancel.

---

## 7. Templates (`resources/smart_templates/`)

Each template must run against the demo config (`tests/test_config.cfg`)
and is exercised by tests.

1. `minimal.py` contains `API_VERSION`, `NAME`, and an `analyze` that only
   calls `ctx.record(mean=...)` and returns `None`.
2. `adaptive_exposure.py` ports the reference example. The base sequence is
   one event, the mode is blocking, each analysis returns the next event
   with an adjusted exposure, and `STOP` is returned after
   `PARAMETERS["n_frames"]` frames.
3. `detect_and_zstack.py` uses async mode. The base sequence is a position
   list. A threshold on the mean or max finds a "hit". The script returns
   `useq.MDASequence(stage_positions=[(x, y, z)], z_plan=..., channels=[...])`
   with `priority="next"`, using `frame.metadata` or `frame.event` for the
   position. It shows how to build follow-up acquisitions.
4. `stop_when.py` runs a long time-lapse base and returns `STOP` once a
   recorded metric crosses a threshold. It shows `ctx.state` history.

---

## 8. Implementation phases (each ends green: tests, ruff, mypy, pyright)

**Phase 0: groundwork.**

- G3 fork patch: done and pushed (`5ff067a`). Confirm that `uv.lock`
  includes it (see G3).
- G8 lazy package exports and G9 `freeze_support`, with the "no Qt on
  import" test.
- `RunOwnership`.
- The `AcquireViewersManager` changes from 6.3, with tests that use a
  generator run.
- *Acceptance:* `core.run_mda(iter([...]))` in a GUI test produces a viewer
  that follows the `t` index, and existing tests pass.

**Phase 1: headless engine (no UI).**

- `smart/_api.py`, `_loader.py`, `_worker.py`, `_executors.py` (both
  modes), `_scheduler.py`, `_log.py`, and `_controller.py`.
- *Acceptance:* in a test without widgets but with a `qapp`, each template
  runs end to end on the demo core in **both** execution modes, with the
  expected `frames.jsonl` and `analysis.jsonl` contents. Cancel mid-run in
  blocking mode stops within 1 s and the run is logged as `cancelled`. A
  script that raises applies the error policy. A script that calls
  `os._exit(1)` in process mode is reported as fatal and the run stops
  cleanly.

**Phase 2: tab skeleton and window integration.**

- `SmartMDAWidget`, `SmartMicroscopyPage` with the script panel (load,
  validate, parameters, execution settings), the main-window changes from
  6.2, and **the page's own viewer workspace** (Section 5.8, "Viewer").
- *Acceptance:*
  - The user loads a template and presses Run. The window stays on Smart
    Microscopy, and a live viewer **pops up inside the Smart Microscopy
    tab** at run start and follows each new frame.
  - No viewer is created on the Acquire page for that run.
  - An MDA started from the console, or from Acquire's Run button, still
    switches to Acquire and opens its viewer there, unchanged.
  - Other tabs are disabled during the run and re-enabled after it.
  - Closing mid-run cancels the run and terminates the worker.

**Phase 3: monitor.**

- The `_monitor.py` table, log, and counters, the "Open folder" button,
  and the "Re-use MDA" routing from the smart viewer back to the base
  sequence.

**Phase 4: authoring loop.**

- Templates and "New from template", "Open in editor", the file watcher
  with the modified badge, "Test on last image", and settings persistence.

**Phase 5: hardening and docs.**

- Safety limits in the UI, timeouts, PyInstaller check (build the bundle
  and run a process-mode smart run from it), and
  `docs/architecture/SMART_MICROSCOPY.md`. That doc covers the user guide,
  the API reference, the thread-versus-process trade-offs, the
  frame-mapping explanation, and limitations.

Commit at the end of each phase. Per the attribution reminder, commit
messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

---

## 9. Tests

Place them in `tests/smart/`. Follow `tests/conftest.py`: the `mmcore`
fixture provides a fresh demo core per test, settings are patched, and the
layouts directory is temporary. Mark process-mode tests
`@pytest.mark.slow` only if they exceed about 5 s, because spawn costs about
1–2 s per test. Share one spawned executor per test where possible.

- `test_api.py`: the `normalise_response` table (3.3), including the cap,
  generator expansion, and bad types.
- `test_loader.py`: valid templates. Errors with line numbers: a syntax
  error, a missing `analyze`, a wrong arity, `async def`, a non-literal
  `PARAMETERS`, an unknown `API_VERSION`, and bad `EXECUTION` or `SYNC`
  values. Also check that the loader **never executes** the script: a
  module-level `raise` must not trigger.
- `test_worker.py`: `_ScriptHost` with sibling imports, `ctx.log` and
  `ctx.record` buffering, an exception captured into `error`, and a
  read-only image view in thread mode.
- `test_executors.py`: parametrised over `["thread", "process"]` and checks
  the same results from the same script. Also covers setup errors that
  abort `start`, a crashing child that sets `broken`, `stop()` being
  idempotent and terminating a hung child, and spawn context only.
- `test_scheduler.py`: uses a fake runner (`status`,
  `event_seconds_elapsed`). Covers blocking holding events, async
  pass-through, `next` pre-empting a far-future base event, `end` ordering,
  `drop_base`, `stop`, FINISHING raising `StopIteration` within `poll_s`,
  `max_total_events`, and the natural end condition with pending analyses.
- `test_controller.py`: runs on the demo core, through each template, in
  both modes. Covers timing rebase (an injected time-plan sequence is
  delayed by about its interval, not fired at once), tagging,
  `frames.jsonl` `t_index` agreeing with the sink's `t` length, cancel
  status (G3 workaround), late results marked `dropped`, and the
  `DirectConnection` handler running off the main thread (assert
  `threading.current_thread() is not threading.main_thread()` inside a
  patched hook).
- `test_run_owner.py`: claim, release, `accepts`, and claim refused while
  running.
- `test_smart_page.py`: uses `qtbot`. Covers loading a script and checking
  the form fields, the run lock enabling and disabling controls, "Test on
  last image" executing no events, and the watcher showing the modified
  badge.
- Additions to `tests/test_main_window.py`: the tab count and labels, the
  window staying on Smart during a smart run, other tabs disabled, the
  Acquire run still switching to Acquire, and close mid-run.
- `tests/test_settings.py`: the new section round-trips, and old files
  without it still load.
- An import hygiene test (G8).

Run commands. Plain `uv run` re-syncs, and the user develops the forks
editable:

```text
uv run --no-sync pytest tests -n auto --max-worker-restart=0
uv run --no-sync ruff check . && uv run --no-sync ruff format --check .
uv run --no-sync mypy src
uv run --no-sync pyright --pythonpath .venv/bin/python
```

---

## 10. Decisions taken (and why)

| Decision | Why |
|---|---|
| Scripts cannot touch the core; they act only by returning events | Makes thread and process modes semantically identical. Keeps hardware control on the runner thread, which avoids races with the engine. Makes runs reproducible from `frames.jsonl`. |
| Static (`ast`) metadata loading | The GUI never executes untrusted code merely by *loading* a script. The parameter form and validation work instantly and without side effects. |
| One analysis worker | Preserves result order and makes `ctx.state` meaningful. Parallel workers can come later for stateless scripts. |
| Spawn, never fork | Forking a Qt and multithreaded process is unsafe. Behaviour is the same on all OSes. |
| Process warm-up before acquisition | Interpreter start and imports take seconds. Setup errors must surface before the hardware moves. |
| Base events held back until nearly due | Lets analysis pre-empt far-future base events. The runner cannot be interrupted once it has an event. |
| Engine hardware sequencing off for smart runs | One event, one frame, one decision. Triggered bursts would hide frames from the feedback loop. |
| Our own `frames.jsonl` instead of fixing the sink shape | Smart runs are unbounded and irregular by nature, and ome-writers needs a fixed shape for every dim except the first. A sidecar is simple and verifiable. |
| Separate `RunOwnership` instead of one more flag on the pages | Both the window and the viewer managers need the same answer. Keeping it in one place avoids drift. |

## 11. Future work (out of scope for v1)

- Shared-memory frame transfer for process mode at high frame rates.
- Live plots of `ctx.record` values in the monitor.
- Optional safety envelope: reject injected XY or Z moves outside a
  user-set region, or outside the base positions plus a margin. Stage
  limits are the user's responsibility in v1, so document this prominently.
- Re-opening a smart run as a viewer with channel and position reconstructed
  from `frames.jsonl`, by integrating with `_acquisition_loader.py`.
- An external interpreter for process mode, for scripts that need
  libraries the bundle does not ship.
- Parallel stateless workers, and GPU-friendly batching.
- Exposing the API as a pymmcore-plus feature upstream (`run_mda` with an
  analyzer), if it proves useful outside the GUI.
