# Smart Microscopy (event-driven acquisition)

<!-- markdownlint-disable MD013 -->
<!-- Long lines are kept in tables and code. -->

The **Smart Microscopy** tab runs acquisitions that react to their own
images. A Python script analyzes each frame as it arrives. Based on the
result, it decides what to acquire next: change the exposure, add a z-stack
where something interesting was found, skip ahead, or stop.

You define three things:

1. A **base acquisition** in the tab's MDA editor. It can be a full plate
   scan, or a single event that starts a purely reactive loop.
2. An **analysis script**: a `.py` file with an `analyze()` function.
3. **How the script runs**: in a background thread or a separate process,
   and whether acquisition waits for each analysis.

---

## 1. Quick start

1. Open **Smart Microscopy** and choose **New from template**. Save the copy
   somewhere, and it opens in your text editor.
2. In the left-hand MDA editor, set up the base acquisition. It needs at
   least one channel or position.
3. Adjust the parameters in the middle column, if the script has any.
4. Optional: **Snap** on the Acquire tab, then **Test on last image** to see
   what the script would request. Nothing is acquired.
5. Press **Run**. The live viewer opens on this tab. The monitor below it
   lists every frame, what analysis made of it, and the script's messages.

Every run writes its records to a **run folder**. The folder sits next to
the data when saving to disk, otherwise in a temporary folder. **Open run
folder** shows it. See §5.

---

## 2. Writing a script

A script is a single Python file. The GUI reads its settings **without
running it**, by parsing the source. Module-level settings must therefore be
plain literals: numbers, strings, lists, and dicts.

```python
import numpy as np
import useq

from pymmcore_gui.smart import STOP, AnalysisContext, FrameInfo, Response

API_VERSION = 1                      # required
NAME = "My experiment"               # optional: shown in the GUI
DESCRIPTION = "What it does."        # optional
EXECUTION = "thread"                 # optional default: "thread" | "process"
SYNC = "blocking"                    # optional default: "blocking" | "async"
ANALYZE = {"channels": ["FITC"], "every_nth": 1, "origins": ["base"]}
PARAMETERS = {
    "threshold": {"default": 1000.0, "min": 0, "max": 65535, "step": 10,
                  "label": "Threshold", "tooltip": "Max intensity of a hit"},
    "n_max": 10,                     # a plain value works too
    "mode": {"default": "fast", "choices": ["fast", "careful"]},
}


def setup(ctx: AnalysisContext) -> None:        # optional, once per run
    ctx.state["hits"] = 0


def analyze(image: np.ndarray, frame: FrameInfo, ctx: AnalysisContext):
    hit = float(image.max()) > ctx.params["threshold"]
    ctx.record(max=float(image.max()), hit=hit)  # shown in the monitor
    if not hit:
        return None                              # nothing to add
    ctx.state["hits"] += 1
    if ctx.state["hits"] > ctx.params["n_max"]:
        return STOP                              # finish the run
    return frame.event.model_copy(update={"exposure": 50.0})


def teardown(ctx: AnalysisContext) -> None:     # optional, always called
    ctx.log(f"{ctx.state['hits']} hits")
```

### What `analyze` receives

| Argument | Contents |
|---|---|
| `image` | The frame as a numpy array. **Read-only:** do not modify it. |
| `frame.frame_id` | 0-based acquisition order. It is also the frame's index along the data's `t` axis. |
| `frame.event` | The `useq.MDAEvent` that produced it: channel, exposure, x/y/z... |
| `frame.metadata` | Recorded state: `position` (x/y/z), `pixel_size_um`, `exposure_ms`, `camera_device`, `runner_time_ms`, `property_values`. |
| `frame.origin` | `"base"` or `"analysis"` (the event was requested by a script). |
| `frame.parent_frame_id` | For analysis-origin frames, the frame whose analysis requested it. |
| `ctx.params` | The parameter values chosen in the GUI (read-only). |
| `ctx.state` | A dict that persists across calls within one run. |
| `ctx.run_dir` | A folder for your own outputs (masks, tables...). |
| `ctx.log(msg, level)` | A message shown in the monitor's log and kept in `analysis.jsonl`. |
| `ctx.record(**values)` | Scalars for this frame: monitor "Results" column, and `analysis.jsonl`. |

### What `analyze` may return

| Return | Meaning |
|---|---|
| `None` | No change. The acquisition continues. |
| an `MDAEvent` | Acquire it next. |
| a list (or any iterable) of `MDAEvent` | Acquire them next, in order. |
| an `MDASequence` | Acquire all its events next, for example a z-stack at the current position. |
| `Response(...)` | Full control, see below. |
| `STOP` | Finish after the event that is currently running. |

```python
Response(
    events=...,             # MDAEvent list or MDASequence
    priority="next",        # "next": before the remaining base events
                            # "end":  after them
    timing="relative",      # "relative": a returned time-lapse starts its
                            #   own clock now; "absolute": as given
    stop=False,             # finish the run
    drop_base=False,        # discard the remaining base events
)
```

At most 1000 events per response are accepted. The whole run is capped by
**Max events** in the GUI.

### Rules

- **Scripts never control the microscope directly.** They act only by
  returning events. Everything the hardware does goes through the
  acquisition engine on its own thread. This is also what lets the same
  script run in a separate process. Device settings an event can carry:
  `channel`, `exposure`, `x_pos`/`y_pos`/`z_pos`, `properties`, `roi`,
  `slm_image`.
- **Device settings an event applies stay applied** for every event after
  it, including the remaining base events. A follow-up that changes the
  objective, a filter, or a light source must change it back at the end.
- To change settings **without taking an image**, give the event
  `action=useq.CustomAction(name=...)`. It still moves the stage and applies
  `properties`, but acquires nothing. The *Detect and act* template uses
  this to switch objective before a follow-up and back after it.
- Changing objective also changes the pixel size: `frame.metadata` reports
  the one in effect, and so does `frames.jsonl`.
- Injected stage moves are **not** checked against any safe region. Check
  coordinates in your script if your stage needs it.
- A `useq.Channel` in an `MDASequence` is a different type from an event's
  `channel`. To reuse an event's channel in a sequence, rebuild it:
  `useq.Channel(config=c.config, group=c.group, exposure=...)`.
- A sequence with no axes, `useq.MDASequence()`, contains **no events**.
  Give it at least one channel or position.
- Modules next to the script can be imported (`from helpers import ...`).
- In the packaged application, scripts can only import libraries bundled
  with it, such as numpy and useq. Use a Python installation of
  pymmcore-gui to analyze with anything else.

---

## 3. Thread or process

| | Thread | Process |
|---|---|---|
| Start-up | Instant | A few seconds, paid *before* acquisition starts |
| Per frame | No copy | The frame is copied to the process |
| A crash in the script's libraries | Takes the application down | Contained: the run stops cleanly |
| A hung analysis | Cannot be interrupted (the run can still stop) | The process is terminated |
| Pure-Python CPU work | Competes with the GUI | Separate CPU |

Start with **Thread** while developing. Use **Process** for heavy, slow, or
fragile analysis, such as deep learning or C extensions. The script is
identical in both modes.

## 4. Blocking or async

- **Blocking:** no new event is acquired until the pending analysis has
  finished, so every acquisition sees the latest decision. Use it for
  feedback loops, such as adaptive exposure or autofocus-like logic.
- **Async:** the base events keep being acquired while frames are analyzed,
  and requested events are inserted as results arrive. Use it for
  scan-and-zoom-in experiments.

In both modes the run lasts until all three of these are true: the base
events are done, the queue of requested events is empty, and no analysis is
pending. It also ends on `STOP`, **Stop after current**, **Cancel**, an
analysis error (with *On error: Stop the run*), or a timeout.

**Analysis timeout** applies in blocking mode only, and always stops the
run. One worker runs analyses in order, so nothing queued behind a hung
call could finish anyway.

**Frames to analyze** limits which frames reach `analyze`, by channel
preset, every *N*th frame, and origin. The default comes from the script's
`ANALYZE`. A frame that is not analyzed never blocks.

---

## 5. Data and the run folder

A smart run is saved through the usual Saving section. Because a run's
shape is not known in advance, every frame is stored along **one `t` axis**
in acquisition order, whatever its channel or position. That is what lets
any irregular run be saved and viewed live. The viewer shows it as a single
`t` slider.

The run folder records what each frame was:

| File | Contents |
|---|---|
| `run.json` | Settings, parameters, base sequence, versions, start and end times, status, counts. |
| `frames.jsonl` | One line per frame: `frame_id` (== `t` index), origin, parent frame, the full event, position, exposure, pixel size, time. |
| `analysis.jsonl` | One line per call (`setup`/`analyze`/`teardown`): duration, records, logs, the response, errors. |
| `script.py` | The exact code that ran. |

When saving to `/data/exp.ome.zarr`, the folder is `/data/exp_smart/`.
Without saving, it is a temporary folder: use **Open run folder** to keep
it.

Possible run statuses: `completed`, `stopped_by_script`, `stopped_by_user`,
`cancelled`, `error`, `analysis_timeout`, `max_events_reached`.

---

## 6. For developers

```text
SmartMDAWidget.run_mda()                       widgets/_smart/_mda.py
  └─ execute_mda(output) → SmartMicroscopyPage._start_run(seq, output)
       ├─ reload + validate script             _smart/_loader.py (ast only)
       ├─ SmartController.prepare() on a thread
       │    └─ create_executor(mode).start()   _smart/_executors.py
       │         └─ _ScriptHost: import + setup()   _smart/_worker.py
       └─ RunOwnership.claim(SMART); SmartController.start()
            └─ core.run_mda(SmartEventIterator(base), output=output)

runner thread ── next() ──▶ SmartEventIterator    _smart/_scheduler.py
runner thread ── frameReady (DirectConnection) ──▶ _on_frame_ready
                   └─ frames.jsonl; executor.submit(image, frame)
executor thread ── _on_result: normalise → inject → analysis.jsonl
```

Points that are easy to get wrong:

- **Passing an iterator** makes the runner skip the engine's own event
  iterator (no hardware sequencing). It reports an empty
  `GeneratorMDASequence`, and its sink becomes an unbounded `(t, y, x)`
  array. `AcquireViewersManager` detects this and follows such runs by
  frame count, in grayscale.
- **Qt delivers `frameReady` to plain callables on the GUI thread.** The
  MDA signals are Qt signals in this app, so the controller connects with
  `Qt.DirectConnection` to stay on the runner thread.
- **The runner blocks inside `SmartEventIterator.__next__`.** Every wait
  there is bounded and re-checks the runner's phase, so Cancel works while
  waiting for analysis. Far-future base events are held back so a
  `priority="next"` event can overtake them.
- **Relative timing** is applied when an event is handed out, not when it
  is queued. useq's `reset_event_timer` flags are consumed there and never
  reach the runner, whose clock the base events depend on.
- **Scripts are compiled from the inspected source**, not imported with the
  normal loader. The bytecode cache keys on whole-second mtimes, so a quick
  edit could otherwise run stale code. It also guarantees that `script.py`
  is what ran.
- **`pymmcore_gui.smart` and `_smart` (except the controller) must stay
  Qt-free.** A spawned worker imports them. `tests/test_import_hygiene.py`
  enforces this. The package's top-level exports are lazy for the same
  reason.
- **`RunOwnership`** decides which page a run belongs to. The window keeps
  the owner's tab on screen, and only the owner's viewers open a viewer for
  it. Unclaimed runs, such as a console run, belong to Acquire.

Tests: `tests/test_smart_*.py`, `tests/test_run_owner.py`, and the
iterator-run cases in `tests/test_ndv_viewers.py`.
