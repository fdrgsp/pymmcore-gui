# pymmcore-plus: upstream PRs

`fdrgsp/pymmcore-plus@cite` against `pymmcore-plus/pymmcore-plus@main`, merge
base `94db4e3` (2026-11-24). See [README.md](README.md) for the conventions and
the cross-repository chains.

The fork adds:

- **Small fixes** to the config model, discovery, install, runner and logging
  (PLUS-1 to PLUS-9, PLUS-13).
- **Data sink access** for viewers and exports: `get_sink`, `release_sink`,
  per-frame metadata (PLUS-10 to PLUS-12).
- **Autofocus**: a `pymmcore_plus.autofocus` package (scores, searches,
  routines, a registry), the engine running software autofocus, a hardware
  Z search, and keeping a hardware autofocus locked (PLUS-15 to PLUS-20).
- **UniMMCore**: saving Python devices in a config file, and loading a config
  past a broken Python device (PLUS-21 to PLUS-24).

## Summary

<!-- markdownlint-disable MD013 -->
| ID | Type | Title | Size | Needs | Pairs with |
| --- | --- | --- | --- | --- | --- |
| PLUS-1 | fix | Model reads a stage's focus direction | S | – | – |
| PLUS-2 | fix | Model saves every Core role | S | – | – |
| PLUS-3 | fix | Model writes an assigned COM port once | S | – | – |
| PLUS-4 | fix | Model links peripherals by hub label | S | – | – |
| PLUS-5 | fix | Discovery forgets uninstalled Micro-Manager | S | – | – |
| PLUS-6 | fix | `install(test_adapters=True)` honors `log_msg` | S | – | – |
| PLUS-7 | fix | A cancel during event iteration stays CANCELED | S | – | – |
| PLUS-8 | fix | `iterProperties` filters types via the overridable getter | S | – | – |
| PLUS-9 | fix | CoreLog in its own file, rotated by MMCore | S | upstream #630, #618 | – |
| PLUS-10 | feat | Export `SingleOutput`, `OmeWritersSink`, `SinkProtocol`, `frame_meta_to_ome` | S | – | – |
| PLUS-11 | feat | `MDARunner.get_sink()` and `release_sink()` | S | – | – |
| PLUS-12a | feat | `OmeWritersSink` accessors | S | PLUS-10 | – |
| PLUS-12b | feat | `OmeWritersSink.frame_metadata` | S | PLUS-12a | – |
| PLUS-12c | feat | Store each frame's event in unbounded runs | S | PLUS-12b | – |
| PLUS-13 | fix | Re-engage continuous focus once, and only if it was on | S | – | – |
| PLUS-14 | fix | Full sequence axes in root acquisition order | S | USEQ-10 | – |
| PLUS-15 | feat | `AutofocusResult` and `autofocusFinished` | S | – | – |
| PLUS-16 | feat | Hardware autofocus Z search | S | USEQ-5, PLUS-15 | WDG-36c |
| PLUS-17a | feat | Image filters and focus scores | M | – | – |
| PLUS-17b | feat | Focus searches: Brent, Z stack, peak fit | M | – | – |
| PLUS-18a | feat | Capture state and routine settings | M | PLUS-17a | – |
| PLUS-18b | feat | Routine registry, `oughtafocus`, `run_software_autofocus` | M | PLUS-15, 17b, 18a | – |
| PLUS-18c | feat | `jaf` and `duo` routines | M | PLUS-18b | – |
| PLUS-19 | feat | The engine runs `SoftwareAutofocus` | M | USEQ-6, PLUS-18b | WDG-36d |
| PLUS-20a | feat | Lock after autofocus and restore at the end | S | USEQ-7b, PLUS-13 | WDG-36e |
| PLUS-20b | feat | Hold the lock across XY moves | M | PLUS-20a | – |
| PLUS-20c | feat | Suspend the lock for Z stacks and channel offsets | M | PLUS-20b | – |
| PLUS-20d | feat | No hardware-sequenced burst spans a focus change | S | PLUS-20c | – |
| PLUS-21 | fix | Model saves UniMMCore Python devices | M | upstream #609 | – |
| PLUS-22 | fix | Model offers a Python hub's peripherals | S | PLUS-4, PLUS-21 | – |
| PLUS-23 | fix | UniMMCore emits no events until a config load ends | S | upstream #609 | – |
| PLUS-24a | fix | Unloading a Python device always completes | M | upstream #609 | – |
| PLUS-24b | feat | Load a config past broken Python devices | M | PLUS-24a | – |
<!-- markdownlint-enable MD013 -->

---

## Independent fixes

### PLUS-1: model reads a stage's focus direction

*fix · S · `9aac6cc`*

`STAGE_DEVICE_GETTERS` maps `getFocusDirection` to `labels` instead of
`focus_direction`. So `Microscope.create_from_core()` always leaves the
direction Unknown, and saving writes `FocusDirection,<stage>,0`.

### PLUS-2: model saves every Core role

*fix · S · `070fae0`*

`ROLES` gains XYStage, AutoFocus, SLM, Galvo and ImageProcessor. Without a Roles
line, MMCore gives a role to the last loaded device of its type, so a config
with two XY stages reloads with the wrong one.

### PLUS-3: model writes an assigned COM port once

*fix · S · `421a7ce`*

A COM port can be both in `assigned_com_ports` and in `devices`, and writing it
twice makes a file MMCore refuses to load ("label COM3 already in use"). Write
it once, and skip its pre-init properties in the later section so they don't
overwrite the values in use.

### PLUS-4: model links peripherals by hub label

*fix · S · `5509df6`*

`get_available_devices` looked hubs up by adapter name where it needs the
loaded label.

### PLUS-5: discovery forgets uninstalled Micro-Manager

*fix · S · `7e37364`*

Evict cached installations whose folder is gone, and make
`find_micromanager(return_first=False)` return what this scan found, not
everything the cache has ever seen.

### PLUS-6: `install(test_adapters=True)` honors `log_msg`

*fix · S · `f0c1c92`*

`_test_dev_install` ignored the caller's logger and always drew rich progress.

### PLUS-7: a cancel during event iteration stays CANCELED

*fix · S · `5ff067a`*

When the event iterator blocks (e.g. an event-driven generator) and the user
cancels, ending the iterator overwrote `FinishReason.CANCELED` with
`COMPLETED`.

### PLUS-8: `iterProperties` filters types via the overridable getter

*fix · S · `51e81ef`*

`iterProperties(property_type=...)` called `super().getPropertyType`, which
raises for any UniMMCore Python device. Use `self.getPropertyType`.

### PLUS-9: CoreLog in its own file, rotated by MMCore

*fix · S · `e98feb2`, `4148192`, `dbf683d`, `2a543d1` · coordinate with
upstream #630 and #618*

Each core writes `<log>-core.log` next to the Python log, rotated by MMCore
(`setPrimaryLogFileRotation`, where it exists) with the same limits, and follows
the file when `configure_logging` moves it. Two handles appending to one file
break rotation on Windows. `mmcore logs --core` shows the CoreLog.

Upstream #630 fixes the same Windows rollover failure another way, and #618
unifies the logging in MMCore. Comment there first and agree on one fix. Drop
this PR if theirs lands.

### PLUS-13: re-engage continuous focus once, and only if it was on

*fix · S · `c27304e` and the `_af_was_engaged` line of `b827e0c`*

Two bugs in the engine's autofocus:

- `setup_sequence` records `isContinuousFocusLocked()` where it means
  `isContinuousFocusEnabled()`: a device that is on but not locked yet at the
  start reads as off.
- `exec_event` calls `enableContinuousFocus(True)` before *every* image after a
  successful autofocus, instead of once after the autofocus.

It comes before PLUS-20, which rewrites that code.

## Data sink access

The gui's viewers, its "Save" export and its acquisition loader use these
(`_ndv_viewers.py`, `_mda_export.py`, `_acquisition_loader.py`).

### PLUS-10: export the sink API

*feat · S · `e4ebcdb` and the `mda/__init__.py` part of `6e782d8`*

`from pymmcore_plus.mda import SingleOutput, OmeWritersSink, SinkProtocol,
frame_meta_to_ome`. Rename `_frame_meta_to_ome` to `frame_meta_to_ome`, so code
that replays frames into a new stream writes the same per-frame metadata as the
live sink.

### PLUS-11: `MDARunner.get_sink()` and `release_sink()`

*feat · S · `7387d69` and the `_runner.py` part of `6e782d8`*

- `get_sink()` returns the sink of the current or last run. Upstream's
  deprecation of `get_output_handlers` already tells users to call
  `get_sink()`, which doesn't exist yet.
- `release_sink(sink=None) -> bool` drops the runner's reference once the data
  is no longer needed (e.g. its viewer closed), so its memory and temporary
  files can be freed. With `sink=...` it only drops that sink, never a newer
  run's.
- The view protocol documents the optional `dims`, `coords` and
  `coords_changed`.

### PLUS-12a: `OmeWritersSink` accessors

*feat · S · from `6e782d8` · needs PLUS-10*

Read-only `settings` (resolved after `setup()`), `summary_meta` and `stream`.

### PLUS-12b: `OmeWritersSink.frame_metadata`

*feat · S · `aee0198` · needs PLUS-12a*

The exact dict written with each frame slot, in acquisition order, with `None`
for skipped slots. Rebuilding this list from `frameReady` misses skipped frames
(so later entries shift). Each `setup()` starts a new list, so a list kept from
an earlier run stays intact.

### PLUS-12c: store each frame's event in unbounded runs

*feat · S · from `6e782d8` · needs PLUS-12b*

When the sink falls back to one unbounded axis (iterator-driven runs, or
sequences it can't map to a fixed shape), the axes no longer say which channel,
z or position a frame is. Then `stores_events` is true and each frame's metadata
carries `mda_event` as a JSON string: OME-TIFF map values must be strings.

## Sub-sequence axes

### PLUS-14: full sequence axes in root acquisition order

*fix · S · `16d385f` · needs USEQ-10*

`get_full_sequence_axes` returns the root and sub-sequence axes in the root
`axis_order`, and `position_sizes` follows it. This is only correct once useq
iterates sub-sequence axes in the root order: before USEQ-10 it would be wrong.

## Autofocus

The design is described in pymmcore-gui's
`docs/architecture/SOFTWARE_AUTOFOCUS.md` (sections 2 to 9). Copy its summary
into the PLUS-17a description: it explains why the routines are
reimplementations of MMStudio's and where they deliberately differ.

### PLUS-15: `AutofocusResult` and `autofocusFinished`

*feat · S · the `_result.py`, signal and hardware parts of `5fe4917`*

- A new `pymmcore_plus.autofocus` package, holding only `AutofocusResult` for
  now: kind, method, focus device, Z before and after, succeeded, message.
- `PMDASignaler.autofocusFinished(event, result)`, on both the psygnal and the
  Qt signaler.
- The engine's hardware autofocus is moved into `_exec_hardware_autofocus`,
  which emits the signal (a failing listener can't break the run) and sets
  `engine.last_autofocus_result`.

### PLUS-16: hardware autofocus Z search

*feat · S · `_search_for_focus` and `_focus_search_positions` from `5fe4917` ·
needs USEQ-5, PLUS-15 · pairs with WDG-36c*

If `fullFocus` fails, step the focus drive below the start position first, then
above (as MMStudio's HardwareFocusExtender does), retrying at each step. If
nothing locks, return Z to where it started before raising. The search stops
when the run is cancelled (`_should_cancel`).

### PLUS-17a: image filters and focus scores

*feat · M (`_filters.py`, `_fft.py`, `_scoring.py`, ~400 lines) · from
`5fe4917`*

Pure NumPy, no core: 3×3 convolution, median, sharpen, Sobel, FFT band-pass
power, and `ScoringMethod` / `score_image` / `jaf_score` with MMStudio's scores
(edges, sharp edges, mean, normalized std dev and variance, Redondo, Vollath,
median edges, Tenengrad, FFT band-pass). Everything is computed in float64:
MMStudio writes intermediate results back into 8 or 16-bit images, and the
module docstring says why that matters.

### PLUS-17b: focus searches

*feat · M (`_optimizers.py`, ~340 lines) · from `5fe4917`*

`brent_search`, `zstack_search`, `fit_peak`, `FocusSearchResult` and
`AutofocusCancelled`. Each takes a `measure(z) -> score` callable, so the tests
need no hardware.

### PLUS-18a: capture state and routine settings

*feat · M (`_capture.py`, `_settings.py`, ~400 lines) · from `5fe4917`,
`54c4a00`, `5de1561` · needs PLUS-17a*

- `capture_state`: borrow channel, exposure, ROI crop and an open shutter for a
  focus run, and restore them exactly once, including on failure.
- `OughtaFocusSettings`, `JAFSettings` and `DuoSettings`: plain dataclasses, so
  the settings dict round-trips through JSON and a GUI can build a form from
  `dataclasses.fields()`.

### PLUS-18b: routine registry, `oughtafocus`, `run_software_autofocus`

*feat · M (`_registry.py` and `oughtafocus` in `_routines.py`) · from
`5fe4917`, `4662b90`, `01e7153`, `3fd94cf`, `5de1561` · needs PLUS-15, 17b,
18a*

- `register_software_autofocus`, `available_methods` (most generally useful
  first), `get_method`, `settings_model`.
- `run_software_autofocus(core, method, settings)` for a "focus now" button. It
  prepares the microscope exactly as the engine does (stops live, switches a
  locked hardware autofocus off).
- `oughtafocus` keeps the stage within its limits and goes back to its start
  on failure or cancel.

### PLUS-18c: `jaf` and `duo` routines

*feat · M · the `jaf` and `duo` parts of PLUS-18b's commits · needs PLUS-18b*

`jaf` is MMStudio's JAF. `duo` runs two routines in turn, by default a wide
coarse `oughtafocus` scan then a narrow precise one. These are the remaining
built-in routines.

### PLUS-19: the engine runs `SoftwareAutofocus`

*feat · M · `_exec_software_autofocus` from `5fe4917`, `7bf69ea` · needs
USEQ-6, PLUS-18b · pairs with WDG-36d*

- The routine comes from the registry. The settings are checked once before
  anything moves, then the routine runs with `action.max_retries` attempts,
  each starting from the same Z.
- A cancel is reported as cancelled, not as a failed run.
- On success, the Z correction for that position is applied to every later
  event there, but only when the routine moved the focus device the Z plan
  uses.
- Emits `autofocusFinished`.

### PLUS-20a: lock after autofocus and restore at the end

*feat · S · from `5c03806`, `b827e0c` · needs USEQ-7b, PLUS-13 · pairs with
WDG-36e*

After a successful hardware autofocus, switch continuous focus on and wait for
the lock (`MDAEngine(lock_timeout=5.0)`) when the action has `keep_locked` or
focus was on at the start. `teardown_sequence` restores the starting state, also
after an error.

### PLUS-20b: hold the lock across XY moves

*feat · M · from `b827e0c`, `f52488e` · needs PLUS-20a*

`_prepare_focus` runs in `setup_event`. While focus is on, or the event's plan
asks for it (`plans_at`), an XY move applies the plan's autofocus offset and
waits for the lock. If the lock fails, the engine searches once with the plan;
if that fails too, it switches focus off and doesn't retry at that location
(`_lock_or_recover`, `_give_up_lock`).

### PLUS-20c: suspend the lock for Z stacks and channel offsets

*feat · M · from `b7c59be`, `e630c4b`, `b827e0c` · needs PLUS-20b*

A Z excursion is an image away from the Z the lock holds: a plane of a Z stack,
or a channel with a Z offset. Its reference is the drive position under the
lock, sampled once per timepoint and location. An ordinary drive gets focus
switched off and back on after the excursion. A continuous-focus drive (e.g. a
PFS offset) moves with focus on. `event_iterator` looks one event ahead so that
`teardown_event` knows whether the excursion continues, and sequenced Z
positions are taken relative to the reference.

### PLUS-20d: no hardware-sequenced burst spans a focus change

*feat · S · from `f52488e`, `b091430` · needs PLUS-20c*

With hardware sequencing, `event_iterator` splits the events wherever focus may
change (`_focus_boundary`), so a burst never crosses one. A keep-locked
time-lapse is still sequenced across timepoints.

## UniMMCore

pymmcore-gui's `cite-unicore` branch uses these. Upstream #609 (tlambert03)
refactors UniMMCore heavily: wait for it, then rebase. Also read upstream #637
(hinderling), which touches the same event code.

### PLUS-21: model saves UniMMCore Python devices

*fix · M · `e303a71`*

The model's getters were unbound `CMMCorePlus` methods, which skip UniMMCore's
overrides, so a Python device came back with no library, class, properties or
role. `CoreObject.core_values` now calls the core's own method of that name.
`Device.is_py_device` is new. `dump()` writes those devices, and every line that
names them, as UniMMCore's `#py` lines, so the file round-trips through
UniMMCore and still loads, without them, in the standard core. Output for a
standard core is unchanged.

### PLUS-22: model offers a Python hub's peripherals

*fix · S · `6e582a5` · needs PLUS-4, PLUS-21*

A Python hub has no entry among the C++ adapters' devices, so
`available_peripherals()` offered nothing. Build the hub's entry from the core.

### PLUS-23: UniMMCore emits no events until a config load ends

*fix · S · `67dd9ce`*

The Python loader goes through `defineConfig`, `setProperty` and the other
overrides line by line, emitting each event at once, so listeners queried a
half-loaded system. Hold the core's events back during the load (with either
signal backend) and emit `systemConfigurationLoaded` at the end, as MMCore's
loader does.

### PLUS-24a: unloading a Python device always completes

*fix · M · the `_device_manager.py` and unload parts of `74bfcbc`*

Unloading completes even when `shutdown()` raises, and clears the device's
roles, cached state, presets, pixel calibrations and Core role settings in
presets. A missing dependency of a device module keeps its
`ModuleNotFoundError`. Adds `getPyDeviceInitializationError(label)`.

### PLUS-24b: load a config past broken Python devices

*feat · M · the `_config.py` part of `74bfcbc` · needs PLUS-24a*

`loadSystemConfiguration(path, on_device_error=callback)` no longer aborts on a
Python device that fails to import, build, initialize or respond. The device is
skipped together with its hub's peripherals and every line that names it, and
each failure reaches the callback as a `PyDeviceLoadError`. The whole file is
read first, so devices declared later are known too. C++ errors still fail the
load, and without a callback nothing changes.
