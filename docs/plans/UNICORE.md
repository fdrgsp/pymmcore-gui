# Plan: Python device adapters (UniMMCore) in pyMM

Status: proposal, not started. Written 2026-11-15 against `cite` and
pymmcore-plus `cite` (`6e582a5`). **Revised the same day after an external
review:** runtime core swapping is dropped. The core is chosen once, at
startup, and the `--py-devices` flag is gone. §14 records why. A second
review tightened the folder-repair rule, module ownership, the loader's
error boundary and the safeguards before a load.

## TL;DR

- **The core is chosen once, at startup, and never swapped.** pyMM runs on
  `UniMMCore` when started with `-u/--unicore`, or when the startup
  configuration declares Python devices (`#py` lines). Otherwise it runs on
  the standard `CMMCorePlus`. The standalone app never uses UniMMCore.
- **No path flag.** The device folder comes from the configuration (a new
  `#@PyDevicesPath` line, §4.1) or from the Hardware tab's *Python devices*
  row (label | path | Browse… | Load). The row is always visible, but only
  enabled on UniMMCore.
- **In-session loads never change the core.** On UniMMCore any configuration
  loads, standard or not. On the standard core, a Python-device configuration
  can't be fully loaded. pyMM explains how to get them: close pyMM and reopen
  it with `pymm -c scope.cfg` (or `pymm -u`). It offers "Load without the
  Python devices" or "Cancel". pyMM doesn't restart itself.
- **Broken Python devices are skipped, never fatal.** That tolerance has to
  live **inside pymmcore-plus's loader** (§4.3). A device can fail at import,
  construction, initialization or later in the load, and the loader keeps
  running device code after initialization. Checking in the GUI after the
  load is too late. C++ device failures still fail the load, as today.
- **Missing dependencies get a precise message.** It names the package and
  the Python environment it must be installed in (§3.4). Dependencies must be
  installed in pyMM's environment before running `pymm`.
- **Saving already works in pymmcore-plus `cite`** (`e303a71` and `6e582a5`
  merged and pinned). The GUI adds the folder line and overwrite warnings
  (§8).

Typical workflow:

1. Start with `pymm -u`.
2. In Hardware Setup, browse to the device folder and press **Load**.
3. Add devices and save the configuration.
4. Later launches run `pymm -c scope.cfg`. The `#py` lines select UniMMCore,
   and the saved path restores the folder.

---

## 1. Where things stand

**gui `cite`** already chooses the core at startup (`03382b8`):

- The `-u/--unicore` bool flag and `create_mmgui(unicore=True)` exist.
- `startup_core` (`_unicore.py`) also picks UniMMCore when the startup config
  has `#py` lines. It doesn't replace a global core the caller already set up.
- The Hardware tab asks before loading a `#py` config on the standard core
  (`_confirm_python_devices_dropped`).
- Missing: device discovery, a device folder, tolerant loading, and a guard
  for the frozen app.

**pymmcore-plus `cite`** (pinned at `6e582a5`):

- `pymmcore_plus.model` keeps Python devices. `Microscope.create_from_core`
  shows them, and `Microscope.save` writes them as `#py` lines (§8).
- `UniMMCore.loadSystemConfiguration` raises on the first bad line, then
  unloads everything.
- `UniMMCore.loadDevice(label, module, cls)` does `__import__(module)`. There
  is no discovery API.
- `PyDeviceManager.initialize` swallows `initialize()` errors and records them
  in the device's initialization state. The loader then carries on:
  post-init properties, `System/Startup`, then `waitForSystem()`, which calls
  `busy()` on every Python device, including one that failed to initialize.
  The review reproduced this: a Python shutter whose `initialize()` failed
  then raised from `busy()`, so the load raised and unloaded everything.
- `_get_py_device_instance` rewords an `ImportError` as `type(e)("<module> is
  not a known Micro-manager DeviceAdapter, or an importable python module")`.
  That loses `ModuleNotFoundError.name`.
- `Device.type()` and `Device.name()` are classmethods.

## 2. Behaviour contract

<!-- markdownlint-disable MD013 -->
| Situation | Source run | Standalone app |
| --- | --- | --- |
| `pymm` | standard core | standard core |
| `pymm -u` | UniMMCore; the startup dialog still asks for a config | exits: "Python devices need pyMM run from source" |
| `pymm -c std.cfg` | standard core | standard core |
| `pymm -u -c std.cfg` | UniMMCore | exits, as above |
| `pymm -c py.cfg`, or the startup dialog / auto-load picks it | UniMMCore; folder from the cfg's `#@PyDevicesPath`; broken devices skipped and reported | standard core, C++ part loaded, message |
| GUI: load any cfg on UniMMCore | loads on UniMMCore, broken devices skipped and reported | — |
| GUI: load `py.cfg` on the standard core | asks: "Load without them" / "Cancel". The text: "To use them, close pyMM and reopen it with `pymm -c <path to py.cfg>` or `pymm --unicore` (`-u`)." | asks: "Load without them" / "Cancel"; the text says they need pyMM run from source |
<!-- markdownlint-enable MD013 -->

`-u` exists for starting on UniMMCore *without* a Python-device config yet,
e.g. to build one. A `#py` startup config selects UniMMCore by itself. That
happens before the window exists, so it costs nothing, and `cite` already
does it. The rule for
"Python-device config" stays the textual one (`config_needs_unicore`: any line
starting with `#py`). It doesn't depend on whether the devices import. A
UniMMCore with no working Python devices is harmless, and the user can fix the
folder from the Hardware tab and reload.

All GUI loads go through one function, `load_configuration(path)` (startup
dialog, auto-load, `-c`, `--demo-config`, the Hardware tab's **Load…**). It
activates the folder, loads with tolerance, and reports. A script that calls
`core.loadSystemConfiguration()` from the console is not intercepted.

## 3. The device folder

### 3.1 What counts as a device file

- Top-level `*.py` files. The module name is the file stem, so a cfg line
  reads `#py pyDevice,<label>,<stem>,<ClassName>`. That is what `UniMMCore`
  writes back (`cls.__module__`).
- Files starting with `_` are helpers. They aren't scanned, but sibling files
  can import them.
- Subpackages aren't scanned.
- A device is a class defined in that module (`cls.__module__ ==
  module.__name__`), a non-abstract subclass of `unicore.Device`, and not
  `_`-prefixed. It shows `cls.type()`, `cls.name()` and the first docstring
  line.
- Devices from installed packages (`#py pyDevice,Cam,mypkg.cameras,MyCam`)
  need no folder.

### 3.2 One owner for the folder

A small `PyDeviceFolder` session object, owned by the main window, is the only
code that touches `sys.path` or `sys.modules` for device files:

- It holds the folder path and the `sys.path` entry it appended.
- **The folder's modules are defined by location:** every module in
  `sys.modules` whose file (`__spec__.origin`) resolves inside the folder,
  worked out at purge time. That includes `_helper.py` files imported by a
  device, modules imported by the config loader, imports done later inside a
  device's code, and subpackages. Tracking only the names the scan imported
  would miss them, so after a folder switch a device could import a stale
  helper from the old folder. Checking every loaded module is cheap.
- **Appending** to `sys.path`, rather than prepending, means a file can't
  shadow an installed module. A clash is reported instead (§3.4).
- **It never purges a module that a loaded device uses.** Purging is allowed
  only when no loaded device comes from the folder: at configuration load,
  after the old devices are unloaded, or on **Load** with no such device
  loaded. Otherwise **Load** only imports files not yet imported and reports
  "changes to `my_laser.py` apply after reloading the configuration". This
  prevents old and new code generations from mixing inside one device.
- **Switching to another folder is refused while devices from the current
  one are loaded.** The message: "Remove Laser and Filter, or start a New
  configuration, before switching folders." Otherwise a saved config would
  point at a folder that doesn't contain its devices.
- **Activation timing.** At startup, the folder is activated before the
  window exists, when nothing is running. For a GUI load, the user confirms
  first; the old devices are unloaded next; only then does the folder change
  and the new config load. A refused or cancelled load therefore never
  changes import resolution under running hardware.

### 3.3 Scanning by import

**Load** imports each file and inspects its classes:

- Import catches what parsing can't: missing dependencies, errors at import
  time, base classes imported under an alias or through another module.
- The device type comes from `cls.type()`.
- `UniMMCore.loadDevice` later finds the module already in `sys.modules`.

Importing runs the folder's code. The help text under the row says so, and
the user docs say "don't touch hardware at import time; do it in
`initialize()`". There is no file watcher: rescanning is an explicit **Load**.

### 3.4 Errors, and missing dependencies in particular

Each problem becomes a `PyDeviceProblem(file_or_label, stage, exc)`
dataclass. A scan problem has `stage="import"`. A load problem wraps the
loader's `PyDeviceLoadError` (§4.3), whose `stage` is passed through. The
message is formatted from the stage and the exception:

<!-- markdownlint-disable MD013 -->
| Cause | Message |
| --- | --- |
| invalid module name / keyword | "not a valid module name: rename it to e.g. `my_cam.py`" |
| clash with an installed module | "`camera` is already the installed module at `<origin>`: rename the file" |
| `SyntaxError` | "line 42: invalid syntax" |
| `ModuleNotFoundError`, `e.name` ≠ the file's own module | **missing dependency**, below |
| other exception at import | "error while importing: `ValueError: …` (line 17)"; full traceback under Details |
| class missing / not a `Device` | "no class `MyCam` in `my_cam.py`" / "`MyCam` is not a UniMMCore device" |
| constructor raised | "could not create `MyCam`: `…`" |
| `initialize()` raised | "failed to initialize: `…`" |
| setting failed after init (`property`, `startup`) | "`Power` = 10 on Laser failed: `…`; the device was kept" |
| `busy()` raised while waiting (`wait`) | "stopped responding while loading: `…`" |
| `shutdown()` raised while removing it (`unload`) | appended to the device's entry: "…and its shutdown also failed: `…`" |
<!-- markdownlint-enable MD013 -->

**Missing dependency.** Device files often need third-party packages (vendor
SDKs, `pyserial`, `nidaqmx`, …). They must be installed in pyMM's own Python
environment before running `pymm`. pyMM never installs anything. The message:

> **my_laser.py** needs the Python package **`nidaqmx`**, which isn't
> installed in the environment pyMM is running in
> (`/path/to/.venv/bin/python`). Install it there, then press Load again
> (or restart pyMM).
>
> Note: the import name can differ from the name to install
> (e.g. `cv2` → `opencv-python`).

- The package name is `e.name.split(".")[0]`; the interpreter is
  `sys.executable`. Naming the interpreter matters with `uv tool install` or
  `pipx`, where pyMM runs in an isolated environment.
- The user docs list how to install into that environment: `uv pip install X`
  in its venv, `uv tool install pymmcore-gui --with X`,
  `uv run --with X pymm`, `pipx inject pymmcore-gui X`.

**How problems are shown:**

- **Load** shows one dialog listing every failing file, with a *Details*
  section holding the tracebacks.
- A persistent amber line above the Available devices table ("⚠ 2 files
  couldn't be used. Details") stays until the next Load.
- A configuration load that skipped devices shows one summary dialog.

## 4. The `.cfg`

### 4.1 The path line

UniMMCore's format names only a module (`#py pyDevice,Laser,my_laser,
MyLaser`), not where it lives. This plan adds one GUI line, following the
existing `#@LightSource` / `#@Exposure` convention:

```text
#@PyDevicesPath,/Users/me/microscope/py_devices
```

- Both parsers treat it as a comment: it starts with `#`, not with the `#py`
  prefix (which includes a trailing space).
- It is written as an absolute path. A hand-edited relative path resolves
  against the cfg's directory, so a config and its folder can move together.
- Save writes it whenever a folder is active.
- **On load, which folder is used:**
  - If the cfg's path exists, it replaces the active folder (subject to
    §3.2).
  - A cfg without the line keeps the active folder.
  - **If the cfg's path doesn't exist** (the folder moved), the active folder
    is used when there is one, and the summary says "`/old/path` not found,
    used `/new/path`". With no active folder, the devices that needed it are
    skipped, and the summary offers **Locate folder…**: browse to the new
    location, which becomes the active folder, and retry the load.
  - Either way, the next Save writes the new path, which repairs the config.
    A stale path never overrides a folder the user has just chosen.

### 4.2 Loading in the GUI

`load_configuration(path)` runs these steps in order. **Nothing is unloaded
until steps 1–4 pass**, and stopping at any of them leaves the session as it
was:

1. **Validate and read the file.** It must exist and be readable. Parse its
   `#@PyDevicesPath` and `#py` lines now, so every later decision uses this
   one read.
2. **Refuse during hardware work.** Refuse while an acquisition, pixel
   calibration, autofocus or white balance runs, with a message naming the
   operation.
3. **Confirm unsaved edits.** Ask about unsaved hardware edits (the Hardware
   tab's `_confirm_discard`) and configuration edits (the Configurations
   page's dirty state). Startup skips this step: nothing is edited yet.
4. **Standard core with `#py` lines:** ask (§2). "Load without them" loads
   the C++ part with the native loader, which skips `#py` lines. The message
   gives the exact relaunch command, with the cfg's full path, so it can be
   copied.
5. **Stop live mode**, then on UniMMCore unload the current devices, resolve
   and apply the folder (§4.1, §3.2), and call the tolerant loader (§4.3),
   collecting `PyDeviceLoadError`s.
6. **Show one summary** listing each skipped device and its reason (§3.4
   wording, Details expandable), plus folder notes from §4.1. Remember the
   skipped labels for the save warning (§8).

### 4.3 Tolerant loading belongs in the loader (pymmcore-plus)

Filtering lines before the load can't catch constructor or initialization
failures. Checking after the load is too late, because the loader keeps
running code on the failed device (§1). So the tolerance goes into
`UniMMCore.loadSystemConfiguration`:

```python
Stage = Literal[
    "import", "construct", "initialize", "property", "startup", "wait", "unload"
]


@dataclass(frozen=True)
class PyDeviceLoadError:
    label: str
    stage: Stage
    exc: BaseException
    line_no: int | None = None  # None for stages not tied to one line
    line: str = ""


def loadSystemConfiguration(
    self,
    fileName: str | Path = "MMConfig_demo.cfg",
    *,
    on_device_error: Callable[[PyDeviceLoadError], None] | None = None,
) -> None: ...
```

**Without a callback, behaviour is kept as it is today.** That isn't simply
"raises": today an `initialize()` error is only logged, and the load then
raises or not depending on what touches the device next.

**With a callback**, the loader:

- **Reads the whole file first.** It parses every line before running any,
  so it knows each Python device's parent even when the `Parent` line comes
  after the device lines.
- **Skips a Python device that fails, and continues:**
  - `import` / `construct`, at its `pyDevice` line: import, class lookup,
    constructor or `loadPyDevice` fails.
  - `initialize`, right after `Core,Initialize,1`: every Python device's
    initialization state is checked. Failed devices are unloaded *before*
    any later line, `System/Startup` or waiting touches them.
  - `wait`: in tolerant mode the final wait goes through Python devices one
    at a time, so a `busy()` that raises is pinned to its device, which is
    then unloaded.
- **Failed hub:** its Python peripherals are unloaded too, peripherals before
  the hub, even if they were already created. Each is reported against the
  hub's failure.
- **Every later `#py` line that references a skipped label is skipped:** its
  properties, delay, focus direction, labels, config-group and pixel-size
  settings, `Core` roles whose value is the label, and `Parent`.
- **A setting that fails after initialization** on a *healthy* Python device
  (a post-init `Property` line or a `System/Startup` setting) skips only that
  line. It is reported as `property` / `startup`, and the device is kept.
- **Removal always completes.** If a failed device's `shutdown()` raises
  while it is being unloaded, the error is reported (`unload`) and the device
  is removed anyway: the manager drops it in a `finally`.
- **`exc` is the original exception.** A `ModuleNotFoundError` for a
  dependency keeps its `.name` (this also fixes the rewording in
  `_get_py_device_instance`).
- **C++ devices are out of scope:** a C++ failure still fails the load, as
  today.
- `systemConfigurationFile()` and the `systemConfigurationLoaded` event stay
  correct, because the core does its own bookkeeping.

A callback, rather than a new return value, keeps the base class's
signature. The GUI wraps each `PyDeviceLoadError` in a `PyDeviceProblem`
(§3.4).

### 4.4 Adding a device in the Hardware tab

The Add flow loads, then initializes, one device at a time:

- A constructor error surfaces through the tab's existing `_warn` path.
- `initializeDevice` doesn't raise for a Python device, so the tab checks
  `getDeviceInitializationState(label)` right after initializing. On failure
  it unloads the device and warns, before anything calls `busy()` on it.

## 5. Startup and CLI

- `-u/--unicore` stays a bool flag (already on `cite`). `create_mmgui(
  unicore=True)` stays. There is no path flag.
- `startup_core(unicore, mm_config)`: UniMMCore if `unicore` or
  `config_needs_unicore(mm_config)`, and only if not frozen. A global core
  the caller already set up is still not replaced. This is today's logic plus
  the frozen guard.
- The startup dialog and auto-loading the last config pick their config
  before the window exists, so they follow the same rule.
- In the frozen app, `-u` exits with code 1 and a red message.
- The startup config goes through `load_configuration`. Its skipped-devices
  summary shows after the window appears, as `_load_startup_config` does
  today.
- Help text for `-u`: "Run on pymmcore-plus's UniMMCore, which can also load
  Python device adapters (`.py` files; set their folder in Hardware Setup).
  Their dependencies must be installed in this Python environment. Chosen
  automatically when the startup config declares Python devices. Not
  available in the standalone app."

## 6. No runtime swap

pyMM never replaces its core while the window is open. `MicroManagerGUI`,
its pages and their widgets keep the core they were built with for the life
of the window. §14 gives the reasons.

The Hardware tab's **Load** therefore never changes the core. On the
standard core the row is disabled (§7).

A small permanent "Python devices" chip in the status bar shows when the
session runs on UniMMCore. Its tooltip names the folder. The About dialog
names the core class.

## 7. Hardware tab

**The Python devices row** sits at the bottom of the left pane, below the
Available devices table, and is always visible:

```text
┌ Available devices ─────────────────────────────┐
│ …                                              │
│ Utilities   State Device Shutter  ShutterDevice│
└────────────────────────────────────────────────┘
 ⚠ 1 file in the Python devices folder couldn't be used. Details
 Python devices [/Users/me/scope/py_devices     ] [Browse…] [Load]
 Load runs the .py files in this folder. Install their dependencies
 in pyMM's Python environment first.
```

- **On UniMMCore:** enabled. The path is empty until set, or filled from the
  loaded cfg's `#@PyDevicesPath`. **Browse…** only fills the field. **Load**
  activates the folder (§3.2), scans it (§3.3), shows problems, and adds the
  devices to Available devices with a "Py" badge. The existing filter covers
  them.
- **On the standard core:** disabled. Tooltip: "Start pyMM with `pymm -u` to
  use Python devices."
- **In the standalone app:** disabled. Tooltip: "Python devices need pyMM run
  from source (`pip install pymmcore-gui`)."
- Clearing the field and pressing **Load** deactivates the folder (refused
  while its devices are loaded, §3.2).
- The folder is owned by the window's `PyDeviceFolder`, not the page.

## 8. Saving

- The Hardware tab saves through `pymmcore_plus.model.Microscope.save`. With
  `e303a71` (now pinned) it writes Python devices and every line that
  references them as `#py`. With `6e582a5`, Python hubs offer their
  peripherals. `Microscope.create_from_core` also shows Python devices
  correctly, which the tab's Installed devices table relies on.
- After `Microscope.save`, the GUI writes `#@PyDevicesPath`, next to where it
  already writes the channel comments. `write_channel_comments` must keep the
  line; add a test.
- **Overwrite warning.** Save regenerates the file from the live system. So
  saving over a file whose Python devices were skipped this session (failed
  import, failed construction or initialization, standard core, standalone
  app) would delete them from the file. The warning reads: "Saving over
  `scope.cfg` removes 2 Python devices that didn't load (Laser, Filter). Save
  anyway, or Save As…?"

## 9. Standalone app

`unicore_supported() = not sys.frozen`. In the frozen app:

- `-u` is refused. A `#py` startup config never selects UniMMCore.
- Nothing imports `pymmcore_plus.experimental.unicore`. `is_unicore` checks
  `sys.modules`, so it never triggers the import.
- The row is disabled (§7). A `#py` config loads its C++ part, with the
  message from §2.
- The overwrite warning (§8) applies.

## 10. pymmcore-plus work

1. **Done.** `e303a71` and `6e582a5` were merged into `cite` and pinned in the
   gui (`3a96413`).
2. **Required.** Tolerant loading with `on_device_error` and
   `PyDeviceLoadError` (§4.3): read the whole file first, check
   initialization right after `Core,Initialize,1`, cascade from a failed hub,
   wait on devices one at a time, guarantee removal when `shutdown()` raises,
   and pass on the original exception for missing dependencies. Test it
   there, including the review's failing-`busy()` shutter.

Nothing else is needed upstream. Without a swap, there is no reassignment of
the global core and no hand-made copy of the loader's bookkeeping.

## 11. From `cite-unicore`: kept and dropped

<!-- markdownlint-disable MD013 -->
| `cite-unicore` | This plan |
| --- | --- |
| fixed folder `USER_DATA_DIR/devices` + `MMGUI_PY_DEVICES_DIR` | **dropped**: folder from the cfg line or the Hardware tab row |
| AST scan, base-name table | **dropped**: import scan (§3.3) |
| file watcher | **dropped**: explicit Load |
| "Python Devices…" button that opens the folder | **replaced** by the path row |
| core chosen only at startup | **kept** (§6) |
| frozen-app guard, CLI refusal | **kept** |
| folder appended to `sys.path`; module-name / keyword / clash checks | **kept** (§3.2, §3.4) |
| `is_unicore` via `sys.modules` | **kept** |
| amber problems banner + Details dialog | **kept** |
| "Py" devices in Available, Python hubs + peripherals | **kept** |
| temp uv override pinning plus `cite-unicore` | **dropped** (no longer needed) |
<!-- markdownlint-enable MD013 -->

## 12. Phases

1. **plus**: tolerant loader (§10.2). Relock the gui.
2. **`_unicore.py`** (no Qt): `PyDeviceFolder`, the import scan,
   `PyDeviceProblem` and its message formatting, reading and writing
   `#@PyDevicesPath`, and the frozen guard.
3. **Startup**: `startup_core` with the frozen guard, `load_configuration`,
   the skipped-devices summary.
4. **Hardware tab**: the row, Load, banner, Py badge, the Add-flow
   initialization check, the standard-core prompt, the
   path line on save, the overwrite warning, the status bar chip.
5. **Standalone app**: disabled row, messages, a `test_bundle` case.
6. **Docs**: a user guide (writing a device file, installing its dependencies
   into pyMM's environment, the cfg format, what Load runs).

## 13. Tests

- **Loader (plus).** A Python device that fails at each stage (`import`,
  `construct`, `initialize`, `wait`, including one whose `busy()` raises
  after a failed `initialize()`) is skipped, and the callback gets the right
  `stage`. Every `#py` line referencing it is skipped. A hub that fails at
  initialization unloads its already-created peripherals, even when the
  `Parent` lines come after the device lines. A device whose `shutdown()`
  raises during removal is still removed, and an `unload` error is reported.
  A failing post-init property on a healthy device skips only that line.
  C++ devices and the rest of the file load. With no callback, behaviour
  matches today's.
- **Scan.** Valid device, `_helper.py` skipped, a class from another module
  not listed, syntax error, missing dependency (the message names the package
  and `sys.executable`), exception at import, name clash, bad file name.
- **Folder.** Load re-imports an edited file when no device from it is
  loaded, and doesn't purge a module a loaded device uses. A `_helper.py`
  imported by a device is purged on a folder switch, so it isn't served stale
  to the new folder's devices. Switching folders is refused while the current
  folder's devices are loaded. A cancelled GUI load leaves `sys.path` and
  `sys.modules` unchanged. **Moved folder:** a cfg whose saved path no longer
  exists uses the active folder and says so. With no active folder, **Locate
  folder…** retries the load. The next save writes the new path.
- **Startup matrix.** Every row of §2, with `sys.frozen` monkeypatched for
  the standalone column.
- **GUI loads.** The standard-core prompt and its "load without them" path.
  The message names the relaunch command with the cfg's path. Each pre-unload
  check (missing or unreadable file, refused during an acquisition or
  calibration, cancelled unsaved-edits prompt) leaves the current devices
  loaded and the folder unchanged. Live mode is stopped before unloading.
  On UniMMCore, a standard cfg loads, and a `#py` cfg with a broken device
  shows the summary.
- **Add flow.** A device whose `initialize()` raises is unloaded and reported
  before anything polls it.
- **Save.** Round-trip of `#py` lines plus `#@PyDevicesPath`. The channel
  comments and the path line coexist. The overwrite warning appears.
- **Smoke test by hand on UniMMCore** (pymmcore-widgets may call C++-only
  APIs): snap, live, MDA with a Python camera, Stage Explorer with a Python XY
  stage, pixel calibration, Configurations page editing a Python device's
  presets.

## 14. Decided

- **No runtime core swap.** About 35 modules hold a core reference. The
  GUI's core signals use the Qt backend (`PYMM_SIGNALS_BACKEND=qt`), so a
  disconnect doesn't stop callbacks that were already queued. The
  Configurations editors hold uncommitted edits, which serializing the core
  wouldn't capture. Every intermediate failure would need its own recovery
  state. Closing pyMM and reopening it with the right config costs one
  launch and none of that risk.
- **Flag:** `-u/--unicore` (bool), for starting on UniMMCore without a
  Python-device config yet. No `--py-devices`. The folder lives in the cfg
  and in the Hardware tab row.
- **A `#py` startup config selects UniMMCore automatically**, with or without
  `-u` (`-c`, the startup dialog, auto-load).
- **Loading a `#py` config on the standard core** offers "Load without
  them" or "Cancel", and tells the user to close pyMM and reopen it with
  `pymm -c <cfg>` or `pymm -u`. pyMM doesn't restart itself.
- **One device folder only**, and it is **not remembered in Settings**.
