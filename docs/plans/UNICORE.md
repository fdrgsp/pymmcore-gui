# Plan: Python device adapters (UniMMCore) in pyMM

Status: proposal, not started. Written 2026-11-15 against `cite` (`36bff4d`),
pymmcore-plus `cite` (`6c1f2b7`) and `cite-unicore` (`6e582a5`). Revised the
same day after review: row placement, the prompt when a standard config is
loaded, flag name, saving explained, and a safety-first swap procedure (§6).

## TL;DR

- **One rule picks the core.** pyMM runs on `UniMMCore` when Python devices
  are in play: either a device folder is active (`--py-devices DIR`, or
  **Load** in the Hardware tab), or the loaded `.cfg` declares Python devices
  that can be imported. Otherwise it runs on the standard `CMMCorePlus`. The
  standalone app never uses UniMMCore.
- **The core is swapped at runtime, carefully.** Loading a Python-device
  config on the standard core switches to UniMMCore. Loading a standard config
  while on UniMMCore **asks**: switch back to the standard core, or keep
  UniMMCore. The swap follows a fixed, crash-safe sequence (§6): stop
  everything, disconnect every slot from the old core, release its hardware,
  delete the old pages at a controlled moment, then build the new ones. At
  startup the core is chosen before the window exists, so the common path
  never swaps.
- **The Python devices row is always visible**, at the bottom of the left
  pane, below the Available devices table: *label | path | Browse… | Load*.
  The path is empty by default. **Load** imports the files, adds their devices
  to the Available devices table, and swaps the core if needed.
- **Broken devices are skipped, never fatal.** A file that fails to import
  (syntax error, **missing dependency**, exception at import time) or a device
  that fails to initialize is reported with its reason, and everything else
  loads. A missing dependency gets its own message naming the package and the
  Python environment it must be installed in (§3.4).
- **The `.cfg` remembers the folder.** The UniMMCore format has no path in it
  today (§4.1). Save adds a `#@PyDevicesPath,<dir>` line, which a later load
  uses to restore the folder and refill the path field.
- **Saving needs a pymmcore-plus fix.** `UniMMCore` itself can save, but the
  Hardware tab saves through `pymmcore_plus.model`, which only understands
  Python devices with `e303a71` (on plus `cite-unicore`, not `cite`). The
  details are in §8.
- **Standalone app:** it refuses `--py-devices`, shows the row disabled with
  an explanation, and loads only the C++ part of a Python-device config. It
  tells the user which devices were left out.

---

## 1. Where things stand

**gui `cite`** already has a first cut (`03382b8`):

- `-u/--unicore` is a **bool** flag, and `create_mmgui(unicore=...)` takes
  it.
- `_unicore.py` picks the core **once, at startup** (`startup_core`).
- When the Hardware tab loads a `#py` config on the standard core, it asks
  "load anyway, without them?" (`_confirm_python_devices_dropped`).
- No device discovery, no device folder, and no frozen-app guard.

**gui `cite-unicore`** (`53d84e9`, `d3ba651`) went further. What this plan
keeps from it, and what it drops, is in §11.

**pymmcore-plus** (`experimental/unicore`):

- `UniMMCore.loadSystemConfiguration` reads `#py` lines. It **raises on the
  first bad line** and then unloads everything, so on its own it can't "load
  what we can".
- `UniMMCore.saveSystemConfiguration` exists on plus `cite` and writes Python
  devices as `#py` lines. The Hardware tab doesn't call it, though; see §8.
- `UniMMCore.loadDevice(label, module, cls)` does `__import__(module)`. There
  is **no discovery API**: the gui has to find device classes itself.
- A Python device whose `initialize()` raises does **not** raise from
  `initializeDevice` / `initializeAllDevices`. `PyDeviceManager.initialize`
  logs the error and stores it in the device's initialization state, so the
  gui has to check for it (§4.4).
- `Device.type()` and `Device.name()` are classmethods, so an imported class
  reports its type without being instantiated.
- On an `ImportError`, `_get_py_device_instance` raises
  `type(e)("<module> is not a known Micro-manager DeviceAdapter, or an
  importable python module")`. That drops `ModuleNotFoundError.name`, which is
  exactly what's needed to say which dependency is missing.

## 2. Behaviour contract

"Folder active" means a device folder was set this session, with
`--py-devices` or with **Load** in the Hardware tab. "Resolvable" means at
least one `#py pyDevice` line in the config imports successfully (§4.2).

<!-- markdownlint-disable MD013 -->
| Situation | Source run | Standalone app |
| --- | --- | --- |
| `pymm` | standard core | standard core |
| `pymm --py-devices DIR` | UniMMCore, folder active; the startup dialog still asks for a config | exits: "Python devices need pyMM run from source" |
| `pymm -c std.cfg` | standard core | standard core |
| `pymm --py-devices DIR -c std.cfg` | UniMMCore (the flag is explicit) | exits, as above |
| `pymm -c py.cfg`, devices resolvable | UniMMCore; folder taken from the cfg's `#@PyDevicesPath` | standard core, `#py` part skipped, message |
| `pymm -c py.cfg`, nothing resolvable (no path, bad path, all imports fail) | standard core, C++ part loaded, message listing each skipped device and why | same |
| `pymm --py-devices DIR -c py.cfg` | the flag wins over the cfg's path; the message says so if the two differ | — |
| GUI: load `py.cfg` on the standard core | swap to UniMMCore, then load | load the C++ part, message |
| GUI: load `std.cfg` on UniMMCore | **ask**: "Switch to the standard core" / "Keep UniMMCore". Switching deactivates the folder but leaves its path in the field | — |
| Hardware tab: **Load** a folder on the standard core | swap to UniMMCore, **carrying the current configuration over** | row disabled |
| Hardware tab: clear the path and **Load** | if Python devices are loaded, confirm removing them; swap to the standard core, carrying the C++ part over | — |
<!-- markdownlint-enable MD013 -->

**The prompt when loading a standard config on UniMMCore:**

- The default button is "Keep UniMMCore". The cheapest answer (no swap) is
  also the safe one, and the folder's devices stay available to add to the new
  config.
- The prompt only appears in GUI loads. Startup never asks, because the flags
  and the config already answer the question.

**One entry point.** The startup dialog, auto-loading the last config, `-c`,
`--demo-config` and the Hardware tab's **Load…** all go through the same
window-level function (§6.1). A script that calls
`core.loadSystemConfiguration()` from the console is **not** intercepted (no
monkey-patching, unlike napari-micromanager). It loads on whichever core is
running, and the console docs say so.

## 3. The device folder

### 3.1 What counts as a device file

- Top-level `*.py` files in the folder. The module name is the file stem, so a
  cfg line reads `#py pyDevice,<label>,<stem>,<ClassName>`. That is also the
  name `UniMMCore` writes back, because it saves `cls.__module__`.
- Files starting with `_` are helpers. They aren't scanned, but sibling files
  can import them because the folder is on `sys.path`.
- Subpackages aren't scanned in v1.
- A device is a class **defined in that module** (`cls.__module__ ==
  module.__name__`), a subclass of `unicore.Device`, not abstract, and not
  `_`-prefixed. Its type comes from `cls.type()`, its name from `cls.name()`,
  and its description from the first line of its docstring.
- Devices from installed packages (`#py pyDevice,Cam,mypkg.cameras,MyCam`)
  also work with no folder, because `UniMMCore` just imports them. Only folder
  devices need the folder.

### 3.2 Import path

- Activating a folder **appends** it to `sys.path`. Appending means a file
  can't shadow an installed module; the clash is reported instead (§3.4).
- Deactivating or switching folders removes the folder from `sys.path` and
  purges, from `sys.modules`, every module whose `__file__` is inside it.
- **Load** re-imports from scratch: it purges the folder's modules, calls
  `importlib.invalidate_caches()`, then imports again. That way an edited file
  is picked up. Devices that are already loaded keep their old code until the
  configuration is reloaded, and the status bar says so.
- There is no file watcher. Scanning runs the files' code, which shouldn't
  happen on every save in an editor. Rescanning is an explicit **Load**.

### 3.3 Scanning by import, not by parsing

`cite-unicore` parsed files with `ast` and matched base-class names. This plan
**imports** them instead, on **Load** only:

- Import is the only way to catch what actually breaks a device: missing
  dependencies, errors at import time, a base class imported under an alias,
  or a base class one level removed through another module.
- The device type comes straight from `cls.type()` instead of a hand-kept
  table of base-class names.
- `UniMMCore.loadDevice` later finds the module already in `sys.modules`, so
  nothing is imported twice.

The cost is that it runs the folder's code. That is acceptable because the
user chose the folder and pressed Load. The user docs say: **don't touch
hardware at import time; do it in `initialize()`**. The help text under the
row says that Load runs the files' code.

### 3.4 Errors, and missing dependencies in particular

Each file that can't be used produces a `(file, reason, details)` entry. Its
devices are skipped and every other file still loads. Reasons:

<!-- markdownlint-disable MD013 -->
| Cause | Message |
| --- | --- |
| invalid module name (`my-cam.py`, `2cam.py`, keyword) | "not a valid module name: rename it to e.g. `my_cam.py`" |
| clashes with an installed module | "`camera` is already the installed module at `<origin>`: rename the file" |
| `SyntaxError` | "line 42: invalid syntax" |
| `ModuleNotFoundError` with `e.name` ≠ the file's own module | **missing dependency**, below |
| any other exception at import | "error while importing: `ValueError: …` (line 17)". The line is the last traceback frame inside the file. The full traceback goes under Details |
| class from a cfg line missing / not a `Device` subclass | "no class `MyCam` in `my_cam.py`" / "`MyCam` is not a UniMMCore device" |
<!-- markdownlint-enable MD013 -->

**Missing dependency.** Device files commonly need third-party packages
(vendor SDKs, `pyserial`, `nidaqmx`, …). These must be installed **in the
same Python environment pyMM runs in, before running `pymm run`**. pyMM
never installs anything itself. The message:

> **my_laser.py** needs the Python package **`nidaqmx`**, which isn't
> installed in the environment pyMM is running in
> (`/path/to/.venv/bin/python`). Install it there, then press Load again
> (or restart pyMM).
>
> Note: the import name can differ from the name to install
> (e.g. `cv2` → `opencv-python`).

- The package name is `e.name.split(".")[0]`. The interpreter path is
  `sys.executable`. Naming the interpreter matters: with
  `uv tool install pymmcore-gui` or `pipx`, pyMM runs in an isolated
  environment, and `pip install nidaqmx` in another shell won't help.
- The user docs list the ways to install into pyMM's environment:
  `uv pip install X` inside its venv, `uv tool install pymmcore-gui --with X`,
  `uv run --with X pymm`, `pipx inject pymmcore-gui X`.
- The gui's own scan detects the error, so it never relies on UniMMCore's
  reworded `ImportError`. The same classification is used when a cfg's
  `#py pyDevice` line fails (§4.2).

**How problems are shown:**

- On **Load** (an explicit action), one dialog lists every file that failed
  and why, with a *Details* section holding the tracebacks.
- A persistent amber line above the Available devices table reads "⚠ 2 files
  in the Python devices folder couldn't be used. Details". It stays until the
  next Load. This is `cite-unicore`'s `set_problems` banner, reused.
- When a cfg load skips devices, a single summary dialog lists them (§4.3).

## 4. The `.cfg`

### 4.1 Format, and the path line

**Today a Python-device config contains no path.** UniMMCore's format only
names a module:

```text
#py pyDevice,Laser,my_laser,MyLaser
```

`my_laser` must already be importable: an installed package, or a folder on
`PYTHONPATH`. Nothing in the file says where it lives.

This plan adds one gui line, following the existing `#@LightSource` /
`#@Exposure` convention (`_light_sources.py`):

```text
#@PyDevicesPath,/Users/me/microscope/py_devices
...
Device,Camera,DemoCamera,DCam
#py pyDevice,Laser,my_laser,MyLaser
#py Property,Laser,Power,10
#py ConfigGroup,Channel,GFP,Laser,Power,20
#py Property,Core,Shutter,Laser
```

- Both the C++ parser and UniMMCore's parser treat it as a comment: it starts
  with `#` but not with the `#py` prefix (which includes a trailing space).
- It is written as an absolute path. On load, a relative path resolves
  against the cfg's own directory, so a hand-edited relative path makes a
  config and its device folder portable together.
- It is written whenever a folder is active, so reopening the file restores
  the folder and the path field.
- Files saved elsewhere (scripts, `UniMMCore.saveSystemConfiguration`) won't
  have it. Those fall back to the active folder, or report the devices as not
  found (§4.2).

### 4.2 Preflight, before choosing the core

`preflight(cfg, cli_folder) -> Preflight` is pure Python and unit-testable.
It:

1. Reads `#@PyDevicesPath`, every `#py pyDevice,<label>,<module>,<class>`
   line, and the `Parent,<label>,<hub>` lines, so hub→peripheral cascades are
   known.
2. Resolves the folder. Precedence: **`--py-devices` > the cfg's
   `#@PyDevicesPath` > the folder already active in the session**. The last
   one covers a cfg without the line. Loading a cfg from the GUI that names a
   folder makes that folder active and refills the path field. A path that
   doesn't exist becomes a reported problem, not an error.
3. Activates the folder (§3.2), then for each `pyDevice` line imports the
   module and checks the class (§3.4 classification).
4. Returns `resolvable` labels, `skipped` (label → reason) and the folder.
   A skipped Python hub also skips its Python peripherals.

The core decision is `use_unicore = folder_active or bool(resolvable)`, never
in the frozen app. At startup this happens **before** the window is built.

### 4.3 Tolerant load

The core loads the config **without** the skipped devices and without every
`#py` line that references them: their `Property`, `Delay`, `FocusDirection`,
`Label` lines, `ConfigGroup`/`ConfigPixelSize` settings, `Core` roles whose
value is the label, and `Parent` lines.

- **Recommended:** add `skip_devices: Collection[str] = ()` to
  `UniMMCore.loadSystemConfiguration` in plus `cite` (§10). It keeps the
  core's own bookkeeping (`systemConfigurationFile()`, the
  `systemConfigurationLoaded` event) correct and is easy to test there.
- Fallback without an upstream change: filter the text in the gui, call the
  private `unicore.core._config.load_from_string`, then set
  `_last_sys_config` and emit the event by hand. **Not** a filtered temp file,
  because `systemConfigurationFile()` would then point at the temp file, and
  Save and the recent-configs list read it.
- Standard core (standalone app, or nothing resolvable): the native loader
  already skips every `#py` line, so this is just
  `loadSystemConfiguration(path)` plus the message.

After a load that skipped anything, a single dialog appears: "Loaded
`scope.cfg` without 2 Python devices:" with each label and its reason
(§3.4 wording), a *Details* section, and, when nothing could be resolved,
"Running on the standard core." At startup this replaces today's
`_load_startup_config` warning.

### 4.4 Devices that fail to initialize

Because UniMMCore swallows `initialize()` errors, after every load (and after
each **Add** in the Hardware tab) the gui checks
`getDeviceInitializationState(label)` for each Python device. For each one
that failed:

- `unloadDevice(label)`. UniMMCore's `_cleanup_pydevice_state` already drops
  its role and its `_py_config_groups` entries.
- Add it to the skipped list with "failed to initialize: `<exception>`".
- Delete presets that this leaves empty. The C++ side keeps a preset even when
  its only settings were on that device.

In the Hardware tab's add flow, `dev.initialize(...)` returning normally does
not mean success for a Python device. Check the state and use the same
`_warn` path the tab uses for C++ devices.

## 5. Startup

### 5.1 CLI

- Replace the bool `-u/--unicore` with `--py-devices DIR` (alias
  `--py-devs`; no single-letter form), on both `run` and the root callback.
  It takes `exists=True, file_okay=False, resolve_path=True`. `-u` and
  `--unicore` were never released, so there is nothing to keep compatible.
- Help text: "Folder of Python device adapters (`.py` files) for
  pymmcore-plus's UniMMCore. Their dependencies must be installed in this
  Python environment. Not available in the standalone app."
- In the standalone app the flag exits with code 1 and a red message
  (`cite-unicore`'s check, kept).

### 5.2 `create_mmgui`

- Replace `unicore: bool` with `py_devices: Path | str | None`.
- Keep `mmcore=` passing a `UniMMCore` as the way for embedders to force it.
- `startup_core` becomes: preflight the startup config (or no config), then
  `UniMMCore()` if `use_unicore`, else leave `mmcore=None`. The existing
  "don't replace a caller's global core" rule stays.
- The startup dialog's choice and the stored auto-load config are known
  before the window is built, so they also choose the core without a swap.
  The skipped-devices summary shows after the window appears, as
  `_load_startup_config` does today, so it isn't lost in a frozen build.

## 6. Swapping the core at runtime

The swap is the delicate part. About 35 modules hold a core reference, all
of them below the three core pages (Hardware, Configurations, Acquire), the
MDA status widget and the console. Those include pymmcore-widgets widgets,
ndv/vispy canvases, `QTimer`s and the MDA runner thread. The window's own
shell (menu, mode tabs, theme, notifications, Installation page) doesn't
depend on the core.

### 6.1 Principles

1. **Swap as rarely as possible.** Startup chooses the core before the
   window exists. Loading a config on the core it already needs never
   swaps. "Keep UniMMCore" is the default answer to the §2 prompt.
2. **Only from a clean call stack on the GUI thread.** Every request is
   queued with `QTimer.singleShot(0, ...)` and runs from the event loop:
   never from inside a core signal handler, a worker thread, or a slot of a
   page that the swap is about to delete (the Hardware tab's own buttons).
   `MicroManagerGUI.load_configuration(path)` is the only caller of
   `set_core`; it runs preflight, decides, swaps if needed, loads, and
   reports.
3. **Nothing can reach a dead object.** Every slot is cut from the old core
   before any widget is deleted. The old core is kept alive, empty, until the
   old widgets are really gone. A stray timer or late callback therefore
   hits a valid, empty core, never freed memory.
4. **Deletion and GC happen at a moment we choose,** never during a paint.
   The real app has already crashed once from Python's GC destroying widgets
   mid-paint (see the GC note in `tests/conftest.py`).
5. **Hardware is opened by one core at a time.** The old core releases its
   devices before the new core loads any, because serial ports and cameras
   are exclusive.

### 6.2 `MicroManagerGUI.set_core(make_core, *, carry_over)`

Each step finishes before the next one starts. A busy overlay covers the
window throughout.

1. **Quiesce, or refuse.** Refuse if an MDA, pixel calibration, autofocus or
   white balance is running. If the MDA is only finishing a cancel, wait for
   `sequenceFinished`, reusing `closeEvent`'s `_close_pending` mechanism.
   Stop live mode (`stopSequenceAcquisition`). Ask only if something would be
   lost: open viewers holding unsaved snaps (the Acquire page is rebuilt), or
   unsaved hardware or configuration edits when not carrying them over.
2. **Snapshot.** Save the current tab, the Acquire layout
   (`store_session_layout`), and the Hardware and Configurations dirty flags.
   With `carry_over`, also save the current system as text: the same model
   dump that Save writes.
3. **Detach the old core.** Call `old.events.disconnect()`,
   `old.mda.events.disconnect()` and `old.events.property_callbacks.clear()`.
   psygnal's `SignalGroup.disconnect()` with no argument removes every slot,
   and `devicePropertyChanged` callbacks go through `propertyChanged`. That
   one step cuts every connection anyone made (window, pages,
   pymmcore-widgets, viewers) without listing them.
4. **Release the hardware.** Call `old.unloadAllDevices()` on the GUI thread.
5. **Tear down the old UI.** Close viewers through their normal close path,
   so ndv/vispy release their GL resources the way a user-closed tab does.
   Then hide the core pages and the MDA status widget, `setParent(None)`,
   and `deleteLater()`. Flush with
   `QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)`
   (`processEvents()` does not run deferred deletes), then `gc.collect()`.
   This is the controlled point from principle 4.
6. **Drop the old core.** Release the window's reference and check, through
   a `weakref`, that the object is gone. A survivor is logged as a leak, not a
   crash: it is already empty and disconnected.
7. **Create and install the new core.** It is created only now, so two cores
   (and UniMMCore's sequence buffer) never exist side by side. Point the
   global instance at it: `_array_viewer` and the console read
   `CMMCorePlus.instance()`. There is no public setter; `startup_core`
   already uses the private `_mmcore_plus._instance` (§10, item 4).
8. **Build the new UI.** This means factoring the core-bound half of
   `MicroManagerGUI.__init__` into `_build_core_ui(core)`. The swap then
   runs the same constructors as every launch, not a second code path. Then
   restore the layout and tab, re-push `mmcore` into the console namespace,
   and restore the dirty flags.
9. **Load.** Load the carried-over text or the requested cfg, with the
   tolerant load (§4.3). If loading fails, the new core stays up, empty, with
   the error shown. It is not rolled back, because a rollback would be a
   second swap.

### 6.3 Cost

- The real cost is hardware: the old devices are unloaded and the new core
  initializes devices again. When the swap comes with loading a *new*
  config, that config would be initialized anyway, so the swap adds only
  unload time. Only **Load** of a folder with hardware already loaded pays to
  re-initialize the same devices. Two `CMMCore` instances can't share a
  device handle, so this can't be avoided.
- The UI part costs about building three pages at startup. Measure it in
  phase 4 with the demo config.
- The window shell, Installation page, menus and notifications are not
  rebuilt, so there is no window flash and window geometry is kept.

### 6.4 Fallback if in-place proves fragile

If the swap stress test (§13) shows crashes we can't fix, relaunch the
process instead. Write the current system to a temp cfg, then `os.execv`
pymm with `--py-devices`, `-c` and `-l`. That is crash-proof, but slower,
and it loses open viewers, so it is the fallback, not the first choice.

### 6.5 Making it visible

The status bar shows a small permanent "Python devices" chip while on
UniMMCore. Its tooltip names the folder. The About dialog names the core
class. Sentry gets a `core` tag.

## 7. Hardware tab

**The Python devices row** is always visible, at the bottom of the left pane,
below the Available devices table:

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

- The path is empty by default. It is filled from `--py-devices`, or from
  the cfg's `#@PyDevicesPath` after a load.
- **Browse…** uses `QFileDialog.getExistingDirectory` and only fills the
  field.
- **Load** scans (§3), shows the problems dialog if needed, swaps to
  UniMMCore if needed (with carry-over, so a half-built config survives), and
  adds the devices to Available devices. They show their module as the
  library and a small "Py" badge, and the existing filter covers them.
- Clearing the field and pressing **Load** turns Python devices off (§2).
- In the standalone app the row is shown **disabled**. Its tooltip reads:
  "Python devices need pyMM run from source (`pip install pymmcore-gui`)".
- The active folder is **owned by the window**, not the page, because the
  page is rebuilt on a swap and has to refill its field from it.
- `_confirm_python_devices_dropped` and its "load anyway without them" dialog
  are removed. Source runs now swap instead, and the standalone app uses §9.

## 8. Saving

There are **two independent save paths** in pymmcore-plus, and the gui uses
the one that doesn't handle Python devices yet:

1. **`UniMMCore.saveSystemConfiguration(path)`** (in
   `experimental/unicore/core/_config.py`, **already on plus `cite`**). It
   writes the live core's state, with Python devices as `#py` lines.
2. **`pymmcore_plus.model.Microscope.save(path)`** is what the Hardware tab
   uses (`HardwareSetupPage.save_to`), and it also uses
   `Microscope.create_from_core` to *display* the installed devices. On plus
   `cite` the model reads devices through unbound `CMMCorePlus` methods,
   which bypass UniMMCore's overrides. A Python device therefore comes back
   with no library, class, properties or role. It would show up broken in the
   Installed devices table and be saved wrong or not at all.

`e303a71` (on plus `cite-unicore`) fixes path 2: the model calls the core's
own methods and writes Python devices with `#py` syntax. `6e582a5` adds
Python hubs' peripherals. Simply switching Save to path 1 on UniMMCore would
not be enough, because the tab still has to *show* Python devices through the
model. Merging the fixes into plus `cite` is therefore required (§10.1).

On top of that, the gui:

- adds `#@PyDevicesPath` after `Microscope.save`, next to where it already
  writes the channel comments. `write_channel_comments` must keep that line;
  add a test.
- warns before **saving over a file whose devices were skipped this
  session** (failed import, failed init, standalone app). Save regenerates
  the file from the live system, so those devices would silently disappear
  from it. The warning reads: "Saving over `scope.cfg` removes 2 Python
  devices that didn't load (Laser, Filter). Save anyway, or Save As…?". The
  same applies to a standard-core save over a `#py` file.

## 9. Standalone app

`unicore_supported() = not sys.frozen`, kept from `cite-unicore`. In the
frozen app:

- `--py-devices` is refused (§5.1) and the row is disabled (§7). Nothing
  imports `pymmcore_plus.experimental.unicore`, and `is_unicore` checks
  `sys.modules` so it never triggers the import.
- A `#py` config loads its C++ part through the native loader. The message
  reads: "`scope.cfg` declares Python devices (Laser, Filter), which need pyMM
  run from source (`pip install pymmcore-gui`). The rest of the configuration
  was loaded."
- The save warning in §8 applies.

## 10. pymmcore-plus work

1. **Required. Done 2026-11-15:** plus `cite` was fast-forwarded to
   `cite-unicore` (`6e582a5`) and pushed, and the gui's `uv.lock` was relocked
   to `6e582a5`.
2. **Recommended.** `UniMMCore.loadSystemConfiguration(path, *,
   skip_devices=())` (§4.3).
3. **Recommended.** In `_get_py_device_instance`, keep the original
   `ModuleNotFoundError` (and `.name`) when the failure is a *dependency* of
   the module rather than the module itself.
4. **Optional.** A public way to replace the global instance
   (`CMMCorePlus.set_instance(core)`), so the gui stops reading
   `_mmcore_plus._instance`.

## 11. From `cite-unicore`: kept and dropped

<!-- markdownlint-disable MD013 -->
| `cite-unicore` | This plan |
| --- | --- |
| fixed folder `USER_DATA_DIR/devices` + `MMGUI_PY_DEVICES_DIR` | **dropped**: user-chosen folder (flag / path field / cfg line) |
| core chosen only at startup | **dropped**: runtime swap (§6) |
| AST scan, base-name table | **dropped**: import scan (§3.3) |
| file watcher | **dropped**: explicit Load |
| "Python Devices…" button that opens the folder | **replaced** by the path row |
| frozen-app guard (`unicore_supported`, CLI refusal) | **kept** |
| folder appended to `sys.path`; module-name / keyword / clash checks | **kept** (§3.2, §3.4) |
| `is_unicore` via `sys.modules` (no import) | **kept** |
| amber problems banner + Details dialog | **kept** (§3.4) |
| "Py" devices in Available, Python hubs + peripherals | **kept** (needs §10.1) |
| temp uv override pinning plus `cite-unicore` | **dropped** once §10.1 lands |
<!-- markdownlint-enable MD013 -->

## 12. Phases

1. **plus**: §10.1, then §10.2–3. Relock the gui.
2. **`_unicore.py`**: folder activation, import scan with error
   classification, cfg preflight, `#@PyDevicesPath` read/write, tolerant
   load. No Qt; fully unit-tested.
3. **Startup**: `--py-devices`, `create_mmgui(py_devices=)`, preflight-based
   `startup_core`, skipped-devices summary, frozen refusal.
4. **Core swap**: factor `_build_core_ui`, then `set_core` (§6.2) and
   `load_configuration`. Route every load path through it. The swap stress
   test (§13) must pass before phase 5 starts.
5. **Hardware tab**: path row, Load, banner, Py badge, init-failure check on
   Add, save path line and save warnings, the standard-config prompt.
6. **Standalone app**: disabled row, messages, a `test_bundle` case.
7. **Docs**: user guide (writing a device file, installing its dependencies
   into pyMM's environment, the cfg format, what Load runs), and
   `docs/architecture/UNICORE.md` for the swap.

## 13. Tests

- **Scan.** Valid device, `_helper.py` skipped, a class from another module
  not listed, syntax error, missing dependency (the message names the package
  and `sys.executable`), exception at import, name clash, bad file name, and
  re-Load picking up an edit.
- **Preflight.** Folder precedence (flag > cfg > active), relative
  `#@PyDevicesPath`, missing folder, hub skip cascading to peripherals,
  nothing resolvable → standard core.
- **Tolerant load.** Skipped devices' lines (properties, presets, roles,
  `Parent`) are removed. The rest loads. `systemConfigurationFile()` is the
  real path.
- **Init failure.** A device that raises in `initialize()` is unloaded and
  reported. Presets left empty by it are removed.
- **Swap behaviour.** standard → Uni → standard with carry-over keeps
  devices, groups, pixel sizes and the dirty flags. The global instance
  follows the swap. The old core's devices are unloaded before the new core
  loads. The swap is refused during an MDA. The standard-config prompt
  defaults to "Keep UniMMCore".
- **Swap stress (gate for phase 4).** Run 20 alternating swaps with the demo
  config loaded and a viewer open, with **automatic GC enabled for this
  test** (`conftest.py` turns it off) and run natively, not offscreen. After
  each swap check that the old core's `weakref` is dead, that the exception
  log is empty, and that emitting on a kept reference to the old core's
  events reaches nothing. Also run the swap from a Hardware-tab button click,
  to exercise the deferred call.
- **Save.** Round-trip of `#py` lines plus `#@PyDevicesPath`. The channel
  comments and the path line coexist. The skipped-devices overwrite warning
  appears.
- **CLI.** `--py-devices` / `--py-devs` validation, and frozen refusal
  (monkeypatch `sys.frozen`).
- **Smoke test by hand on UniMMCore** (pymmcore-widgets may call C++-only
  APIs): snap, live, MDA with a Python camera, Stage Explorer with a Python XY
  stage, pixel calibration, Configurations page editing a Python device's
  presets.

## 14. Decided

All earlier open questions are settled (2026-11-15):

1. **One device folder only.** No `os.pathsep`-separated list.
2. **The folder is not remembered in Settings** across launches. Without
   `--py-devices` pyMM starts on the standard core. The cfg's
   `#@PyDevicesPath` line restores the folder when that config is reopened.
