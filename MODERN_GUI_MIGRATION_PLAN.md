# Replace the legacy GUI with the modern GUI

## Implementation status

Implemented locally on branch `modern-gui-only` on 2026-10-02. Changes remain
uncommitted. The sections below retain the original handoff instructions and
inspection baseline for reference.

- The modern window is now the public `MicroManagerGUI` in `_main_window.py`;
  `MainWindow` is an alias. Python and CLI launches use the same implementation.
- All 30 modern source files were moved. Legacy window/viewer/toolbars/stage-grid
  implementations, the unused pygfx preview, and `--old` were removed.
- All 176 modern GUI test functions and seven window-settings test functions
  were retained. Legacy menu/action-window tests were replaced by modern panel
  and window integration tests; the always-skipped pygfx test was removed.
- Shared viewer helpers have identical implementation ASTs. Of the 30 moved
  files, 27 retain identical implementation ASTs after excluding imports and
  docstrings. The window additionally sets its resource icon and exposes an
  accurate non-optional core type. Viewer cleanup now also connects to owner
  destruction so runner callbacks are disconnected before Qt deletes children.
  Acquisition-page lifecycle fixes keep inactive cached MDA/stage editors
  parented to the page and stop the topology timer/remove splitter event filters
  during shutdown. Regression tests confirmed the previous ownership/timer gaps.
- Settings keys, versions, dock identifiers, dependency versions, and the
  pre-existing `uv.lock` contents were preserved. Passive legacy settings remain
  loadable; modern-only, legacy-only, and mixed file round trips are tested.
- The user had already deleted the acquisition-reopen plan and relocated this
  plan into `TEMP_MD` before implementation. That deletion was preserved;
  this plan was moved back to the root when removing the runtime package.

Validation completed:

- Baseline: 426 passed, four skipped; Ruff, mypy, and Pyright passed.
- Migrated full suite: 416 passed, three skipped. The lower count reflects
  retired legacy tests. The macOS streaming skip predates this migration.
- Additional stage-factory and built-bundle checks: two passed.
- Parallel-crash follow-up: seven targeted lifecycle/layout tests passed, then
  two consecutive full `pytest -n auto --max-worker-restart=0` runs each passed
  with 419 passed and two skipped. Twenty additional layout-and-deletion cases passed
  across four workers. The exact reported native worker crash did not reproduce;
  the fixes address demonstrated lifecycle gaps, rather than a proven native
  crash stack. A preview test double also gained the required `detach()` method
  after parallel execution exposed its delayed destruction callback.
- Fresh-process imports and CLI help/version checks passed.
- Ruff checks/formatting, mypy, Pyright, configured Markdown lint, spelling,
  TOML/YAML validation, project metadata validation, whitespace, file-size,
  and `git diff --check` checks passed. Pyright was run with
  `--pythonpath .venv/bin/python` to select the installed project environment.
- The macOS bundle built and passed its existing launch smoke test. An ordinary
  launch with an explicit test configuration and telemetry disabled also stayed
  running, without pytest-only startup flags or startup errors in its log.

Windows, PySide6, and other Python versions still require the existing CI matrix;
those environments were not available for local verification. No branch was
pushed and no release was published.

## Objective and priority

Make the GUI implemented on this branch the only built-in application GUI.
Remove the legacy window and its duplicate controllers, and incorporate the
modern implementation into a package structure close to `main`.

Priority order:

1. Preserve every implemented modern GUI capability and regression fix.
2. Preserve the established public launch API and reusable infrastructure.
3. Keep familiar file locations from `main` where they fit the modern design.
4. Remove duplicate implementations and obsolete launch paths.

This is an implementation handoff, not an instruction to redesign the UI.
Move working code and consolidate ownership before attempting simplifications.
Do not rebuild modern features using the older implementations.

## Inspection baseline

Prepared on 2026-10-02 from these local Git references:

<!-- markdownlint-disable MD013 -->

| Reference | Commit |
| --- | --- |
| Current branch, `cite` | `34618677d07d8d4aef5995b51cac0e5fda81d956` |
| Local `main` | `1418f603dd91eebd2f97d3f7801a4e25f33d8847` |
| Local remote-tracking `origin/main` | `4a17f3151418800a85a38ed80df1802a202149ee` |

<!-- markdownlint-enable MD013 -->

These are local snapshots; no remote fetch was performed. The local `main`
and `origin/main` differ. Use their structure as a reference, while using the
current working branch as the source of feature implementations. Do not reset
files to either main reference or merge main as part of this migration.

The working tree already has a modified `uv.lock`. Preserve that change.
This planning task changes no application code and runs no application tests.
An implementing agent must establish its own test baseline before editing.
No applicable `AGENTS.md` was found during inspection; check again when
starting.

### What is currently duplicated or connected

- `_main_window.py` contains the legacy `MicroManagerGUI`.
- `_modern_gui/_main_win.py` contains the modern `MainWindow`.
- `_cli.py` defaults to `pymmcore_gui._modern_gui.MainWindow`, but `run --old`
  selects the legacy window.
- `_app.create_mmgui()` defaults to the legacy `MicroManagerGUI`; the package
  also exports that legacy class. CLI and Python launches therefore disagree.
- `_ndv_viewers.py` contains the legacy `NDVViewersManager`, but also three
  helpers imported by the modern viewer manager. It cannot simply be deleted.
- `widgets/_toolbars.py` contains legacy optical-config and shutter toolbars;
  `_modern_gui/_acquire_toolbar.py` supplies their modern counterparts.
- `widgets/_stage_control.py` supplies the old all-device grid;
  `_modern_gui/_acquire_stages.py` supplies the modern per-device dock panel.
- Shared MDA, calibration, stage-explorer, channel-table, and array-viewer code
  imports modern theme or control code. Modern functionality extends well beyond
  the `_modern_gui` directory.
- `_modern_gui/_panels.py` still imports property-browser and exception-log
  factories from `actions/widget_actions.py`.
- `Settings.window` and `Settings.modern_window` contain incompatible dock-state
  formats. Modern dock names and panel keys are persisted identifiers.

## Required end state

### Application and public API

- `pymmcore_gui.MicroManagerGUI` is the modern window, implemented in
  `pymmcore_gui/_main_window.py`.
- Rename the modern class to `MicroManagerGUI`. Keep
  `MainWindow = MicroManagerGUI` in that module as a lightweight alias for the
  newer name; both names identify the same class and implementation.
- `create_mmgui()` with no `window_cls` launches that class.
- `mmgui`, `mmgui run`, `python -m pymmcore_gui`, and the bundled executable all
  reach that same default implementation.
- Remove `--old`. Passing it must fail as an unknown CLI option, with no launch.
- Keep `mmgui layouts`, `--layout/-l`, `--config/-c`, `--demo-config`,
  `--no-telemetry`, `--version`, and the settings commands.
- Keep the existing `create_mmgui()` parameters, including the custom
  `window_cls` extension point and dotted-string resolution. Custom windows
  supplied by callers are not a second built-in GUI.
- Retain public exports `create_mmgui`, `MicroManagerGUI`, `ActionInfo`,
  `CoreAction`, `WidgetAction`, and `__version__`.
- Remove the private `_modern_gui` package after moving its implementations.
  Update all project imports and dotted strings. Do not retain a parallel tree
  of forwarding modules just to preserve private import paths.
- Keep the modern window's Qt `objectName`, `pyMMGUI`. The Python class rename
  does not justify changing Qt identities or persistence keys.

### Compatibility boundaries

The old window's menu and toolbar dictionaries, `get_widget()`, `get_action()`,
and automatic insertion of registered widget actions into a Window menu are
legacy extension behavior. Do not recreate the old window controller to retain
them. Document their removal and point custom-panel authors to `PanelInfo` and
`PANELS`. Retaining the action registry API does not imply retaining its old
automatic menu integration.

Preserve modern APIs such as `window.acquire`, its MDA editor accessors, viewer
manager, and console namespace variables. Preserve constructor parameters and
Qt signals on moved modern widgets unless an actual conflict requires change.

## Target structure and complete move map

Paths in the following table are relative to `src/pymmcore_gui/`. Entries ending
in `/` mean move the complete package, including its `__init__.py` and children.

Keep existing root infrastructure, `actions/`, `_qt/`, `widgets/`, and
`resources/`, as on main. Put page widgets in `widgets/`, with the theme and
viewer/ROI controllers at the root. Preserve the hardware page's existing
internal package boundaries rather than flattening its several panes.

<!-- markdownlint-disable MD013 -->

| Current implementation | Final location | Action |
| --- | --- | --- |
| `_modern_gui/_main_win.py` | `_main_window.py` | Replace legacy module; rename class, preserve modern bodies |
| `_modern_gui/_theme/` | `_theme/` | Move entire theme package unchanged in behavior |
| `_modern_gui/_acquire.py` | `widgets/_acquire.py` | Move modern Acquire page |
| `_modern_gui/_acquire_viewers.py` | `_ndv_viewers.py` | Replace legacy manager, retaining shared helpers |
| `_modern_gui/_camera_roi_sync.py` | `_camera_roi_sync.py` | Move modern ROI controller |
| `_modern_gui/_acquire_toolbar.py` | `widgets/_toolbars.py` | Replace legacy toolbar implementations |
| `_modern_gui/_acquire_stages.py` | `widgets/_stage_control.py` | Replace legacy grid with modern `StagesPanel` |
| `_modern_gui/_acquire_presets.py` | `widgets/_acquire_presets.py` | Move modern group/preset selector |
| `_modern_gui/_panels.py` | `widgets/_panels.py` | Move panel registry and factories |
| `_modern_gui/_configurations.py` | `widgets/_configurations.py` | Move configuration page |
| `_modern_gui/_hardware/` | `widgets/_hardware/` | Move page, panes, setup pane, peripherals, and exports |
| `_modern_gui/_installation.py` | `widgets/_installation.py` | Move installation page and release dialog |
| `_modern_gui/_mda_status.py` | `widgets/_mda_status.py` | Move acquisition status widget |
| `_modern_gui/_preferences.py` | `widgets/_preferences.py` | Move preferences dialog |
| `_modern_gui/_startup.py` | `widgets/_startup.py` | Move startup dialog and choice types |
| `_modern_gui/_busy.py` | `widgets/_busy.py` | Move busy overlay and helpers |
| `_modern_gui/_sidebar.py` | `widgets/_sidebar.py` | Move page sidebar primitives |
| `_modern_gui/_tab_page.py` | `widgets/_tab_page.py` | Move shared page shell |
| `_modern_gui/_toolbar.py` | `widgets/_tab_toolbar.py` | Move page toolbar strip; distinguish it from acquisition controls |
| `_modern_gui/__init__.py` | No replacement package | Remove once canonical exports/imports work |

<!-- markdownlint-enable MD013 -->

The theme package includes `__init__.py`, `_dark.py`, `_light.py`, `_fonts.py`,
`_qt.py`, `_scaled_view.py`, `_style.py`, and `_types.py`. The hardware package
includes `__init__.py`, `_page.py`, `_panes.py`, `_setup_pane.py`, and
`_peripherals.py`. Preserve all of them.

Retain these branch-added implementations in their current locations:

- `_array_viewer.py`, `_channel_luts.py`, `_acquisition_loader.py`,
  `_mda_export.py`, `_ome_tiff_wrapper.py`, `_ome_zarr_wrapper.py`.
- `_layouts.py`, `_light_sources.py`, and the complete `_pixel_calibration/`
  package.
- `widgets/_mda_widget.py`, `_active_channel_table.py`, `_stage_explorer.py`,
  `_pixel_configuration.py`, and `_pixel_calibration_panel.py`.
- `widgets/image_preview/_preview_base.py` and `_ndv_preview.py`.
- `_app.py`, `_cli.py`, `_settings.py`, `_notification_manager.py`,
  `_sentry.py`,
  `_utils.py`, `_qt/`, and the shared console, exception-log, about, and
  notification widgets, with necessary import/API updates.

Move the three `TEMP_MD` documents outside the runtime package:

<!-- markdownlint-disable MD013 -->

| Current basename under `_modern_gui/TEMP_MD/` | Destination |
| --- | --- |
| `DATA_SAVING.md` | `docs/architecture/DATA_SAVING.md` |
| `PIXEL_CALIBRATION.md` | `docs/architecture/PIXEL_CALIBRATION.md` |
| `ACQUISITION_REOPEN_AND_REUSE_MDA.md` | `docs/architecture/ACQUISITION_REOPEN_AND_REUSE_MDA.md` |

<!-- markdownlint-enable MD013 -->

Create `docs/architecture/` during implementation; it does not currently exist.
Update source references in these documents using the move map. The acquisition
reopen document describes an earlier implementation plan: mark it as historical
and link to the implemented code/tests, so it is not mistaken for unfinished
work. Keep this migration plan at the repository root.

## Mandatory feature-preservation checklist

Each row is a migration acceptance requirement. Existing tests are the primary
specification; update import paths without weakening their assertions.

<!-- markdownlint-disable MD013 -->

| Area | Behavior that must survive | Existing evidence |
| --- | --- | --- |
| Startup | Config/layout chooser, recent configs, cancellation, saved appearance before dialogs, demo and explicit config paths, visible load failures, correct initial mode | `test_startup.py`, `test_app.py`, startup tests in `test_new_gui.py` |
| Main shell | Installation, Hardware Setup, Configurations, Acquire; theme toggle, zoom shortcuts, notification bell, exception reporting, OpenGL first-viewer behavior | `test_new_gui.py`, `test_notification_manager.py` |
| Installation | Lazy install widget, release selection/cancel, active-install switching, unloading devices before active uninstall, errors and missing-install routing | installation tests in `test_new_gui.py` |
| Hardware | Devices, setup/peripherals, core configuration loading, save workflow, pending-add cancellation, unsaved-change prompts | hardware tests in `test_new_gui.py` |
| Configuration editing | Groups/presets and pixel configurations, selected-tab commit, independent dirty state, save-file cancel before core mutation, reentrancy protection and event coalescing | configuration tests in `test_new_gui.py` |
| Pixel calibration | Resolution-specific capture settings, selectable camera/stage, affine/pixel-size fitting, diagnostics, cancellation, rejected-result protection, restoration of hardware state, commit rollback | `test_pixel_calibration.py`, `test_pixel_configuration_calibration.py` |
| Acquisition layout | Registry-driven buttons; lazy tools; hide/customize; dock pin/move/autohide; reset; named layouts and Last Session; restored widths and bounded repair/settling | `test_layouts.py`, `test_new_gui_settings.py`, layout tests in `test_new_gui.py` |
| Stage controls | Both XYZ and per-device presentations; in-place switching; add/remove and restore selected devices; auto-snap behavior | stage tests in `test_new_gui.py` |
| Stage Explorer | Lazy creation, positions sent to MDA, acquisition lock, pixel-size-dependent geometry refresh, stop/cleanup | explorer tests in `test_new_gui.py`, legacy lifecycle assertions to port |
| Snap/live controls | Preview created before capture, selected-row settings, exposure/intensity updates while live, channels, shutters, auto-shutter behavior | acquisition-toolbar/channel tests in `test_new_gui.py` |
| Camera ROI | Bidirectional widget/viewer synchronization, correct ROI-session setup and teardown, live restart, auto-snap, unused ROI discarded, reopened data excluded from live-camera observations | ROI tests in `test_new_gui.py`, `test_acquisition_reopen.py` |
| MDA editor | Both collapsible and topbar presentations; switch without losing work; sequence round trips; positions, grids, autofocus offsets, time/z/channel setup; saving/execution controls | MDA tests in `test_new_gui.py` |
| Channel/light sources | Explicit cfg-comment declarations; numeric property selection; active-channel capture settings; sequence metadata precedence; stale/malformed declarations ignored; complete cfg round trip | channel/light-source tests in `test_new_gui.py` |
| Configuration saves | Preserve declared light sources when requested; discard orphaned declarations; save destination chosen before commits | configuration/light-source tests in `test_new_gui.py` |
| Acquisition execution | Memory and disk outputs, scratch limits/spill directory, progress overlays, pause/cancel, editor locks, status display and polling recovery | `test_mda_status.py`, execution tests in `test_new_gui.py` |
| Viewer presentation | Nested viewer dock manager; tab/split/close without disturbing tools; LUT recall; composite channels; histogram; axis rotation, scales, crosshair and ROI controls | `test_array_viewer.py`, `test_channel_luts.py`, `test_ndv_preview.py`, viewer tests in `test_new_gui.py` |
| Live MDA data | Sink-backed data, GUI-thread signal delivery, growing dimensions, flattened position/grid following, frame metadata captured even with follow disabled | `test_ndv_viewers.py`, viewer tests in `test_new_gui.py` |
| Data lifetime | Closing a viewer releases its own sink; closing an old viewer preserves a newer run; running sink release is deferred; stream callbacks and loader handles cleaned up | lifetime tests in `test_new_gui.py`, `test_acquisition_reopen.py` |
| Saving/export | OME-TIFF and OME-Zarr, supported TIFF layouts/extensions, multiposition output, partial/unbounded acquisitions, metadata, overwrite and cancellation semantics | `test_mda_export.py`, `test_ome_wrappers.py`, saving tests in `test_new_gui.py` |
| Reopening/reuse | Supported dropped datasets, queued background reads, separate tabs, meaningful titles, file-handle release, Re-use MDA confirmation/locking, safe save destination reuse | `test_acquisition_loader.py`, `test_acquisition_reopen.py` |
| Console | Lazy kernel, actual owning core, `window`, `acquire`, `mdawidget`/`mda_widget`, `mmc`/`core`/`mmcore`, runner; history thread and kernel cleanup | console tests in `test_new_gui.py` |
| Safe close | Decline cancel, cancel-and-wait, stopping/force-quit path, unsaved configuration prompts, calibration lock, viewer/explorer/live cleanup | close tests in `test_new_gui.py` |
| Distribution | PyQt/PySide wrappers, resource icons, bundled console/startup, DLL handling, READY signaling and graceful test process shutdown | `test_bundle.py`, existing CI matrix |

<!-- markdownlint-enable MD013 -->

“Preserve” means the same current capability, including existing limitations.
For example, the saving documentation records an RGB/RGBA gap; this refactor
does not promise to add that feature or alter writer formats.

## Implementation phases

### Phase 1: Establish the preservation baseline

1. Read this plan, applicable repository instructions, and the architecture
   documents before editing. Inspect any changes since the recorded commit.
2. Record `git status --short` and review existing modifications. Do not amend
   the pre-existing lockfile change or other user work.
3. Inventory imports, string-based lookups, tests, and serialized identifiers.
   Search for `_modern_gui`, `MicroManagerGUI`, `MainWindow`,
   `NDVViewersManager`, `--old`, `window_cls`, and the three replaced modules.
4. Run the current test suite using the existing development environment and
   Micro-Manager test adapters. Record failures, skips, and platform issues
   before changing code. Do not treat an unrun suite as a passing baseline.
5. Record existing modern test functions and parameterization. At inspection,
   `test_new_gui.py` has 176 `test_` function definitions and
   `test_new_gui_settings.py` has seven. These are definitions, not executed
   case counts. A move must not silently discard tests.

Exit condition: a baseline and feature inventory exist, and all planned
deletions have identified consumers or an explicit absence of consumers.

### Phase 2: Cut over the launch paths and public window name

Perform the window replacement and default-launch changes together so there
is no checkpoint where the public API points to the wrong implementation.

1. Replace `_main_window.py` with the modern window implementation. Rename
   `MainWindow` to `MicroManagerGUI`, add the alias described above, and retain
   every modern helper class and function from `_main_win.py`.
2. Preserve `RESOURCES` and `ICON` at `_main_window.py` for `_app.py` to import.
   Compute paths relative to the root package, not the former modern directory.
   Set the window icon using that same resource if direct construction needs it.
3. Update `__init__.py` exports and `_app.py` so the default window is modern.
   Preserve the Windows DLL preload and `PYMM_SIGNALS_BACKEND` setup before
   GUI/core imports.
4. Remove `_cli.run.old` and its branching. Call `create_mmgui()` through the
   default API rather than another hard-coded modern-class string.
5. Remove `_check_layout(..., old=...)` and its legacy warning branch; preserve
   the unknown-name warning and fallback behavior.
6. Preserve the modern startup hook, configuration-load error dialog,
   `on_startup_configuration_loaded()`, delayed `restore_state()`, and
   `mm_config=False` semantics. Explicit config skips the chooser; a layout
   flag alone preselects the chooser; cancel exits before constructing a window.
7. Keep validation and string resolution for caller-supplied `window_cls`.
   Forward the `layout` keyword only to windows supporting the established
   startup/layout hook. Do not require every custom window's `restore_state()`
   to accept modern-only keywords.
8. `_decide_configuration()` and `LoadConfigDialog` currently provide fallback
   startup behavior for windows with no modern startup hook. Retain them only
   for that existing custom-window extension behavior, with updated comments.
   They must never select or instantiate a legacy application window. The
   built-in window always uses `StartupDialog`.

Update launch tests at this point. In particular,
`test_failed_startup_config_load_shows_a_dialog` currently relies on the old
default auto-load flow. Rewrite it to make the modern startup prompt select the
missing file; retain the assertions about the warning and visible error.

Exit condition: direct Python construction, default Python launch, CLI launch,
and canonical dotted-string launch all use the modern class; no built-in legacy
class remains; `--old` cannot launch anything.

### Phase 3: Move modern modules into the target structure

1. Apply the complete move map. Prefer moves and narrowly scoped import edits,
   not rewrites of widget bodies. For destination collisions, save the needed
   legacy helpers before replacing the module.
2. Update imports according to each destination, not a global prefix
   substitution. Former modern siblings are now split across the root and
   `widgets/`; `_toolbar` specifically becomes `_tab_toolbar`.
3. Update shared code outside the former modern package, especially:
   `_array_viewer.py`, `_app.py`, `widgets/_mda_widget.py`,
   `_active_channel_table.py`, `_stage_explorer.py`, and
   `_pixel_calibration_panel.py`.
4. Keep `widgets/__init__.py` lightweight. The current modern package uses lazy
   loading to avoid cycles: widget modules import the theme while the main
   window imports widgets. Do not replace that with eager page re-exports.
5. Keep widget factory imports local where they are currently local. Preserve
   lazy IPython/qtconsole and installation-widget construction.
6. Update module references in `TYPE_CHECKING` blocks, docstrings, monkeypatch
   strings, `patch()` targets, CLI tests, and architecture documentation.
7. Remove `_modern_gui/__init__.py` and the now-empty directory only after all
   implementations and documents have been transferred.

Exit condition: source/test imports resolve without `_modern_gui`; all modern
implementation bodies have a mapped destination; cold package, leaf-widget,
theme, and CLI imports work in fresh processes.

### Phase 4: Consolidate viewer, toolbar, stage, and panel ownership

#### Viewer manager

- The final `_ndv_viewers.py` contains `AcquireViewersManager` and its modern
  worker task, record, sink compatibility, release, reopening, and cleanup code.
- Keep the modern class name `AcquireViewersManager`; do not force its richer
  interface into the legacy manager's constructor or signal signatures.
- Transfer `_StreamSignalBridge`, `_extract_scales`, and
  `_add_follow_lock_button` from the original `_ndv_viewers.py` into the final
  module. Remove the modern manager's import of those helpers from itself.
- Remove the legacy `NDVViewersManager` body and its old viewer bookkeeping.
- Preserve `_runner_sink()` and `_release_runner_sink()` compatibility for
  released pymmcore-plus versions. “Legacy” in those helpers concerns the
  dependency API, not the removed GUI.
- Keep data export, acquisition reading, array presentation, and channel LUT
  memory in their existing shared modules; do not fold them into the manager.

#### Toolbars and stage controls

- The final `widgets/_toolbars.py` contains modern `SnapButton`, `LiveButton`,
  `ShuttersBar`, `PanelButtonBar`, and their helpers. Delete `OCToolBar` and
  `ShuttersToolbar`; modern presets/shutters are their retained replacement.
- The final `widgets/_stage_control.py` contains modern `StagesPanel` and its
  helpers. Delete the old `_Group` and `StagesControlWidget` implementation.
- Preserve the separate upstream `XYZStageWidget` option. It and `StagesPanel`
  represent intentionally supported presentations, not duplicate GUIs.
- Update `actions.widget_actions.create_stage_widget()` and its return type to
  construct `StagesPanel`. For this standalone compatibility factory, populate
  it with loaded XY/Z devices through `add_stages()`; the ordinary modern
  Acquire
  per-device factory must still start empty. Use local imports to avoid the
  panel-registry/stage-control import cycle.

#### Panel factories and actions

- Keep one modern Acquire panel registry, `widgets/_panels.py`, with unchanged
  `PanelKey` values, `PanelInfo.dock_name`, stage/MDA kinds, and factories.
- Preserve `actions/` as a public reusable action API. It is not itself another
  GUI; deleting it would break package exports and shared tests unnecessarily.
- Use its existing property-browser and exception-log factories where useful;
  avoid inventing a second registry or generic action/docking framework.
- Update factories referencing replaced modules. Remove old-window-specific
  comments and assumptions, while keeping correct owning-core resolution.
- `_get_core()` and console injection must keep preferring an explicitly
  supplied owning-window core over the global singleton. Preserve this with a
  real distinct-core test, including a nested child widget.
- Keep `pyMMGUI` recognized in factory/console ownership lookup. Recognition of
  `MicroManagerGUI` as a compatibility object name is harmless and does not
  preserve an old window implementation.
- Do not alter the shared MDA base merely because two MDA presentations use it.
  `MemoryMDAWidget` and `TopbarMemoryMDAWidget` are both required functionality.

Exit condition: one built-in window, one viewer manager, one acquisition toolbar
implementation, and one modern panel registry; intentional presentation choices
and dependency compatibility remain supported.

### Phase 5: Preserve settings and serialized state

Use a deliberately conservative persistence strategy for this migration:

1. Keep the JSON section name `modern_window`, its model and field names, and
   settings version `1.0`. Update Python-path descriptions, not serialized keys.
2. Keep `Settings.window`/`WindowSettingsV1` as passive compatibility data for
   existing files. No built-in window reads or writes it. Retain
   `auto_load_last_config` and `fallback_to_demo_config` for settings
   compatibility and custom-window fallback startup.
3. Document those legacy fields as compatibility fields. A small passive schema
   is preferable to a new settings migration or losing existing preferences;
   it does not retain a second GUI or duplicate runtime controllers.
4. Preserve current modern-only behavior when both sections exist. Never feed
   the legacy `dock_manager_state` or `window_state` into the modern dock
   manager.
5. A legacy-only file launches the modern GUI with its normal defaults and
   existing shared preferences/config history. Do not invent a legacy-to-modern
   dock translation in this task.
6. Preserve `pmm_settings.json`, user-data locations, source precedence,
   environment prefixes, forgiving validation, `MMGUI_NO_SETTINGS`, scratch
   preferences, LUT memory, and recent-config behavior.
7. Preserve named layout JSON fields/version, reserved names, ADS object names,
   nested stage-device persistence, and `MMGUI_LAYOUTS_DIR`.
8. Preserve MDA presentation state wherever current code stores it. Do not add
   a new settings field just because `mda_kind` appears in named/session layout
   data rather than the window settings model.

`CONTRIBUTING.md` requires a schema-version bump for renamed/removed settings.
The strategy above avoids those breaking changes. A later settings cleanup is a
separate task requiring a versioned migration and tests, not an implicit part
of removing the GUI.

Exit condition: existing modern preferences/layouts restore identically,
legacy-only and mixed files still load, and modern saves leave passive classic
window state untouched.

### Phase 6: Migrate tests and finish deletion

1. Rename `test_new_gui.py` to `test_gui.py`, and
   `test_new_gui_settings.py` to `test_window_settings.py`. Preserve all tests,
   helpers, fixtures, parametrization, and assertions; initially avoid splitting
   the large GUI file into a new test hierarchy.
2. Keep `test_main_window.py` as focused canonical-window integration coverage.
   Replace its old fixture and old `get_widget()`/menu assumptions. Port useful
   behavior assertions to `window.acquire` and the modern controllers.
3. Preserve or port old lifecycle checks for snap, live streaming, MDA data,
   notifications, save/restore, and stopping Stage Explorer on close. Delete
   assertions whose sole purpose is exercising removed menus/toolbars.
4. Keep the GUI-thread `_StreamSignalBridge` test in `test_ndv_viewers.py`.
   Rewrite its legacy-manager test for the modern constructor, dock manager,
   records, and teardown; do not keep a legacy manager just to pass that test.
5. Update `test_cli.py` to check default/canonical launches, CLI arguments, and
   unknown-option rejection for `--old`. Keep layout warnings/listing tests.
6. Update `test_actions.py`: keep registry validation and owning-core tests;
   replace the automatic-Window-menu test with the modern panel registration
   path. Ensure any injected global registry entry is cleaned up after the test.
7. Keep `test_window_settings.py`'s assertion that passive classic settings are
   untouched. Add file-level compatibility cases for modern-only, legacy-only,
   and mixed settings, with real round trips through the settings reader.
8. Update startup/application tests to exercise `create_mmgui()` without an
   explicit modern `window_cls`; keep explicit class/string override tests too.
9. `test_widgets.py` currently tries to import a nonexistent old pygfx path and
   skips. Remove this obsolete test and the unused
   `widgets/image_preview/_pygfx_image.py` after confirming there are no real
   callers. Keep NDV preview and its base, with their existing tests.
10. Keep the unused branch-added `widgets/_joystick.py` out of this migration's
    cleanup: it is not an alternate GUI or a duplicate of a retained controller.
    Any separate prototype removal should be explicit, not incidental.
11. Update `tests/conftest.py` to initialize `pymmcore_gui._theme`, preserving
    zoom reset, settings/layout isolation, core cleanup, icon mocking, and Qt
    session shutdown. Do not force a QApplication into the first app test.

Search results and test collection must show no dependency on deleted private
paths or legacy runtime classes. Lower test counts must be explained by removal
of specific obsolete tests, not by lost modern coverage or broader skips.

### Phase 7: Documentation, packaging, and verification

1. Update README launch/customization guidance and CONTRIBUTING structure
   references. State that the modern interface is the only built-in GUI; show
   the existing default `create_mmgui()` API and canonical window import.
2. Document removed `--old`, private `_modern_gui` imports, and legacy
   menu/widget
   extension methods. Document the canonical panel registry and preserved
   modern console accessors.
3. Preserve dependency versions and current `cite` source overrides. Do not
   remove PyQt6Ads, ndv/vispy, ome-writers, pymmcore-widgets, qtconsole, psutil,
   or other dependencies solely because the legacy window is gone.
4. Inspect `app/mmgui.spec` and hooks for string/lazy import collection after
   the moves. Modern widget factories and theme imports must be discoverable.
   Add a narrowly required hidden import only if bundle analysis demonstrates
   it is missing.
5. Preserve resources, DLL filtering/preloading, Qt API wrappers, splash
   handling, console redirection, READY output, and graceful test shutdown.
   Do not mix unrelated packaging/crash work into this refactor.
6. Build and smoke-test the bundle on the current platform, then use the
   existing
   macOS/Windows bundle workflow and Python/Qt CI matrix for platform coverage.
   Building an artifact is insufficient without launching it.

## Verification commands and acceptance gates

Commands assume the existing environment is available. Prefer `uv run --no-sync`
for local verification so running checks does not rewrite the user's lockfile.
Install missing prerequisites separately and report environmental failures.
Tests require Micro-Manager test adapters as configured in CI.

### Static and import checks

```sh
uv run --no-sync ruff check --no-fix src tests
uv run --no-sync ruff format --check src tests
uv run --no-sync mypy src tests
uv run --no-sync pyright
```

Use the configured pre-commit hooks as the final repository checks, including
Markdown and manifest validation where applicable. The `just lint` recipe
currently invokes `pre-commit`; confirm tool availability rather than assuming
it follows the newer `prek` dependency name.

Run independent fresh-process imports of:

- `pymmcore_gui` and `pymmcore_gui._main_window`.
- `pymmcore_gui._theme`.
- `pymmcore_gui.widgets._mda_widget` and `_pixel_calibration_panel`.
- `pymmcore_gui._ndv_viewers` and `_camera_roi_sync`.
- `pymmcore_gui._cli`, and `python -m pymmcore_gui --help`/`--version`.

These checks specifically catch circular imports masked by a previously
imported test process. Imports should not require constructing a window or
loading a hardware configuration.

### Focused regression suites after the moves

```sh
uv run --no-sync pytest tests/test_app.py tests/test_cli.py \
  tests/test_startup.py
uv run --no-sync pytest tests/test_main_window.py tests/test_gui.py
uv run --no-sync pytest tests/test_window_settings.py tests/test_settings.py \
  tests/test_layouts.py
uv run --no-sync pytest tests/test_ndv_viewers.py tests/test_ndv_preview.py \
  tests/test_array_viewer.py tests/test_channel_luts.py
uv run --no-sync pytest tests/test_mda_export.py tests/test_ome_wrappers.py \
  tests/test_acquisition_loader.py tests/test_acquisition_reopen.py
uv run --no-sync pytest tests/test_pixel_calibration.py \
  tests/test_pixel_configuration_calibration.py tests/test_mda_status.py
uv run --no-sync pytest
```

Respect the ordered first-application test. Do not diagnose full-suite behavior
using only a process in which a QApplication was created manually beforehand.
Record executed cases and skips. Existing CI describes platform-specific native
shutdown issues; compare with the baseline and investigate new failures without
adding blanket skips or weakening unrelated checks.

### Bundle verification

```sh
uv run --no-sync pyinstaller app/mmgui.spec --clean --noconfirm
uv run --no-sync pytest -v tests/test_bundle.py
```

Use the test's current launch/readiness conventions. Also smoke-test an ordinary
bundle startup without test-only environment flags, since READY alone does not
prove the user-facing startup dialog works.

### Manual demo-session acceptance

Use demo/test hardware and temporary output directories. Check:

1. Launch from Python and CLI; choose/cancel startup; load explicit/demo
   configs;
   confirm failures remain visible when a config is missing.
2. Visit all four modes, edit configurations, cancel a save, save/apply, switch
   installations, and confirm dirty-state prompts.
3. Snap/live with channel settings and shutters; open/close preview; synchronize
   camera ROI through both widget and viewer.
4. Open each lazy tool, switch stage and MDA presentations, customize buttons,
   dock/autohide, save a named layout, close, and restore theme/zoom/layout.
5. Run memory and disk MDA, pause/cancel, inspect status/locks, split/close
   viewers, then run another acquisition to check sink ownership and cleanup.
6. Save/export TIFF and Zarr; reopen/drop them; Re-use MDA; confirm a reused
   saved sequence does not overwrite its original acquisition on the next run.
7. Exercise calibration preview, cancellation, and successful application with
   suitable demo/synthetic capture data; verify geometry refresh and
   restoration.
   Use automated synthetic calibration tests for numerical acceptance rather
   than assuming every demo camera image is calibratable.
8. Open the console and verify owning-core/window/editor variables; close during
   an acquisition and exercise the cancel/wait behavior.

### Final deletion audit

```sh
rg -n '_modern_gui|NDVViewersManager|OCToolBar' \
  src tests app .github README.md CONTRIBUTING.md docs
rg -n 'ShuttersToolbar|StagesControlWidget' \
  src tests app .github README.md CONTRIBUTING.md docs
rg -n -- '--old' src tests README.md CONTRIBUTING.md docs
git diff --check
git status --short
```

Expected matches are limited to deliberate removal documentation and the CLI
test that asserts `--old` is rejected. There must be no source implementation or
test import of a removed runtime class/path. The passive legacy settings model
and custom-window startup fallback are deliberate compatibility exceptions.

## Completion checklist for the implementing agent

- [x] All built-in entry points use the same modern `MicroManagerGUI`.
- [x] `--old`, the old window body, and duplicate viewer/toolbars/stage grid are
      removed.
- [x] Every row of the source move map is complete, including theme/hardware
      children and retained docs; the pre-existing reopen-plan deletion was honored.
- [x] Shared modern acquisition, saving/reopen, ROI, calibration, LUT, and
      layout code remains.
- [x] Every feature-preservation row passes its existing or ported local tests.
- [x] Cold imports and custom-core resolution pass.
- [x] Existing settings and layout data restore without schema or identifier
      changes.
- [x] Modern tests survive relocation; obsolete test removals are explained.
- [x] Static checks, full tests, and current-platform bundle launch pass, or
      specific baseline/environment limitations are documented.
- [ ] Cross-platform CI/bundle results are checked before declaring platform
      coverage complete.
- [x] README, CONTRIBUTING, architecture documents, and customization guidance
      match the final code.
- [x] Pre-existing working-tree changes, especially `uv.lock`, remain intact.

The final implementation report should list removed implementations, new
canonical locations, compatibility decisions, validation results, and any
unverified platform behavior. Do not claim completion solely from successful
file moves or test collection.
