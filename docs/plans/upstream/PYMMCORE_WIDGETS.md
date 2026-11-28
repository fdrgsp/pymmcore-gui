# pymmcore-widgets: upstream PRs

`fdrgsp/pymmcore-widgets@cite` against `pymmcore-plus/pymmcore-widgets@main`,
merge base `b2559077` (2026-11-26): 373 commits, +34k / −3.2k lines, of which
about 19k are source. See [README.md](README.md) for the conventions and the
cross-repository chains.

Every PR here targets `main`. Four of them already exist as your open PRs
there: #522, #529, #531 and #482 (WDG-18, WDG-54, WDG-57, WDG-58). Update those
rather than opening new ones.

## Overwrite, v2 class, or version flag?

### Recommendation

**Overwrite in place, on `main`.** Don't add `MDAWidgetV2` or
`MDAWidget(version=2)`. The one real "v2" is the stage widget.

A script compared every modified file's public classes (methods, parameters and
base classes) between upstream `main` and `cite`. The fork **only adds** API,
with these exceptions:

<!-- markdownlint-disable MD013 -->
| Change on `cite` | Impact | Plan |
| --- | --- | --- |
| `StageWidget(levels=, position_label_below=)` and its "Set as Default" radio removed | **Breaking** | Not ported: see WDG-23 to WDG-25 |
| `PropertyBrowser` is a `QWidget`, no longer a `QDialog` | **Breaking** for callers of `exec()` | WDG-56, flagged in the title |
| `GridPlanWidget` is a `QWidget`, no longer a `QScrollArea` (the tab page scrolls instead) | **Breaking** for callers of the scroll-area API | WDG-40d, flagged in the title |
| `StageMovementButtons(levels=)` removed | Internal | – |
| `MDAButton` removed (`useq_widgets/_positions.py`) | Not exported; the position table's delegate paints that cell | WDG-32b |
| `CoreMDATabs` moved from `mda/_core_mda.py` to `mda/_core_tabs.py` | Now exported from `pymmcore_widgets.mda` | WDG-40a |
| `MarkVisit`, `get_vispy_scene_bounds`, `well_coords_affine` removed | Private module helpers | – |
<!-- markdownlint-enable MD013 -->

So almost everything can land in place without breaking callers. There are two
kinds of PR, both to `main`:

1. **Features and fixes that keep the default look**: small PRs, in whatever
   order their dependencies allow.
2. **Visible changes** (the MDA widget's tabs, cards and footer; the grid and Z
   editor layouts; new defaults; marked *visible* in the summary): also in
   place, but one visible change per PR, each with before/after screenshots and
   a changelog line. Release them together in one minor version, so users meet
   one redesign rather than a trickle of them. Upstream CI already runs the
   downstream napari-micromanager and pymmcore-gui tests, which catch what a
   layout change breaks.

### Why not a version flag or a V2 class

- **A `version=` flag** puts two layouts in one class: every layout method
  branches on it, some attributes exist in one version only, the type of `tabs`
  depends on a runtime argument, and the tests double.
- **A `MDAWidgetV2`** would duplicate about 6k lines (`mda/` plus
  `useq_widgets/`) for a difference that is only layout. The features live in
  the shared sub-editors (channel, position, Z and grid editors, autofocus),
  which both classes would use. The fork already isolates the layout behind
  hooks: `MDASequenceWidget(tab_widget=...)`, `MDAWidget._create_tab_widget()`
  and `MDATabs._build_tabs()`. Another presentation is a subclass of the
  container: that's how `CollapsibleMDAWidget` works.
- **Use a new class** only when the API breaks (StageWidget), when a second
  presentation should exist next to the default one (`CollapsibleMDAWidget`),
  or for genuinely new widgets.

### Per widget

<!-- markdownlint-disable MD013 -->
| Widget | What changed on `cite` | API | Plan |
| --- | --- | --- | --- |
| `MDAWidget`, `MDASequenceWidget`, `MDATabs` | Sub-editor features (below); new layout: icon-on-top tabs, a Settings card, Autofocus and Saving tabs, a footer with Save/Load/Run; axes start off and empty | Additive | Features in place. Layout in place, in four visible PRs (WDG-40a to WDG-40d). Defaults: WDG-42, an issue first |
| `CollapsibleMDAWidget`, `CollapsibleCoreMDATabs` | New | New classes | WDG-41, after WDG-40a to WDG-40c |
| `AutofocusAxis` | Checkable group box; hardware and software modes, every-N, Z search, keep locked | `value()` kept, `plan()` and `setPlan()` added | In place (WDG-36), following the useq and plus chain |
| `ZPlanWidget`, `CoreConnectedZPlanWidget` | Stack graphic, mode radios, boxed controls; navigation and live Z | Additive | Layout in place on top of Talley's #542 (WDG-30). Navigation (WDG-31) |
| `GridPlanWidget` | Mode radios and two columns; bounds fixes | Base class `QScrollArea` → `QWidget` | Fixes first (WDG-27), then the layout (WDG-28). The base class changes with the tab pages (WDG-40d) |
| `PositionTable`, `CoreConnectedPositionTable` | Model/view (speed), Surface Focus, copy Z/AF, HCS lock | `MDAButton` gone (not exported) | In place |
| `ChannelTable`, `CoreConnectedChannelTable` | Light source and intensity columns, per-channel properties, Core Channel combo | Additive | In place. The new columns are hidden by default upstream (on `cite` the light source column starts visible) |
| `StageWidget` | Compass of 8 arrows, a single step, no "Set as Default", wheel Z, polling pause, Enable Editing menu | **Breaking** | Keep its API and port the behavior only (WDG-23). The new look ships as `XYZStageWidget` (WDG-24), which can grow into StageWidget's replacement (WDG-25) |
| `XYZStageWidget` | New | New class | WDG-24 |
| `CameraRoiWidget` | Value round-trip, live-view selection, ROI visibility | Additive | In place (WDG-22) |
| `PropertyBrowser` | Device type toolbar on top; a `QWidget` | Base class `QDialog` → `QWidget` | Toolbar (WDG-55), base class (WDG-56) |
| `PixelConfigurationWidget` | Device type toolbar, dirty tracking, read-only mode, camera hints | Additive (the changed signals are on private classes) | In place (WDG-16, WDG-17, WDG-55) |
| `ConfigGroupsEditor` | Read-only mode, editor fixes, slider editor | Additive | In place (WDG-15, WDG-16) |
| `ConfigWizard` | Roles, labels, devices pages; cleanup and device reload | Additive | Your #522 (WDG-18) |
| `StageExplorer`, `StageViewer` | Speed, memory cap, MDA markers, ROI menu, flip menu | Additive | In place (WDG-43 to WDG-48); flip menu is your #531 (WDG-57) |
| `HCSWizard` and plate views | Wells page, versioned files, Cover Area, automated calibration, small plates | Additive; old files still load | In place (WDG-49 to WDG-53); small plates are your #482 (WDG-58) |
| `InstallWidget`, `CoreLogWidget`, `ShuttersWidget` | Signals, log rotation, icons | Additive | In place |
| `ChannelSettingsTable`, `CustomPlateWidget`, `BusyOverlay`, `ColumnFitter`, `RecordTableModel`, `DeviceButtonToolbar` | New | New | One PR each |
<!-- markdownlint-enable MD013 -->

## Summary

IDs are labels, not an order: follow the Needs column. *Visible* marks a PR
that changes what users see by default.

<!-- markdownlint-disable MD013 -->
| ID | Type | Title | Size | Needs | Pairs with |
| --- | --- | --- | --- | --- | --- |
| **Groundwork** | | | | | |
| WDG-1 | chore | Require Python 3.11 | S | – | – |
| WDG-2 | fix | PySide6: no slot calls on freed widgets | S | – | – |
| WDG-3 | fix | Number cells in data tables render blank | S | – | – |
| WDG-4 | fix | A garbage-collected vispy canvas segfaults | S | – | – |
| WDG-5 | feat | One combo-box row height for every table | S | – | – |
| WDG-6 | feat | No wheel scrolling in combo and spin boxes of editors | S | – | – |
| WDG-7 | feat | `BusyOverlay` with progress | S | – | – |
| WDG-8a | feat | DataTable: move rows | S | – | – |
| WDG-8b | feat | DataTable: lock a column | S | – | – |
| WDG-8c | perf | DataTable: batched fills, shared button icons | S | WDG-7 | – |
| WDG-8d | feat | `ComboColumn` | S | – | – |
| WDG-9 | fix | Follow the camera FOV on camera, binning and mirror changes | M | – | – |
| WDG-10 | feat | Default tiling overlap of 10% | S | issue first | – |
| WDG-11 | fix | Log widget follows the log file through a rotation | S | after #546 | – |
| WDG-12 | feat | InstallWidget signals and cancel | M | – | – |
| WDG-13 | feat | Shutter state as an open or closed aperture | S | after #561 | – |
| WDG-14 | feat | "Enable Editing" menu for fields that move the stage | S | – | – |
| WDG-54 | fix | Save names after multi-position OME-TIFF runs (your #529) | S | – | – |
| **Devices, configuration, channels, camera, stage** | | | | | |
| WDG-15a | feat | Slider editor for preset values with limits | S | – | – |
| WDG-15b | fix | Preset cell editors: first click, closing mid-click, undo | M | – | – |
| WDG-15c | fix | Preset metadata refresh on device update; selection tint | S | – | – |
| WDG-16 | feat | Read-only groups/presets and pixel editors | S | – | – |
| WDG-55 | feat | `DeviceButtonToolbar` in PropertyBrowser and the pixel editor | M | – | – |
| WDG-56 | refactor! | `PropertyBrowser` is a QWidget, not a QDialog | S | issue first | – |
| WDG-17a | feat | PixelConfigurationWidget dirty tracking | S | – | – |
| WDG-17b | refactor | Pixel property table as model/view | M | WDG-17a | – |
| WDG-17c | feat | Warn when a pixel size may be another camera's | S | WDG-9 | – |
| WDG-18 | feat | ConfigWizard pages and device reload (your #522) | M | – | – |
| WDG-19 | feat | Light source helpers | S | – | – |
| WDG-20a | feat | Light source and intensity columns in the MDA channel table | M | WDG-19, 8d | – |
| WDG-20b | feat | Per-channel properties applied during the run | M | WDG-20a | – |
| WDG-20c | feat | "Core Channel" group combo, advanced columns toggle | S | WDG-20a | – |
| WDG-21 | feat | `ChannelSettingsTable` | L | WDG-19, 5 | – |
| WDG-22a | feat | `CameraRoiWidget` value round-trip | S | – | – |
| WDG-22b | feat | `CameraRoiWidget` live-view selection hooks | S | WDG-22a | – |
| WDG-23 | feat | StageWidget: polling pause, wheel Z, halt any device | S | WDG-14 | – |
| WDG-24 | feat | `XYZStageWidget` | M | WDG-23 | – |
| WDG-25 | feat | `XYZStageWidget` with fixed devices, StageWidget deprecated | S | WDG-24, issue first | – |
| **MDA and useq editors** | | | | | |
| WDG-26 | fix | Time plan: zero intervals | S | – | – |
| WDG-27 | fix | Grid bounds: edits, Move, the half-FOV overshoot | S | – | – |
| WDG-28 | feat | Grid editor layout (*visible*) | M | WDG-27 | – |
| WDG-29 | feat | Cover Area points plan | M | USEQ-8a | USEQ-8a |
| WDG-30 | feat | Z editor graphic and layout (*visible*) | M | #542 | – |
| WDG-31 | feat | Z navigation, reference reset, live Z | M | WDG-30, 14 | – |
| WDG-32a | feat | `RecordTableModel` and `RecordTableView` | L | WDG-5 | – |
| WDG-32b | perf | Position table on model/view | M | WDG-32a | – |
| WDG-33a | feat | Copy a position's Z or AF offset to all positions | S | WDG-32b | – |
| WDG-33b | fix | Grid sub-sequence popup: non-modal, wide enough, freed | S | – | – |
| WDG-33c | feat | Sub-sequence popup edits only a grid plan | S | issue first | USEQ-10 |
| WDG-33d | feat | HCS positions locked behind "Unlock" | S | WDG-32b | – |
| WDG-34 | fix | Plate views: themes, styled parents, plate changes | S | – | – |
| WDG-35 | feat | `CustomPlateWidget` | M | – | – |
| WDG-36a | refactor | `AutofocusAxis` as a checkable group box with `plan()` | S | – | – |
| WDG-36b | feat | Autofocus every-N intervals | S | WDG-36a, USEQ-4 | USEQ-4 |
| WDG-36c | feat | Hardware autofocus Z search controls | S | WDG-36a, PLUS-16 | PLUS-16 |
| WDG-36d-1 | feat | Software autofocus mode and method picker | M | WDG-36a, PLUS-19 | USEQ-6, PLUS-19 |
| WDG-36d-2 | feat | `AutofocusSettingsDialog` | L | WDG-36d-1 | – |
| WDG-36d-3 | feat | Test a routine from its settings | S | WDG-36d-2 | PLUS-18b |
| WDG-36e | feat | "Keep locked" | S | WDG-36a, PLUS-20a | USEQ-7b, PLUS-20 |
| WDG-36f | feat | Run refuses autofocus without an axis | S | WDG-36a | – |
| WDG-37a | feat | `ZSurfaceDialog` | L | USEQ-9a | USEQ-9a |
| WDG-37b | feat | Surface Focus in the position table | M | WDG-37a, 36a, USEQ-9b | USEQ-9b |
| WDG-38 | fix | MDA run fixes: live, dialogs, axis order, pause | S | PLUS-10 | – |
| WDG-39 | fix | Plate positions keep z unset; relative Z anchored | S | USEQ-1 | USEQ-1 |
| WDG-40a | feat | MDA Settings card and footer (*visible*) | S | – | – |
| WDG-40b | feat | MDA Saving tab (*visible*) | S | WDG-40a | – |
| WDG-40c | feat | MDA Autofocus tab (*visible*) | S | WDG-40a, WDG-36a | – |
| WDG-40d | feat! | MDA icon-on-top axis tabs; `GridPlanWidget` becomes a QWidget (*visible*) | M | WDG-40c, WDG-28 | – |
| WDG-41 | feat | `CollapsibleMDAWidget` | L | WDG-40a/b/c | – |
| WDG-42 | feat | MDA axes start off and empty (*visible*) | S | issue first | – |
| **Stage Explorer** | | | | | |
| WDG-43 | feat | Map memory cap | M | – | – |
| WDG-44a | perf | Downsample tiles when zoomed out | S | – | – |
| WDG-44b | perf | Draw tiles in view from a pool of images | M | WDG-44a | – |
| WDG-44c | perf | Prepare map images off the GUI thread, in batches | M | WDG-44b | – |
| WDG-44d | perf | Multiscale overview | L | WDG-44c | – |
| WDG-45 | feat | Show MDA positions on the map | M | – | – |
| WDG-46 | feat | ROI tools and `sendToMDARequested` | M | – | – |
| WDG-47 | feat | Stage position overlay | S | – | – |
| WDG-48 | fix | MM affine convention; camera format changes | S | WDG-9 | – |
| WDG-57 | feat | Flip menu in StageViewer (your #531) | S | – | – |
| **HCS** | | | | | |
| WDG-58 | fix | Calibrate plates with fewer than 3 wells (your #482) | S | – | – |
| WDG-49 | fix | Calibration fit, FOV on load, snake order | S | WDG-58 | – |
| WDG-50 | feat | Wells on their own page; wizard sizing and icons | M | WDG-49 | – |
| WDG-51 | feat | Versioned wizard save files | S | – | – |
| WDG-52 | feat | Cover Area on the FOV page | M | WDG-29, USEQ-8b | USEQ-8b |
| WDG-53a | feat | NumPy well detection | M | – | – |
| WDG-53b | feat | Plate survey scan | M | WDG-53a | – |
| WDG-53c | feat | Automated plate calibration tab (experimental) | M (×2) | WDG-53b, 50 | – |
<!-- markdownlint-enable MD013 -->

---

## Groundwork

### WDG-1: require Python 3.11

*chore · S · `f356d1f6`*

useq-schema and pymmcore-plus require 3.11 and use `StrEnum`, so a 3.10 install
can't import its own dependencies. Change the metadata, the ruff target and the
lower CI bound.

### WDG-2: PySide6: no slot calls on freed widgets

*fix · S · `7e2bd64c`*

On PySide6, a connection to a method that overrides a parent's `@Slot` isn't
tied to the receiver. A signal queued from another thread then reaches a freed
widget: a segfault, a hang, or an odd AttributeError. Drop `@Slot` from the
StageExplorer and PixelConfigurationWidget methods that subclasses override.

### WDG-3: number cells in data tables render blank

*fix · S · `6b58f341`*

With `NoButtons`, the `QAbstractSpinBox { border: none; }` style sheet in
`_column_info.py` collapses the edit field to 1 px, so number cells look empty
and can't be edited (Qt 6.4 and 6.12, PyQt6 and PySide6, macOS, Fusion and
Windows styles). Upstream `main` has the same style sheet. Use
`setFrame(False)` instead.

### WDG-4: a garbage-collected vispy canvas segfaults

*fix · S · `eafdbfd8`*

vispy's Qt backend leaves a private GL context current on the main thread, owned
only by the Python canvas. When the garbage collector frees a dead canvas on a
worker thread (e.g. the MDA runner), the next surface destroyed on the main
thread segfaults. `own_vispy_init_context(canvas)` in `_util.py`, called by
`ImageWidget`, the stack viewer and `StageViewer`.

### WDG-5: one combo-box row height for every table

*feat · S · `5a7e1d65`, `4cd0dd04`, `3d9744be`, `ce0a34af`, `130006d0`*

`use_combo_row_height(table)` and `combo_box_height(widget)` keep a table's rows
as tall as a combo box at every font and zoom, instead of Qt's default (38 px at
the GUI's zoom), which stretches cell widgets. Apply it to the MDA, install,
config presets, pixel, property, group preset and well calibration tables.
Setting a time plan no longer selects its last row.

### WDG-6: no wheel scrolling in combo and spin boxes of editors

*feat · S · `df621ef9`*

`disable_wheel_scroll(widget)` and `disable_wheel_scroll_recursive(widget)`
swallow wheel events on combo and spin boxes in the tables and editors, so
scrolling past one no longer changes its value by accident.

### WDG-7: `BusyOverlay` with progress

*feat · S · `c7114a6a`, `9478d6af`, `d8df3de8`*

`BusyOverlay(parent)` and the context manager `busy(overlay, message)` cover a
widget during blocking work. `progress(done, total)` draws a bar under the
message, repainted at each whole percent without processing events. Filling a
DataTable reports each row, and loading MDA settings shows it.

### WDG-8a: DataTable: move rows

*feat · S · `079b14bc`, `1fb94d74`*

Move-up and move-down actions in the toolbar, after a separator.

### WDG-8b: DataTable: lock a column

*feat · S · `1c0dbd08`*

`DataTable.setColumnLocked(key, locked)` disables a column's cell widgets, also
in rows added later and when the column is rebuilt.

### WDG-8c: DataTable: batched fills, shared button icons

*perf · S · `32b2ac55`, `792a1c99`, `7c980509` · needs WDG-7*

`batchUpdate()` hides a visible table while many rows are set, so the cell
widgets are laid out once (signals stay on). Each `ButtonColumn` row reuses one
icon instead of building its own, and the column is sized once, to a
deterministic width.

### WDG-8d: `ComboColumn`

*feat · S · from `_column_info.py`*

A column whose cells are combo boxes (`ComboColumn`, `TableComboBox`), exported
from `useq_widgets`.

### WDG-9: follow the camera FOV on camera, binning and mirror changes

*fix · M · `fbee6fe8`, `9890f35d`, `144ad68d`*

Switching cameras or binning emits neither `roiSet` nor `pixelSizeChanged`, so
the HCS FOV page, the MDA grid, the position grids and the Stage Explorer kept
the previous camera's FOV, and a Multi Camera's second camera overwrote the
first one's Explorer tiles. `FovChangeObserver` in `_util.py` replaces each
widget's own camera, ROI, binning and pixel size connections with one refresh
on the GUI thread.

### WDG-10: default tiling overlap of 10%

*feat · S · `b03429ff` · open an issue first*

`DEFAULT_OVERLAP = 10.0` for the grid plan, the grid and random-points
sub-sequence, the HCS overlap and the Explorer's scan menu, resets included.
This changes the default plans, so ask first.

### WDG-11: log widget follows the log file through a rotation

*fix · S · `2f9ba8e5`, `1df3aaf6`, `c97d1632`, `0d6c88c3`, and the `_log.py`
part of `09e9ea9e` · after #546*

- The reader only noticed a rotation if the watcher fired while the new file was
  smaller than the read position; otherwise it stayed on the renamed file and
  stopped updating. On every poll, check whether the path now holds a different
  file, and finish the old one before switching.
- The view starts at the left edge of the loaded lines by moving the text
  cursor to the start, instead of a `QTimer.singleShot` that could fire after
  the widget was deleted (`c97d1632`, `0d6c88c3`).
- Each test gets its own log file.

Rebase on #546 if it lands first.

### WDG-12: InstallWidget signals and cancel

*feat · M · `3ce24674`, `4fcabbc6`, `1eb9c2fc`*

- `installStarted(str)`, `installFinished(int)`, `install()`,
  `cancel_install()` and `is_installing()`, so an app can follow and drive an
  install.
- The version combo lists only releases that `mmcore install --release` accepts
  on this platform: full nightly builds on Windows and Intel macOS, test-adapter
  releases elsewhere. Before, most of the list failed with "Release ... not
  found". Offline, only the dated releases are lost.
- A failed GitHub response is closed: it raised a `ResourceWarning` on
  Python 3.14.

### WDG-13: shutter state as an open or closed aperture

*feat · S · `ca447a24` · after #561*

The default icons are a wide-open aperture and one closed to a pinhole, rather
than one recolored circle.

### WDG-14: "Enable Editing" menu for fields that move the stage

*feat · S · `9a017305`*

`RightClickMenu` in `_util.py`: the StageWidget position boxes and the grid's
Absolute Bounds fields start disabled (read-only looked the same as editable),
and a right-click offers "Enable Editing", which the tooltips mention. WDG-31
uses the same pattern for the Z fields and Move buttons.

### WDG-54: save names after multi-position OME-TIFF runs (your #529)

*fix · S · `_util.py` on `cite` · update your #529*

`get_next_available_path` counted the `_###` of *every* file with the same
extension, so `test.ome.tiff` next to an unrelated `other_009.ome.tiff` became
`test_010.ome.tiff`. #529 already fixes that. Refresh it with the fork's
version on top of #568 (merged): only names whose stem without its counter is
the requested stem count; a requested counter is kept when higher; and a
multi-position OME-TIFF output folder (`stem` or `stem_###`) counts, compared
without case.

## Devices, configuration, channels, camera, stage

### WDG-15a: slider editor for preset values with limits

*feat · S · `593270ab`*

`PropertySettingDelegate` edits and paints a property with limits as a slider.

### WDG-15b: preset cell editors: first click, closing mid-click, undo

*fix · M · `c4ce2cdd`, `a9350ee6`, `f3a8513f`*

- The click that opens a cell's editor now reaches it, so a slider, spin box or
  text field no longer needs a second click.
- If committing the value closes the editor mid-click, the rest of the click is
  dropped instead of reaching the deleted control.
- With an undo stack, each value change rebuilt the pivot table and closed the
  editor, so a slider drag stopped after one step and added one undo entry per
  step. A value change now updates only its cell, and consecutive changes of
  one value merge into one undo step.

### WDG-15c: preset metadata refresh on device update; selection tint

*fix · S · `3de7e6b2`, `5f9ea889`*

- Settings keep a snapshot of their property's allowed values, so a device
  loaded later (e.g. a new shutter adding a Core-Shutter choice) was missing
  from existing editors. `update_from_core` refreshes it, keeping values and
  unsaved edits.
- Selected cells get a translucent tint over the controls instead of the
  view's solid fill, which hid slider grooves.

### WDG-16: read-only groups/presets and pixel editors

*feat · S · `35151aa7`*

`ConfigGroupsEditor.setReadOnly()` and `PixelConfigurationWidget.setReadOnly()`:
browse without changing. The models drop the editable flag, the presets table
opens no menus, and the edit actions sit in a disabled `QActionGroup`.

### WDG-55: `DeviceButtonToolbar` in PropertyBrowser and the pixel editor

*feat · M · `475417c8`, `00ad8ae2`*

A new `DeviceButtonToolbar` (`device_properties/_device_type_toolbar.py`) puts
one icon button per device type on top of the PropertyBrowser, replacing the
device filters inside `_device_property_selector.py` (about 100 lines go). The
pixel configuration editor uses the same toolbar.

### WDG-56: `PropertyBrowser` is a QWidget, not a QDialog

*refactor! · S · `728dc192` · open an issue first*

So an app can dock it or embed it. A caller that runs it with `exec()` or relies
on its dialog flags breaks: say so in the title and the changelog, and check
napari-micromanager's use in the downstream CI job.

### WDG-17a: PixelConfigurationWidget dirty tracking

*feat · S · from `7620d351`*

`isClean()`, `setClean()` and `cleanChanged(bool)`, like `ConfigGroupsEditor`,
so an app can ask before discarding edits.

### WDG-17b: pixel property table as model/view

*refactor · M · `43853938`, `1759b48f`, `ab82e0df` · needs WDG-17a*

The property value table becomes a model and view, like the config presets
editor, which the read-only mode and the shared row height rely on. Only open
it if WDG-16 or the review needs it: it is the largest part of the
`_pixel_configuration_widget.py` diff.

### WDG-17c: warn when a pixel size may be another camera's

*feat · S · `f33e9838`, `f38a537e`, `4be77ee3`, `235a1d78`, `74295a10`,
`c08616cc` · needs WDG-9*

Micro-Manager can't read a camera's pixel size, so with several cameras a pixel
size preset without Core-Camera applies to every camera. `pixel_presets_hint`,
`pixel_size_camera_warning` and `fov_pixel_size_warning` say so, ignore the
Multi Camera, and the HCS points page is disabled until the pixel size is
certain.

### WDG-18: ConfigWizard pages and device reload (your #522)

*feat · M · update your #522 · from `8c1cb4d5`, `ecef7ba8`, `8332ee08`,
`e68072a6`, `d4baff99` and the later `hcwizard/` changes*

Your #522 ("better mimic the Java wizard") already targets `main` and touches
the same files. Refresh it with `cite`'s `hcwizard/`:

- On finish, the wizard cleans the configuration up and reloads the devices.
- The labels page reads state labels from the hardware, and the roles page the
  focus directions.
- Devices page, device setup dialog and the simple property table updates.

If it grows past M, split it by page.

### WDG-19: light source helpers

*feat · S · `_light_sources.py`, `46bdc53d`*

`find_light_sources(core)`, `light_source_limits`, `IntensitySpinBox` and
`IntensityColumn`. Lumencor's legacy LIDA driver takes intensity backwards (0 is
full power): the widgets flip it within the property limits, so 100 is always
full power.

### WDG-20a: light source and intensity columns in the MDA channel table

*feat · M · `ba4c7607`, `cede0d15`, `8da8810f`, `f8f92930`, `23bfa941`,
`7620d351`, `4e94bad0` · needs WDG-19, WDG-8d*

`CoreConnectedChannelTable` gets Light Source and Intensity columns after
Exposure (several properties per channel allowed), with
`setLightSourceVisible()`, hidden by default upstream, and a color indicator.
`refresh()` re-reads the core.

### WDG-20b: per-channel properties applied during the run

*feat · M · `_channel_properties.py` · needs WDG-20a*

`useq.Channel` has no device properties, so the settings travel in
`metadata[PYMMCW_METADATA_KEY]["channel_properties"]`, and `MDAWidget.value()`
returns a `ChannelPropertiesSequence` (a `useq.MDASequence` subclass) whose
events carry the matching `MDAEvent.properties`. `setValue()` restores them.

In the description, ask whether `useq.Channel` should get a first-class
`properties` field instead. If yes, that is a small USEQ PR, and this one
shrinks.

### WDG-20c: "Core Channel" group combo, advanced columns toggle

*feat · S · `5882d08d`, `049a47e4`, `e9353cda` · needs WDG-20a*

A group chosen in the table's combo sets the core's channel group, and the combo
follows the core. A toggle shows or hides the advanced columns.

### WDG-21: `ChannelSettingsTable`

*feat · L (~700 lines) · `cf8f7b63`, `5882d08d`, `dc8e3f5a` · needs WDG-19,
WDG-5*

A new control widget: one row per preset of a channel group, with its exposure
and light source, and `ChannelSettings` / `apply_channel_settings` in
`_channel_settings.py`. A click on the row's dot or name applies the preset;
clicks elsewhere only edit. The table fits its rows. Add
`examples/channel_settings_table.py`.

Split: (1) `ChannelSettings`, `apply_channel_settings`, and the table showing
presets with the current one highlighted and applied on click
(`CurrentPresetColumn`, `CurrentPresetDelegate`); (2) editing exposure and
light source in the table.

### WDG-22a: `CameraRoiWidget` value round-trip

*feat · S · `ecfad4cb`*

`CameraRoiValue` (camera, x, y, width, height: JSON-friendly), `roiValue()`,
`setRoiValue()` (restores a planned ROI in the editor without applying it),
`fullFrameValue()` and `applyFullFrame()`. An MDA can save and restore its
camera ROI.

### WDG-22b: `CameraRoiWidget` live-view selection hooks

*feat · S · `bce35dee`, `378a4e30`, `7449ad8b`, `9127d93d` · needs WDG-22a*

`roiSelectionRequested(bool)`, `setLiveSelectionActive()` and
`setRoiSelectionAvailable()` let a host app with an image viewer offer "select
the ROI in live view". Also ROI visibility and accent colors for the crop and
selection icons.

### WDG-23: StageWidget: polling pause, wheel Z, halt any device

*feat · S · `a688fe0d`, `195dd070`, the behavior parts of `5c0193eb` · needs
WDG-14*

Keep the constructor and the look. Add:

- polling stops while the widget is hidden;
- an XY position read that raises no longer breaks polling;
- the mouse wheel moves a Z stage by one step (a `_WheelPollStageWidget` base,
  shared with WDG-24);
- `HaltButton.setDevice()`.

### WDG-24: `XYZStageWidget`

*feat · M (~400 lines) · `f4536b56`, `a785dafa`, `bf5bf1ab`, `488e5b5b` · needs
WDG-23*

A new widget for the core's *default* XY stage and focus device, following
changes of either (a placeholder replaces the controls of a missing default).
It has the new look: a compass of 8 arrows, one step per axis, position boxes
below, and the wheel for Z. `examples/xyz_stage_widget.py`.

### WDG-25: `XYZStageWidget` with fixed devices, StageWidget deprecated

*feat · S · new `xy_device` and `z_device` parameters · open an issue first ·
needs WDG-24*

Optional. The fork's per-device StageWidget (pymmcore-gui's Stages panel) uses
the same compass. Rather than overwrite StageWidget, give `XYZStageWidget`
optional `xy_device` and `z_device` (None follows the core, `""` hides the
axis), and deprecate `StageWidget`'s `levels` and `position_label_below`. This
is the one real v2 in this plan.

## MDA and useq editors

### WDG-26: time plan: zero intervals

*fix · S · `7b7a503e`, `b63a5c28`*

A time plan with a zero interval raised `ZeroDivisionError` in the duration
estimate.

### WDG-27: grid bounds: edits, Move, the half-FOV overshoot

*fix · S · `b0165340`, `41ebb5d1`, `5d7a4593`, `457641dd`, `825d0f03`*

- Editing or marking a bound updated neither the summary nor `valueChanged`
  (`CoreConnectedGridPlanWidget` replaced `bounds_wdg` without connecting it).
- A mark outlived a hand edit of its field: Move went to the old mark. Without a
  mark, Move centered the FOV on the outer image edge.
- `CoreXYBoundsControl._visit` overshot marked corners by half a FOV.
- An absolute grid shows its first FOV in the disabled X/Y cells, and its labels
  are fixed.

Port onto the current layout. WDG-28 then moves them.

### WDG-28: grid editor layout (visible)

*feat · M · `90ec7b37`, `4d018e25`, `74c497bc`, `f2417908`, `e33c3e19`,
`98b00212`, `db144a3f`, `fff0ae48`, `d730ad18`, `199e6c08` · needs WDG-27*

The mode becomes a row of icon radios (Rows & Columns, Width & Height,
Absolute Bounds, and Polygon when one is set), as in the Z editor. Overlap,
order buttons, Edges/Corners and a summary go in a fixed left column, and
the mode's own fields in rows on the right, at the same place in every mode.
Icons and dividers follow the palette. `SeparatorWidget` can be vertical.
`GridPlanWidget` stays a `QScrollArea` here: WDG-40d changes that.

### WDG-29: Cover Area points plan

*feat · M · `_cover_area_widget.py`, `_common.py`, `_points_plan_selector.py`,
`_well_graphics_view.py` · needs USEQ-8a*

`CoverAreaWidget` edits a `useq.GridFromShape` (shape, size, cover or inside,
overlap, order), and `RelativePointPlanSelector` offers it next to Rows &
Columns and Random. The well view draws the area and its tiles.

### WDG-30: Z editor graphic and layout (visible)

*feat · M · `8c7c3aa1`, `b4196774`, `42b7834d`, `25911ef2`, `805542aa`,
`55c42f31`, `680007b8`, `211eb59e` · needs #542*

`ZStackViz` draws the stack next to boxed controls with a mode radio row. The
height stays steady across modes and zooms. Talley's #542 (on `main`) refactors
the same widget: build on it instead of competing with it, or fold this into it
with his agreement.

### WDG-31: Z navigation, reference reset, live Z

*feat · M · `33b3e7ec`, `8a18d2f5`, `9f434397`, `e27b9a96`, `9018022b`,
`42dac0ad`, `a21ee393`, `62174fd6` · needs WDG-30, WDG-14*

`CoreConnectedZPlanWidget`:

- Move to top, bottom and center; return to the reference Z.
- Reset the reference (or the center) to the current position.
- An opt-in "Poll" shows the live Z on the graphic. Reads run off the GUI
  thread, and polling pauses while hidden or unchecked.
- Top and bottom fields need "Enable Editing", and Move buttons "Enable Moving",
  so a stray click doesn't move the stage.

### WDG-32a: `RecordTableModel` and `RecordTableView`

*feat · L (~570 lines) · the `_record_model.py` part of `371338a6` · needs
WDG-5*

A generic table of plain records, with a `ColumnDelegate` that paints cells and
creates an editor only for the cell being edited. On `cite`,
`docs/plans/POSITION_TABLE_MODEL_VIEW.md` holds the design: put its summary in
the description.

Split: (1) `RecordTableModel` (~200 lines, testable without a view);
(2) `ColumnDelegate` and `RecordTableView`.

### WDG-32b: position table on model/view

*perf · M · `371338a6`, `1a6635da`, `7568c299` · needs WDG-32a*

The position table held a widget per cell (spin boxes, arrow and grid buttons),
so an HCS import of 2,400 positions froze the app for about 8.5 s. It now uses
WDG-32a. The painted arrows follow the theme. `MDAButton` goes; it wasn't
exported.

### WDG-33a: copy a position's Z or AF offset to all positions

*feat · S · `de778597` · needs WDG-32b*

A right-click on a row's Z or AF set button offers to copy that value to every
position.

### WDG-33b: grid sub-sequence popup: non-modal, wide enough, freed

*fix · S · `cc6eea28`, `69914bc6`, `5566c5e9`*

- The popup ran a modal `exec()`, which blocked live mode and the stage while
  the bounds were set. It is a non-modal window now, raised again by its
  button.
- It opens wide enough for the whole editor.
- Lambdas holding the popup kept it alive after close, which crashed PySide6 at
  teardown. Bound methods replace them.

### WDG-33c: sub-sequence popup edits only a grid plan

*feat · S · `bc1f3ed1`, `c0a0bdc6` · open an issue first · pairs with USEQ-10*

A position's sub-sequence editor shows only the grid plan, with "Same as main"
checkboxes for the axes it mirrors. Upstream iterates other sub-sequence axes
inside each position, out of the root order (see USEQ-10), so the full editor
invites surprising acquisitions. Discuss it together with USEQ-10.

### WDG-33d: HCS positions locked behind "Unlock"

*feat · S · `b072cb1f`, `32b2ac55` · needs WDG-32b*

Positions generated by the HCS wizard are locked, with an orange "Unlock" button
and a tooltip on it and on the table saying how to edit them. (`d85cd41d`'s
button-cell event filter is not needed: WDG-32b paints those cells.)

### WDG-34: plate views: themes, styled parents, plate changes

*fix · S · `6cb89080`, `4e91b310`, `6894799b`, `8822bb92`, `46cb3e8b`*

- Plate and FOV graphics stay legible in light and dark themes, and the first
  point is drawn green.
- The well shape and the random points widget follow plate changes.
- `WellPlateView` and `WellView` no longer fail under a parent with a style
  sheet: they set their own style sheet after creating the items it recolors.
- Adds `setClearButtonVisible` and `setUnselectedColor`.

### WDG-35: `CustomPlateWidget`

*feat · M (~400 lines) · `916a295b`, `0aa4176c`, `ba3c88d7`, `9c69c3ef`,
`53ceedba`, `f49cc218`*

A dialog, opened from `WellPlateWidget`, to define, save and delete custom well
plates. They are registered with `useq.register_well_plates`, so they can be
used by name anywhere, and a saved plate shows in the list at once.

### WDG-36a: `AutofocusAxis` as a checkable group box with `plan()`

*refactor · S · `7fc06654` and the hardware part of `80b5fea8`*

`AutofocusAxis` becomes a checkable `QGroupBox`: `isChecked()` says whether
autofocus runs, and Qt disables the controls when it's unchecked (the old
hand-made enablement had a stale-state bug). `plan(axes)` and `setPlan(plan)`
replace building `AxesBasedAF` in `MDASequenceWidget`. Hardware only, no new
useq fields. Every later autofocus PR builds on this. Open a comment on #553
(gcharvin) linking this chain.

### WDG-36b: autofocus every-N intervals

*feat · S · `9000849f`, `69dbbba3` · needs WDG-36a, USEQ-4 · pairs with USEQ-4*

An "every N" spin box per axis checkbox (positions, grid tiles, timepoints).

### WDG-36c: hardware autofocus Z search controls

*feat · S · `684efd43`, `b57f9887` · needs WDG-36a, PLUS-16 · pairs with
PLUS-16*

A "Search" checkbox turns the below, above and step fields on, which keep their
values while it is off. The step defaults to 5 µm and can't be 0. Nothing
searches by default.

### WDG-36d-1: software autofocus mode and method picker

*feat · M · `80b5fea8`, `9c6a795a`, `486a2e9d`, `059e0f77`, `95205296`,
`44c45b7e`, `e3f8b3b7` · needs WDG-36a, PLUS-19 · pairs with USEQ-6, PLUS-19*

- Hardware and Software radios, then a method combo filled from
  `pymmcore_plus.autofocus.available_methods()`, with each routine's
  description as tooltip.
- With Software, per-position autofocus offsets are disabled (they are a
  hardware motor offset), and a per-position offset is only read from a
  hardware plan.
- The section's enabled state is recomputed when the configuration changes,
  and the summary stays visible during a run.

### WDG-36d-2: `AutofocusSettingsDialog`

*feat · L (~670 lines) · `3f9d2818`, `afc26338`, `7bb7e5a8`, `382f858c`,
`ee9c5675` · needs WDG-36d-1*

A form generated from the routine's settings dataclass
(`pymmcore_plus.autofocus.settings_model`), with tooltips from the field docs. A
field naming another routine (`duo`) offers that choice with its own Settings
button, and channel fields list the microscope's presets. Settings come back
exactly as given: no rounding by the display precision, no dropping of values
the user didn't touch.

Split: (1) `SettingsForm` for plain fields and the dialog around it;
(2) the chained-routine editor (`_StepEditor`) and channel choices
(`_config_choices`).

### WDG-36d-3: test a routine from its settings

*feat · S · `318abbdb` · needs WDG-36d-2 · pairs with PLUS-18b*

A Test button runs the routine at the current position with the edited settings
(`run_software_autofocus`) and reports which device moved, from where to where,
and how many images it took.

### WDG-36e: "Keep locked"

*feat · S · `3c2d4be3`, `b9b31f6e`, `bac0906d`, `26aeefb4` · needs WDG-36a,
PLUS-20a · pairs with USEQ-7b, PLUS-20*

A "Keep locked" row builds a hardware plan with `keep_locked=True` (also with
no axis checked), shown in the collapsed summary. The tooltip says the lock is
released for Z stacks and channel offsets.

### WDG-36f: Run refuses autofocus without an axis

*feat · S · `5d2b0ab3` · needs WDG-36a*

With Autofocus checked but no axis (and no Keep locked), autofocus would never
run. Run now says so and stops.

### WDG-37a: `ZSurfaceDialog`

*feat · L (~730 lines) · `70efe7e2`, `8898a64e`, `f6778b47`,
`6698e3f2`, `4d03cc91`, `a5c6e386`, `f3b8d804`, `6f7fc05f` · needs USEQ-9a ·
pairs with USEQ-9a*

A non-modal dialog to set up a `useq.FocusSurface`:

- A points table whose XY and Z are each set from the stage.
- The method, and a map of the surface with a colorbar.
- A Test button that moves Z to the surface at the current XY; a double-click
  on the map moves the stage there.
- `ZSurface` in `_z_surface.py` builds the surface once.

Split: (1) `ZSurface`, the points table and the method; (2) the map, Test and
double-click moves.

### WDG-37b: Surface Focus in the position table

*feat · M · `70efe7e2`, `39cd9d9a`, `708e1436`, `abaa032d` · needs WDG-37a,
WDG-36a, USEQ-9b · pairs with USEQ-9b*

A "Surface Focus" checkbox and gear button in the position table set
`MDASequence.focus_surface`, also for HCS plates. Conflicting choices can't be
made, and the tooltip says why: Top & Bottom Z mode, and hardware autofocus
(Surface Focus works with software autofocus).

### WDG-38: MDA run fixes: live, dialogs, axis order, pause

*fix · S · `eba2869a`, `03b16684`, `4d82187f`, `3590e39a` · needs PLUS-10*

- Stop live before HCS wizard positions are added, and disable the dialogs the
  widget owns during a run (dialogs are windows, so disabling the editors left
  the HCS wizard usable).
- The axis order combo no longer resets when a run disables the Positions table.
- Pause is disabled, with a tooltip, while a hardware-triggered sequence runs.
- Type `prepare_mda` and `execute_mda` with `SingleOutput`.

### WDG-39: plate positions keep z unset; relative Z anchored

*fix · S · `680411e6` · needs USEQ-1 · pairs with USEQ-1*

`execute_mda` anchors a relative Z plan of a `WellPlatePlan` to the current
focus, as the equivalent absolute plan (`ZTopBottom` when it visits the same
planes, else `ZAbsolutePositions`), so the plate plan is kept for the writers.
`value()` keeps the plate plan and the relative plan for Save MDA Settings.

### WDG-40a: MDA Settings card and footer (visible)

*feat · S · from `7610eb4d`, `5750ec92`, `e5023384`*

`MDASequenceWidget` moves the axis order from the row above the tabs into a
Settings card under them, with the keep-shutter-open settings. The bottom row
becomes a footer: the time estimate, then Save and Load MDA Settings, and
`MDAWidget` appends Run, Pause and Cancel to that row instead of adding a row
of its own. `CoreMDATabs` moves to
`mda/_core_tabs.py`, exported from `pymmcore_widgets.mda`.

### WDG-40b: MDA Saving tab (visible)

*feat · S · from `5750ec92` · needs WDG-40a*

The save options become a Saving tab after the axes
(`CoreMDATabs.add_supporting_sections`), instead of a group box above the tabs.

### WDG-40c: MDA Autofocus tab (visible)

*feat · S · from `9be78165`, `26231f50`, `f67ad62f` · needs WDG-40a, WDG-36a*

Autofocus moves from under the tabs to a tab before Saving
(`MDATabs.add_autofocus`). The tab's checkbox switches autofocus, and its
tooltip says why when autofocus can't be used. Its mode is picked with icon
radios, as in the Grid and Z tabs.

### WDG-40d: MDA icon-on-top axis tabs; `GridPlanWidget` becomes a QWidget (visible)

*feat! · M · from `5750ec92`, `92970fe4`, `9be78165`, `2169713f` · needs
WDG-40c, WDG-28*

- `MDATabs` shows icon-on-top tab buttons with a checkbox each (`_AxisTabBar`,
  `_AxisTabButton`, built by `_build_tabs()`), with Grid after Time.
- Every page insets its editor, and the Grid and Z editors sit in a card inside
  a scroll area (`_CardFrame`). So `GridPlanWidget` no longer needs to be a
  `QScrollArea` and becomes a `QWidget`: flag that in the title and the
  changelog, since code that used its scroll-area API breaks.

### WDG-41: `CollapsibleMDAWidget`

*feat · L (~690 lines) · `f0637330`, `8247917d`, `085921f4`, `c52ddfd1`,
`92970fe4`, `b17407fb`, `d7279d3c` · needs WDG-40a, WDG-40b, WDG-40c*

A new class that swaps the tabs for collapsible, checkable sections with a
summary each (`CollapsibleCoreMDATabs`, `CollapsibleAcquisitionSection`,
`SectionMetrics`). `examples/mda_widget_collapsible.py`.

Split: (1) `CollapsibleAcquisitionSection` and `SectionMetrics`, a standalone
widget (~240 lines); (2) `CollapsibleCoreMDATabs` and `CollapsibleMDAWidget`.

### WDG-42: MDA axes start off and empty (visible)

*feat · S · `6ef8beb4`, `8cc06af5` · open an issue first*

No axis is checked by default (Channels was), the position, channel and time
tables start without rows, and in the collapsible layout Channels starts
collapsed. Scripts and napari-micromanager may rely on the old defaults.

## Stage Explorer

pymmcore-gui's large HCS scans drove these. WDG-44 holds the speed work, the
others features.

### WDG-43: map memory cap

*feat · M · `dd926345`, `1b35df5b`, `63863563`, `f60f9581`*

The map's images are capped at a share of RAM: 20% by default, at least
0.5 GB, set from a `MapMemoryMenu`. A banner appears when the cap is reached,
and allocation errors are handled. `max_map_memory_mb` and `map_memory_bytes`.
**Adds a `psutil` dependency**: say so in the description.

### WDG-44a: downsample tiles when zoomed out

*perf · S · `4fcc0b2e`, `_image_processing.py`*

Once the view settles after a zoom, visible tiles are uploaded as a block mean
at the largest power of two that still gives one texture pixel per screen
pixel. A 10×10 grid of 2048² tiles zoomed out to fit uploads about 12.5 MB
instead of the full resolution.

### WDG-44b: draw tiles in view from a pool of images

*perf · M · `1a0cf0cc`, `8b2b1be6` · needs WDG-44a*

vispy pushes every camera move through every visual, drawn or not, so panning
slowed down with map size (~54 ms per step at 3000 tiles). Tiles become plain
records, drawn by `Image` visuals lent from a small pool while in view and
culled with one NumPy test: about 1 ms at any size. An unchanged cull no longer
repaints.

### WDG-44c: prepare map images off the GUI thread, in batches

*perf · M · `a1d15689`, `6ea7e3a1`, `fdec0905`, the explorer part of
`3494ce0a`, `e6c0de67` · needs WDG-44b*

Thumbnails, reductions and overview chunks are prepared on one worker thread,
coalesced per location, and applied in bounded GUI batches. Results are
immutable, and stale ones are rejected. Visible tiles are downsampled in
batches: one at a time cost about N² draws. Fixes acquisition stalls during a
scan.

### WDG-44d: multiscale overview

*perf · L (~550 lines) · `79d60dee`, `2fa83f01`, `9ef01ce7`, `6750849f` · needs
WDG-44c*

With more than 32 tiles in view, draw the coarsest level of a chunked pyramid of
reduced copies, as napari's multiscale does: ~760 tiles took ~95 ms per frame
before. A moved tile repaints only its chunks.

Split: (1) `OverviewData`, the pyramid as plain NumPy with no vispy, tested
alone; (2) `OverviewPyramid`, which draws it, and the switch in `StageViewer`.

### WDG-45: show MDA positions on the map

*feat · M · `1fffa5db`, `34ed46f1`, `918eb09d`, `01227817`, `794c658f`,
`_mda_positions.py`*

`StageExplorer.set_mda_positions(value)` takes a position table's value (a list
of positions or a `WellPlatePlan`), with a "Show MDA Positions" toggle, on by
default. The explorer stays decoupled: the app connects the table to it. Grids
of up to 50,000 FOVs are marked. Bigger ones get a note in the corner rather
than vanishing.

### WDG-46: ROI tools and `sendToMDARequested`

*feat · M · `08842675`, `67ad4fc0`, `9780ac71`, `12548763`, `eada592d`,
`9e65bf21`*

- A cog button holds the scan options (overlap, order), and a button deletes
  only the selected ROI.
- Right-clicking an ROI offers the scan options and "Send to MDA" for that ROI.
- `sendToMDARequested(list, bool)` carries the positions, and the example shows
  how to feed the MDA table.

### WDG-47: stage position overlay

*feat · S · `3889b813`, `bf9ddd62`*

The X/Y position is shown in the map's bottom-right corner instead of the
toolbar. The position indicator menu stays in sync.

### WDG-48: MM affine convention; camera format changes

*fix · S · `66484510`, `c6859d74`, `7a23ca15` · needs WDG-9*

- A calibrated pixel affine is used as is, and the camera mirror flags apply
  only without a calibration, as Micro-Manager's XYNavigator does. Binning no
  longer turns the default identity into a 2 µm/px calibration.
- When a camera switches between mono and RGB, the explorer asks before mixing
  formats in the map, and restores the pixel type if the user declines.

### WDG-57: flip menu in StageViewer (your #531)

*feat · S · `6037a106` · update your #531*

A context menu on the map to flip the view horizontally or vertically. #531
targets `main` with the same change: refresh it with the fork's
`_stage_viewer.py` and tests.

## HCS

### WDG-58: calibrate plates with fewer than 3 wells (your #482)

*fix · S · `76de502a` · update your #482*

With a plate of one well or two (e.g. `coverslip-18-mm-square`), the calibration
page demanded at least 3 calibrated wells, so the wizard could not get to the
FOV page (#481). #482 targets `main` with this fix: refresh it with the fork's
`_plate_calibration_widget.py` and its tests.

### WDG-49: calibration fit, FOV on load, snake order

*fix · S · `f1090e95`, `546bf2b0`, the HCS part of `3494ce0a`, `a9ff5c24` ·
needs WDG-58*

- A1 and rotation are a least-squares fit at the plate's nominal spacing (the
  plan uses that spacing): on a real 2×2 scan the largest deviation drops from
  176 to 68 µm.
- Loading a plan uses the current camera FOV and no longer mutates the
  caller's plan.
- Wells are visited in snake order.

### WDG-50: wells on their own page; wizard sizing and icons

*feat · M · `85c460c0`, `f713e744`, `d828f8ef`, `73553de6`*

The first page only picks or creates the plate, and wells are selected on their
own page after the calibration. The wizard sizes itself to fit every page on
first show (it was a fixed 880×720). Its actions get icons.

### WDG-51: versioned wizard save files

*feat · S · the `_hcs_wizard.py` save/load part of `d3630789`*

Saved files become `{"version": 1, ...}` around the `WellPlatePlan`. Files
without a version still load, and newer versions are refused with a clear
message.

### WDG-52: Cover Area on the FOV page

*feat · M · `d3630789` · needs WDG-29, USEQ-8b · pairs with USEQ-8b*

The FOV page gets a "Cover Area" card (`GridFromShape`) with Coverage and Grid
tabs, and the rows/columns grid moves into it. One overlap for x and y
(right-click for separate values), the scan order and "Keep FOVs Inside Area"
sit below. The area starts at the full well and can't exceed it.

### WDG-53a: NumPy well detection

*feat · M · `_image_ops.py`, `_well_detection.py`,
`_well_detection_settings.py`, `1422aa90`*

Pure NumPy, no scikit-image or SciPy (`canny`, `fftconvolve`,
`peak_local_max`, `least_squares` in `_image_ops.py`): finds the wells in the
survey images by edges, template correlation and a circle fit. No widget, so it
can be reviewed and tested alone.

### WDG-53b: plate survey scan

*feat · M · `_plate_survey.py`, `_survey_canvas.py`, `920738ec`, `7809766c` ·
needs WDG-53a*

Scans a rectangle given by 2 corners or 4 edge points. The shutter stays open
for the whole scan (auto-shutter restored afterwards, also on cancel or
failure). Color cameras are surveyed on their luminance.

### WDG-53c: automated plate calibration tab (experimental)

*feat · two M PRs (~1,050 lines in all) · `17aa993c` to `59ae9a2c`,
`f76909b4`, `_automated_plate_calibration_widget.py`, `_plate_calibration.py`,
`docs/hcs-automated-calibration.md` · needs WDG-53b, WDG-50*

An Automated tab next to Manual on the calibration page: select wells, record
the scan area, scan, detect wells (with a settings dialog), then Apply. A
detected but not applied calibration is shown as pending, not as calibrated.
Mark it experimental. Split it:

1. Scan-area table, scan and the survey view.
2. Detection, the settings dialog, the Apply step and the docs page.
