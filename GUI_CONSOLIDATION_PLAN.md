# Plan: retire the classic GUI, promote the modern GUI

> Working document for the GUI consolidation on the `cite-saving` branch.
> Delete this file once the work has landed.

## Context

`pymmcore-gui` currently ships **two** main windows out of one package:

- the classic `MicroManagerGUI` ([_main_window.py](src/pymmcore_gui/_main_window.py),
  a QtAds dock canvas driven by the pydantic [actions/](src/pymmcore_gui/actions/)
  registry), and
- the modern `MainWindow` ([_modern_gui/](src/pymmcore_gui/_modern_gui/)) — three
  mode tabs (Hardware Setup / Configurations / Acquire) with its own theme engine
  and panel registry.

Selection is a single CLI flag (`mmgui run --modern`). The classic GUI is the
default, which means the PyInstaller bundle — the only way most users get the
app — still launches it. Meanwhile the modern GUI carries 66 of the repo's 103
tests, and `--modern` is documented nowhere.

Carrying both is now pure cost: `_theme/` is already shared by `_app.py` and
three `widgets/` modules despite living inside `_modern_gui/`, which forces a
lazy-`__getattr__` cycle-breaker in `_modern_gui/__init__.py`; `_settings.py`
keeps two parallel window sections; and `_array_viewer.py` imports the `actions`
registry only to reach dead code.

**Outcome:** one GUI, living at the top level of the package. `mmgui` and the
bundle both launch it. Source files for widgets the modern GUI hasn't absorbed
yet are kept in place so they can be ported later; everything else the classic
GUI needed is deleted.

## Decisions

1. **Flatten** `_modern_gui/*` up into `src/pymmcore_gui/`. `_main_win.py` →
   `_main_window.py` (name freed by the deletion), `_theme/` →
   `pymmcore_gui/_theme/`.
2. **Delete** four superseded subsystems: `NotificationManager` + toasts,
   `NDVViewersManager`, `ShuttersToolbar`, `PygfxImagePreview`.
3. **Keep in place, untouched:** `widgets/_about_widget.py`,
   `_stage_control.py`, `_stage_explorer.py`, `_toolbars.py` (`OCToolBar` only),
   `_joystick.py`. Verified none of them import `actions/` or `_main_window`, so
   they need zero edits.
4. **Public API:** export `create_mmgui` + `MainWindow`, plus
   `MicroManagerGUI = MainWindow`.

**Order matters:** delete first, *then* move. Deleting frees the
`_main_window.py` name, so Phase 2 is a pure-rename commit — which keeps git
rename detection (and therefore codecov patch coverage) working. Each phase ends
green.

## Phase 0 — Prep

1. Commit or stash the 4 dirty files (`_modern_gui/_acquire.py`,
   `widgets/_active_channel_table.py`, `tests/test_bundle.py`,
   `tests/test_new_gui.py`). `git mv` on a dirty file works but makes Phase 2's
   diff impure.
2. Record the baseline pass count: `uv run pytest -q`.

## Phase 1 — Delete the classic GUI

Ends green with `_modern_gui/` still nested.

### 1.1 New `src/pymmcore_gui/_resources.py`

`ICON`/`RESOURCES` currently live in `_main_window.py:64-65` and are imported by
`_app.py:16` for the app icon. Move them to a new leaf module (2 constants) so
`_app.py` doesn't have to import the whole GUI at module scope:

```python
RESOURCES = Path(__file__).parent / "resources"
ICON = RESOURCES / ("icon.ico" if sys.platform.startswith("win") else "logo.png")
```

`widgets/_about_widget.py:22` and `app/mmgui.spec:38-39` each compute their own
copies — deduplicating those is optional follow-up, not part of this change.

### 1.2 `_array_viewer.py`

This is the edit that actually breaks the import cycle. Delete line 34
(`from ...actions.widget_actions import WidgetAction, _get_mm_main_window`),
delete `_KeyFilter` (48-66) and both `installEventFilter` calls (89-93) — but
**keep** `widget = self.widget()` at line 90, used again at line 100. Trim
`QEvent`/`QObject` from the line-22 import (`Qt` is still used).

*Intentional behavior change:* the old `eventFilter` returned `True` for every
`Key_M` press while its handler body was unreachable (it probed a non-existent
`WidgetAction.STATS_TABLE`), so `M` was being swallowed. It now reaches ndv's
canvas. `tests/test_array_viewer.py` references none of `_key_filter` / `Key_M` /
`_get_roi_data`.

### 1.3 `_modern_gui/_acquire_viewers.py`

Absorb the two survivors of `_ndv_viewers.py`. Copy `_add_follow_lock_button`
and `_extract_scales` in verbatim (add `import weakref`; `ndv`,
`SummaryMetaV1`, `MDASequence` are already in its TYPE_CHECKING block).
Optionally hoist the four function-local imports at `_ndv_viewers.py:228-231`,
but **keep** `_extract_scales`'s local `useq` import — it sits inside a
`suppress(Exception)` guard. `_acquire_viewers.py` already has its own
`_StreamSignalBridge`; don't copy the other one. Then
`git rm src/pymmcore_gui/_ndv_viewers.py tests/test_ndv_viewers.py`.

### 1.4 `_modern_gui/_panels.py`

Inline the three `widget_actions` factories (`create_property_browser`,
`create_camera_roi`, `create_exception_log`) as local `PanelFactory`s matching
the existing `_create_mda` / `_create_console` shape, and delete the
`_ignoring_core` adapter. Update the three `PANELS` entries.

- **Keep `parent=parent`** on all three. `PropertyBrowser` is a `QDialog`;
  parentless it would briefly be a top-level window, which
  `test_new_gui.py::test_acquire_docked_panels_are_reparented_not_windows` and
  `conftest.py::check_leaks` exist to police.
- **Drop** `create_exception_log`'s `setWindowFlags(...)` and
  `resize(800, 400)`: the flags were already a no-op for a docked panel
  (`_acquire.py:337-346`'s own comment explains that `dock.setWidget()`
  reparents, clearing window flags), and `ExceptionLog.__init__` already does
  `resize(800, 600)` at `_exception_log.py:141`. Then trim that `_acquire.py`
  comment — only its `PropertyBrowser`-is-a-`QDialog` half stays true.
- Keep the `ExceptionLog` import **function-local**: `_exception_log.py:10` does
  `from pymmcore_gui import _app` at module scope, so a module-level import here
  would thread it into the `pymmcore_gui/__init__ -> _app -> ...` chain.

### 1.5 `_settings.py` — the delicate one

- Delete line 26 (`WidgetAction` import) → `_settings.py` becomes a true leaf
  module.
- Delete `_default_widgets()` (142-145) and `WindowSettingsV1` (151-173). Drop
  `model_validator` from the line-16 import block (its only user was
  `_migrate_names`).
- **Keep `WidgetNames` (147-148)** — shared by
  `acquire_panels`/`acquire_hidden_panels`.
- Delete `SettingsV1.window` (line 211). Rename class
  `ModernWindowSettingsV1` → `WindowSettingsV1`, but **keep the field named
  `modern_window`** (212-214) so no user loses their saved layout/theme/zoom. Do
  **not** use a pydantic alias: `_good_data_only` matches on `cls.model_fields`
  keys and is alias-unaware, so an aliased field would fall into the unknown-key
  branch and warn. Rewrite the class docstring, which currently explains itself
  in terms of "kept separate from `WindowSettingsV1`".
- **Legacy-key migration.** Existing settings files have a top-level `window`
  key, and `_good_data_only` warns
  `RuntimeWarning: Key 'window' ... not found in model` for unknown keys. That
  warning fires inside `MMGuiUserPrefsSource.__call__` (`_settings.py:94`) —
  i.e. in the pydantic-settings *source*, before any model validation — so a
  `model_validator` will **not** suppress it. Fix it at that layer: a
  module-level `_LEGACY_KEYS = frozenset({"window"})`, popped from `values` in
  `__call__` before the `_good_data_only` call. `_read_settings()` returns a
  fresh dict per call, so mutating it is safe, and `_write_settings` uses
  `exclude_defaults=True`, so the stale key is never written back — the file
  self-heals on first save. Scope it to `window` specifically; the generic
  unknown-key warning is covered by an existing test and must keep working.

### 1.6 `widgets/_toolbars.py`

Delete `ShuttersToolbar` (58-104). Trim the now-unused `cast`, `DeviceType`,
`ShuttersWidget`, `QWidgetAction` imports; keep `OCToolBar`.

### 1.7 `git rm`

`_main_window.py`, the whole `actions/` package (5 files),
`_notification_manager.py`, `widgets/_notifications.py`,
`widgets/image_preview/_pygfx_image.py`.

### 1.8 `__init__.py`

Replace lines 53-64 with `create_mmgui` + `MainWindow` +
`MicroManagerGUI = MainWindow`; `__all__` accordingly.
`ActionInfo`/`CoreAction`/`WidgetAction` drop out of the public API (intended
break). The alias has a concrete payoff:
`widgets/_mm_console.py::_inject_core_vars` splats `pymmcore_gui.__dict__` into
the IPython namespace.

### 1.9 `_app.py`

Line 16 → `from pymmcore_gui._resources import ICON`. Make the default-window
resolution a **function-local** import inside `create_mmgui` so
`import pymmcore_gui._app` stays cheap for `_exception_log.py:10` and
`conftest.py:23`:

```python
if window_cls is None:
    # -> ._main_window in Phase 2
    from pymmcore_gui._modern_gui._main_win import MainWindow

    window_cls = MainWindow
```

Keep the `window_cls` parameter and the `hasattr(win, "restore_state")` branch
at line 194 — both are the documented extension point, exercised by
`test_new_gui_settings.py`. Update the docstring (136-139) and the comment at
155-160, which is phrased around two GUIs sharing widgets; keep its *reason*
(the theme must be installed before any themed widget is constructed).

### 1.10 `_cli.py`

Delete `--modern` (104-108) and the `window_cls` branch (113-117) outright;
`create_mmgui(window_cls=...)` remains the escape hatch and the package is
`Development Status :: 3 - Alpha`. The `"_cli.py" = ["B008"]` ruff ignore stays.

### 1.11 Tests

- `git rm tests/test_main_window.py tests/test_actions.py
  tests/test_ndv_viewers.py tests/test_notification_manager.py
  tests/test_widgets.py`. (That last file's single test uses a stale
  `importorskip` path, so it has been silently always-skipping.)
- `tests/test_cli.py` — drop the `window_cls is None` assertion from
  `test_default_command_forwards_config_to_standard_gui` (and rename it); delete
  `test_modern_flag_uses_modern_gui`.
- **`tests/test_settings.py:39-47`** breaks *both* ways — it hardcodes
  `{"window": {"geometry": ..., "window_state": [1,2,3]}}` and asserts
  `pytest.warns(match="Could not validate key 'window_state'")`. Retarget it to
  `modern_window` / `acquire_dock_state` (same `Base64Bytes | None` shape), then
  **add** a legacy-key case asserting `MMGuiUserPrefsSource(BaseSettings)() == {}`
  for `{"window": ...}` — no `pytest.warns` needed, since
  `filterwarnings = ["error"]` already turns a stray warning into a failure.
- `tests/test_new_gui_settings.py` — delete
  `test_modern_window_does_not_touch_classic_window_settings` (187-204); it
  asserts on `settings.window`. Fix its module docstring (line 7) and the
  comment at 163.
- `tests/conftest.py` — update the `_init_gui_theme` docstring (65-81).
- `tests/test_new_gui.py:2492` — rewrite `test_stage_explorer_style`'s docstring,
  which justifies itself via `actions/widget_actions.py`; it is now that widget's
  only coverage.

**Gate:** `uv run pytest` green; `uv run mmgui` launches the modern GUI;
`mmgui --help` has no `--modern`.

## Phase 2 — Flatten

### 2.1 `git mv`, no content edits in this commit

`_main_win.py` → `_main_window.py`; the 10 sibling modules (`_acquire*.py`,
`_busy.py`, `_configurations.py`, `_panels.py`, `_sidebar.py`, `_tab_page.py`,
`_toolbar.py`) → `src/pymmcore_gui/`; `_hardware/` and `_theme/` →
`src/pymmcore_gui/`; then `git rm _modern_gui/__init__.py`. Relative imports
inside the moved tree (`from ._theme import ...`, `from ._acquire import ...`)
are unaffected.

*Naming nit:* `pymmcore_gui/_toolbar.py` (modern tab toolbar) will sit one
directory from the kept `widgets/_toolbars.py` (`OCToolBar`). Legal, but
confusable.

### 2.2 Mechanical rewrite of absolute imports

Drop `_modern_gui.` from every `pymmcore_gui._modern_gui.X`. Full site list:

- src: `_app.py` (both the local `MainWindow` import and the `_theme` import at
  161); `_hardware/_page.py:17,18`; `_hardware/_panes.py:9`;
  `_hardware/_peripherals.py:7`; `_hardware/_setup_pane.py:14`;
  `widgets/_mda_widget.py:18,19`; `widgets/_stage_explorer.py:15`;
  `_settings.py:177,189,199` (docstrings).
- tests: `conftest.py:62,85`; `test_new_gui.py:18,19,22-28,37,38,68`;
  `test_new_gui_settings.py:15-17`.

**Do not** rewrite `_hardware/*` to `from .._theme import theme`: ruff has `TID`
selected with no `flake8-tidy-imports` override, so the default
`ban-relative-imports = "parents"` makes `..` a lint error. Keep them absolute.

### 2.3 Hoist `_app.py`'s theme import to module scope

Post-move the graph is `_app -> _main_window -> {_acquire -> {_array_viewer,
_panels, _tab_page, _theme}, ...}` with `_theme -> _array_viewer ->
{_mda_export, _qt}` as a one-way edge. No cycle remains once `_array_viewer.py`
stops importing `actions` (1.2), so the deferred import at line 161 can become
top-level. **Leave `_theme -> _array_viewer` alone** — inverting it means
relocating three public helpers across 8 files for no functional gain.

### 2.4 `DATA_SAVING.md`

Move to `src/pymmcore_gui/DATA_SAVING.md`, not a new top-level `docs/`:
`app/mmgui.spec:88`'s `collect_data_files("pymmcore_gui")` currently ships it in
the bundle, and moving it out of the package would silently drop it. Fix its two
hardcoded `src/pymmcore_gui/_modern_gui/_acquire_viewers.py` references at lines
98 and 155.

### 2.5 Optional, separate commit

Rename `tests/test_new_gui.py` → `tests/test_main_window.py` and
`test_new_gui_settings.py` → `test_window_settings.py` ("new" vs "old" is now
meaningless). Skip if you'd rather not disturb `git log` on a 2665-line file.

### 2.6 Clear stale bytecode

`find src -name __pycache__ -type d -exec rm -rf {} +` — not cosmetic. There are
already stale `.pyc` files for modules that no longer exist (`_enums`,
`_viewport`, `_tab_bar`, `_utils`), so after `git rm`-ing
`_modern_gui/__init__.py` the directory survives on disk and
`import pymmcore_gui._modern_gui` would still **succeed** as an implicit
namespace package — masking any rewrite missed in 2.2.

## Phase 3 — Docs, comments, config

- **`README.md` is factually wrong after this in two places.** Line ~137 tells
  users to install adapters via "the `Devices > Install Devices ...` menu" — that
  menu and `InstallWidget` existed only in `actions/widget_actions.py`; delete
  the bullet. Lines ~148-155 describe the classic menu bar, loading a config
  "from the 'Devices' menu", and the Hardware Config Wizard; rewrite around the
  Acquire / Configurations / Hardware Setup pages. Lines 168-205
  (`create_mmgui` recipes) stay accurate.
- **`pyproject.toml:164`** — drop `"pygfx.*"` from the mypy override list;
  **keep** `"rendercanvas.*"` (still used by
  `widgets/image_preview/_ndv_preview.py:14`).
- **Dangling docstrings** naming deleted things: `_acquire_viewers.py:59`
  ("the classic GUI's `NDVViewersManager`"), `_acquire_toolbar.py:211` ("the
  legacy `ShuttersToolbar`"), `_acquire.py:76` ("Mirrors `_main_window.py`'s
  classic-GUI setup").
- **Keep** `MainWindow.setObjectName("MicroManagerGUI")` (`_main_win.py:185`) —
  `widgets/_mm_console.py:101` scans `QApplication.topLevelWidgets()` for
  exactly that objectName. Do not "clean it up".
- **No change needed:** `app/mmgui.spec` (it freezes `__main__.py`, so the new
  CLI default makes the bundle launch the modern GUI automatically), `justfile`,
  `.github/workflows/*`, `[project.scripts]`.

## Phase 4 — Coverage

`codecov.yml` sets `project.threshold: 1%` and `patch.target: 85%` on PRs. ~340
lines drop to zero coverage because their only exercise was the deleted
`test_main_window.py::test_main_window_widget_actions` (which parametrized every
`WidgetAction`). Extend `[tool.coverage.run] omit` with a comment marking them as
kept for a future port: `_utils.py` (its only importer is `_about_widget.py`, so
it dies too), `widgets/_about_widget.py`, `_joystick.py`, `_stage_control.py`,
`_toolbars.py`. Do **not** omit `widgets/_stage_explorer.py` — it keeps coverage
via `test_stage_explorer_style`.

**One new test** — the only real coverage regression:
`widgets/image_preview/_ndv_preview.py` + `_preview_base.py` (257 lines) lose
*all* real coverage, because the modern tests monkeypatch `NDVPreview` to a
`FakePreview` (5 sites) and the deleted `test_main_window.py::test_snap` /
`::test_stream` were the only tests building a real one. Add a test that builds
`AcquirePage`, calls `core.snapImage()`, and asserts
`AcquireViewersManager.ensure_preview()` returned a real `NDVPreview` that got
docked.

## Verification

Cheap → expensive:

1. `uv run python -c "import pymmcore_gui; print(pymmcore_gui.MainWindow,
   pymmcore_gui.MicroManagerGUI)"`
2. Each leaf importable standalone (the residual-cycle check): `_app`, `_theme`,
   `_array_viewer`, `_settings`.
3. `uv run python -c "import pymmcore_gui._modern_gui"` → **must raise
   `ModuleNotFoundError`** (guards the namespace-package trap in 2.6).
4. `uv run pytest tests/test_settings.py tests/test_cli.py -v` — migration + CLI.
5. `uv run pytest tests/test_app.py -v` **alone**, then in the full run — see R1.
6. `uv run pytest` — full suite; also confirms nothing references the 5 deleted
   test files.
7. `uv run mmgui`, `uv run mmgui --help` (no `--modern`),
   `uv run mmgui -c tests/test_config.cfg`.
8. **Manual settings migration** — no automated coverage is possible, because
   `TESTING` drops `MMGuiUserPrefsSource` from the source list entirely. Back up
   `pmm_settings.json` (`uv run mmgui settings --reveal`), confirm it has a
   `window` key, run `uv run python -W error::RuntimeWarning -m pymmcore_gui`,
   confirm no warning and that the saved theme/zoom/layout restore, close,
   confirm `window` is gone from the file.
9. `just lint` — ruff-check runs with `--exit-non-zero-on-fix`, so hand-remove
   the unused imports from 1.2/1.6 first. Also runs mypy, pyright, markdownlint
   (on the moved `DATA_SAVING.md`), typos.
10. `just bundle`, then `uv run pytest -v tests/test_bundle.py`. Scan the
    PyInstaller log: losing the static references to
    `ConfigWizard`/`InstallWidget` is expected, but confirm `PropertyBrowser`,
    `CameraRoiWidget`, `GroupPresetTableWidget`, `PixelConfigurationWidget`,
    `MDAWidget`, `StageExplorer` and `qtconsole` still resolve (modulegraph does
    follow function-level imports, so the inlined `_panels.py` factories are
    fine).
11. **Launch `dist/pymmgui` by hand** and confirm the modern window appears.
    `test_bundle.py` only waits for the `READY` line, which `_app.py:179` prints
    *before* the window class is instantiated.

## Risks

- **R1 (highest) — `tests/test_app.py::test_main_app` now builds the modern
  window.** It is `@pytest.mark.order(0)` and drives the real path end-to-end
  with `MMQApplication.exec` patched to a single `processEvents()`. It will now
  construct a full CDockManager-based `MainWindow` as the first thing in the
  session under offscreen Qt, and `create_mmgui` takes the
  `QTimer.singleShot(0, restore_state)` branch instead of `win.show()`, so
  whether `restore_state` runs at all hinges on that one `processEvents()`.
  `test_new_gui_settings.py:103-110` documents that ADS teardown is fragile
  enough to segfault offscreen. **Test this first, during Phase 1.** If it
  flakes: keep `order(0)`, add `deleteLater()` + `processEvents()` after the
  close loop.
- **R2 — Exception feedback regresses.** Once `NotificationManager` goes,
  `MMQApplication.exceptionRaised` has exactly one consumer left
  (`_exception_log.py:148`), and it only refreshes an *already-open* panel. An
  unhandled exception will produce no visible GUI feedback unless the user
  happens to have the Exception Log panel open. Worth a follow-up issue.
- **R3 — Dead-code drift.** `_about_widget.py`, `_stage_control.py`,
  `_stage_explorer.py`, `_toolbars.py`, `_joystick.py`, `_utils.py` will have
  zero importers. They stay type-checked and linted (good), but PyInstaller
  won't bundle them — they're genuinely absent from the frozen app. File a
  tracking issue so "port later" doesn't become "port never".
- **Intentional feature loss, for the PR body:** Install Devices, the About
  dialog, Stage Control, Stage Explorer, the Optical Configs toolbar, the
  standalone Hardware Config Wizard (superseded by the Hardware Setup page), the
  classic menu bar, and notification toasts.
