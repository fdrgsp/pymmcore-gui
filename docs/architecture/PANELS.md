# Acquire panels, and how to add one

Every tool on the Acquire page (MDA, Groups and Presets, Properties, Stages,
Camera ROI, Stage Explorer, Console and Exception Log) is a *panel*: one
`PanelInfo` entry in the `PANELS` tuple of
[`widgets/_panels.py`](../../src/pymmcore_gui/widgets/_panels.py).
`AcquirePage` ([`widgets/_acquire.py`](../../src/pymmcore_gui/widgets/_acquire.py))
builds everything else from that entry:

- a checkable, icon-only toggle button in the Acquire toolbar;
- a dock, created the first time the panel is opened;
- an entry in Preferences > Show Widgets, so the button can be hidden;
- saving and restoring whether the panel is open, in the session and in
  named layouts;
- disabling the panel while an acquisition runs.

Adding a panel therefore means writing the widget and registering it. No
window, toolbar, layout or settings code needs to change.

## Adding a panel

### 1. Write the widget

Put application widgets in `src/pymmcore_gui/widgets/`. Any `QWidget` works,
including one from `pymmcore_widgets`. A panel widget should:

- **Use the core it is given.** The factory receives the page's
  `CMMCorePlus`; pass it on and never call `CMMCorePlus.instance()`, which may
  be a different core when the window was created with `mmcore=...`.
- **Release its resources in `closeEvent`.** When the window closes,
  `AcquirePage.shutdown()` calls `close()` on every panel it built. Stop
  timers and threads, and disconnect from core signals, there.
- **Follow the theme.** Read colors and sizes from
  `pymmcore_gui._theme.theme()` (e.g. `qcolor(theme().text_secondary)`,
  `theme().scaled(16)`) instead of hard-coding them. Icons drawn with a
  baked-in color (such as `QIconifyIcon`) must be rebuilt on
  `QEvent.Type.StyleChange`, which fires when the theme or zoom changes; see
  `SnapButton` in [`widgets/_toolbars.py`](../../src/pymmcore_gui/widgets/_toolbars.py).

### 2. Register it

In `widgets/_panels.py`, add a key, a factory and a `PanelInfo`:

```python
class PanelKey:
    ...
    FOCUS_MONITOR: Final = "focus_monitor"


def _create_focus_monitor(parent: QWidget, core: CMMCorePlus) -> QWidget:
    # Import inside the factory: panels are built on demand, so the
    # widget's module (and its dependencies) load only if it is opened.
    from pymmcore_gui.widgets._focus_monitor import FocusMonitor

    return FocusMonitor(parent=parent, mmcore=core)


PANELS: Final[tuple[PanelInfo, ...]] = (
    ...
    PanelInfo(
        key=PanelKey.FOCUS_MONITOR,
        title="Focus Monitor",
        icon="mdi:target",
        tooltip="Focus Monitor — show or hide the focus monitor panel",
        create=_create_focus_monitor,
    ),
)
```

The order of `PANELS` is the order of the toolbar buttons.

### 3. Decide whether it may stay enabled during an acquisition

While an MDA runs, `AcquirePage.set_mda_lock` disables every open panel
except those whose key is in `_MDA_UNLOCKED_PANELS` in `widgets/_acquire.py`.
Leave a new panel out of that set if it can change hardware state. Add it
only if it is read-only, or if, like the MDA editor, it disables its own
controls during a run.

### 4. Test it

The registry tests in `tests/test_gui.py` iterate over `PANELS`, so the new
button's icon, tooltip and default state are checked automatically. Add a test
that opens the panel:

```python
def test_focus_monitor_panel_opens(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    page = AcquirePage(mmcore)
    qtbot.addWidget(page)

    page.panel_button(PanelKey.FOCUS_MONITOR).click()

    assert isinstance(page.panel_widget(PanelKey.FOCUS_MONITOR), FocusMonitor)
    assert PanelKey.FOCUS_MONITOR in page.open_panels()
```

If the panel uses `default_open=True`, also add its dock name to the list of
docks built at startup in `test_acquire_panel_buttons_match_registry`.

## `PanelInfo` fields

<!-- markdownlint-disable MD013 -->

| Field | Default | Meaning |
| --- | --- | --- |
| `key` | required | Stable identifier; see [Keys are persisted](#keys-are-persisted). |
| `title` | required | Dock title. |
| `icon` | required | [Iconify](https://icon-sets.iconify.design/) name for the toolbar button, e.g. `"mdi:crop"`. |
| `tooltip` | required | Toolbar button tooltip, by convention `"Title — what it does"`. |
| `create` | required | Factory `(parent, core) -> QWidget`, called once on first open. |
| `area` | right | `LeftDockWidgetArea` docks into the MDA column; anything else joins the tabbed right column. |
| `default_open` | `False` | Open in the built-in "Default" layout. |
| `unstyle` | `False` | Run `unstyle_widgets()` on the widget; see below. |
| `refresh` | `None` | Called to resync the widget with the core; see below. |
| `always_visible` | `False` | The button cannot be hidden from Preferences > Show Widgets. |

<!-- markdownlint-enable MD013 -->

### `unstyle`

Third-party widgets sometimes set their own style sheets, which override the
app's themed style. `unstyle=True` clears those style sheets on the widget and
all its children (sliders excepted) and gives its buttons the app's look. Set
it for upstream widgets that look out of place; your own widgets should not
need it.

### `refresh`

`refresh(widget)` is called on the next event-loop turn after the panel is
opened, and every time the Acquire tab is shown while the panel is open. Use
it for core state that changes without a signal. Devices added on the
Hardware Setup tab are loaded without `systemConfigurationLoaded`, and
configuration groups saved from the Configurations tab emit no signals. A
widget that only depends on signalled changes does not need it.

## Keys are persisted

The key is saved as-is: the panel's dock is named `acquire_<key>`, which the
dock manager uses to restore positions, and open or hidden panels are stored
by key in the settings file and in every saved layout. Renaming a key loses
the panel's saved position and open state for existing users. Pick a key once
and keep it.

A panel added in a new release appears for existing users with its button
visible and the panel closed, until they open it. Saved layouts that predate
it are unaffected.

## Using the widget outside the Acquire page

`pymmcore_gui.actions.widget_actions` provides public `create_*` factories
(exposed through `pymmcore_gui.WidgetAction`) for some of the same widgets.
Factories for widgets that are also panels call the panel factory with the
core of the window that hosts the parent. To expose a new panel the same way,
add a `WidgetAction` member, a `create_*` function that calls your panel
factory with `_get_core(parent)` (instead of building the widget a second
time), and a `WidgetActionInfo` tying the two together.

## Adding a top-level tab

The window's tabs (Installation, Hardware Setup, Configurations, Acquire,
Smart Microscopy) are pages in a `QStackedWidget`, built in
`MicroManagerGUI.__init__` in
[`_main_window.py`](../../src/pymmcore_gui/_main_window.py). To add one:

1. Subclass `TabPage` ([`widgets/_tab_page.py`](../../src/pymmcore_gui/widgets/_tab_page.py)),
   which provides `toolbar`, `left` and `content` regions, and take the core
   in its constructor.
2. In `MicroManagerGUI.__init__`, create it with `self._mmc` and add it to
   `self._stack`. Add its label to `TAB_LABELS` at the same position: the
   tab index is the stack index.
3. Add it to the pages listed in `_on_mda_running`, which disables every
   tab except the one running the acquisition, and, if it can change
   hardware state, to those in `_on_pixel_calibration_running`.
4. If the page starts its own acquisitions, add a `RunOwner` member in
   [`_run_owner.py`](../../src/pymmcore_gui/_run_owner.py), call
   `RunOwnership.claim()` with it just before starting a run (as the Smart
   Microscopy page does), and return the page for it from
   `MicroManagerGUI._run_owner_page`. The window then keeps the user on that
   page during the run, and other pages' viewers ignore it.
5. If it owns timers, threads or workers, stop them from
   `MicroManagerGUI.closeEvent`, as `self._acquire.shutdown()` and
   `self._smart.shutdown()` do.
