# Scripting the app

You don't need any of this to use the app.  It's here for when you want to automate
something that the interface doesn't do.

## The console

Click the **Console** button in the Acquire toolbar for an IPython console inside the
running app.  It's the same session the app is using, so what you do there happens on
the microscope, and the interface follows.  It starts with these names defined:

| Name | What it is |
| --- | --- |
| `mmc` (also `core`, `mmcore`) | The [`CMMCorePlus`][pymmcore_plus.CMMCorePlus] the app is using |
| `mda` | Its acquisition runner, `mmc.mda` |
| `window` | The main window |
| `acquire` | The Acquire tab |
| `panels` | The Acquire tab's panels, such as `panels.mda` |
| `np`, `useq` | NumPy and useq-schema, along with useq-schema's names, such as `MDASequence` |

For example, this updates the exposure everywhere in the app:

```python
mmc.setExposure(50)
```

And this runs an acquisition, keeping the images in memory and showing them in a new
viewer tab:

```python
mmc.run_mda(MDASequence(time_plan={"interval": 1, "loops": 5}), output="memory")
```

Without `output`, the acquisition still runs, but no viewer opens.  See the
[pymmcore-plus guides](../pymmcore-plus/guides/mda_engine.md) for more about running
acquisitions from code.

## Launching from Python

Start the app from a script with [`create_mmgui`][pymmcore_gui.create_mmgui].  This call
returns when the window closes:

```python
from pymmcore_gui import create_mmgui

create_mmgui(mm_config="path/to/my_scope.cfg")
```

### Customizing before launch

Pass `exec_app=False` to get the window back before the Qt event loop starts, make your
changes, then start the loop yourself.  (`create_mmgui` makes the `QApplication` for you.)

```python
from pymmcore_gui import create_mmgui
from qtpy.QtWidgets import QApplication

window = create_mmgui(exec_app=False)

# customize the window, or use the core it's driving
window.mmcore.setExposure(50)

QApplication.instance().exec()
```

### Using your own core

By default, the app uses the global
[`CMMCorePlus.instance()`][pymmcore_plus.CMMCorePlus.instance], so any script in the same
process reaches the same core.  If you already have one with a configuration loaded, pass
it in with `mm_config=False`, so that the app doesn't load one on top of it:

```python
from pymmcore_gui import create_mmgui  # import this before creating the core
from pymmcore_plus import CMMCorePlus

core = CMMCorePlus()
core.loadSystemConfiguration()

create_mmgui(mmcore=core, mm_config=False)
```

The order of the imports matters: a core created before Qt has been imported delivers its
events on whichever thread they happen on, which the interface isn't built for.
