# Python devices

pyMM can drive devices written in Python, next to Micro-Manager's C++ device
adapters. It does so on pymmcore-plus's `UniMMCore`, and only when pyMM runs
from source (`pip install pymmcore-gui`), never in the standalone app.

## Quick start

1. Start pyMM on UniMMCore: `pymm -u`.
2. In **Hardware Setup**, under *Available devices*, set **Python devices** to
   the folder holding your device files (**Browse…**), then press **Load**.
   Your devices appear in the list with a "(Py)" badge.
3. Add them like any other device, then **Save…** the configuration.
4. From then on, start with `pymm -c scope.cfg`. A configuration with Python
   devices picks UniMMCore by itself and remembers the folder.

The core is chosen when pyMM starts and never changes while it runs. On the
standard core the **Python devices** row is disabled, and loading a
configuration with Python devices offers to load the rest without them. To
use them, close pyMM and reopen it with `pymm -c <cfg>` or `pymm -u`.

## Writing a device file

A device file is a top-level `.py` file in the folder. Its name is its module
name, so it must be a valid Python identifier (`my_laser.py`, not
`my-laser.py`) and must not clash with an installed module (`json.py`,
`serial.py`, …). Each public class it defines that subclasses a `unicore`
device base class is offered as a device. Its first docstring line is the
device's description.

```python
"""my_laser.py"""

from pymmcore_plus.experimental.unicore import ShutterDevice, pymm_property


class MyLaser(ShutterDevice):
    """A 488 nm laser on a serial port."""

    def initialize(self) -> None:
        # open the port here, not at import time
        import serial

        self._port = serial.Serial("/dev/ttyUSB0", 9600)

    def shutdown(self) -> None:
        self._port.close()

    def get_open(self) -> bool: ...

    def set_open(self, open: bool) -> None: ...

    @pymm_property(limits=(0, 100))
    def power(self) -> float: ...

    @power.setter
    def _set_power(self, value: float) -> None: ...
```

- Files starting with `_` are helpers: they aren't scanned, but device files
  can import them (`from _protocol import encode`). Subfolders aren't scanned.
- Classes starting with `_`, abstract classes, and classes imported from
  another module aren't offered.
- A Python hub (`HubDevice`) offers the device classes it lists in
  `get_installed_peripherals()` once it is added, like a C++ hub.

## What Load runs

**Load imports every device file in the folder**, so the code at the top
level of each file runs. Keep it to imports and definitions: touch hardware
in `initialize()`, which runs when the device is added or a configuration is
loaded. There is no file watcher; press **Load** again after editing a file.

While a device from the folder is loaded, **Load** doesn't re-import the files
it uses, so old and new code never mix in one device. Edits apply when the
configuration is loaded again, or on the next **Load** once no device from the
folder is loaded. Switching to another folder is refused until its devices are
removed.

Files that couldn't be used are counted above the row; **Details** lists why,
with the full tracebacks.

## Dependencies

Device files often need other packages: a vendor SDK, `pyserial`, `nidaqmx`.
Install them **in the Python environment pyMM runs in**, before running
`pymm`. pyMM never installs anything. When a package is missing, pyMM names
it and the interpreter it needs to go into. Depending on how pyMM was
installed:

<!-- markdownlint-disable MD013 -->
| pyMM installed with | Add a package with |
| --- | --- |
| a virtual environment | `uv pip install nidaqmx` (or `pip install nidaqmx`) inside it |
| `uv tool install pymmcore-gui` | `uv tool install pymmcore-gui --with nidaqmx` |
| run with `uv run` | `uv run --with nidaqmx pymm` |
| `pipx install pymmcore-gui` | `pipx inject pymmcore-gui nidaqmx` |
<!-- markdownlint-enable MD013 -->

The import name can differ from the name to install: `import cv2` needs
`opencv-python`, `import serial` needs `pyserial`.

## The configuration file

Python devices are saved on lines starting with `#py` and a space, which
Micro-Manager and the standard core read as comments. pyMM also records the
device folder:

```text
#@PyDevicesPath,/Users/me/microscope/py_devices
Property,Core,Initialize,0
Device,Camera,DemoCamera,DCam
#py pyDevice,Laser,my_laser,MyLaser
Property,Core,Initialize,1
#py Property,Laser,power,10
#py Property,Core,Shutter,Laser
```

- `#py pyDevice,<label>,<module>,<class>` loads a device. A module from an
  installed package (`mypkg.lasers`) needs no folder.
- A relative `#@PyDevicesPath` resolves against the configuration's folder,
  so the two can move together. Saving writes the absolute path.
- If the saved folder no longer exists, the active one is used instead and
  the next save records it. With no folder active, the load summary offers
  **Locate folder…**.

## When a device fails to load

A broken Python device never stops a configuration from loading. A device
that fails to import, to be created, to initialize, or to answer while the
configuration loads is skipped, along with every line that refers to it (and,
for a hub, its peripherals). Pixel calibrations depending on a skipped device
are removed because their device conditions can no longer be checked.
A setting that fails on a device that loaded is
skipped alone. One summary lists all of them after the load. C++ device
errors still fail the load, as in Micro-Manager.

If a device's `shutdown()` also fails, it is removed from pyMM anyway, but its
port or camera may still be held. Restart pyMM before loading again.

Saving over a configuration whose Python devices or settings didn't load
would drop them from the file, so pyMM asks first: **Save anyway**, or
**Save As…** to keep the original.
