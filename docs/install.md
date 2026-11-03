# Installing pymmcore-gui

Installing has two parts: the application itself, and the Micro-Manager *device
adapters*, which are the drivers that let it talk to hardware.

## 1. Install the application

=== "Bundled application"

    The easiest way if you don't use Python.  Pre-built bundles include everything
    the app needs.  Download one, extract it, and run the application inside.

    | Platform | Latest nightly build |
    | :--: | :--- |
    | Windows | [pymmgui-Windows.zip][win-zip] |
    | macOS | [pymmgui-macOS.zip][mac-zip] |

    [win-zip]: https://nightly.link/pymmcore-plus/pymmcore-gui/workflows/bundle/main/pymmgui-Windows.zip
    [mac-zip]: https://nightly.link/pymmcore-plus/pymmcore-gui/workflows/bundle/main/pymmgui-macOS.zip

    !!! warning "Unsigned app"

        The bundles aren't signed yet, so your operating system will warn you about
        an app from an unidentified developer.  You'll need to accept that warning
        to run it.

=== "Python package"

    You'll need Python 3.12 or newer, and `git`.  Install into a fresh environment:
    pymmcore-gui requires a specific version of PyQt6, so it shouldn't share one with
    another Qt application.

    ```sh
    pip install git+https://github.com/pymmcore-plus/pymmcore-gui
    ```

    Or, to run it without installing anything permanently, use
    [uv](https://docs.astral.sh/uv/):

    ```sh
    uvx --from git+https://github.com/pymmcore-plus/pymmcore-gui pymm
    ```

    pymmcore-gui isn't released on PyPI yet.  (There *is* a `pymmcore-gui` package
    there, but it's an outdated pre-release, so please install from GitHub.)

    !!! tip

        The GitHub version may change at any time.  If you add it to your own
        project's dependencies, pin a specific commit:
        `pymmcore-gui @ git+https://github.com/pymmcore-plus/pymmcore-gui@<commit>`

## 2. Install the device adapters

Neither option above includes the device adapters, and without them the app has
nothing to control, not even the demo devices.

The simplest way is from inside the app: without any adapters installed, it opens
on the **Installation** tab, where you click **Install**
([more details](microscope.md#installation-tab)).

!!! warning "Apple Silicon Macs and Linux"

    Micro-Manager doesn't publish full builds for these platforms.  Installing there
    gets you the *test* adapters: enough for the demo devices and this documentation's
    examples, but no drivers for real hardware.  To run a real microscope on them,
    you'd need to build the device adapters yourself.  Windows and Intel Macs get the
    full set.

You can also install them from a terminal, or use a Micro-Manager you already have:

- Run `mmcore install` (or `uvx --from "pymmcore-plus[cli]" mmcore install`).
- Install a
  [Micro-Manager nightly build](https://micro-manager.org/Micro-Manager_Nightly_Builds).

See [Installing Micro-Manager device adapters](../pymmcore-plus/install.md#installing-micro-manager-device-adapters)
for more about how they're found.

## 3. Start the app

Double-click the bundled application, or run this in the environment where you
installed it:

```sh
pymm
```

(`mmgui` is the same command under another name.)

Next: [Your first acquisition](quickstart.md).

!!! note "Error reports"

    The first time you launch it, the app asks whether you'd like to send error
    reports to the developers.  This is opt-in: nothing is sent unless you agree, and
    the [privacy policy](https://github.com/pymmcore-plus/pymmcore-gui/blob/main/PRIVACY.md)
    lists what's included.  To skip the question and send nothing, start the app with
    `pymm run --no-telemetry`.

!!! note "License"

    The bundled application includes PyQt6 (GPL-3.0), so the bundle is distributed as
    a combined work under the GPL-3.0.  pymmcore-gui itself is BSD-3-Clause.
