# pymmcore-gui

**pymmcore-gui** (the `pymm` command) is a desktop application for running a
microscope.  You load a hardware configuration, look at a live image, set up a
multi-dimensional acquisition, and run it, all without writing code.  It runs on
the Micro-Manager core, written entirely in Python, and aims to be a replacement
for Micro-Manager's Java GUI (MMStudio).

!!! info "Name pending"

    *pymmcore-gui* is a working title.  The application is usable today, but
    expect rough edges and frequent changes.

![The Acquire tab of pymmcore-gui, showing a two-channel z-stack time series
acquired with the demo devices](images/acquisition.png){ loading=lazy }

## New here?

You don't need a microscope to try it: Micro-Manager's demo devices simulate one.

1. [Install pymmcore-gui](install.md).
2. Follow [Your first acquisition](quickstart.md), which takes about five minutes
   and uses only the demo devices.
3. When you're ready for real hardware, see
   [Setting up your microscope](microscope.md).

## What the app is organized around

The window has four tabs.  You move through them from left to right the first time,
and after that you will mostly stay on **Acquire**.

| Tab | What it's for | How often |
| --- | --- | --- |
| [Installation](microscope.md#installation-tab) | Install the Micro-Manager device adapters. | Once |
| [Hardware Setup](microscope.md#hardware-setup-tab) | Say which devices your microscope has. | When the hardware changes |
| [Configurations](microscope.md#configurations-tab) | Define channels, objectives and pixel sizes. | When the settings change |
| [Acquire](acquiring.md) | Snap, go live, and run acquisitions. | Every day |

## Coming from Micro-Manager?

Your existing `.cfg` files work as they are, so you can skip straight to loading one.
Most ideas carry over: configuration groups, presets and the multi-dimensional
acquisition are all still here.  [Finding your way around](interface.md) shows where
things are.

!!! question "pymmcore-gui or napari-micromanager?"

    [napari-micromanager](../napari-micromanager/index.md) is built from the same
    widgets, but it's a [napari](https://napari.org) plugin in which images become
    napari layers.  If your analysis already lives in napari, it may suit you better.
    Otherwise, start here: pymmcore-gui uses the much lighter
    [ndv](https://github.com/pyapp-kit/ndv) viewer, and it's where new development is
    going.

## Getting help

- Ask a question on the [image.sc forum](https://forum.image.sc/tag/pymmcore-plus).
- Report a bug or request a feature on
  [GitHub](https://github.com/pymmcore-plus/pymmcore-gui/issues).
- Chat with the developers on
  [Zulip](https://imagesc.zulipchat.com/#narrow/channel/442785-pymmcore.5B-plus.5D).
