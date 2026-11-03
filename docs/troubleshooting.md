# Troubleshooting

## The app won't load my configuration

The most common reason, particularly on a new computer, is that the device adapters are
missing or are the wrong version.  Check the
[Installation tab](microscope.md#installation-tab): one version should be marked active,
and installing the latest one is a good first step.

If a particular device fails, check that its vendor's driver is installed, that it's
turned on and connected, and that no other program (such as Micro-Manager) is using it.

## The Demo configuration isn't offered, or won't load

The demo devices come with the device adapters, so you need to
[install those](install.md#2-install-the-device-adapters) first.

## Buttons show a question mark instead of an icon

The app downloads its icons the first time it needs them, so without an internet
connection it can't show those it hasn't seen.  It all still works, and the icons appear
once it can download them.  If you're always offline, run the app once on a network first.

## A panel is missing

- If its button isn't in the toolbar, turn it on under **Show Widgets** in
  [Preferences](interface.md#preferences).
- To reset the whole arrangement, select the **Default** layout in Preferences.

## "No Pixel Size", or the Stage Explorer is unavailable

The current objective has no pixel size.  [Set one](microscope.md#pixel-sizes), or select
an objective that has one.

## An acquisition won't start because it's too big

With [saving off](acquiring.md#saving), a run has to fit in the memory allowed by
**Max in-memory size** in [Preferences](interface.md#preferences).  Turn on **Saving**, or
**Spill to disk when exceeded**, or raise the limit.

## Starting over

To put the app's settings back to their defaults (window layout, recent configurations and
preferences), run:

```sh
pymm settings --reset
```

Your `.cfg` files and saved images aren't affected.

## Reporting a problem

Reports are much more useful with the details of what went wrong.  Please include:

- what you did, and what happened;
- anything in the **Exception Log** panel;
- the output of `pymm --version`.

[Open an issue on GitHub](https://github.com/pymmcore-plus/pymmcore-gui/issues), or ask on
the [image.sc forum](https://forum.image.sc/tag/pymmcore-plus).  If the problem is with a
device rather than with this app, the
[core and device adapters troubleshooting page](../pymmcore-plus/troubleshooting.md) may
help as well.
