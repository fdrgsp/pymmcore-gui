# Command line and settings

`pymm` starts the app.  `mmgui` is the same command under another name.

## Starting the app

`pymm` is short for `pymm run`, which takes these options:

| Option | What it does |
| --- | --- |
| `-c FILE`, `--config FILE` | Load this configuration file, and skip the startup dialog. |
| `--demo-config` | Load Micro-Manager's demo configuration, and skip the startup dialog. |
| `-l NAME`, `--layout NAME` | Open with this saved [layout](interface.md#arranging-the-window).  Without `-c`, this preselects it in the startup dialog. |
| `--no-telemetry` | Don't ask about, or send, error reports. |

For example:

```sh
pymm run -c path/to/my_scope.cfg -l "Two viewers"
```

Spell out `run` for `--demo-config` and `--no-telemetry`.  (`-c` and `-l` work without it.)

## Other commands

| Command | What it does |
| --- | --- |
| `pymm --version` | Print the versions of pymmcore-gui, pymmcore-plus and pymmcore. |
| `pymm layouts` | List your saved layouts, and the folder they're in. |
| `pymm settings --reveal` | Show the settings file in your file browser. |
| `pymm settings --edit` | Open the settings file in your default editor. |
| `pymm settings --reset` | Restore the default settings. |

## What's in the settings file

The window's size and layout, your recent configurations, your
[Preferences](interface.md#preferences), and your answer about error reports.  Your
hardware configuration isn't kept here: that's in your `.cfg` file.
