# TODO

- Remove the 100 frames from the `NDVPreview`; show a single image. ✅

- Stop live mode when pressing `Run` in the `MDAWidget` (add to `_MDAWidget`). ✅

- `ConfigWizard` background is white on Windows 11 and many elements are invisible.

- `ConfigWizard` connection-parameter window is too small and not resizable.

- Bug launching GUI without the config `-c` flag; it gave an error on Christina.

- `LMM5` reload fails in pymmcore-gui but works in Java Micro-Manager; likely a Hub
   reload issue.

- `OCToolBar` should refresh when adding a preset to the channel cfg.

- `ShuttersToolbar` should refresh after creating a new cfg via `ConfigWizard`.

- Subclass and modify `PixelConfigurationWidget`; its window remains open after closing
  the widget.

- `StagesControlWidget` is very large on Windows; adjust sizing.

- Add mouse wheel support to `StagesControlWidget` (ported from QI OpenSPIM).

- `StageExplorer` crashes the GUI when popped out from the main window.

- Unlink the `ndv` viewer when launching a scan from the `StageExplorer`.

- Refresh `StageExplorer` when pixel-size cfgs change (e.g., 10x -> 20x).

- `ndv` tab titles are not visible on Windows when not selected; fix contrast.

- Add z scale to `ndv` viewers for correct z projection.

- Show "file already exists" errors in `NotificationManager`, not only console.
