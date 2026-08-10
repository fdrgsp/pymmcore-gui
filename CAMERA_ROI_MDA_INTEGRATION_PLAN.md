# Camera ROI / MDA / ndv integration plan

## Recommendation

This is a good consolidation. The ROI is an acquisition setting, so it belongs in
the MDA editor rather than in a separate dock. The MDA widget's `value()` and
`setValue()` should include it so an MDA file restores the same crop, but it should
be stored as JSON-safe metadata rather than as a new `useq.MDASequence` field:

```text
metadata["pymmcore_widgets"]["camera_roi"] = {
    "enabled": true,
    "camera": "Camera",
    "x": 0,
    "y": 0,
    "width": 512,
    "height": 512,
}
```

`useq.MDASequence` has no ROI field and ignores unknown fields. The crop mode and
"centered" state need not be persisted because both are derivable from the ROI and
the selected camera. Loading a sequence should restore the planned ROI in the UI
without unexpectedly changing hardware; applying/cropping it should remain an
explicit operation and the acquisition preflight should ensure that the planned
state is actually applied.

## 1. Move Camera ROI into the MDA editor and make it round-trip with the sequence

**UI and ownership**

- Create exactly one `CameraRoiWidget` as a child of `MemoryMDAWidget` in
  `src/pymmcore_gui/widgets/_mda_widget.py`, using the same `CMMCorePlus` instance.
- Wrap it in a `CollapsibleAcquisitionSection` titled **ROI**, with a checkbox and
  collapsed initial state. Insert it after **Time Series** and before **Saving**.
  Keep this as a local pymmcore-gui extension of
  `ActiveChannelCollapsibleCoreMDATabs`; ROI is not a useq axis and should not be
  added to `_AXES`, `axis_order`, or `usedAxes()`.
- Define the checkbox unambiguously: checked means that the saved ROI is part of
  the acquisition plan; unchecked means full-chip/no acquisition ROI. Show a
  compact summary such as `Off · Full chip` or `On · 512 × 512 at (128, 96)` and
  disable the body while unchecked. Avoid having checkbox state and the inner
  **Full Chip** mode represent contradictory states: selecting Full Chip should
  turn ROI off, while enabling ROI should restore the last valid non-full-chip
  rectangle (or initialize a centered rectangle).
- Give `CameraRoiWidget` a small public plan-state API if it does not have one by
  implementation time (for example, a serializable current-ROI getter and a
  signal-blocked setter). Do this upstream in `pymmcore-widgets` if practical;
  otherwise put the compatibility adapter in one local subclass instead of
  spreading accesses to `_get_roi_values()`, `_update_roi_values()`, or ndv/Qt
  internals throughout the GUI.

**Sequence behavior**

- Add one constant/schema helper for the `camera_roi` metadata payload. Validate
  integers, positive size, sensor bounds, and the saved camera name. If the named
  camera is not loaded or its sensor geometry has changed, retain the data but
  show a clear invalid/unavailable state rather than silently applying it to a
  different camera.
- Override `MemoryMDAWidget.value()` after `super().value()` to put the ROI payload
  under the existing `pymmcore_widgets` metadata namespace. Always persist the
  `enabled` flag so checked and unchecked states round-trip; store primitive values,
  not the widget's `CameraInfo` dataclass or its per-session `_cameras` dictionary.
- Extend `MemoryMDAWidget.setValue()` to restore the checkbox and ROI fields after
  the upstream axes/channels have been restored, with signals blocked so loading
  a file does not call `setROI`, snap, or create a burst of intermediate rectangles.
  A sequence without `camera_roi` metadata must remain backward-compatible and
  load with ROI off.
- At MDA preflight/run, validate and apply the enabled planned ROI before the sink
  and first frame are created. When ROI is off, explicitly use full-chip geometry
  so the result does not depend on stale hardware state from an earlier session.
  Keep the existing **Crop** button for immediate manual application and document
  that the applied ROI remains the camera state after the run, matching current
  `CameraRoiWidget` behavior.
- Remove the separate Camera ROI entry from the modern panel registry in
  `src/pymmcore_gui/_modern_gui/_panels.py`. Also remove the legacy toolbar/action
  entry in `src/pymmcore_gui/_main_window.py` and
  `src/pymmcore_gui/actions/widget_actions.py` once the legacy MDA factory uses the
  same embedded editor; do not leave two independently connected ROI widgets.
  Treat the old `camera_roi` persisted panel key as a migration case. Current key
  filtering handles the set of open panels, but the saved ADS byte state may still
  mention the deleted dock, so add a tested one-time fallback/reset (or layout
  version bump) instead of allowing startup restore to fail unpredictably.

**Tests and completion criteria**

- Update `tests/test_new_gui.py`, `tests/test_new_gui_settings.py`, and legacy main
  window tests to verify section order, one ROI widget per MDA editor, removal of
  the standalone panel/action, and safe restoration of layouts saved with the old
  panel.
- Add widget tests for enabled/disabled summaries, Full Chip invariants, YAML/JSON
  save-load and direct `value()`/`setValue()` round trips, old sequences with no
  ROI metadata, invalid/missing cameras, and exact application of `(x, y, w, h)`
  before the first MDA frame.
- This step is complete when the ROI can be configured and saved entirely from the
  MDA panel, reopening a sequence reproduces the plan without mutating hardware,
  and running it produces the intended camera dimensions without a standalone ROI
  dock.

## 2. Add a bidirectional live-view ROI selection session

**Interaction and architecture**

- Add a checkable **Select in Live View** button next to **Crop**. “Live” by itself
  is too ambiguous: this control starts/opens live preview as needed and enters ROI
  selection; it is not a second general-purpose live toggle.
- Have the embedded ROI widget emit a request signal. Let `AcquirePage` handle that
  request because it owns `LiveButton` and `AcquireViewersManager`; the MDA widget
  should not reach through parents or create an ndv viewer itself. Add an
  idempotent `ensure_live()` path to the existing live toolbar logic so this action
  starts live only when stopped and never turns an already-running stream off.
- Restrict camera-ROI editing to the snap/live `NDVPreview`, not MDA result viewers.
  Changing hardware ROI during an acquisition is unsafe, and an MDA viewer's data
  coordinates describe an acquired array rather than the current camera sensor.
- Introduce a small `CameraRoiSyncController` (owned by `AcquirePage` or the preview
  manager) that connects the embedded widget, core, and current preview viewer.
  Keep all ndv-specific adaptation in `MMArrayViewer`/this controller so a future
  ndv API change has one compatibility seam.

**Coordinate model and two-way synchronization**

- Use absolute camera coordinates in the same units accepted by
  `CMMCorePlus.getROI()`/`setROI()` as the canonical planned ROI, and map
  `(x, y, width, height)` to ndv's public bounding box form
  `((x, y), (x + width, y + height))`.
- Entering selection should temporarily display a full-chip live frame while the
  controller preserves the planned rectangle. This matters: after a hardware crop,
  ndv coordinates start at `(0, 0)` inside the cropped frame, so using that frame
  directly would offset the ROI and make it impossible to expand beyond the current
  crop. Ignore the transient full-chip `roiSet` as a plan edit, rebuild the preview,
  then draw the preserved rectangle over the full sensor image.
- Back the overlay with one ndv `RectangularROIModel`. Connect its
  `bounding_box` event to the ROI fields; normalize a drag with floor for the
  minimum corner and ceil for the maximum corner, then clamp/validate against the
  camera bounds. Connect `CameraRoiWidget.roiChanged` to update the same model so
  typed values immediately move/resize the overlay.
- Use a controller reentrancy guard plus signal blockers for programmatic field
  updates. Viewer-to-widget changes must not echo back indefinitely, and dragging
  must only edit the plan: it must not call `core.setROI()` on every mouse move.
  **Crop** (or MDA preflight) remains the commit point.
- Reuse the existing ndv ROI button/functionality through an app-level method such
  as `MMArrayViewer.begin_roi_selection()` rather than reaching into
  `_qwidget.add_roi_btn` from `AcquirePage`. If ndv exposes a public interaction-mode
  API by implementation time, use it directly. Keep both entry points synchronized:
  the MDA button opens/arms the ndv tool, and activating the ndv ROI tool while the
  preview is live attaches the same controller.
- On **Crop**, exit selection and hide the full-sensor overlay after applying the
  plan, because the rebuilt cropped preview has a different coordinate origin and
  shape. Also detach cleanly when the preview closes, live stops, the camera/system
  configuration changes, or an MDA starts. The saved/widget ROI remains intact and
  pressing **Select in Live View** starts a new full-sensor editing session.

**Tests and completion criteria**

- Add Qt tests proving that the button creates/selects the preview, starts live once
  when needed, leaves existing live acquisition running, and is unavailable while
  an MDA is active.
- Test both directions with non-zero origins: ndv drag to integer widget values and
  widget edits to the exact ndv bounding box. Include rounding, clamping, centered
  ROIs, full-chip reset, camera changes, and a regression assertion that one edit
  emits one logical update rather than recursively looping.
- Test the full workflow: begin with an already-cropped camera, enter selection,
  confirm a full-sensor preview without losing the planned crop, draw a larger ROI,
  press Crop, and confirm both `core.getROI()` and the rebuilt preview dimensions.
- This step is complete when either control can start the same selection session,
  the widget and overlay remain visibly synchronized, and only an explicit commit
  changes the hardware ROI.
