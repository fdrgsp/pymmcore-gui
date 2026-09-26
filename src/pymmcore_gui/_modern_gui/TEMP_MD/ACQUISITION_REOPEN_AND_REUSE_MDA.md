# Reopen Acquisitions and Re-use Their MDA Settings

## Scope

Implement this feature in the default modern GUI for the acquisition formats that
pyMM already writes:

- OME-TIFF (`.ome.tif` and `.ome.tiff`)
- OME-Zarr (`.ome.zarr` directories)
- Multi-position OME-TIFF directories produced by pyMM

“Reopen as acquired” means restoring the acquisition's actual dimensions,
coordinates, channel names, physical scales, and recorded `MDASequence`. Transient
NDV state such as manually adjusted contrast is not currently persisted and is
therefore outside this scope.

## 1. Add a metadata-preserving acquisition loader

Create `src/pymmcore_gui/_acquisition_loader.py` with a `LoadedAcquisition` model
that contains:

- A lazy, NDV-compatible data wrapper
- The actual on-disk dimensions and coordinates
- Physical scales
- The original `MDASequence`, when available
- The source path and display title
- A resource-cleanup callback

Extract `mda_sequence` from the metadata locations written by the existing
acquisition/export pipeline:

- OME-TIFF: the `pymmcore_plus` OME map annotation
- OME-Zarr: `pymmcore_plus.summary_metadata.mda_sequence`

Validate the recovered value with `MDASequence.model_validate()`.

The displayed shape must come from the data on disk rather than from the saved
sequence. This ensures that cancelled and partial acquisitions reopen with only the
frames that were actually written. Pixel access should remain lazy so opening a
large acquisition does not materialize the entire array in memory.

Files with valid image data but no pyMM sequence metadata should still open. They
simply will not offer the **Re-use MDA…** action.

## 2. Generalize viewer creation

Refactor `src/pymmcore_gui/_modern_gui/_acquire_viewers.py` so live and reopened
acquisitions use one viewer-creation path.

Extend `_ViewerRecord` to retain:

- The acquisition's original sequence
- Its source title/path
- A loader cleanup callback
- The existing live acquisition and sink information

Add an `open_acquisition(path)` entry point that loads the dataset, creates an
`MMArrayViewer`, attaches the recovered metadata, and creates a viewer dock.

For a dropped dataset, use `path.name` as the dock-tab title. Live acquisitions can
keep the existing `MDA <uid>` title. Continue using a separate unique object name so
two files with the same basename do not collide internally.

When a loaded viewer closes, release any open TIFF handles, memory maps, or other
backing resources through the record's cleanup callback.

## 3. Add application-level drag and drop

Update `src/pymmcore_gui/_modern_gui/_main_win.py` to:

- Enable drops on `MainWindow`
- Accept supported local file and directory URLs
- Route every valid dropped acquisition to
  `AcquireViewersManager.open_acquisition()`
- Switch to the Acquire page after the first dataset opens successfully
- Reject unsupported URLs without intercepting unrelated drag operations
- Report invalid or corrupt datasets through the existing notification/status
  mechanism without crashing the application

Dropping multiple supported datasets may open one viewer tab per dataset.

## 4. Add the NDV context-menu action

Extend the existing event-filter integration in
`src/pymmcore_gui/_array_viewer.py`. The current filter is already installed on both
the NDV widget and its canvas, making it the appropriate compatibility seam for a
right-click menu.

When the viewer has an associated `MDASequence`, its context menu should include:

> Re-use MDA…

Selecting the action should emit or invoke a callback with the exact sequence and
source title belonging to that viewer. The action should be absent or disabled for
snap/live previews and datasets without sequence metadata. It should also be
disabled while an MDA is running.

Expose this through an `AcquireViewersManager.reuseMDARequested(sequence,
source_title)` signal rather than letting the viewer reach directly into the MDA
widget.

## 5. Confirm before replacing the MDA parameters

Connect `reuseMDARequested` in
`src/pymmcore_gui/_modern_gui/_acquire.py` after the `MemoryMDAWidget` has been
created.

The handler should show a confirmation dialog such as:

> Replace the current MDA parameters with those used to acquire `<filename>`?

The safe/default response should be Cancel or No.

If the user declines, leave the current MDA unchanged. If the user confirms:

1. Call `MemoryMDAWidget.setValue(sequence)`.
2. Open and focus the MDA dock so the restored parameters are immediately visible.
3. Preserve the current saving destination and output format.

Preserving the saving controls is important: reusing acquisition parameters must
not make the next run overwrite the dataset that was just opened. The existing
`MemoryMDAWidget.setValue()` override should remain the single path for restoring
the sequence because it already avoids applying the selected channel row to the
microscope during programmatic restoration.

## 6. Tests

### Loader tests

Add focused tests for `src/pymmcore_gui/_acquisition_loader.py`. Generate fixtures
through the real `ome-writers`/`export_acquisition()` path and verify:

- OME-TIFF and OME-Zarr round trips
- Multi-position acquisitions
- Axis order and actual on-disk shape
- Channel and position coordinates
- Physical scales
- Exact `MDASequence` recovery
- Cancelled or partial acquisitions
- Missing and malformed `pymmcore_plus` metadata
- Lazy pixel access
- Cleanup of opened resources

### GUI tests

Extend `tests/test_new_gui.py` and, where appropriate,
`tests/test_array_viewer.py` to cover:

- Accepting supported drag-and-drop paths
- Rejecting unsupported paths
- Routing dropped paths to the viewer manager
- Switching to the Acquire page after a successful drop
- Using the dropped filename as the viewer tab title
- Opening multiple datasets in separate tabs
- Showing **Re-use MDA…** for both live and reopened acquisitions
- Omitting or disabling the action when no sequence is available
- Emitting the sequence associated with the correct viewer
- Confirmation and cancellation behavior
- Leaving the current MDA unchanged after cancellation
- Calling `MemoryMDAWidget.setValue()` after confirmation
- Preserving the saving controls
- Disabling reuse while an acquisition is running
- Releasing file resources when the viewer closes
- Reporting corrupt datasets without crashing

## Acceptance criteria

- Dropping a pyMM-created OME-TIFF or OME-Zarr dataset opens it in a new NDV tab.
- The tab title is the dataset's filename.
- Axes, coordinates, labels, scales, and the actual acquired extent are restored.
- Opening a dataset does not eagerly load the entire acquisition into memory.
- Right-clicking a viewer with sequence metadata offers **Re-use MDA…**.
- The MDA editor is changed only after explicit confirmation.
- Cancelling the confirmation leaves the existing MDA parameters untouched.
- Reusing a sequence does not change the current save destination or format.
- Files without pyMM sequence metadata can still be viewed but cannot overwrite the
  MDA editor through this action.
- Closing a reopened acquisition releases its file-backed resources.
