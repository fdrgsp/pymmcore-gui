# ome-writers: upstream PRs

`fdrgsp/ome-writers@cite` against `pymmcore-plus/ome-writers@main`, merge base
`c8ef044` (2026-11-14). See [README.md](README.md) for the conventions and the
cross-repository chains.

The fork adds:

- **Multi-file OME-TIFF fixes**: per-frame metadata and frame counts went to
  the wrong file, the companion path was absolute, and file names collided
  (OW-1 to OW-5).
- **A `self-contained` multi-file mode**: every TIFF stands on its own (OW-6).
- **Speed with many positions**: one shared OME model, a bounded writer pool,
  concurrent Zarr creation (OW-7 to OW-9).
- **Scratch format**: where spill files go, and deleting them as soon as the
  data is released (OW-10).
- **useq conversion**: position and grid axes may be apart when only one of
  them varies (OW-11).

pymmcore-gui uses OW-6 (export and save options), OW-10 (`spill_dir`, in
`widgets/_mda_widget.py`), and the speed fixes for HCS plates with hundreds of
positions.

## Summary

<!-- markdownlint-disable MD013 -->
| ID | Type | Title | Size | Needs | Pairs with |
| --- | --- | --- | --- | --- | --- |
| OW-1 | fix | Per-frame metadata goes to the file holding the position's Image | S | – | – |
| OW-2 | fix | Frames written are recorded in that file too | S | OW-1 | – |
| OW-3 | fix | `BinaryOnly` metadata file is a relative name | S | – | – |
| OW-4 | test | Strict xfail for tifffile's cross-file read bug | S | – | – |
| OW-5 | fix | Unique OME-TIFF names for grids and plates | M | – | – |
| OW-6 | feat | `multi_file_metadata="self-contained"` | M | after OW-1 | – |
| OW-7 | perf | One OME model shared by redundant-mode files | M | OW-2 | – |
| OW-8a | perf | Create TIFF writers lazily, off the acquisition thread | M | upstream #150 | – |
| OW-8b | perf | Bounded writer pool and `max_writer_threads` | M | OW-8a | – |
| OW-9 | perf | Create OME-Zarr groups and arrays concurrently | M | yaozarrs first | – |
| OW-10a | feat | `ScratchFormat.spill_dir`, delete spill files once released | S | – | – |
| OW-10b | perf | Reopen only the requested scratch position | S | OW-10a | – |
| OW-11 | fix | Position and grid axes need not be adjacent when one doesn't vary | M | – (last test: USEQ-10) | – |
| OW-12 | build | `fallback-version` for tagless git installs | S | – | – |
| OW-13 | test | Read local Zarr arrays without Windows file URLs | S | – | – |
<!-- markdownlint-enable MD013 -->

---

## OW-1: per-frame metadata goes to the file holding the position's Image

*fix · S · `5d882cd`*

An `ome.Plane` is a child of `Pixels`, so per-frame metadata can only be stored
where the position's `Image` lives. Planes were sent to the mirror of the
frame's own *file*. In `master-tiff` and `companion-file` mode that file is a
`BinaryOnly` stub with no `Image`, so the frames were dropped without a word:
master-tiff kept only position 0's frames and companion-file kept none. Route
each position to the mirror and `Image` index that describe it
(`_plane_targets`), and move its `MapAnnotation` with it so the `AnnotationRef`
still resolves.

## OW-2: frames written are recorded in that file too

*fix · S · `0f2a3d6` · needs OW-1*

At close, each position's frame count (outer dimension size, `TiffData
PlaneCount`) went to its own mirror, which in the stub modes has no `Image`. An
unbounded or early-stopped run then reported `SizeT=1`. Use `_plane_targets`
from OW-1. Remove the now unused `OmeXMLMirror.own_pixels`. Upstream #139
(aaronkanzer, closed unmerged) fixed the same bug: credit it.

## OW-3: `BinaryOnly` metadata file is a relative name

*fix · S · `ac0000f`*

`master-tiff` and `companion-file` stored the absolute path of the master or
companion file, so moving or renaming the output folder orphaned every stub.
All files of a set share one folder, so a bare file name is enough, and it is
what the OME spec recommends.

## OW-4: strict xfail for tifffile's cross-file read bug

*test · S · `69d110c`*

`TiffPageSeries.dataoffset` checks that a series' pages are contiguous but not
that they are in one file, so reading a grouped multi-file OME-TIFF returns
another position's pixels (uncompressed only). Our XML is valid. A strict xfail
flags the day tifffile fixes it. Report the bug to tifffile at the same time.

## OW-5: unique OME-TIFF names for grids and plates

*fix · M · `509d812`, `bd2edfa`, `c25556c`*

Files are named by the flat position index (`p000`, `p001`, ...), which mixes
up stage positions and grid tiles, depends on the axis order, and says nothing
of the well. The new rules:

- A file is named after the position's stage identity (its well for a plate,
  else its name), plus `_r{row}_c{col}` when that identity has several tiles,
  e.g. `t_004_A1_p000_r000_c000.ome.tiff`.
- Positions without grid coordinates (`RandomPoints`) are separate positions,
  not tiles.
- A tie-break makes sure two positions never share a path.

This changes file names that users see: put a before/after table in the
description. `test_integration.py` calls the same
`_position_filename_suffixes()` helper, so the tests can't drift from the code.

## OW-6: `multi_file_metadata="self-contained"`

*feat · M (~220 lines) · `82d6471` · after OW-1, which rewrites the same
routing in `_ome_xml.py`*

Writes N independent single-position OME-TIFFs into one folder. Each file
describes only its own position and names no other file, so it can be opened,
moved or deleted on its own. In the other three modes, one file's removal
breaks the rest. The trade-off: readers see N images rather than one N-series
dataset, and a plate layout shrinks to the single well and field each file
holds. Adds a note to `ome-tiff-spec.md` explaining why this is not a fourth
spec mode.

## OW-7: one OME model shared by redundant-mode files

*perf · M · `6a21ef1`, and the TIFF part of `4f747c7` · needs OW-2*

In `redundant` mode every file carries the full N-image OME-XML, differing only
in its root UUID, but each file deep-copied the model and serialized it again,
so preparing and closing took O(N²). The mirrors now share one
`SharedOMEModel` whose XML is cached per change: 256 positions go from ~18 s to
~0.34 s, and 1024 from ~310 s to ~2.2 s. It also fixes stale sizes and missing
planes for sibling positions in each file.

## OW-8a: create TIFF writers lazily, off the acquisition thread

*perf · M · `cf3fd84`, `04221f2` · after upstream #150*

Open a position's TIFF file when its first frame arrives, not at setup for every
position, and never on the thread calling `append`.

Upstream #150 (tlambert03) changes the same writer thread so that its errors
reach `append`, `skip` and `close`: land it first and rebase on it.

## OW-8b: bounded writer pool and `max_writer_threads`

*perf · M · `74f7168`, `68616ca`, `6f81a77` · needs OW-8a*

A fixed pool of writer threads replaces one thread per position. Each position
is pinned to one thread, so its frames stay in order.
`OmeTiffFormat.max_writer_threads` (default 4, `ge=1`) sets the pool size: more
threads help on network storage. The `6f81a77` test fix finds a stalled writer
by thread, since threads are now shared.

## OW-9: create OME-Zarr groups and arrays concurrently

*perf · M · `0518fdd`*

Hundreds of positions meant about 1500 file operations one after another, slow
on network and Windows drives. `_yaozarrs_concurrent.py` creates each level of
the hierarchy concurrently (`thread_map` in `_util.py`) and reads the
`zarr.json` files back concurrently. The output is byte-identical.

It uses private yaozarrs helpers, so **propose it to yaozarrs first**
(`PlateBuilder` and `Bf2RawBuilder` creating levels concurrently). Here, only
the read-back and the writer lookup remain.

## OW-10a: `ScratchFormat.spill_dir`, delete spill files once released

*feat · S · `b817b77`*

- `ScratchFormat.spill_dir`: the parent folder for spill files (default: the
  system temp folder). It is created if missing.
- The spill folder is deleted when the backend is garbage collected
  (`weakref.finalize`), not at process exit. Repeated large in-memory runs used
  to fill the disk (Errno 28).
- After finalize, reads return read-only copies and `get_arrays()` always
  returns proxies, so no caller keeps a file mapped, which blocks deleting it
  on Windows.

## OW-10b: reopen only the requested scratch position

*perf · S · the scratch part of `4f747c7` · needs OW-10a*

After finalize, reading one position memmapped every position's file and
parsed the manifest. Reopen only the one requested, with the dtype kept in the
settings dump.

## OW-11: position and grid axes need not be adjacent when one doesn't vary

*fix · M · `748549b` · last test case needs USEQ-10*

`useq_to_acquisition_settings` flattens `p` and `g` into one position
dimension and rejected an `axis_order` in which they aren't adjacent. That is
only needed when both vary. With a single stage position, a single-tile grid or
no grid, the dimension takes the slot of the axis that varies
(`_position_dim_slot`), e.g. `"ptgc"` with one position. Add
`test_dims_match_event_arrival_order`, which checks every case against the real
event order.

The `single_position_subgrid_not_adjacent` case (a per-position grid in
`"ptgc"`) only holds once useq iterates sub-sequence axes in the root order:
leave it out until USEQ-10 lands.

## OW-12: `fallback-version` for tagless git installs

*build · S · `3431905`*

`[tool.hatch.version] fallback-version`, so `pip install git+...` of a fork or
a shallow clone without tags still builds.

## OW-13: read local Zarr arrays without Windows file URLs

*test · S · `1d11b74`*

With older yaozarrs, `to_zarr_python()` hands zarr-python a `file://` URL,
which Windows turns into a literal `/C:/...` path. The integration test opens
the known local path with `zarr.open_array` instead.
