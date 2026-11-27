# useq-schema: upstream PRs

`fdrgsp/useq-schema@cite` against `pymmcore-plus/useq-schema@main`, merge base
`87349e5` (2026-11-21). See [README.md](README.md) for the conventions and the
cross-repository chains.

The fork adds:

- **Software autofocus**: an action and a plan whose routine the acquisition
  engine defines (USEQ-6).
- **More hardware autofocus options**: every-N intervals, a Z search range,
  keeping the device locked (USEQ-4, 5, 7).
- **`GridFromShape`**: tiles that cover a rectangle or an ellipse (USEQ-8).
- **`FocusSurface`**: each event's Z from manually focused points (USEQ-9).
- **One behavior change**: a position sub-sequence's axes follow the root
  `axis_order` (USEQ-10).

**Iteration.** The fork rewrote v1 `_iter_sequence` from a recursion into
position sub-sequences to one flat merge of per-position branches (USEQ-10).
USEQ-3 and USEQ-9b also change iteration. Port them to upstream's *recursive*
`_iter_sequence`, so that they don't wait for USEQ-10, which needs an issue
first. `useq.v2` is untouched on `cite` (`29596be` restored it), so v2 gets none
of these features: say so in each PR description.

## Summary

<!-- markdownlint-disable MD013 -->
| ID | Type | Title | Size | Needs | Pairs with |
| --- | --- | --- | --- | --- | --- |
| USEQ-1 | fix | Plate positions keep z unset (upstream #276) | S | – | WDG-39 |
| USEQ-2 | refactor | Shared base class for autofocus plans | S | – | – |
| USEQ-3 | fix | Axis triggers keep no state between iterations | S | USEQ-2 | – |
| USEQ-4 | feat | Every-N positions, grid tiles and timepoints | S | USEQ-2 | WDG-36b |
| USEQ-5 | feat | Z search range for hardware autofocus | S | – | PLUS-16, WDG-36c |
| USEQ-6 | feat | Software autofocus action and plans | M | USEQ-2 | PLUS-19, WDG-36d |
| USEQ-7a | feat | `MDASequence.plans_at()` | S | USEQ-2 | PLUS-20 |
| USEQ-7b | feat | `keep_locked` for hardware autofocus | S | USEQ-7a | PLUS-20, WDG-36e |
| USEQ-8a | feat | `GridFromShape` | M | – | WDG-29 |
| USEQ-8b | feat | `WellPlatePlan` rotates a `GridFromShape` with the plate | S | USEQ-8a | WDG-52 |
| USEQ-9a | feat | `FocusSurface` model | M | – | WDG-37a |
| USEQ-9b | feat | `MDASequence.focus_surface` | S | USEQ-9a | WDG-37b |
| USEQ-10 | feat! | Sub-sequence axes follow the root `axis_order` | L | issue first | PLUS-14, OW-11, WDG-33c |
<!-- markdownlint-enable MD013 -->

---

## USEQ-1: plate positions keep z unset

**Don't open a PR: review #276 (alandolt).** It is the fork's `0050879`
(position part): `PositionBase.__add__` turned a missing base z plus a relative
offset z (`RelativePosition.z` defaults to 0) into z=0, so every
`WellPlatePlan` position had z=0 and the engine moved the focus to 0 at every
well. With the fix, a base without z gives a sum without z. If #276 lacks the
fork's `tests/test_stage_positions.py` case, suggest it there.

## USEQ-2: shared base class for autofocus plans

*refactor · S · from part of `4a95391`*

Split the plan code so that later PRs only add code:

- `_autofocus_base.py`: `_AutofocusPlanBase` (`event`, `autofocus_event`,
  abstract `as_action`, `should_autofocus`) and `_AxesTrigger` (`axes`,
  `should_autofocus`).
- `_hardware_autofocus.py`: `AutoFocusPlan` and `AxesBasedAF` on top of them.
- `_autofocus.py`: `AnyAutofocusPlan` (only `AxesBasedAF` for now).

No behavior change, and every public name stays importable from `useq`. Keep
`event()` as it is (it only subtracts a relative Z plan's value): USEQ-7a fixes
the channel offset.

## USEQ-3: axis triggers keep no state between iterations

*fix · S · from `770797e`, `eefc280` · needs USEQ-2*

`AxesBasedAF` stores the previous event's index on itself (`_previous`), which
causes two bugs:

1. Iterating the same sequence a second time, or a deep copy of it (pydantic
   copies private attributes), skips the first autofocus.
2. A position with its own plan (per-position autofocus offsets) only compares
   against its own previous event. In a `tp` acquisition, `p` never changes on a
   revisit, so autofocus runs only at the first timepoint.

The fix adds a stateless `triggers_after(previous, event)` and makes the
iteration remember the previous event of the whole run, whichever plan it had.
`should_autofocus(event)` called on its own keeps its current behavior.

**Port:** the fork keeps `af_previous` in its flat loop. In upstream's recursive
`_iter_sequence`, pass a small mutable holder down the recursion instead.

## USEQ-4: every-N positions, grid tiles and timepoints

*feat · S · from `dd7ceec` and the `every_n_timepoints` part of `4a95391` ·
needs USEQ-2 · pairs with WDG-36b*

`every_n_positions`, `every_n_grid_positions` and `every_n_timepoints` (all
`ge=1`, default 1) on `_AxesTrigger`. `every_n_timepoints` is MMStudio's "skip
frames", useful when each autofocus run is expensive.

## USEQ-5: Z search range for hardware autofocus

*feat · S · from the hardware part of `4a95391` · pairs with PLUS-16, WDG-36c*

`search_below_um`, `search_above_um`, `search_step_um` on `HardwareAutofocus`
and `AutoFocusPlan`, with a validator: a range needs a non-zero step. A device
only locks within a limited range, so after a large move (new well, tilted
sample) `fullFocus` fails, and these let the engine step through a range. They
default to 0, so nothing moves Z unless asked. Also fixes `as_action()`, which
dropped the plan's `max_retries`.

## USEQ-6: software autofocus action and plans

*feat · M · from the rest of `4a95391` · needs USEQ-2 · pairs with PLUS-19,
WDG-36d*

- `SoftwareAutofocus` action: `method`, `focus_device`, `settings` (a dict that
  must serialize to JSON) and `max_retries`.
- `SoftwareAutofocusPlan` and `SoftwareAxesBasedAF`.
- `AnyAutofocusPlan` becomes a discriminated union: a plan that names a
  `method` is a software plan. It must be explicit because `FrozenModel`
  ignores extra fields, so a plain union would parse a software plan as a
  hardware one and drop `method` without a word.
- `_ensure_json_serializable`, shared with `CustomAction.data`.
- `docs/schema/software_autofocus.md`, `docs/schema/event.md` and the mkdocs
  nav entry.

For the description: useq names a routine but doesn't define which exist. That
keeps the schema free of any engine's algorithms, and a saved sequence keeps the
exact settings it ran with. Link pymmcore-widgets #553 (gcharvin), which stores
the settings in widget metadata instead.

## USEQ-7a: `MDASequence.plans_at()`

*feat · S · from the `plans_at` part of `6975368` · needs USEQ-2 · pairs with
PLUS-20*

`plans_at(index) -> EventPlans(autofocus_plan, z_plan, channel, z_shift)`
returns the plans an event was built from. A position's sub-sequence overrides
only the plans it defines. `z_shift` is the relative Z plan value plus the
channel Z offset. `useq.EventPlans` is exported.

It carries one fix: `autofocus_event` now starts from the position's own Z,
removing the channel's Z offset as well as the relative Z plan value, so an
autofocus before an offset channel no longer starts off-focus.

## USEQ-7b: `keep_locked` for hardware autofocus

*feat · S · from `74c4c9e`, `6975368` · needs USEQ-7a · pairs with PLUS-20,
WDG-36e*

`keep_locked: bool = False` on `HardwareAutofocus` and `AutoFocusPlan`. This is
only a policy: the engine keeps continuous focus (e.g. Nikon PFS) on between the
searches the plan schedules, and suspends it only for Z moves it can't make
while locked. With `axes=()` it keeps focus without any scheduled search.

Port only the final form from `6975368`. `74c4c9e`'s "hold" events and
`find_focus` field were removed again.

## USEQ-8a: `GridFromShape`

*feat · M (~350 lines, ~470 test lines to trim) · from `b31d88d` and the
geometry part of `0050879` · pairs with WDG-29*

A relative plan tiling a rectangle or an ellipse centered on the origin (squares
and circles via equal width and height), on a centered lattice of FOVs:

- `placement="cover"` (default) keeps every tile whose footprint touches the
  area; `"inside"` keeps only tiles entirely inside it.
- `rotation` rotates the area, not the tiles, which stay aligned with the stage.
- `tile_coordinates()` is cached by a pure `lru_cache` function, so model
  equality is unaffected, and it refuses more than 1,000,000 candidate cells.
- It joins `RelativeMultiPointPlan` and `MultiPointPlan` *before*
  `GridWidthHeight`, so a dump never reloads as `GridWidthHeight`.
- Adds `examples/grid_from_shape.py`.

## USEQ-8b: `WellPlatePlan` rotates a `GridFromShape` with the plate

*feat · S · from the `_plate.py` part of `b31d88d` · needs USEQ-8a · pairs
with WDG-52*

A `GridFromShape` used as `well_points_plan` takes the plate's rotation unless
it sets its own. Docstring updated.

## USEQ-9a: `FocusSurface` model

*feat · M (~170 lines, ~270 test lines) · from `7ef655b`, `22e10ce`, `8094a5b`,
`7b64e95` · pairs with WDG-37a*

`useq.FocusSurface(points, method)`: a surface `z = f(x, y)` through manually
focused `(x, y, z)` points.

- `"linear"` interpolates on a Delaunay triangulation and continues outside the
  hull with the slope of the least-squares plane. `"plane"` is that plane.
- It is validated on construction: at least 3 points, no two at the same XY,
  not all on a line.
- `evaluate(xy)` returns the Z values and `hull()` the convex hull.
- It uses shapely's `delaunay_triangles`. shapely is already a dependency.

The model alone, with no sequence field, so the geometry gets its own review.

## USEQ-9b: `MDASequence.focus_surface`

*feat · S · from the `_mda_sequence.py` and `_iter_sequence.py` parts of the
same commits · needs USEQ-9a · pairs with WDG-37b*

An event at a position without its own z gets the surface Z at the event's XY
(each grid tile and plate image has its own), then adds its relative Z plan
value and channel offset. A position's own z wins, and so does a position
sub-sequence's own surface. A surface together with an absolute Z plan (global
or per position) is rejected. The surface is evaluated once per distinct XY.

**Port:** add a `surface_z` callable to `_xyzpos` and pass the active surface
down upstream's recursion.

## USEQ-10: sub-sequence axes follow the root `axis_order`

*feat! · L (rewrite of `_iter_sequence.py`, expected events changed in
`tests/fixtures/cases.py`) · from `d471e44`, `16eaf86`, `29596be` · pairs with
PLUS-14, OW-11, WDG-33c*

**Open an issue first.** Upstream runs a position's sub-sequence inside that
position, in the sub-sequence's own order, and the root `axis_order` has no say.
The issue can show both with these sequences (output of upstream `main` and of
`cite` on 2026-11-26):

```python
MDASequence(
    axis_order=order,
    channels=["DAPI", "FITC"],
    stage_positions=[{"sequence": {"grid_plan": {"rows": 1, "columns": 2}}}],
)
# upstream, "pgc" and "pcg" alike: (g0,DAPI) (g1,DAPI) (g0,FITC) (g1,FITC)
# cite "pgc": (g0,DAPI) (g0,FITC) (g1,DAPI) (g1,FITC);  cite "pcg": as upstream

MDASequence(
    axis_order="tpc",
    time_plan={"interval": 0, "loops": 2},
    channels=["DAPI"],
    stage_positions=[
        {"sequence": {"time_plan": {"interval": 0, "loops": 3}}},
        {},
    ],
)
# upstream: 8 events. Position 0 runs all 3 of its timepoints inside each of
#           the 2 root timepoints, so it is imaged 6 times.
# cite:     5 events. t0 p0, t0 p1, t1 p0, t1 p1, t2 p0
```

On `cite`, each position is a sparse branch: its sub-sequence overrides the root
plans it redefines, adds any axis of its own, and the branches are merged lazily
in the root order. An axis that only one position has still respects the root
order without duplicating events for the others. `keep_shutter_open_across`
counts an axis present in only one of two adjacent events as changed
(`16eaf86`). `PositionBase.is_relative` comes with it.

If the maintainers would rather keep v1 as it is, propose the same semantics for
`useq.v2`, whose iteration (`iterate_axes_recursive`) already takes a
`parent_order`. In that case PLUS-14 and the last case of OW-11 wait for it.
