"""Tests for `pymmcore_gui._acquisition_loader` -- no Qt required.

Fixtures are built through the real `ome_writers` path, with a global
"pymmcore_plus" metadata blob shaped exactly like `OmeWritersSink`/
`export_acquisition` write it (see `pymmcore_plus.mda._sink._serialize_summary_meta`).
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pytest
import useq
from ome_writers import (
    AcquisitionSettings,
    OmeTiffFormat,
    OmeZarrFormat,
    create_stream,
    useq_to_acquisition_settings,
)

from pymmcore_gui._acquisition_loader import open_acquisition, supports_path

if TYPE_CHECKING:
    from ome_writers import Channel, Dimension

FORMATS = ("ome-tiff", "ome-zarr")


def _ch(*names: str) -> tuple[useq.Channel, ...]:
    """Real `Channel` objects -- `MDASequence.channels` is statically a
    `tuple[Channel, ...]`; pydantic coerces plain strings at runtime, but the
    stub doesn't reflect that, so tests spell it out explicitly."""
    return tuple(useq.Channel(config=n, exposure=10) for n in names)


def _fmt(name: str) -> OmeTiffFormat | OmeZarrFormat:
    """Same idea as `_ch`, for `AcquisitionSettings.format`."""
    return OmeTiffFormat() if name == "ome-tiff" else OmeZarrFormat()


def _write_with_sequence(
    seq: useq.MDASequence, fmt: str, root: Path, n_frames: int
) -> str:
    """Write `seq`'s acquisition, embedding it as real acquisition metadata."""
    dims = useq_to_acquisition_settings(seq, 8, 8, pixel_size_um=0.325)["dimensions"]
    settings = AcquisitionSettings(
        dimensions=tuple(dims), dtype="uint16", root_path=str(root), format=_fmt(fmt)
    )
    summary = {
        "format": "summary-dict",
        "version": "1.0",
        "mda_sequence": seq.model_dump(mode="json", exclude_unset=True),
    }
    with create_stream(settings) as stream:
        stream.set_global_metadata("pymmcore_plus", {"summary_metadata": summary})
        for i in range(n_frames):
            stream.append(np.full((8, 8), i, dtype="uint16"))
    return settings.output_path


def _write_bare(dims: list[Dimension], fmt: str, root: Path, n_frames: int) -> str:
    """Write an acquisition with no acquisition-level metadata at all."""
    settings = AcquisitionSettings(
        dimensions=tuple(dims), dtype="uint16", root_path=str(root), format=_fmt(fmt)
    )
    with create_stream(settings) as stream:
        for i in range(n_frames):
            stream.append(np.full((8, 8), i, dtype="uint16"))
    return settings.output_path


@pytest.mark.parametrize("fmt", FORMATS)
def test_recovers_sequence_and_lazy_data(tmp_path: Path, fmt: str) -> None:
    seq = useq.MDASequence(
        channels=_ch("DAPI", "FITC"),
        time_plan=useq.TIntervalLoops(interval=timedelta(seconds=0.5), loops=3),
        metadata={
            "pymmcore_widgets": {
                "save_dir": "/data/orig",
                "save_name": "acq",
                "format": "ome-tiff",
                "should_save": True,
            }
        },
    )
    ext = "ome.tiff" if fmt == "ome-tiff" else "ome.zarr"
    out = _write_with_sequence(seq, fmt, tmp_path / f"a.{ext}", n_frames=6)

    assert supports_path(out)
    loaded = open_acquisition(out)
    try:
        assert loaded.sequence is not None
        assert loaded.sequence.replace(uid=seq.uid) == seq
        assert loaded.title == Path(out).name
        assert loaded.source_path == Path(out)
        # the wrapper is real, lazy, ndv-compatible data -- not re-parsed
        c_axis = loaded.wrapper.dims.index("c")
        idx: dict[int, int | slice] = dict.fromkeys(range(len(loaded.wrapper.dims)), 0)
        idx[c_axis] = 1
        for i, name in enumerate(loaded.wrapper.dims):
            if name in ("y", "x"):
                idx[i] = slice(None)
        frame = loaded.wrapper.isel(idx)
        assert frame.shape == (8, 8)
    finally:
        loaded.close()


@pytest.mark.parametrize("fmt", FORMATS)
def test_no_sequence_metadata_still_opens(tmp_path: Path, fmt: str) -> None:
    """A file with valid image data but no pyMM sequence metadata still opens."""
    seq = useq.MDASequence(
        channels=_ch("DAPI"),
        time_plan=useq.TIntervalLoops(interval=timedelta(seconds=1), loops=4),
    )
    dims = useq_to_acquisition_settings(seq, 8, 8, pixel_size_um=0.325)["dimensions"]
    ext = "ome.tiff" if fmt == "ome-tiff" else "ome.zarr"
    out = _write_bare(dims, fmt, tmp_path / f"nometa.{ext}", n_frames=4)

    loaded = open_acquisition(out)
    try:
        assert loaded.sequence is None
        assert loaded.wrapper.sizes()["t"] == 4
    finally:
        loaded.close()


def test_malformed_sequence_metadata_yields_none(tmp_path: Path) -> None:
    seq = useq.MDASequence(channels=_ch("DAPI"))
    dims = useq_to_acquisition_settings(seq, 8, 8, pixel_size_um=0.325)["dimensions"]
    settings = AcquisitionSettings(
        dimensions=tuple(dims),
        dtype="uint16",
        root_path=str(tmp_path / "bad.ome.zarr"),
        format=OmeZarrFormat(),
    )
    with create_stream(settings) as stream:
        stream.set_global_metadata(
            "pymmcore_plus",
            {"summary_metadata": {"mda_sequence": {"channels": "not-a-sequence"}}},
        )
        stream.append(np.zeros((8, 8), dtype="uint16"))

    loaded = open_acquisition(settings.output_path)
    try:
        assert loaded.sequence is None  # malformed -- recovered as None, not raised
    finally:
        loaded.close()


def test_cancelled_partial_ome_tiff_shows_only_written_frames(tmp_path: Path) -> None:
    """OME-TIFF only ever has pages for frames actually written."""
    seq = useq.MDASequence(
        channels=_ch("DAPI"),
        time_plan=useq.TIntervalLoops(interval=timedelta(seconds=1), loops=10),
    )
    out = _write_with_sequence(
        seq, "ome-tiff", tmp_path / "partial.ome.tiff", n_frames=3
    )

    loaded = open_acquisition(out)
    try:
        # the file only has 3 pages even though the recovered sequence asked
        # for 10 -- the displayed shape comes from disk, not the sequence
        assert loaded.wrapper.sizes()["t"] == 3
        assert loaded.sequence is not None
        time_plan = loaded.sequence.time_plan
        assert getattr(time_plan, "loops", None) == 10
    finally:
        loaded.close()


def test_cancelled_partial_ome_zarr_reports_declared_shape(tmp_path: Path) -> None:
    """Known limitation: an OME-Zarr array pre-declares its full nominal shape.

    Unlike OME-TIFF (a flat page list with one entry per frame actually
    written), `ome_writers`' zarr backend allocates the array at its full
    declared extent up front; a cancelled run's unwritten region is simply
    missing *chunks* (sparse storage), not a shorter `zarr.json` shape. This
    wrapper reports the array's declared shape as-is; reconstructing the true
    high-water mark from chunk presence isn't implemented.
    """
    seq = useq.MDASequence(
        channels=_ch("DAPI"),
        time_plan=useq.TIntervalLoops(interval=timedelta(seconds=1), loops=10),
    )
    out = _write_with_sequence(
        seq, "ome-zarr", tmp_path / "partial.ome.zarr", n_frames=3
    )

    loaded = open_acquisition(out)
    try:
        assert loaded.wrapper.sizes()["t"] == 10
    finally:
        loaded.close()


def test_supports_path_rejects_unsupported(tmp_path: Path) -> None:
    not_a_dataset = tmp_path / "random_dir"
    not_a_dataset.mkdir()
    assert not supports_path(not_a_dataset)
    assert not supports_path(tmp_path / "does-not-exist.ome.tiff")


def test_open_acquisition_raises_for_unsupported_path(tmp_path: Path) -> None:
    not_a_dataset = tmp_path / "random_dir"
    not_a_dataset.mkdir()
    with pytest.raises(ValueError, match="Not a supported acquisition"):
        open_acquisition(not_a_dataset)


def test_close_releases_resources(tmp_path: Path) -> None:
    seq = useq.MDASequence(channels=_ch("DAPI"))
    out = _write_with_sequence(
        seq, "ome-tiff", tmp_path / "closeme.ome.tiff", n_frames=1
    )

    loaded = open_acquisition(out)
    tf = loaded.wrapper._tf  # type: ignore[attr-defined]
    assert tf.filehandle.closed is False
    loaded.close()
    assert tf.filehandle.closed is True


# --------------------------- record (Save-button re-export) -----------------


@pytest.mark.parametrize("fmt", FORMATS)
def test_record_has_real_dims_scale_and_channel_names(tmp_path: Path, fmt: str) -> None:
    """The Save button must re-export real metadata, not the current mic state.

    Per-frame `delta_t` is included, matching a real acquisition (pymmcore-
    plus always captures it in `frameReady`): OME-TIFF has no "nominal
    interval" concept independent of actually-observed per-frame timestamps
    (unlike OME-Zarr, which bakes the writer's declared scale into its NGFF
    metadata regardless), so recovering a real `t` scale from a plain OME-TIFF
    with no per-frame metadata isn't possible -- this exercises the realistic,
    always-present case instead.
    """
    seq = useq.MDASequence(
        channels=_ch("DAPI", "FITC"),
        time_plan=useq.TIntervalLoops(interval=timedelta(seconds=0.5), loops=3),
    )
    dims = useq_to_acquisition_settings(seq, 8, 8, pixel_size_um=0.325)["dimensions"]
    ext = "ome.tiff" if fmt == "ome-tiff" else "ome.zarr"
    settings = AcquisitionSettings(
        dimensions=tuple(dims),
        dtype="uint16",
        root_path=str(tmp_path / f"a.{ext}"),
        format=_fmt(fmt),
    )
    summary = {
        "format": "summary-dict",
        "version": "1.0",
        "mda_sequence": seq.model_dump(mode="json", exclude_unset=True),
    }
    with create_stream(settings) as stream:
        stream.set_global_metadata("pymmcore_plus", {"summary_metadata": summary})
        for t in range(3):
            for _c in range(2):
                stream.append(
                    np.zeros((8, 8), dtype="uint16"),
                    frame_metadata={"delta_t": t * 0.5},
                )

    loaded = open_acquisition(settings.output_path)
    try:
        assert loaded.record is not None
        by_name = {d.name: d for d in loaded.record.settings.dimensions}
        assert by_name["t"].scale == pytest.approx(0.5)
        assert by_name["x"].scale == pytest.approx(0.325)
        assert by_name["y"].scale == pytest.approx(0.325)
        channel_coords = cast("list[Channel]", by_name["c"].coords or [])
        assert [c.name for c in channel_coords] == ["DAPI", "FITC"]
        # a channel/position axis never gets a spurious physical scale, even
        # though ndv's own axis_scales() would compute one from numeric-
        # looking string labels (e.g. tifffile's zero-padded position index)
        assert by_name["c"].scale is None
        assert loaded.record.summary_meta is not None
        recovered_seq = loaded.record.summary_meta.get("mda_sequence")
        assert recovered_seq is not None
        assert recovered_seq.replace(uid=seq.uid) == seq
    finally:
        loaded.close()


@pytest.mark.parametrize("fmt", FORMATS)
def test_record_roundtrips_through_export_acquisition(tmp_path: Path, fmt: str) -> None:
    """Reopening a dataset and re-exporting it (e.g. to the other format) works."""
    from pymmcore_gui._mda_export import export_acquisition

    seq = useq.MDASequence(
        channels=_ch("DAPI", "FITC"),
        time_plan=useq.TIntervalLoops(interval=timedelta(seconds=0.5), loops=3),
    )
    ext = "ome.tiff" if fmt == "ome-tiff" else "ome.zarr"
    out = _write_with_sequence(seq, fmt, tmp_path / f"a.{ext}", n_frames=6)

    loaded = open_acquisition(out)
    try:
        assert loaded.record is not None
        other_fmt: Literal["ome-tiff", "ome-zarr"] = (
            "ome-zarr" if fmt == "ome-tiff" else "ome-tiff"
        )
        other_ext = "ome.zarr" if fmt == "ome-tiff" else "ome.tiff"
        resaved = export_acquisition(
            loaded.record, tmp_path / f"resaved.{other_ext}", other_fmt
        )
        assert resaved is not None

        reloaded = open_acquisition(resaved)
        try:
            assert reloaded.sequence is not None
            assert reloaded.sequence.replace(uid=seq.uid) == seq
            assert reloaded.wrapper.channel_names(reloaded.wrapper.dims.index("c")) == {
                0: "DAPI",
                1: "FITC",
            }
            assert reloaded.wrapper.axis_scales()["x"] == pytest.approx(0.325)
        finally:
            reloaded.close()
    finally:
        loaded.close()


def test_record_for_multiposition_file_without_sequence_metadata(
    tmp_path: Path,
) -> None:
    """A bare multi-position file (no recovered sequence) still gets a usable record."""
    seq = useq.MDASequence(
        channels=_ch("DAPI"),
        stage_positions=(useq.Position(x=0, y=0), useq.Position(x=10, y=10)),
    )
    dims = useq_to_acquisition_settings(seq, 8, 8, pixel_size_um=0.325)["dimensions"]
    settings = AcquisitionSettings(
        dimensions=tuple(dims),
        dtype="uint16",
        root_path=str(tmp_path / "bare.ome.tiff"),
        format=OmeTiffFormat(),
    )
    with create_stream(settings) as stream:
        for i in range(2):
            stream.append(np.full((8, 8), i * 100, dtype="uint16"))

    loaded = open_acquisition(Path(settings.output_path))
    try:
        assert loaded.sequence is None
        assert loaded.record is not None
        by_name = {d.name: d for d in loaded.record.settings.dimensions}
        assert by_name["p"].type == "position"
        assert by_name["p"].scale is None  # never a spurious scale for position
        assert by_name["p"].count == 2
    finally:
        loaded.close()
