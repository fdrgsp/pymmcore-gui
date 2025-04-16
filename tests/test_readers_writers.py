from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import tifffile
import useq
from pymmcore_widgets.useq_widgets._mda_sequence import PYMMCW_METADATA_KEY

from pymmcore_gui.readers import OMEZarrReader, TensorstoreZarrReader
from pymmcore_gui.writers import OMETiffWriterMM, OMEZarrWriterMM, TensorStoreWriterMM

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot


# fmt: off
files = [
    # indexers, expected files, file_to_check, expected shape
    ({}, ["p0.tif", "p0.json", "p1.tif", "p1.json"], "p0.tif", (3, 2, 512, 512)),
    ({"p": 0}, ["test.tiff", "test.json"], "test.tiff", (3, 2, 512, 512)),
    ({"p": 0, "t": 0}, ["test.tiff", "test.json"], "test.tiff", (2, 512, 512)),
    # when a tuple is passed, first element is indexers and second is the kwargs
    (({"p": 0}, {"p": 0}), ["test.tiff", "test.json"], "test.tiff", (3, 2, 512, 512)),
    (({"p": 0}, {"t": 0}), ["test.tiff", "test.json"], "test.tiff", (2, 512, 512)),
]

MDA = useq.MDASequence(
    axis_order=["p", "t", "c"],
    channels=["FITC", "DAPI"],
    stage_positions=[(0, 0), (0, 1)],
    time_plan={"loops": 3, "interval": 0.1},
)
ZARR_META = {"format": "ome-zarr", "save_name": "z.ome.zarr"}
TIFF_META = {"format": "ome-tiff", "save_name": "t.ome.tiff"}
TENSOR_META = {
    "format": "tensorstore-zarr",
    "save_name": "ts.tensorstore.zarr",
}

writers = [
    # meta, name, writer, reader
    (TIFF_META, "t.ome.tiff", OMETiffWriterMM, None),
    (ZARR_META, "z.ome.zarr", OMEZarrWriterMM, OMEZarrReader),
    (TENSOR_META, "ts.tensorstore.zarr", TensorStoreWriterMM, TensorstoreZarrReader),
]
# fmt: on


# NOTE: the tensorstore reader works only if we use the internal TensorStoreWriterMM
# TODO: fix the main TensorStoreHandler because it does not write the ".zattrs"
@pytest.mark.parametrize("writers", writers)
@pytest.mark.parametrize("kwargs", [True, False])
@pytest.mark.parametrize("files", files)
def test_writers_readers(
    qtbot: QtBot,
    mmcore: CMMCorePlus,
    tmp_path: Path,
    writers: tuple,
    files: tuple,
    kwargs: bool,
):
    meta, name, writer, reader = writers
    indexers, expected_files, file_to_check, expected_shape = files

    mda = MDA.replace(
        metadata={
            PYMMCW_METADATA_KEY: {
                **meta,
                "save_dir": str(tmp_path),
                "should_save": True,
            },
            # we remove this before saving json since it is not serializable
            "mm_handler": {writer},
        }
    )

    dest: Path = tmp_path / name
    writer = writer(path=dest) if writer else dest
    with qtbot.waitSignal(mmcore.mda.events.sequenceFinished):
        mmcore.mda.run(mda, output=writer)

    assert dest.exists()

    if reader is None:  # for OME-TIFF
        assert dest.is_dir()
        assert dest.name == name
        dir_files = [f.name for f in dest.iterdir()]
        assert "_metadata.json" in dir_files
        assert "_useq_MDASequence.json" in dir_files
        for p in range(len(MDA.stage_positions)):
            assert f"t_p{p}.ome.tiff" in dir_files
    else:
        w = reader(data=dest)
        assert w.store
        assert w.sequence
        assert w.path == Path(dest)
        assert (
            w.metadata
            if isinstance(w, TensorstoreZarrReader)
            else w.metadata()
            if isinstance(w, OMEZarrReader)
            else None
        )

        # test that the reader can accept the actual store as input on top of the path
        w1 = reader(data=w.store)
        assert isinstance(w1, type(w))
        assert w1.sequence == w.sequence
        assert w1.path

        assert w.isel({"p": 0}).shape == (3, 2, 512, 512)
        assert w.isel({"p": 0, "t": 0}).shape == (2, 512, 512)
        _, metadata = w.isel({"p": 0}, metadata=True)
        assert metadata

        # test saving as tiff
        dest = tmp_path / "test"

        if not indexers and "ome.zarr" in name:
            return  # skipping since the no 'p' index error will be raised

        # if indexers is a tuple, use one as indexers and the other as kwargs
        if isinstance(indexers, tuple):
            # skip if kwargs is False since we don't want to test it twice
            if not kwargs:
                return
            w.write_tiff(dest, indexers[0], **indexers[1])
        # depends om kwargs (once as dict and once as kwargs)
        else:
            w.write_tiff(dest, **indexers) if kwargs else w.write_tiff(dest, indexers)
        # all files in dest
        parent = dest.parent if indexers else dest
        dir_files = [f.name for f in parent.iterdir()]
        assert all(f in dir_files for f in expected_files)
        # open with tifffile and check the shape
        with tifffile.TiffFile(parent / file_to_check) as tif:
            assert tif.asarray().shape == expected_shape
