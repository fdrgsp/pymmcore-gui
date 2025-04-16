from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

from pymmcore_plus.mda.handlers import TensorStoreHandler

if TYPE_CHECKING:
    from collections.abc import Mapping
    from os import PathLike
from pymmcore_plus.metadata.serialize import json_dumps, json_loads

TsDriver: TypeAlias = Literal["zarr", "zarr3", "n5", "neuroglancer_precomputed"]

WAIT_TIME = 10  # seconds


class TensorStoreWriterMM(TensorStoreHandler):
    """Tensorstore handler for writing MDA sequences.

    This is a performant and shape-agnostic handler for writing MDA sequences to
    chunked storages like zarr, n5, backed by tensorstore:
    <https://google.github.io/tensorstore/>

    By default, the handler will store frames in a zarr array, with a shape of
    (nframes, *frame_shape) and a chunk size of (1, *frame_shape), i.e. each frame
    is stored in a separate chunk. To customize shape or chunking, override the
    `get_full_shape`, `get_chunk_layout`, and `get_index_domain` methods (these
    may change in the future as we learn to use tensorstore better).

    Parameters
    ----------
    driver : TsDriver, optional
        The driver to use for the tensorstore, by default "zarr".  Must be one of
        "zarr", "zarr3", "n5", or "neuroglancer_precomputed".
    kvstore : str | dict | None, optional
        The key-value store to use for the tensorstore, by default "memory://".
        A dict might look like {'driver': 'file', 'path': '/path/to/dataset.zarr'}
        see <https://google.github.io/tensorstore/kvstore/index.html#json-KvStore>
        for all options. If path is provided, the kvstore will be set to file://path
    path : str | Path | None, optional
        Convenience for specifying a local filepath. If provided, overrides the
        kvstore option, to be `file://file_path`.
    delete_existing : bool, optional
        Whether to delete the existing dataset if it exists, by default False.
    spec : Mapping, optional
        A spec to use when opening the tensorstore, by default None. Values provided
        in this object will override the default values provided by the handler.
        This is a complex object that can completely define the tensorstore, see
        <https://google.github.io/tensorstore/spec.html> for more information.

    Examples
    --------
    ```python
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.mda.handlers import TensorStoreHandler
    from useq import MDASequence

    core = CMMCorePlus.instance()
    core.loadSystemConfiguration()

    sequence = MDASequence(
        channels=["DAPI", {"config": "FITC", "exposure": 1}],
        stage_positions=[{"x": 1, "y": 1, "name": "some position"}, {"x": 0, "y": 0}],
        time_plan={"interval": 2, "loops": 3},
        z_plan={"range": 4, "step": 0.5},
        axis_order="tpcz",
    )

    writer = TensorStoreHandler(path="example_ts.zarr", delete_existing=True)
    core.mda.run(sequence, output=writer)
    ```

    """

    def __init__(
        self,
        *,
        driver: TsDriver = "zarr",
        kvstore: str | dict | None = "memory://",
        path: str | PathLike | None = None,
        delete_existing: bool = False,
        spec: Mapping | None = None,
    ) -> None:
        super().__init__(
            driver=driver,
            kvstore=kvstore,
            path=path,
            delete_existing=delete_existing,
            spec=spec,
        )

    # # override this method to make sure the ".zattrs" file is written
    def finalize_metadata(self) -> None:
        """Finalize and flush metadata to storage."""
        if not (store := self._store) or not store.kvstore:
            return  # pragma: no cover

        # NOTE: we need to remopve the "mm_handler" key from the metadata
        # because it is not serializable
        frames_meta = []
        for f in [m[1] for m in self.frame_metadatas]:
            ev = f.get("mda_event")
            if ev and ev.sequence:
                seq_meta = dict(ev.sequence.metadata)
                seq_meta.pop("mm_handler", None)
                new_ev_seq = ev.sequence.model_copy(update={"metadata": seq_meta})
                ev_clean = ev.model_copy(update={"sequence": new_ev_seq})
                f["mda_event"] = ev_clean
            frames_meta.append(f)

        metadata = {"frame_metadatas": frames_meta}

        if not self._nd_storage:
            metadata["frame_indices"] = [
                (tuple(dict(k).items()), v)  # type: ignore
                for k, v in self._frame_indices.items()
            ]

        if self.ts_driver.startswith("zarr"):
            store.kvstore.write(
                ".zattrs", json_dumps(metadata).decode("utf-8")
            ).result()
        elif self.ts_driver == "n5":  # pragma: no cover
            attrs = json_loads(store.kvstore.read("attributes.json").result().value)
            attrs.update(metadata)
            store.kvstore.write("attributes.json", json_dumps(attrs).decode("utf-8"))
