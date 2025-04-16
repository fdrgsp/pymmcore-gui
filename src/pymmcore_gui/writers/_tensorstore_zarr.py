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
