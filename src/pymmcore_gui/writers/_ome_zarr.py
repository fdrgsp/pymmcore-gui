from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pymmcore_plus.mda.handlers import OMEZarrWriter
from pymmcore_plus.metadata.serialize import to_builtins

if TYPE_CHECKING:
    import os
    from collections.abc import MutableMapping

    import numpy as np
    import zarr
    from fsspec import FSMap
    from pymmcore_plus.mda.handlers._ome_zarr_writer import (
        ArrayCreationKwargs,
        ZarrSynchronizer,
    )


class OMEZarrWriterMM(OMEZarrWriter):
    def __init__(
        self,
        path: MutableMapping | str | os.PathLike | FSMap | None = None,
        *,
        overwrite: bool = False,
        synchronizer: ZarrSynchronizer | None = None,
        zarr_version: Literal[2, 3, None] = None,
        array_kwargs: ArrayCreationKwargs | None = None,
        minify_attrs_metadata: bool = False,
    ):
        super().__init__(
            path,
            overwrite=overwrite,
            synchronizer=synchronizer,
            zarr_version=zarr_version,
            array_kwargs=array_kwargs,
            minify_attrs_metadata=minify_attrs_metadata,
        )

    def new_array(self, key: str, dtype: np.dtype, sizes: dict[str, int]) -> zarr.Array:
        """Create a new array in the group, under `key`."""
        dims, shape = zip(*sizes.items(), strict=False)
        ary: zarr.Array = self._group.create(
            key,
            shape=shape,
            chunks=(1,) * len(shape[:-2]) + shape[-2:],  # single XY plane chunks
            dtype=dtype,
            **self._array_kwargs,
        )

        # add minimal OME-NGFF metadata
        scales = self._group.attrs.get("multiscales", [])
        scales.append(self._multiscales_item(ary.path, ary.path, dims))
        self._group.attrs["multiscales"] = scales
        ary.attrs["_ARRAY_DIMENSIONS"] = dims
        if seq := self.current_sequence:
            # NOTE: we need to remove the "mm_handler" key from the metadata
            seq_meta = dict(seq.metadata)
            seq_meta.pop("mm_handler", None)
            updated_seq = seq.model_copy(update={"metadata": seq_meta})
            ary.attrs["useq_MDASequence"] = to_builtins(updated_seq)
        return ary

    def finalize_metadata(self) -> None:
        """Called by superclass in sequenceFinished.  Flush metadata to disk."""
        # flush frame metadata to disk
        self._populate_xarray_coords()
        while self.frame_metadatas:
            key, metas = self.frame_metadatas.popitem()
            if key in self.position_arrays:
                # NOTE: we need to remove the "mm_handler" key from the metadata
                # because it is not serializable
                updated_metas = []
                for frame_meta in metas:
                    ev = frame_meta.get("mda_event")
                    if ev and ev.sequence:
                        seq_meta = dict(ev.sequence.metadata)
                        seq_meta.pop("mm_handler", None)
                        new_ev_seq = ev.sequence.model_copy(
                            update={"metadata": seq_meta}
                        )
                        ev_clean = ev.model_copy(update={"sequence": new_ev_seq})
                        frame_meta["mda_event"] = ev_clean
                    updated_metas.append(frame_meta)
                self.position_arrays[key].attrs["frame_meta"] = to_builtins(
                    updated_metas
                )

        if self._minify_metadata:
            self._minify_zattrs_metadata()
