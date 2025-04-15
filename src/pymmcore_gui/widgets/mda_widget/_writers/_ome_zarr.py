# from __future__ import annotations

# from pymmcore_plus.mda.handlers import OMEZarrWriter
# from pymmcore_plus.metadata.serialize import to_builtins


# class _OMEZarrWriter(OMEZarrWriter):
#     ...


#     def new_array(self, key: str, dtype: np.dtype, sizes: dict[str, int]) -> zarr.Array:
#         """Create a new array in the group, under `key`."""
#         dims, shape = zip(*sizes.items())
#         ary: zarr.Array = self._group.create(
#             key,
#             shape=shape,
#             chunks=(1,) * len(shape[:-2]) + shape[-2:],  # single XY plane chunks
#             dtype=dtype,
#             **self._array_kwargs,
#         )

#         # add minimal OME-NGFF metadata
#         scales = self._group.attrs.get("multiscales", [])
#         scales.append(self._multiscales_item(ary.path, ary.path, dims))
#         self._group.attrs["multiscales"] = scales
#         ary.attrs["_ARRAY_DIMENSIONS"] = dims
#         if seq := self.current_sequence:
#             ary.attrs["useq_MDASequence"] = to_builtins(seq)

#         return ary