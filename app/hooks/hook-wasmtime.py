from PyInstaller.utils.hooks import collect_dynamic_libs

# wasmtime loads its native library with ctypes from a path it builds itself
# (wasmtime/<platform>-<arch>/), which PyInstaller's analysis never sees. It is
# imported by itkwasm, through ngff_zarr, when multiview-stitcher is.
binaries = collect_dynamic_libs("wasmtime")
