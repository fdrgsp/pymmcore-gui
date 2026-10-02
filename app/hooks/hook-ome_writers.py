from PyInstaller.utils.hooks import collect_submodules

# ome_writers picks its backend at runtime (ome_writers._stream.BACKENDS) and loads
# it with importlib.import_module("ome_writers._backends._<name>") from a string, so
# PyInstaller's static analysis never sees the backend modules (scratch, tifffile,
# tensorstore, ...). Without this, create_stream() fails in the frozen app with the
# misleading "Backend 'scratch' requested but 'scratch' package is not installed".
hiddenimports = collect_submodules("ome_writers")
