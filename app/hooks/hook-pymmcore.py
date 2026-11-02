from pathlib import Path

from PyInstaller.utils.hooks import get_package_paths

# pymmcore's Windows wheels keep their private DLLs in `pymmcore.libs`, named after
# a hash of their contents, and pymmcore only registers that directory if it exists.
# Another package can ship a DLL with the same name (shapely has the same
# msvcp140-<hash>.dll): PyInstaller then collects just one copy, under that package's
# directory, and `_pymmcore_swig` fails to load. So always collect this directory.
_, _package_dir = get_package_paths("pymmcore")
binaries = [
    (str(dll), "pymmcore.libs")
    for dll in (Path(_package_dir).parent / "pymmcore.libs").glob("*.dll")
]
