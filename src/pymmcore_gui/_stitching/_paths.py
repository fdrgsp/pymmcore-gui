"""Output naming and cleanup shared by the GUI and worker."""

from __future__ import annotations

import logging
import shutil
from typing import TYPE_CHECKING

from pymmcore_gui._stitching._models import tmp_paths_for

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable
    from pathlib import Path

logger = logging.getLogger(__name__)

_EXTENSIONS = (".ome.zarr", ".ome.tiff", ".ome.tif", ".zarr", ".tiff", ".tif")


def acquisition_stem(source: Path) -> str:
    """Remove an acquisition's format extension from its name."""
    for extension in _EXTENSIONS:
        if source.name.lower().endswith(extension):
            return source.name[: -len(extension)]
    return source.name


def unique_path(
    output: Path, taken: Collection[Path] = (), *, check_temporary: bool = True
) -> Path:
    """Choose an unused output name, including queued or partial mosaics."""
    stem = acquisition_stem(output)
    extension = output.name[len(stem) :]
    candidate = output
    number = 2
    while (
        candidate in taken
        or candidate.exists()
        or (check_temporary and any(p.exists() for p in tmp_paths_for(candidate)))
    ):
        candidate = output.with_name(f"{stem}-{number}{extension}")
        number += 1
    return candidate


def remove_path(path: Path) -> None:
    """Remove a file or directory, ignoring a missing path."""
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def cleanup_paths(paths: Iterable[Path]) -> None:
    """Remove temporary paths, logging failures and continuing with the rest."""
    for path in paths:
        try:
            remove_path(path)
        except OSError:
            logger.exception("Could not remove stitching temporary %s", path)
