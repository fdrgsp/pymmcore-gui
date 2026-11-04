from __future__ import annotations

import urllib.error
import urllib.request
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np
import psutil

from pymmcore_gui._qt.QtCore import Qt

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from pymmcore_plus import CMMCorePlus
    from useq import MDAEvent

GH_REPO_URL = "http://github.com/pymmcore-plus/pymmcore-gui"


def connect_direct(signal: Any, callback: Callable[..., Any]) -> None:
    """Connect `callback` on the emitting thread, whatever signaler `signal` is.

    A psygnal signaler has no connection-type argument; it calls back on the
    emitting thread regardless, which is what the direct connection asks Qt for.
    """
    try:
        signal.connect(callback, Qt.ConnectionType.DirectConnection)
    except TypeError:  # pragma: no cover - psygnal signaler
        signal.connect(callback)


_BYTES_PER_GB = 1024**3

_AUTOFOCUS_ACTIONS = {
    "hardware_autofocus": "hardware",
    "software_autofocus": "software",
}


def autofocus_kind(event: MDAEvent) -> str | None:
    """Return `"hardware"` or `"software"` if `event` is an autofocus event, else None.

    Matched on the action's ``type`` string rather than the useq classes, so this
    keeps working against a useq that predates software autofocus.
    """
    return _AUTOFOCUS_ACTIONS.get(str(getattr(event.action, "type", "")))


def current_channel(core: CMMCorePlus) -> str:
    """Return the selected preset of the channel group, or `""` if there is none."""
    try:
        group = str(core.getChannelGroup())
        return str(core.getCurrentConfig(group)) if group else ""
    except Exception:
        return ""


def system_memory_gb() -> tuple[float, float]:
    """Return (total, available) system RAM in GB.

    ``available`` is what the OS considers free for new allocations right
    now (unlike ``total - used``, it accounts for reclaimable caches/buffers).
    """
    vm = psutil.virtual_memory()
    return vm.total / _BYTES_PER_GB, vm.available / _BYTES_PER_GB


def isel_positions(
    indexers: Mapping[int, int | slice],
    n_positions: int,
    sizes: tuple[int, ...],
    dtype: np.dtype,
    read_one: Callable[[int, Mapping[int, int | slice]], np.ndarray],
) -> np.ndarray:
    """Read one frame per selected position, stacking only for a slice index.

    Per `DataWrapper.isel`'s contract an int index for the synthetic position
    axis selects (and squeezes) a single position -- the common case, since ndv
    slider-indexes hidden dims with a plain int. A slice keeps it a real,
    stacked axis and is honored over its full range.
    """
    pos_idx = indexers.get(0, slice(None))
    inner = {i - 1: v for i, v in indexers.items() if i != 0}
    if isinstance(pos_idx, slice):
        frames = [read_one(i, inner) for i in range(n_positions)[pos_idx]]
        if frames:
            return np.stack(frames, axis=0)
        return np.empty(indexed_shape(sizes, indexers), dtype=dtype)
    return read_one(pos_idx, inner)


def indexed_shape(
    shape: Sequence[int], indexers: Mapping[int, int | slice]
) -> tuple[int, ...]:
    """Return the shape after basic integer/slice indexing, without reading data."""
    result = []
    for axis, size in enumerate(shape):
        index = indexers.get(axis, slice(None))
        if isinstance(index, slice):
            result.append(len(range(size)[index]))
        elif not -size <= index < size:
            raise IndexError(
                f"Index {index} out of bounds for axis {axis} of size {size}"
            )
    return tuple(result)


def get_treeish() -> str:
    """Return the git treeish of the current version.

    https://setuptools-scm.readthedocs.io/en/latest/usage/#default-versioning-scheme
    setuptools_scm versioning will look like one of these
    {tag}
    {next_version}.dev{distance}+{scm letter}{revision_hash}
    {tag}+dYYYYMMDD
    {next_version}.dev{distance}+{scm letter}{revision_hash}.dYYYYMMDD

    ... and we want to return the revision_hash when present, and the `tag` otherwise.

    """
    from . import __version__

    tag = __version__
    if "+" in __version__:
        right_part = __version__.split("+")[1].split(".")[0]
        tag = right_part[1:]  # remove the scm letter, e.g. 'g' from 'g1234567'
    return tag


@cache
def gh_link(
    file: str | None = None,
    line_no: int | tuple[int, int] | None = None,
    root: str = GH_REPO_URL,
    treeish: str | None = None,
    check_404: bool = True,
) -> str:
    """Return a link to the root, file, or line number on the github repository.

    Parameters
    ----------
    file : str | None
        The file to link to. If None, the link will point to the root of the repository.
    line_no : int | tuple[int, int] | None
        The line number or range of (start, end) numbers to link to. If None, the link
        will point to the file.
    root : str
        The root url of the repository. Default is the pymmcore-plus/pymmcore-gui
        repository.
    treeish : str | None
        The git treeish of the version to link to (such as "main", "v0.1.0", or a
        commit hash). If None, the treeish is determined from the current version.
    check_404 : bool
        If True, check if the link is 404 and fallback to the main url. Default is True.
    """
    href = root
    href = f"{root}/tree/{treeish or get_treeish()}"
    if file is not None:
        href += f"/{file}"
        if line_no is not None:
            if isinstance(line_no, tuple):
                href += f"#L{line_no[0]}-L{line_no[1]}"
            else:
                href += f"#L{line_no}"

    if check_404:
        # check if the link is 404 and fallback to the main url
        try:
            urllib.request.urlopen(href)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                href = root

    return href
