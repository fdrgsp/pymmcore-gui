"""Remember user-selected LUTs by microscope channel identity."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any

from cmap import Colormap

from pymmcore_gui._settings import Settings

if TYPE_CHECKING:
    import ndv
    from useq import MDASequence


class ChannelLUTMemory:
    """Bind live MDA LUT models to persistent microscope-channel preferences.

    ndv identifies LUTs by array index, but those indices are local to one
    viewer.  Micro-Manager channel group/preset pairs are stable across runs,
    so they are the appropriate identity for an application-level preference.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings.instance()

    def bind_live_mda(self, viewer: ndv.ArrayViewer, sequence: MDASequence) -> None:
        """Apply remembered LUTs and observe later user changes in ``viewer``."""
        luts = getattr(viewer.display_model, "luts", None)
        if luts is None:
            return
        for index, channel in enumerate(sequence.channels):
            group = str(channel.group or "")
            preset = str(channel.config or "")
            if not group or not preset or (lut := luts.get(index)) is None:
                continue

            if remembered := self._settings.channel_lut(group, preset):
                # A stale/third-party cmap name in settings must never prevent
                # an acquisition viewer from opening.  The next valid user
                # selection will replace it through the callback below.
                with suppress(Exception):
                    lut.cmap = Colormap(remembered)

            def _remember(
                cmap: Colormap, _old: Any, *, group: str = group, preset: str = preset
            ) -> None:
                self._settings.remember_channel_lut(group, preset, cmap.name)

            lut.events.cmap.connect(_remember)
