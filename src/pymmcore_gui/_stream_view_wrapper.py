"""ndv DataWrapper for a live `ome_writers` `StreamView`.

ndv's generic `ArrayLikeWrapper` falls back to treating the *smallest*
dimension (when <= `MAX_CHANNELS`) as the channel axis if none is named like a
channel. A live MDA view always has real axis names (``p``, ``t``, ``z``,
``c``, ...), so that fallback misfires: e.g. a 14-position run with no
channels gets its ``p`` axis shown as 14 composite channels instead of a
position slider. This wrapper only ever uses an axis *named* like a channel,
matching what `OMETiffWrapper`/`OMEZarrWrapper` do for reopened acquisitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard

from ndv.models._data_wrapper import ArrayLikeWrapper

if TYPE_CHECKING:
    from collections.abc import Hashable


# Needed otherwise running a MDASequence with no channel plan and multiple positions
# would incorrectly treat the position axis as channels.
class StreamViewWrapper(ArrayLikeWrapper):
    """`ArrayLikeWrapper` whose channel axis comes from axis names only."""

    # checked before ArrayLikeWrapper (PRIORITY 100), which also supports it
    PRIORITY = 40

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[Any]:
        from ome_writers._stream_view import StreamView

        return isinstance(obj, StreamView)

    def guess_channel_axis(self) -> Hashable | None:
        """Use axis names rather than guessing from dimension sizes."""
        for dim in self.dims:
            if str(dim).lower() in self.COMMON_CHANNEL_NAMES:
                return self.normalize_axis_key(dim)
        return None
