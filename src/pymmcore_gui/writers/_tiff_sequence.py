"""Simple Image sequence writer for MDASequences.

Writes each frame of an MDA to a directory as individual TIFF files by default,
but can write to other formats if `imageio` is installed or a custom writer is
provided.
"""

from __future__ import annotations

from itertools import count
from typing import TYPE_CHECKING, Any

from pymmcore_plus.mda.handlers import ImageSequenceWriter
from pymmcore_plus.mda.handlers._5d_writer_base import _NULL
from pymmcore_plus.mda.handlers._util import get_full_sequence_axes
from pymmcore_plus.metadata.serialize import json_dumps

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import TypeAlias  # py310

    import numpy as np
    import numpy.typing as npt
    import useq
    from pymmcore_plus.metadata import FrameMetaV1, SummaryMetaV1

    ImgWriter: TypeAlias = Callable[[str, npt.NDArray], Any]

FRAME_KEY = "frame"


class TiffSequenceWriterMM(ImageSequenceWriter):
    """Write each frame of an MDA to a directory as individual image files.

    This writer assumes very little about the sequence, and simply writes each frame
    to a file in the specified directory as a tif file. A subfolder is created for each
    position. It is a good option for ragged or sparse sequences, or where the exact
    number of frames is not known in advance.

    The metadata for each frame is stored in a JSON file in the directory (by default,
    named "_frame_metadata.json").  The metadata is stored as a dict, with the key
    being the index string for the frame (see index_template), and the value being
    the metadata dict for that frame.

    The metadata for the entire MDA sequence is stored in a JSON file in the directory
    (by default, named "_useq_MDASequence.json").

    Parameters
    ----------
    directory: Path | str
        The directory to write the files to.
    extension: str
        The file extension to use.  By default, ".tif".
    prefix: str
        A prefix to add to the file names.  By default, no prefix is added.
    imwrite: Callable[[str, npt.NDArray], Any] | None
        A function to write the image data to disk. The function should take a filename
        and image data as positional arguments. If None, a writer will be selected based
        on the extension. For the default extension `.tif`, this will be
        `tifffile.imwrite` (which must be installed).
    overwrite: bool
        Whether to overwrite the directory if it already exists.  If False, a
        FileExistsError will be raised if the directory already exists.
    include_frame_count: bool
        Whether to include a frame count item in the template (`{frame:05}`). This
        will come after the prefix and before the indices. It is a good way to
        ensure unique keys. by default True
    imwrite_kwargs: dict | None
        Extra keyword arguments to pass to the `imwrite` function.
    """

    def __init__(
        self,
        directory: Path | str,
        extension: str = ".tif",
        prefix: str = "",
        *,
        imwrite: ImgWriter | None = None,
        overwrite: bool = False,
        include_frame_count: bool = True,
        imwrite_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            directory,
            extension,
            prefix,
            imwrite=imwrite,
            overwrite=overwrite,
            include_frame_count=include_frame_count,
            imwrite_kwargs=imwrite_kwargs,
        )

    # TODO: also override the sequenceStarted to set the correct args. this should be
    # in pymmcore_plus
    def sequenceStarted(
        self, seq: useq.MDASequence, meta: SummaryMetaV1 | object = _NULL
    ) -> None:
        """Store the sequence metadata and reset the frame counter."""
        self._counter = count()  # reset counter
        self._frame_metadata = {}  # reset metadata
        self._directory.mkdir(parents=True, exist_ok=True)

        self._current_sequence = seq
        axes = get_full_sequence_axes(seq)
        self._first_index = dict.fromkeys(axes, 0)
        if seq is not None:
            self._name_template = self.fname_template(
                axes,
                prefix=self._prefix,
                extension=self._ext,
                delimiter=self._delimiter,
                include_frame_count=self._include_frame_count,
            )
            # NOTE: we need to remove the "mm_handler" key from the metadata
            seq_meta = dict(seq.metadata)
            seq_meta.pop("mm_handler", None)
            updated_seq = seq.model_copy(update={"metadata": seq_meta})
            # make directory and write metadata
            self._seq_meta_file.write_text(
                updated_seq.model_dump_json(exclude_unset=True, indent=2)
            )

    def frameReady(
        self, frame: np.ndarray, event: useq.MDAEvent, meta: FrameMetaV1
    ) -> None:
        """Write a frame to disk."""
        frame_idx = next(self._counter)
        if self._name_template:
            if FRAME_KEY in self._name_template:
                indices = {**self._first_index, **event.index, FRAME_KEY: frame_idx}
            else:
                indices = {**self._first_index, **event.index}
            filename = self._name_template.format(**indices)
        else:
            # if we don't have a sequence, just use the counter
            filename = f"{self._prefix}_fr{frame_idx:05}.tif"

        pos_name = event.pos_name or f"p{event.index.get('p', 0)}"
        _dir = self._directory / pos_name
        if not _dir.exists():
            _dir.mkdir(parents=True, exist_ok=True)

        # WRITE DATA TO DISK
        self._imwrite(str(_dir / filename), frame, **self._imwrite_kwargs)

        # NOTE: we need to remove the "mm_handler" key from the metadata
        meta_ev = meta.get("mda_event")
        if meta_ev is not None and meta_ev.sequence is not None:
            seq_meta = dict(meta_ev.sequence.metadata)
            seq_meta.pop("mm_handler", None)
            new_ev_seq = meta_ev.sequence.model_copy(update={"metadata": seq_meta})
            ev_clean = meta_ev.model_copy(update={"sequence": new_ev_seq})
            meta["mda_event"] = ev_clean

        # store metadata
        self._frame_metadata[filename] = meta
        # write metadata to disk every 10 frames
        if frame_idx % 10 == 0:
            self._frame_meta_file.write_bytes(
                json_dumps(self._frame_metadata, indent=2)
            )
