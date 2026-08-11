from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
from ndv.models import RingBuffer

import pymmcore_gui.widgets.image_preview._ndv_preview as preview_module
from pymmcore_gui.widgets.image_preview._ndv_preview import NDVPreview

if TYPE_CHECKING:
    import pytest


def test_shape_change_defers_empty_buffer_assignment() -> None:
    """A Camera ROI change must not give ndv an empty replacement buffer."""
    preview = SimpleNamespace(
        _init_buffer=Mock(),
        _apply_viewer_settings=Mock(),
        _buffer_applied=True,
        _core_dtype=("uint16", (512, 512)),
        _get_core_dtype_shape=Mock(return_value=("uint16", (64, 64))),
    )

    NDVPreview._setup_viewer(preview)  # type: ignore[arg-type]

    preview._init_buffer.assert_called_once_with(("uint16", (64, 64)))
    preview._apply_viewer_settings.assert_not_called()
    assert not preview._buffer_applied


def test_first_new_shape_frame_is_applied_then_fitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first populated replacement frame swaps data and recenters the canvas."""

    old_buffer = RingBuffer(
        max_capacity=1,
        dtype=("uint16", (512, 512)),
    )
    viewer = SimpleNamespace(
        reset_zoom=Mock(),
        display_model=SimpleNamespace(current_index={}),
        data_wrapper=SimpleNamespace(data_changed=SimpleNamespace(emit=Mock())),
    )
    preview = SimpleNamespace(
        _buffer=old_buffer,
        _buffer_applied=True,
        _core_dtype=("uint16", (512, 512)),
        _viewer=viewer,
        process_events_on_update=False,
    )

    def init_buffer(dtype_shape: tuple[str, tuple[int, ...]]) -> None:
        preview._core_dtype = dtype_shape
        preview._buffer = RingBuffer(max_capacity=1, dtype=dtype_shape)

    def apply_settings() -> None:
        assert len(preview._buffer) == 1
        preview._buffer_applied = True

    preview._init_buffer = Mock(side_effect=init_buffer)
    preview._apply_viewer_settings = Mock(side_effect=apply_settings)
    monkeypatch.setattr(
        preview_module,
        "QTimer",
        SimpleNamespace(singleShot=lambda _delay, callback: callback()),
    )

    NDVPreview.append(preview, np.zeros((8, 12), dtype=np.uint16))  # type: ignore[arg-type]

    preview._init_buffer.assert_called_once_with(("uint16", (8, 12)))
    preview._apply_viewer_settings.assert_called_once_with()
    viewer.reset_zoom.assert_called_once_with()
    viewer.data_wrapper.data_changed.emit.assert_called_once_with()


def test_late_roi_set_keeps_buffer_populated_by_auto_snap() -> None:
    buffer = RingBuffer(max_capacity=1, dtype=("uint16", (64, 64)))
    buffer.append(np.zeros((64, 64), dtype=np.uint16))
    preview = SimpleNamespace(
        _buffer=buffer,
        _buffer_applied=True,
        _core_dtype=("uint16", (64, 64)),
        _get_core_dtype_shape=Mock(return_value=("uint16", (64, 64))),
        _init_buffer=Mock(),
    )

    NDVPreview._setup_viewer(preview)  # type: ignore[arg-type]

    preview._init_buffer.assert_not_called()
    assert preview._buffer is buffer
    assert preview._buffer_applied
