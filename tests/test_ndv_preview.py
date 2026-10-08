from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest
import useq
from cmap import Colormap
from ndv.models import ChannelMode, LUTModel, RingBuffer

import pymmcore_gui.widgets.image_preview._ndv_preview as preview_module
from pymmcore_gui.widgets.image_preview._ndv_preview import NDVPreview

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot


def test_failed_viewer_construction_detaches_core_callbacks(
    mmcore: CMMCorePlus,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial preview must not receive the snap that exposed its failure."""
    appended: list[np.ndarray] = []

    def fail_viewer(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("viewer initialization failed")

    monkeypatch.setattr(preview_module, "MMArrayViewer", fail_viewer)
    monkeypatch.setattr(NDVPreview, "append", lambda _self, data: appended.append(data))

    with pytest.raises(RuntimeError, match="viewer initialization failed"):
        NDVPreview(mmcore)

    mmcore.snapImage()
    qtbot.wait(0)
    assert not appended


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
        _update_channel_name=Mock(),
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


def test_apply_viewer_settings_keeps_preview_composite() -> None:
    """Applying a new mono or RGB buffer must not reset the GUI's viewer mode."""
    for is_rgb, expected_axis in ((False, None), (True, 3)):
        lut = LUTModel(cmap=Colormap("green"))
        display_model = SimpleNamespace(
            visible_axes=None,
            channel_axis=None,
            channel_mode=None,
            luts={0: lut},
        )
        preview = SimpleNamespace(
            _viewer=SimpleNamespace(data=None, display_model=display_model),
            _buffer=object(),
            _buffer_applied=False,
            _is_rgb=is_rgb,
            _apply_control_visibility=Mock(),
        )

        NDVPreview._apply_viewer_settings(preview)  # type: ignore[arg-type]

        assert display_model.channel_axis == expected_axis
        assert display_model.channel_mode is ChannelMode.COMPOSITE
        expected_cmap = "green" if is_rgb else "gray"
        assert lut.cmap.name.endswith(expected_cmap)
        assert preview._buffer_applied


def test_preview_lut_uses_current_channel_name() -> None:
    core = Mock()
    core.getChannelGroup.return_value = "Channel"
    core.getCurrentConfig.return_value = "FITC"
    lut = LUTModel()
    preview = SimpleNamespace(
        _mmc=core,
        _is_rgb=False,
        _viewer=SimpleNamespace(display_model=SimpleNamespace(luts={0: lut})),
    )

    NDVPreview._update_channel_name(preview)  # type: ignore[arg-type]

    assert lut.name == "FITC"
    core.getCurrentConfig.return_value = "DAPI"
    NDVPreview._update_channel_name(preview)  # type: ignore[arg-type]
    assert lut.name == "DAPI"


def test_autofocus_images_are_shown_during_an_acquisition(
    mmcore: CMMCorePlus, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A routine's `show_images` is pointless if the preview drops the images.

    Acquisition frames go to the run's own viewer, which is why the preview
    ignores snaps during a run -- but an autofocus routine's images have nowhere
    else to go, and asking to see them is the only reason they are announced.
    """
    appended: list[np.ndarray] = []
    monkeypatch.setattr(NDVPreview, "append", lambda _self, data: appended.append(data))
    shown = Mock()
    preview = NDVPreview(mmcore)
    qtbot.addWidget(preview)
    preview.snapShown.connect(shown)

    events = mmcore.mda.events
    af_event = useq.MDAEvent(action=useq.SoftwareAutofocus(method="oughtafocus"))

    # during a run, an ordinary snap is somebody else's business ...
    preview._is_mda_running = True
    mmcore.snapImage()
    qtbot.wait(0)
    assert not appended

    # ... but the images of an autofocus event are shown
    events.eventStarted.emit(af_event)
    mmcore.snapImage()
    qtbot.wait(0)
    assert len(appended) == 1
    # without pulling the user off the viewer of the run in progress
    shown.assert_not_called()

    # the window closes when the routine reports, and again at the next event
    events.autofocusFinished.emit(af_event, Mock())
    mmcore.snapImage()
    qtbot.wait(0)
    assert len(appended) == 1

    events.eventStarted.emit(af_event)
    events.eventStarted.emit(useq.MDAEvent())
    mmcore.snapImage()
    qtbot.wait(0)
    assert len(appended) == 1

    # outside a run, every snap is shown, and is worth bringing to the front
    preview._is_mda_running = False
    mmcore.snapImage()
    qtbot.wait(0)
    assert len(appended) == 2
    shown.assert_called_once()


def test_a_snap_from_the_acquisition_thread_is_displayed_on_the_gui_thread(
    mmcore: CMMCorePlus, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`append` drives the GPU canvas, so it must not run on the runner thread."""
    threads: list[int] = []
    monkeypatch.setattr(
        NDVPreview, "append", lambda _self, _data: threads.append(threading.get_ident())
    )
    preview = NDVPreview(mmcore)
    qtbot.addWidget(preview)

    def _snap_off_thread() -> None:
        mmcore.snapImage()

    worker = threading.Thread(target=_snap_off_thread)
    worker.start()
    worker.join()
    qtbot.waitUntil(lambda: bool(threads))

    assert threads == [threading.get_ident()]
