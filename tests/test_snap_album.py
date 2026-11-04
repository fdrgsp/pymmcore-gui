from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import numpy as np
import pytest
import tifffile
from ndv.models import ChannelMode
from pytestqt.qt_compat import qt_api

import pymmcore_gui._ndv_viewers as viewers_module
from pymmcore_gui._array_viewer import MMArrayViewer
from pymmcore_gui._channel_settings_store import (
    _STORES,
    channel_settings_store,
    current_light_intensity,
)
from pymmcore_gui._mda_export import export_acquisition
from pymmcore_gui._ndv_viewers import AcquireViewersManager
from pymmcore_gui._qt.QtAds import CDockManager
from pymmcore_gui._qt.QtCore import QEvent, QPoint, Qt
from pymmcore_gui._qt.QtWidgets import (
    QApplication,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from pymmcore_gui.widgets._pixel_calibration_panel import _FrameCore
from pymmcore_gui.widgets.image_preview._snap_album import (
    ALBUM_CAPACITY,
    SnapAlbum,
    SnapAlbumPopup,
    SnappedImage,
    _thumbnail,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot


def _snap(value: int = 0, **kwargs: object) -> SnappedImage:
    return SnappedImage(np.full((8, 12), value, dtype=np.uint16), **kwargs)  # type: ignore[arg-type]


@pytest.fixture
def manager(mmcore: CMMCorePlus, qtbot: QtBot) -> AcquireViewersManager:
    parent = QWidget()
    qtbot.addWidget(parent)
    dock_manager = CDockManager(parent)
    QVBoxLayout(parent).addWidget(dock_manager)
    return AcquireViewersManager(dock_manager, mmcore, parent=parent)


def _album_button(manager: AcquireViewersManager) -> QPushButton:
    assert manager.preview is not None
    widget = manager.preview.viewer.widget()
    layout = widget._btn_layout
    save_index = layout.indexOf(manager.preview.viewer._save_btn)
    button = layout.itemAt(save_index - 1).widget()
    assert isinstance(button, QPushButton)
    return button


def _open_popup(button: QPushButton) -> SnapAlbumPopup:
    button.click()
    popup = QApplication.activePopupWidget()
    assert isinstance(popup, SnapAlbumPopup)
    return popup


def test_album_drops_the_oldest_snap_and_lists_the_newest_first(qapp: object) -> None:
    album = SnapAlbum(capacity=3)
    snaps = [_snap(i) for i in range(5)]
    for snap in snaps:
        album.add(snap)
    assert len(album) == 3
    assert list(album) == [snaps[4], snaps[3], snaps[2]]


def test_description_names_when_and_what_with() -> None:
    snap = _snap(
        timestamp=datetime(2026, 11, 3, 14, 32, 5), channel="DAPI", exposure_ms=100.0
    )
    assert snap.title == "DAPI · 100 ms · 14:32:05"
    assert snap.description == "2026-11-03 14:32:05\nDAPI · 100 ms · 12 x 8"
    # nothing to report beyond the shape, without a dangling separator
    bare = _snap(timestamp=snap.timestamp)
    assert bare.title == "Snap 14:32:05"
    assert bare.description.endswith("05\n12 x 8")


def test_title_and_description_give_the_intensity_without_a_unit() -> None:
    snap = _snap(
        timestamp=datetime(2026, 11, 3, 14, 32, 5),
        channel="DAPI",
        exposure_ms=100.0,
        light_intensity=55.0,
    )
    assert snap.title == "DAPI · 100 ms · I=55 · 14:32:05"
    assert snap.description == "2026-11-03 14:32:05\nDAPI · 100 ms · I=55 · 12 x 8"


def test_capture_records_the_intensity_only_if_its_column_is_shown(
    mmcore: CMMCorePlus,
) -> None:
    image = np.zeros((8, 12), dtype=np.uint16)
    store = channel_settings_store(mmcore)
    mmcore.setConfig("Channel", "DAPI")
    # no light source picked for this preset
    assert SnappedImage.capture(mmcore, image).light_intensity is None

    source = "Camera · TestProperty2"
    store.set(
        "Channel", "DAPI", {"exposure": 20, "light_source": source, "intensity": 55}
    )
    assert store.apply_current_channel()
    assert SnappedImage.capture(mmcore, image).light_intensity == 55

    store.setLightSourcesVisible(False)
    assert SnappedImage.capture(mmcore, image).light_intensity is None


def test_capture_does_not_create_the_settings_store(mmcore: CMMCorePlus) -> None:
    assert mmcore not in _STORES
    assert current_light_intensity(mmcore) is None
    assert mmcore not in _STORES


@pytest.mark.parametrize("same_timestamp", [False, True])
def test_snaps_with_the_same_settings_open_in_their_own_tabs(
    manager: AcquireViewersManager,
    same_timestamp: bool,
) -> None:
    first, second = (
        _snap(
            i,
            channel="DAPI",
            exposure_ms=5.0,
            timestamp=datetime(2026, 11, 3, 14, 32, 5, 0 if same_timestamp else i),
        )
        for i in range(2)
    )
    assert first.title == second.title

    assert manager.open_snap(first) is not manager.open_snap(second)
    docks = [d for d, r in manager._records.items() if r.snap in (first, second)]
    assert len(docks) == 2
    assert {d.windowTitle() for d in docks} == {"DAPI · 5 ms · 14:32:05"}
    assert docks[0].objectName() != docks[1].objectName()


@pytest.mark.parametrize(
    "image",
    [
        np.arange(2048 * 1024, dtype=np.uint16).reshape(1024, 2048),
        np.zeros((16, 16), dtype=np.uint16),  # flat: must not divide by zero
        np.full((16, 16), 65535, dtype=np.uint16),
        np.zeros((512, 512, 3), dtype=np.uint8),
    ],
)
def test_thumbnail_is_small_and_keeps_the_aspect_ratio(
    qapp: object, image: np.ndarray
) -> None:
    pixmap = _thumbnail(image)
    assert not pixmap.isNull()
    assert max(pixmap.width(), pixmap.height()) < 2 * 56
    height, width = image.shape[:2]
    assert pixmap.width() / pixmap.height() == pytest.approx(width / height, rel=0.1)


def test_thumbnail_stretches_contrast(qapp: object) -> None:
    """A dim 16-bit image must not come out black."""
    image = np.zeros((16, 16), dtype=np.uint16)
    image[:, 8:] = 300
    frame = _thumbnail(image).toImage()
    assert frame.pixelColor(0, 0).value() == 0
    assert frame.pixelColor(15, 0).value() == 255


def test_album_button_sits_left_of_save_and_survives_closing_the_preview(
    mmcore: CMMCorePlus, qtbot: QtBot, manager: AcquireViewersManager
) -> None:
    manager.ensure_preview()
    assert "Recent snaps" in _album_button(manager).toolTip()

    mmcore.snapImage()
    qtbot.wait(0)
    assert len(manager._snap_album) == 1

    assert manager._preview_dock is not None
    manager._preview_dock.closeDockWidget()
    qtbot.wait(0)
    assert manager.preview is None

    manager.ensure_preview()
    popup = _open_popup(_album_button(manager))
    assert popup.list_widget.count() == 1
    popup.close()


def test_album_keeps_only_the_last_snaps_of_a_long_session(
    mmcore: CMMCorePlus, qtbot: QtBot, manager: AcquireViewersManager
) -> None:
    manager.ensure_preview()
    for _ in range(ALBUM_CAPACITY + 2):
        mmcore.snapImage()
    qtbot.wait(0)
    assert len(manager._snap_album) == ALBUM_CAPACITY


def test_empty_album_says_so_and_opens_nothing(
    qtbot: QtBot, manager: AcquireViewersManager
) -> None:
    manager.ensure_preview()
    popup = _open_popup(_album_button(manager))
    view = popup.list_widget
    assert view.count() == 1
    item = view.item(0)
    assert item is not None
    assert item.text() == "No snapped images yet"
    assert item.flags() == Qt.ItemFlag.NoItemFlags
    popup.close()


def test_double_click_opens_the_snap_in_a_new_tab(
    qtbot: QtBot, manager: AcquireViewersManager
) -> None:
    """Sent through the popup's window, as the platform would deliver it."""
    manager.ensure_preview()
    older, newer = _snap(1, channel="DAPI", exposure_ms=5.0), _snap(2)
    manager._snap_album.add(older)
    manager._snap_album.add(newer)

    popup = _open_popup(_album_button(manager))
    qtbot.waitExposed(popup)
    view = popup.list_widget
    assert view.count() == 2
    newest, oldest = view.item(0), view.item(1)
    assert newest is not None and oldest is not None
    assert not newest.icon().isNull()

    window = popup.windowHandle()
    viewport = view.viewport()
    assert window is not None and viewport is not None
    # the older snap is listed second
    center = viewport.mapTo(popup, view.visualItemRect(oldest).center())
    qt_api.QtTest.QTest.mouseDClick(
        window, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, center
    )
    qtbot.waitUntil(lambda: bool(manager._records))
    assert QApplication.activePopupWidget() is None

    [(dock, record)] = manager._records.items()
    assert record.snap is older
    assert record.viewer.data is older.image
    assert dock.windowTitle() == older.title
    assert dock.isCurrentTab()
    lut = record.viewer.display_model.luts[0]
    assert (lut.name, lut.cmap.name) == ("DAPI", "matlab:gray")


def test_escape_closes_the_popup_without_opening_anything(
    qtbot: QtBot, manager: AcquireViewersManager
) -> None:
    manager.ensure_preview()
    manager._snap_album.add(_snap())
    popup = _open_popup(_album_button(manager))
    qtbot.waitExposed(popup)

    with qtbot.waitSignal(popup.destroyed):
        qt_api.QtTest.QTest.keyClick(popup.list_widget, Qt.Key.Key_Escape)
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not manager._records


def _click_outside(popup: SnapAlbumPopup, global_pos: QPoint) -> None:
    """Click as the platform would: a popup's window grabs every mouse event."""
    window = popup.windowHandle()
    assert window is not None
    qt_api.QtTest.QTest.mouseClick(
        window,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        window.mapFromGlobal(global_pos),
    )


def test_clicking_away_closes_the_popup(
    qtbot: QtBot, manager: AcquireViewersManager
) -> None:
    parent = manager._parent_widget
    assert parent is not None
    parent.resize(800, 600)
    parent.show()
    qtbot.waitExposed(parent)
    manager.ensure_preview()
    manager._snap_album.add(_snap())
    popup = _open_popup(_album_button(manager))
    qtbot.waitExposed(popup)

    target = parent.mapToGlobal(parent.rect().bottomRight() - QPoint(5, 5))
    assert not popup.geometry().contains(target)
    _click_outside(popup, target)
    qtbot.wait(50)
    assert QApplication.activePopupWidget() is None
    assert not manager._records


def test_clicking_the_album_button_again_closes_the_popup(
    qtbot: QtBot, manager: AcquireViewersManager
) -> None:
    """Rather than closing it and reopening it in the same click."""
    parent = manager._parent_widget
    assert parent is not None
    parent.resize(800, 600)
    parent.show()
    qtbot.waitExposed(parent)
    manager.ensure_preview()
    manager._snap_album.add(_snap())
    button = _album_button(manager)
    popup = _open_popup(button)
    qtbot.waitExposed(popup)

    # a new viewer's panels are still animating, which moves the button under it
    target = button.mapToGlobal(button.rect().center())
    popup.move(target + QPoint(0, -popup.height() - 50))
    qtbot.wait(0)
    _click_outside(popup, target)
    qtbot.wait(50)
    assert QApplication.activePopupWidget() is None


def test_opening_a_snap_twice_goes_back_to_its_tab(
    qtbot: QtBot, manager: AcquireViewersManager
) -> None:
    first, second = _snap(1), _snap(2)
    viewer = manager.open_snap(first)
    other = manager.open_snap(second)
    assert other is not viewer
    assert len(manager._records) == 2

    assert manager.open_snap(first) is viewer
    assert len(manager._records) == 2
    dock = next(dw for dw, r in manager._records.items() if r.snap is first)
    assert dock.isCurrentTab()

    # closing it and opening it again gets a fresh tab
    dock.closeDockWidget()
    qtbot.wait(0)
    assert len(manager._records) == 1
    assert manager.open_snap(first) is not viewer


def test_opened_snap_never_passes_for_a_live_viewer(
    mmcore: CMMCorePlus, qtbot: QtBot, manager: AcquireViewersManager
) -> None:
    created = Mock()
    manager.mdaViewerCreated.connect(created)
    manager.open_snap(_snap())
    created.assert_not_called()
    assert manager.active_viewer is None


def test_saving_an_opened_snap_keeps_the_scale_it_was_snapped_with(
    mmcore: CMMCorePlus,
    qtbot: QtBot,
    manager: AcquireViewersManager,
    tmp_path: Path,
) -> None:
    """Not whatever the microscope is set to by the time it is saved."""
    image = np.arange(8 * 12, dtype=np.uint16).reshape(8, 12)
    viewer = manager.open_snap(SnappedImage(image, pixel_size_um=0.25))
    assert isinstance(viewer, MMArrayViewer)
    record = viewer._acquisition_record
    assert record is not None
    assert record.summary_meta is None

    out = export_acquisition(record, tmp_path / "snap.ome.tiff", "ome-tiff")
    assert out is not None
    with tifffile.TiffFile(out) as tif:
        assert (tif.asarray() == image).all()
        assert 'PhysicalSizeX="0.25"' in str(tif.ome_metadata)


def test_rgb_snap_opens_as_rgb(qtbot: QtBot, manager: AcquireViewersManager) -> None:
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    viewer = manager.open_snap(SnappedImage(image))
    assert viewer.display_model.channel_mode is ChannelMode.RGBA
    assert viewer.display_model.channel_axis == 2
    # routed to the plain TIFF path instead: ome_writers has no RGB sample axis
    assert isinstance(viewer, MMArrayViewer)
    assert viewer._acquisition_record is None


def test_real_rgb_camera_snap_reaches_the_album(
    mmcore: CMMCorePlus, qtbot: QtBot, manager: AcquireViewersManager
) -> None:
    mmcore.setProperty("Camera", "PixelType", "32bitRGB")
    manager.ensure_preview()
    mmcore.snapImage()
    qtbot.wait(0)
    [snap] = manager._snap_album
    assert snap.is_rgb
    assert snap.image.shape[-1] == 3
    manager.open_snap(snap)


def test_autofocus_test_images_stay_out_of_the_album(
    mmcore: CMMCorePlus,
    qtbot: QtBot,
    manager: AcquireViewersManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager.ensure_preview()
    mmcore.snapImage()
    qtbot.wait(0)
    assert len(manager._snap_album) == 1

    def routine(core: CMMCorePlus, *_: object, **__: object) -> SimpleNamespace:
        for _i in range(ALBUM_CAPACITY + 1):
            core.snapImage()
        return SimpleNamespace(succeeded=True, z_after=core.getZPosition())

    monkeypatch.setattr(viewers_module, "run_software_autofocus", routine)
    manager.run_software_autofocus_test(
        "oughtafocus", {"show_images": True}, lambda: False
    )
    qtbot.wait(0)
    assert len(manager._snap_album) == 1

    # and what follows the routine is an ordinary snap again
    assert manager.preview is not None
    assert not manager.preview._autofocus_running
    mmcore.snapImage()
    qtbot.wait(0)
    assert len(manager._snap_album) == 2


def test_failed_autofocus_test_does_not_leave_the_album_shut(
    mmcore: CMMCorePlus,
    qtbot: QtBot,
    manager: AcquireViewersManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def routine(*_: object, **__: object) -> None:
        raise RuntimeError("no focus device")

    monkeypatch.setattr(viewers_module, "run_software_autofocus", routine)
    with pytest.raises(RuntimeError, match="no focus device"):
        manager.run_software_autofocus_test(
            "oughtafocus", {"show_images": True}, lambda: False
        )
    assert manager.preview is not None
    assert not manager.preview._autofocus_running


def test_calibration_frames_are_not_announced_as_snaps(
    mmcore: CMMCorePlus, qtbot: QtBot, manager: AcquireViewersManager
) -> None:
    """A calibration run takes dozens: enough to empty the album of the user's own."""
    manager.ensure_preview()
    frames: list[object] = []
    proxy = _FrameCore(mmcore, mmcore.getXYStageDevice(), frames.append)

    proxy.snapImage()
    image = proxy.getImage()
    qtbot.wait(0)

    assert len(frames) == 1
    assert image.shape == (mmcore.getImageHeight(), mmcore.getImageWidth())
    assert not len(manager._snap_album)


def test_saving_an_rgb_snap_renames_its_tab(
    qtbot: QtBot, manager: AcquireViewersManager, tmp_path: Path
) -> None:
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    viewer = manager.open_snap(SnappedImage(image))
    assert isinstance(viewer, MMArrayViewer)
    path = tmp_path / "saved-rgb.tiff"
    with patch(
        "pymmcore_gui._array_viewer.QFileDialog.getSaveFileName",
        return_value=(str(path), ""),
    ):
        viewer._save_data()
    [dock] = manager._records
    assert dock.windowTitle() == path.name
    assert viewer.source_title == path.name
    np.testing.assert_array_equal(tifffile.imread(path), image)


def test_disconnect_does_not_announce_album_snaps_as_live_viewers(
    qtbot: QtBot, manager: AcquireViewersManager
) -> None:
    closed = Mock()
    manager.mdaViewerClosed.connect(closed)
    manager.open_snap(_snap())
    manager._disconnect()
    closed.assert_not_called()


def test_snap_intensity_reports_the_captured_value_not_the_preset_default(
    mmcore: CMMCorePlus,
) -> None:
    store = channel_settings_store(mmcore)
    mmcore.setConfig("Channel", "DAPI")
    store.set(
        "Channel",
        "DAPI",
        {"exposure": 20, "light_source": "Camera · TestProperty2", "intensity": 55},
    )
    store.apply_current_channel()
    mmcore.setProperty("Camera", "TestProperty2", 7)
    image = np.zeros((8, 12), dtype=np.uint16)
    assert SnappedImage.capture(mmcore, image).light_intensity == 7


def test_snap_intensity_requires_multi_property_values_to_agree(
    mmcore: CMMCorePlus,
) -> None:
    for prop in ("TestProperty2", "TestProperty4"):
        mmcore.defineConfig("_lights", "levels", "Camera", prop, "0.0")
    store = channel_settings_store(mmcore)
    mmcore.setConfig("Channel", "DAPI")
    store.set(
        "Channel",
        "DAPI",
        {"exposure": 20, "light_source": "_lights", "intensity": 12},
    )
    assert store.apply_current_channel()
    assert store.light_intensity() == 12
    mmcore.setProperty("Camera", "TestProperty4", 7)
    assert store.light_intensity() is None
