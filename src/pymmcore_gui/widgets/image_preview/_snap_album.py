"""A short in-memory history of snapped images.

The Preview only ever holds the latest frame, so each snap overwrites the one
before it. The album keeps the last few around to get one of them back.
"""

from __future__ import annotations

from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from pymmcore_gui._channel_settings_store import current_light_intensity
from pymmcore_gui._qt.QtCore import QObject, QPoint, QSize, Qt, Signal
from pymmcore_gui._qt.QtGui import QIcon, QImage, QPixmap
from pymmcore_gui._qt.QtWidgets import (
    QFrame,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pymmcore_gui._utils import current_channel

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pymmcore_plus import CMMCorePlus

# Full frames are kept, so this is what bounds the album's memory use.
ALBUM_CAPACITY = 10

_THUMBNAIL_SIZE = 56
_MAX_VISIBLE_ROWS = 6
_POPUP_WIDTH = 300


@dataclass(frozen=True, eq=False)
class SnappedImage:
    """One snapped frame and the state of the microscope when it was taken."""

    image: np.ndarray
    timestamp: datetime = field(default_factory=datetime.now)
    channel: str = ""
    exposure_ms: float | None = None
    pixel_size_um: float | None = None
    light_intensity: float | None = None

    @classmethod
    def capture(cls, core: CMMCorePlus, image: np.ndarray) -> SnappedImage:
        """Describe `image`, which was just snapped on `core`.

        Call this on the thread that snapped: by the time the GUI thread gets
        round to it, the microscope may already have moved on.
        """
        exposure = pixel_size = light_intensity = None
        with suppress(Exception):
            exposure = core.getExposure()
        with suppress(Exception):
            pixel_size = core.getPixelSizeUm() or None
        with suppress(Exception):
            light_intensity = current_light_intensity(core)
        return cls(
            image,
            channel=current_channel(core),
            exposure_ms=exposure,
            pixel_size_um=pixel_size,
            light_intensity=light_intensity,
        )

    @property
    def is_rgb(self) -> bool:
        return self.image.ndim == 3

    @property
    def _settings(self) -> list[str]:
        """Return what this was snapped with, leaving out what is not known."""
        settings = [self.channel] if self.channel else []
        if self.exposure_ms is not None:
            settings.append(f"{self.exposure_ms:g} ms")
        if self.light_intensity is not None:
            settings.append(f"I={self.light_intensity:g}")
        return settings

    @property
    def title(self) -> str:
        time = f"{self.timestamp:%H:%M:%S}"
        settings = self._settings
        return " · ".join([*settings, time]) if settings else f"Snap {time}"

    @property
    def description(self) -> str:
        """Return when this was snapped, then what with, on a second line."""
        height, width = self.image.shape[:2]
        details = " · ".join([*self._settings, f"{width} x {height}"])
        return f"{self.timestamp:%Y-%m-%d %H:%M:%S}\n{details}"


class SnapAlbum(QObject):
    """The most recent snaps, oldest dropped first.

    Outlives any one Preview, which can be closed and reopened. A QObject so that
    `add`, connected to a signal emitted on the thread that snapped, runs on the
    GUI thread that reads the album.
    """

    def __init__(
        self, capacity: int = ALBUM_CAPACITY, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._snaps: deque[SnappedImage] = deque(maxlen=capacity)

    def add(self, snap: SnappedImage) -> None:
        self._snaps.append(snap)

    def __len__(self) -> int:
        return len(self._snaps)

    def __iter__(self) -> Iterator[SnappedImage]:
        """Iterate from the newest snap to the oldest."""
        return reversed(self._snaps)


def _thumbnail(image: np.ndarray) -> QPixmap:
    """Render a small contrast-stretched copy of `image`."""
    step = max(1, max(image.shape[:2]) // _THUMBNAIL_SIZE)
    small = image[::step, ::step].astype(np.float32)
    low, high = np.percentile(small, (0.5, 99.5))
    scaled = np.clip((small - low) / max(high - low, 1), 0, 1) * 255
    pixels = scaled.astype(np.uint8)
    height, width = pixels.shape[:2]
    is_rgb = pixels.ndim == 3
    fmt = QImage.Format.Format_RGB888 if is_rgb else QImage.Format.Format_Grayscale8
    stride = width * 3 if is_rgb else width
    # fromImage copies, and QImage does not own this buffer; hold on to it until then
    buffer = pixels.tobytes()
    return QPixmap.fromImage(QImage(buffer, width, height, stride, fmt))


class SnapAlbumPopup(QFrame):
    """A popup listing an album's snaps; double-click or Enter opens one.

    The list sits inside a plain frame rather than being the popup itself: QWidget
    is what closes a popup on Escape or a click outside it, and item views handle
    those events without ever getting there.
    """

    snapActivated = Signal(object)  # SnappedImage

    def __init__(self, album: SnapAlbum, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Popup)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        # A click outside only closes the popup, as with a menu. Where the platform
        # replays that click (not macOS), it would otherwise hit the button that
        # opened the popup and open it straight back up.
        self.setAttribute(Qt.WidgetAttribute.WA_NoMouseReplay)

        self.list_widget = QListWidget(self)
        self.list_widget.setIconSize(QSize(_THUMBNAIL_SIZE, _THUMBNAIL_SIZE))
        self.list_widget.setUniformItemSizes(True)
        for snap in album:
            # the selected row must not tint its image with the highlight color
            icon = QIcon()
            pixmap = _thumbnail(snap.image)
            icon.addPixmap(pixmap, QIcon.Mode.Normal)
            icon.addPixmap(pixmap, QIcon.Mode.Selected)
            item = QListWidgetItem(icon, snap.description)
            item.setData(Qt.ItemDataRole.UserRole, snap)
            self.list_widget.addItem(item)
        if not self.list_widget.count():
            item = QListWidgetItem("No snapped images yet")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.list_widget.addItem(item)
        self.list_widget.setCurrentRow(0)
        self.list_widget.itemActivated.connect(self._on_item_activated)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.list_widget)

    def show_near(self, anchor: QWidget) -> None:
        """Open above `anchor` (the viewer's buttons run along its bottom edge)."""
        view = self.list_widget
        rows = min(view.count(), _MAX_VISIBLE_ROWS)
        height = view.sizeHintForRow(0) * rows + 2 * view.frameWidth()
        self.resize(_POPUP_WIDTH, height)

        pos = anchor.mapToGlobal(QPoint(0, -height))
        if (screen := anchor.screen()) is not None:
            bounds = screen.availableGeometry()
            if pos.y() < bounds.top():
                pos = anchor.mapToGlobal(QPoint(0, anchor.height()))
            pos.setX(min(pos.x(), bounds.right() - _POPUP_WIDTH))
        self.move(pos)
        self.show()
        view.setFocus()

    def _on_item_activated(self, item: QListWidgetItem) -> None:
        snap = item.data(Qt.ItemDataRole.UserRole)
        self.close()
        if snap is not None:
            self.snapActivated.emit(snap)
