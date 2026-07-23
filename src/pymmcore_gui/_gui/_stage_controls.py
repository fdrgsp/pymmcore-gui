"""Compact, rearrangeable controls for all loaded XY and Z stages."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from pymmcore_plus import CMMCorePlus, DeviceType
from superqt.utils import create_worker

from pymmcore_gui._qt.QtCore import QMimeData, QPoint, Qt, QTimer
from pymmcore_gui._qt.QtGui import QDrag
from pymmcore_gui._qt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from pymmcore_gui._qt.QtGui import (
        QDragEnterEvent,
        QDropEvent,
        QMouseEvent,
    )

_STAGE_MIME = "application/x-pymmcore-stage-card"


def _position_box() -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(-10_000_000, 10_000_000)
    box.setDecimals(3)
    box.setSuffix(" µm")
    box.setKeyboardTracking(False)
    return box


def _move_button(text: str, tooltip: str) -> QPushButton:
    button = QPushButton(text)
    button.setToolTip(tooltip)
    button.setAutoRepeat(True)
    button.setAutoRepeatDelay(350)
    button.setAutoRepeatInterval(100)
    button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    button.setMinimumSize(34, 30)
    return button


class StageCard(QFrame):
    """One small control card for an XY or Z stage."""

    def __init__(
        self, device: str, device_type: DeviceType, core: CMMCorePlus
    ) -> None:
        super().__init__()
        self.device = device
        self.device_type = device_type
        self._core = core
        self._drag_start: QPoint | None = None
        self._position_boxes: list[QDoubleSpinBox] = []

        self.setObjectName("stageCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(7, 7, 7, 7)
        root.setSpacing(5)

        header = QHBoxLayout()
        self._drag_handle = QLabel("⠿")
        self._drag_handle.setToolTip("Drag to rearrange this stage")
        self._name_label = QLabel(device)
        self._name_label.setToolTip(device)
        kind = QLabel("XY" if device_type is DeviceType.XYStage else "Z")
        kind.setObjectName("stageKind")
        self._snap = QCheckBox("Snap")
        self._snap.setToolTip("Snap an image after each move")
        self._poll = QCheckBox("Poll")
        self._poll.setChecked(True)
        header.addWidget(self._drag_handle)
        header.addWidget(self._name_label)
        header.addWidget(kind)
        header.addStretch()
        header.addWidget(self._snap)
        header.addWidget(self._poll)
        root.addLayout(header)

        self._position_layout = QGridLayout()
        self._position_layout.setContentsMargins(0, 0, 0, 0)
        self._position_layout.setSpacing(4)
        root.addLayout(self._position_layout)

        self._movement_layout = QGridLayout()
        self._movement_layout.setContentsMargins(0, 0, 0, 0)
        self._movement_layout.setSpacing(2)
        root.addLayout(self._movement_layout)

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step"))
        self._step = QDoubleSpinBox()
        self._step.setRange(0.001, 10_000_000)
        self._step.setDecimals(3)
        self._step.setValue(10)
        self._step.setSuffix(" µm")
        self._step.setKeyboardTracking(False)
        step_row.addWidget(self._step, 1)
        root.addLayout(step_row)

        # subclasses add their invert checkbox(es) here, then call
        # _finish_options_row() to append the stretch + Stop button
        self._options_layout = QHBoxLayout()
        self._options_layout.setContentsMargins(0, 0, 0, 0)
        root.addLayout(self._options_layout)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(500)
        self._poll_timer.timeout.connect(self.refresh_position)
        self._poll.toggled.connect(self._toggle_polling)
        self._poll_timer.start()

    @property
    def poll_checkbox(self) -> QCheckBox:
        return self._poll

    @property
    def step_box(self) -> QDoubleSpinBox:
        return self._step

    def _finish_options_row(self) -> None:
        """Append the stretch + Stop button after any subclass invert checkboxes."""
        self._options_layout.addStretch()
        stop_btn = QPushButton("Stop")
        stop_btn.setToolTip(f"Halt motion on {self.device}")
        stop_btn.setProperty("variant", "danger")
        stop_btn.clicked.connect(self._stop)
        self._options_layout.addWidget(stop_btn)

    def _stop(self) -> None:
        try:
            self._core.stop(self.device)
        except Exception as error:
            self._report_error(error)

    def _maybe_snap(self) -> None:
        """Snap an image in the background if the "Snap" checkbox is on."""
        if self._snap.isChecked():
            create_worker(self._core.snap, _start_thread=True)

    def _toggle_polling(self, enabled: bool) -> None:
        if enabled:
            self.refresh_position()
            self._poll_timer.start()
        else:
            self._poll_timer.stop()

    def _set_position(self, box: QDoubleSpinBox, position: float) -> None:
        if not box.hasFocus():
            box.setValue(position)

    def _report_error(self, error: Exception) -> None:
        self.setToolTip(f"Last stage error: {error}")

    def dispose(self) -> None:
        """Stop callbacks before this card is discarded."""
        self._poll_timer.stop()

    def refresh_position(self) -> None:
        """Update position inputs from the stage."""
        raise NotImplementedError

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if (
            event is not None
            and event.button() is Qt.MouseButton.LeftButton
            and event.position().y() <= 38
        ):
            self._drag_start = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if (
            event is not None
            and self._drag_start is not None
            and event.buttons() & Qt.MouseButton.LeftButton
            and (event.position().toPoint() - self._drag_start).manhattanLength()
            >= 8
        ):
            drag = QDrag(self)
            mime = QMimeData()
            mime.setData(_STAGE_MIME, self.device.encode())
            drag.setMimeData(mime)
            self._drag_start = None
            drag.exec(Qt.DropAction.MoveAction)
        super().mouseMoveEvent(event)


class XYStageCard(StageCard):
    """D-pad, absolute positioning, step, and polling for an XY stage."""

    _MOVES = (
        ("↖", -1, 1, 0, 0),
        ("↑", 0, 1, 0, 1),
        ("↗", 1, 1, 0, 2),
        ("←", -1, 0, 1, 0),
        ("•", 0, 0, 1, 1),
        ("→", 1, 0, 1, 2),
        ("↙", -1, -1, 2, 0),
        ("↓", 0, -1, 2, 1),
        ("↘", 1, -1, 2, 2),
    )

    def __init__(self, device: str, core: CMMCorePlus) -> None:
        super().__init__(device, DeviceType.XYStage, core)
        self._x = _position_box()
        self._y = _position_box()
        self._position_boxes.extend((self._x, self._y))
        go = QPushButton("Go")
        go.setToolTip(f"Move {device} to the entered X/Y position")
        go.clicked.connect(self._go_absolute)
        self._position_layout.addWidget(QLabel("X"), 0, 0)
        self._position_layout.addWidget(self._x, 0, 1)
        self._position_layout.addWidget(QLabel("Y"), 1, 0)
        self._position_layout.addWidget(self._y, 1, 1)
        self._position_layout.addWidget(go, 0, 2, 2, 1)

        for text, dx, dy, row, column in self._MOVES:
            button = _move_button(text, f"Move X {dx:+} and Y {dy:+} steps")
            if not (dx or dy):
                button.setEnabled(False)
            else:
                button.clicked.connect(
                    lambda _checked=False, x=dx, y=dy: self._move_relative(x, y)
                )
            self._movement_layout.addWidget(button, row, column)

        self._invert_x = QCheckBox("Invert X")
        self._invert_y = QCheckBox("Invert Y")
        self._invert_x.setToolTip("Reverse the X direction of the buttons above")
        self._invert_y.setToolTip("Reverse the Y direction of the buttons above")
        self._options_layout.addWidget(self._invert_x)
        self._options_layout.addWidget(self._invert_y)
        self._finish_options_row()

        self._core.events.XYStagePositionChanged.connect(
            self._on_position_changed
        )
        self.refresh_position()

    def _move_relative(self, dx: int, dy: int) -> None:
        step = self._step.value()
        if self._invert_x.isChecked():
            dx = -dx
        if self._invert_y.isChecked():
            dy = -dy
        try:
            self._core.setRelativeXYPosition(
                self.device, dx * step, dy * step
            )
            self.refresh_position()
            self._maybe_snap()
        except Exception as error:
            self._report_error(error)

    def _go_absolute(self) -> None:
        try:
            self._core.setXYPosition(
                self.device, self._x.value(), self._y.value()
            )
            self.refresh_position()
            self._maybe_snap()
        except Exception as error:
            self._report_error(error)

    def _on_position_changed(
        self, device: str, x_position: float, y_position: float
    ) -> None:
        if device == self.device:
            self._set_position(self._x, x_position)
            self._set_position(self._y, y_position)

    def refresh_position(self) -> None:
        try:
            x_position, y_position = self._core.getXYPosition(self.device)
            self._set_position(self._x, x_position)
            self._set_position(self._y, y_position)
        except Exception as error:
            self._report_error(error)

    def dispose(self) -> None:
        super().dispose()
        with suppress(Exception):
            self._core.events.XYStagePositionChanged.disconnect(
                self._on_position_changed
            )


class ZStageCard(StageCard):
    """Up/down control, absolute positioning, step, and polling for a Z stage."""

    def __init__(self, device: str, core: CMMCorePlus) -> None:
        super().__init__(device, DeviceType.Stage, core)
        self._z = _position_box()
        self._position_boxes.append(self._z)
        go = QPushButton("Go")
        go.setToolTip(f"Move {device} to the entered Z position")
        go.clicked.connect(self._go_absolute)
        self._position_layout.addWidget(QLabel("Z"), 0, 0)
        self._position_layout.addWidget(self._z, 0, 1)
        self._position_layout.addWidget(go, 0, 2)

        down = _move_button("↓", "Move down one step")
        up = _move_button("↑", "Move up one step")
        down.clicked.connect(lambda: self._move_relative(-1))
        up.clicked.connect(lambda: self._move_relative(1))
        self._movement_layout.addWidget(up, 0, 0)
        self._movement_layout.addWidget(down, 1, 0)

        self._invert = QCheckBox("Invert")
        self._invert.setToolTip("Reverse the direction of the Up/Down buttons above")
        self._options_layout.addWidget(self._invert)
        self._finish_options_row()

        self._core.events.stagePositionChanged.connect(
            self._on_position_changed
        )
        self.refresh_position()

    def _move_relative(self, direction: int) -> None:
        if self._invert.isChecked():
            direction = -direction
        try:
            self._core.setRelativePosition(
                self.device, direction * self._step.value()
            )
            self.refresh_position()
            self._maybe_snap()
        except Exception as error:
            self._report_error(error)

    def _go_absolute(self) -> None:
        try:
            self._core.setPosition(self.device, self._z.value())
            self.refresh_position()
            self._maybe_snap()
        except Exception as error:
            self._report_error(error)

    def _on_position_changed(self, device: str, position: float) -> None:
        if device == self.device:
            self._set_position(self._z, position)

    def refresh_position(self) -> None:
        try:
            self._set_position(self._z, self._core.getPosition(self.device))
        except Exception as error:
            self._report_error(error)

    def dispose(self) -> None:
        super().dispose()
        with suppress(Exception):
            self._core.events.stagePositionChanged.disconnect(
                self._on_position_changed
            )


class StageGrid(QWidget):
    """Drop target that lays stage cards out in a configurable column count."""

    def __init__(self, columns: int = 1) -> None:
        super().__init__()
        self._columns = max(1, columns)
        self._cards: list[StageCard] = []
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)
        self._layout.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.setAcceptDrops(True)

    @property
    def cards(self) -> tuple[StageCard, ...]:
        return tuple(self._cards)

    @property
    def columns(self) -> int:
        return self._columns

    def set_columns(self, columns: int) -> None:
        self._columns = max(1, columns)
        self._relayout()

    def set_cards(self, cards: list[StageCard]) -> None:
        self._cards = cards
        for card in cards:
            card.setParent(self)
        self._relayout()

    def move_card(self, device: str, index: int) -> None:
        """Move a device card to ``index`` and reflow the grid."""
        source = next(
            (i for i, card in enumerate(self._cards) if card.device == device),
            -1,
        )
        if source < 0:
            return
        card = self._cards.pop(source)
        self._cards.insert(max(0, min(index, len(self._cards))), card)
        self._relayout()

    def _relayout(self) -> None:
        while self._layout.count():
            self._layout.takeAt(0)
        for index, card in enumerate(self._cards):
            self._layout.addWidget(
                card, index // self._columns, index % self._columns
            )
        for column in range(self._columns):
            self._layout.setColumnStretch(column, 1)

    def dragEnterEvent(self, event: QDragEnterEvent | None) -> None:
        if event is not None and event.mimeData().hasFormat(_STAGE_MIME):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent | None) -> None:
        if event is None or not event.mimeData().hasFormat(_STAGE_MIME):
            super().dropEvent(event)
            return
        device = bytes(event.mimeData().data(_STAGE_MIME)).decode()
        position = event.position()
        if not self._cards:
            target = 0
        else:
            target = min(
                range(len(self._cards)),
                key=lambda i: (
                    self._cards[i].geometry().center().x() - position.x()
                )
                ** 2
                + (
                    self._cards[i].geometry().center().y() - position.y()
                )
                ** 2,
            )
        self.move_card(device, target)
        event.acceptProposedAction()


class StageControls(QWidget):
    """Discover, show, and arrange every loaded XY and Z stage."""

    def __init__(
        self, core: CMMCorePlus, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._core = core
        self._cards_by_device: dict[str, StageCard] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(5)

        controls = QHBoxLayout()
        title = QLabel("Stages")
        controls.addWidget(title)
        controls.addStretch()
        controls.addWidget(QLabel("Columns"))
        self._columns = QComboBox()
        self._columns.addItems(["1", "2", "3"])
        self._columns.setCurrentText("1")
        self._columns.currentTextChanged.connect(
            lambda value: self._grid.set_columns(int(value))
        )
        controls.addWidget(self._columns)
        root.addLayout(controls)

        self._empty_label = QLabel("No XY or Z stages loaded")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._empty_label)

        self._grid = StageGrid()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        scroll.setWidget(self._grid)
        root.addWidget(scroll, 1)

        self._core.events.systemConfigurationLoaded.connect(
            self.refresh_devices
        )
        self.refresh_devices()

    @property
    def cards(self) -> tuple[StageCard, ...]:
        return self._grid.cards

    def refresh_devices(self, *_: object) -> None:
        """Re-scan the core, preserving card order where possible."""
        xy_devices = list(
            self._core.getLoadedDevicesOfType(DeviceType.XYStage)
        )
        z_devices = list(
            self._core.getLoadedDevicesOfType(DeviceType.Stage)
        )
        device_types = {
            **dict.fromkeys(xy_devices, DeviceType.XYStage),
            **dict.fromkeys(z_devices, DeviceType.Stage),
        }

        old_order = [card.device for card in self._grid.cards]
        order = [device for device in old_order if device in device_types]
        order.extend(device for device in device_types if device not in order)

        for device in set(self._cards_by_device) - set(device_types):
            card = self._cards_by_device.pop(device)
            card.dispose()
            card.deleteLater()

        for device, device_type in device_types.items():
            if device not in self._cards_by_device:
                if device_type is DeviceType.XYStage:
                    card: StageCard = XYStageCard(device, self._core)
                else:
                    card = ZStageCard(device, self._core)
                self._cards_by_device[device] = card

        cards = [self._cards_by_device[device] for device in order]
        self._grid.set_cards(cards)
        self._empty_label.setVisible(not cards)
