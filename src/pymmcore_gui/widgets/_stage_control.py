from __future__ import annotations

import time
from contextlib import suppress
from typing import TYPE_CHECKING

from pymmcore_plus import CMMCorePlus, DeviceType
from pymmcore_widgets.control._q_stage_controller import QStageMoveAccumulator
from superqt import QIconifyIcon

from pymmcore_gui._qt.Qlementine import (  # type: ignore[attr-defined]
    MouseState,
    QlementineStyle,
    SegmentedControl,
)
from pymmcore_gui._qt.QtCore import QSize, Qt, QThread, QTimer, Signal
from pymmcore_gui._qt.QtGui import QColor, QFont, QPalette
from pymmcore_gui._qt.QtWidgets import (
    QApplication,
    QCheckBox,
    QDial,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMenu,
    QPushButton,
    QSizePolicy,
    QStyleFactory,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from qtpy.QtGui import QContextMenuEvent, QKeyEvent, QWheelEvent

ICON_SIZE = QSize(22, 22)
BTN_SIZE = 58
Z_BTN_HEIGHT = (BTN_SIZE * 3 + 2 * 2 - 2) // 2
POLL_INTERVAL_MS = 500
MAX_WHEEL_STEP = 1.0  # µm - cap scroll-wheel Z moves

_STEP_PRESETS: list[tuple[float, str]] = [
    (0.1, "0.1"),
    (1.0, "1"),
    (10.0, "10"),
    (100.0, "100"),
    (1000.0, "1k"),
]
_DEFAULT_STEP_INDEX = 2  # 10 µm


def _qlementine_style() -> QlementineStyle | None:
    style = QApplication.instance().style()  # type: ignore [union-attr]
    return style if isinstance(style, QlementineStyle) else None


def _set_label_color(label: QLabel, color: QColor) -> None:
    pal = label.palette()
    pal.setColor(QPalette.ColorRole.WindowText, color)
    label.setPalette(pal)


def _mono_font(size: int = 11) -> QFont:
    font = QFont("Menlo, Consolas, monospace", size)
    font.setStyleHint(QFont.StyleHint.Monospace)
    return font


def _clear_layout(layout: QLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if (w := item.widget()) is not None:
            w.deleteLater()
        elif (sub := item.layout()) is not None:
            _clear_layout(sub)


# ── Stage position poller (background thread) ───────────────────────


class _StagePoller(QThread):
    """Polls Z stage positions off the UI thread."""

    # list of (device_name, z_value)
    positionReady = Signal(object)

    def __init__(self, mmc: CMMCorePlus, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mmc = mmc
        self._running = True

    def run(self) -> None:
        while self._running:
            positions: list[tuple[str, float]] = []
            for dev in self._mmc.getLoadedDevicesOfType(DeviceType.Stage):
                try:
                    z = self._mmc.getPosition(dev)
                    positions.append((dev, z))
                except Exception:
                    pass
            self.positionReady.emit(positions)
            self.msleep(POLL_INTERVAL_MS)

    def stop(self) -> None:
        self._running = False


# ── Position spinbox ─────────────────────────────────────────────────


class _PositionSpinBox(QDoubleSpinBox):
    """Always-visible position display that doubles as an absolute-move input.

    Normally displays the current position. Click to edit, press Enter to
    move the stage to the entered value.
    """

    goToRequested = Signal(float)

    def __init__(self, parent: QWidget | None = None, *, suffix: str = " µm") -> None:
        super().__init__(parent)
        self.setRange(-1e7, 1e7)
        self.setDecimals(2)
        self.setSuffix(suffix)
        self.setFont(_mono_font(12))
        self.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.setMinimumHeight(20)
        self.setKeyboardTracking(False)
        self.setReadOnly(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._last_core_value = 0.0

        self.editingFinished.connect(self._on_editing_finished)

    def set_core_value(self, v: float) -> None:
        """Update from core - only touches the display if not being edited."""
        self._last_core_value = v
        if not self.hasFocus():
            self.setValue(v)

    def focusInEvent(self, event: object) -> None:
        self.setReadOnly(False)
        self.selectAll()
        super().focusInEvent(event)  # type: ignore [arg-type]

    def focusOutEvent(self, event: object) -> None:
        self.setReadOnly(True)
        self.setValue(self._last_core_value)
        super().focusOutEvent(event)  # type: ignore [arg-type]

    def _on_editing_finished(self) -> None:
        if self.isReadOnly():
            return
        val = self.value()
        self._last_core_value = val  # prevent focusOutEvent from snapping back
        self.setReadOnly(True)
        self.clearFocus()
        self.goToRequested.emit(val)


# ── Dial widget (rotation stage) ────────────────────────────────────


class _DialWidget(QWidget):
    """Circular dial for rotation stages that expose DeviceUnitsPerRevolution."""

    stepRequested = Signal(float)   # relative delta in degrees (for setRelativePosition)

    def __init__(self, units_per_rev: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._units_per_rev = units_per_rev

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        box = QGroupBox(self)
        grid = QGridLayout(box)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(2)

        def _lbl(text: str) -> QLabel:
            lbl = QLabel(text, box)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            font = lbl.font()
            font.setPointSize(font.pointSize() - 1)
            lbl.setFont(font)
            return lbl

        grid.addWidget(_lbl("180°"), 0, 1)
        grid.addWidget(_lbl("90°"), 1, 0)
        grid.addWidget(_lbl("270°"), 1, 2)
        grid.addWidget(_lbl("0°"), 2, 1)

        self._dial = QDial(box)
        self._dial.setWrapping(True)
        self._dial.setMinimumSize(110, 110)
        self._dial.setMinimum(0)
        self._dial.setMaximum(359)
        self._dial.setNotchesVisible(True)
        self._dial.setNotchTarget(23)
        self._dial.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # Force Fusion style so Qlementine doesn't break wrapping or wheel events
        if (fusion := QStyleFactory.create("Fusion")):
            self._dial.setStyle(fusion)
        self._dial.valueChanged.connect(self._on_dial_changed)
        grid.addWidget(self._dial, 1, 1)

        outer.addWidget(box)

        self._last_sent_degrees: float = 0.0
        self._initialized: bool = False
        self._last_move_at: float = 0.0  # time.monotonic() of last goToRequested
        self._pending_delta: float = 0.0

        # Debounce: accumulate all rapid ticks and flush as ONE setRelativePosition.
        # Without this, dragging 90° fires 90 individual calls and the device drops most.
        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.setInterval(80)  # ms quiet-time before flushing
        self._flush_timer.timeout.connect(self._flush_pending)

    def set_angle(self, degrees: float) -> None:
        """Update dial display without emitting goToRequested."""
        if not self._initialized:
            self._last_sent_degrees = degrees
            self._initialized = True
        # Don't visually snap the dial back while the stage is still moving.
        # The stage typically takes ~1 s; suppress poll-driven updates for 1.5 s
        # after the last command (mirrors x.py which never updates dial from poll).
        if time.monotonic() - self._last_move_at > 1.5:
            self._dial.blockSignals(True)
            self._dial.setValue(int(degrees) % 360)
            self._dial.blockSignals(False)

    def cancel_pending(self) -> None:
        """Cancel any buffered dial motion."""
        self._flush_timer.stop()
        self._pending_delta = 0.0

    def sync_to(self, degrees: float) -> None:
        """Sync tracking when an external source (e.g. lineedit) commands a move."""
        self._last_sent_degrees = degrees  # keep unbounded — no % 360
        self._last_move_at = time.monotonic()
        self._dial.blockSignals(True)
        self._dial.setValue(int(degrees) % 360)
        self._dial.blockSignals(False)

    def _on_dial_changed(self, value: int) -> None:
        """Accumulate dial ticks; flush as a single stepRequested after 80 ms idle."""
        print(f"[dial] valueChanged → {value}°")
        # Shortest-path delta so 359→0 moves forward, not backwards 359°
        prev = self._last_sent_degrees % 360
        delta = float(value) - prev
        if delta > 180:
            delta -= 360
        elif delta < -180:
            delta += 360

        self._last_sent_degrees += delta
        self._last_move_at = time.monotonic()
        self._pending_delta += delta

        # Keep the visual in sync with what the user just dialled
        self._dial.blockSignals(True)
        self._dial.setValue(int(self._last_sent_degrees) % 360)
        self._dial.blockSignals(False)

        # (Re)start the debounce timer — fires once the user stops dragging
        self._flush_timer.start()

    def _flush_pending(self) -> None:
        """Send the accumulated delta as a single relative-position command."""
        if self._pending_delta:
            print(f"[dial] stepRequested.emit({self._pending_delta:+.2f}°)")
            self.stepRequested.emit(self._pending_delta)
            self._pending_delta = 0.0


def _is_rotation_stage(mmc: CMMCorePlus, device: str) -> bool:
    """Return True if the device exposes DeviceUnitsPerRevolution."""
    try:
        mmc.getProperty(device, "DeviceUnitsPerRevolution")
        return True
    except Exception:
        return False


def _units_per_rev(mmc: CMMCorePlus, device: str) -> float:
    try:
        return float(mmc.getProperty(device, "DeviceUnitsPerRevolution"))
    except Exception:
        return 360.0


# ── Z buttons ─────────────────────────────────────────────────────────


class _ZButtons(QWidget):
    """Z-axis up/down buttons."""

    moveRequested = Signal(int)
    wheelScrolled = Signal(int)  # direction: +1 / -1

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._up = self._make_btn("mdi:chevron-up")
        self._down = self._make_btn("mdi:chevron-down")

        self._up.clicked.connect(lambda: self.moveRequested.emit(1))
        self._down.clicked.connect(lambda: self.moveRequested.emit(-1))

        layout.addWidget(self._up)
        layout.addWidget(self._down)

    def flash_button(self, direction: int) -> None:
        btn = self._up if direction > 0 else self._down
        btn.setDown(True)
        QTimer.singleShot(100, lambda: btn.setDown(False))

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        if event is None:
            return
        delta = event.angleDelta().y()
        if delta > 0:
            self.wheelScrolled.emit(1)
        elif delta < 0:
            self.wheelScrolled.emit(-1)
        event.accept()

    def _make_btn(self, icon_key: str) -> QPushButton:
        btn = QPushButton(self)
        btn.setIcon(QIconifyIcon(icon_key, color="#aaa"))
        btn.setIconSize(QSize(26, 26))
        btn.setFixedSize(BTN_SIZE, Z_BTN_HEIGHT)
        btn.setAutoRepeat(True)
        btn.setAutoRepeatDelay(400)
        btn.setAutoRepeatInterval(80)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        return btn


# ── Step size (SegmentedControl) ─────────────────────────────────────


class _StepSizeBar(QWidget):
    """Qlementine SegmentedControl for step-size selection."""

    stepChanged = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        layout.addStretch()

        lbl = QLabel("STEP", self)
        if qs := _qlementine_style():
            _set_label_color(
                lbl,
                qs.labelCaptionForegroundColor(MouseState.Normal),
            )
        font = lbl.font()
        font.setPointSize(font.pointSize() - 1)
        lbl.setFont(font)
        layout.addWidget(lbl)

        self._seg = SegmentedControl()
        self._seg.setMaximumWidth(280)
        self._seg.setMaximumHeight(34)
        for value, text in _STEP_PRESETS:
            self._seg.addItem(text, itemData=value)
        self._seg.setCurrentIndex(_DEFAULT_STEP_INDEX)
        self._seg.currentIndexChanged.connect(self._on_index_changed)  # pyright: ignore[reportAttributeAccessIssue]
        layout.addWidget(self._seg)

        unit = QLabel("µm", self)
        if qs:
            _set_label_color(
                unit,
                qs.labelCaptionForegroundColor(MouseState.Normal),
            )
        layout.addWidget(unit)

        layout.addStretch()

    def _on_index_changed(self) -> None:
        data = self._seg.currentData()
        if data is not None:
            self.stepChanged.emit(float(data))

    def current_step(self) -> float:
        data = self._seg.currentData()
        if data is not None:
            return float(data)
        return _STEP_PRESETS[_DEFAULT_STEP_INDEX][0]


# ── Main widget ──────────────────────────────────────────────────────


class StagesControlWidget(QWidget):
    """Z stage control for one or more focus devices."""

    def __init__(
        self, *, parent: QWidget | None = None, mmcore: CMMCorePlus | None = None
    ) -> None:
        super().__init__(parent=parent)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._mmc = mmcore or CMMCorePlus.instance()
        self._invert_z = False
        self._z_accums: dict[str, QStageMoveAccumulator] = {}
        self._z_btns_map: dict[str, _ZButtons] = {}
        self._z_spin_map: dict[str, _PositionSpinBox] = {}
        self._rot_spin_map: dict[str, _PositionSpinBox] = {}
        self._z_dial_map: dict[str, _DialWidget] = {}

        self._build_ui()
        self._connect_signals()
        self._on_cfg_loaded()
        self._stage_poller = _StagePoller(self._mmc)
        self._stage_poller.positionReady.connect(self._on_polled_position)
        if self._poll_cb.isChecked():
            self._stage_poller.start()

    def _disconnect(self) -> None:
        self._stage_poller.stop()
        self._stage_poller.wait()
        self._disconnect_accumulators()
        evts = self._mmc.events
        evts.systemConfigurationLoaded.disconnect(self._on_cfg_loaded)
        evts.stagePositionChanged.disconnect(self._on_z_pos_changed)

    # ── UI construction ──────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # Step size bar
        self._step_bar = _StepSizeBar(self)
        root.addWidget(self._step_bar)

        # Z buttons row (rebuilt dynamically per device)
        self._z_btns_layout = QHBoxLayout()
        self._z_btns_layout.setSpacing(8)
        root.addLayout(self._z_btns_layout)

        # Keyboard hint
        self._hint = QLabel("Esc to stop", self)
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if qs := _qlementine_style():
            _set_label_color(
                self._hint,
                qs.labelCaptionForegroundColor(MouseState.Normal),
            )
        font = self._hint.font()
        font.setPointSize(font.pointSize() - 1)
        self._hint.setFont(font)
        root.addWidget(self._hint)

        # STOP button
        self._stop_btn = QPushButton(
            QIconifyIcon("glyphs:stop-sign-bold", color="white"), "", self
        )
        self._stop_btn.setIconSize(QSize(32, 32))
        self._stop_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._stop_btn.setToolTip("Stop all stage movement (Esc)")
        self._stop_btn.setStyleSheet(
            "QPushButton { background: #a8222a; color: white; font-weight: bold;"
            " font-size: 13px; border-radius: 4px; padding: 4px; }"
            "QPushButton:hover { background: #c0333b; }"
            "QPushButton:pressed { background: #8a1a22; }"
        )
        root.addWidget(self._stop_btn)

        # Z position row (rebuilt dynamically per device)
        self._z_pos_layout = QHBoxLayout()
        self._z_pos_layout.setSpacing(8)
        root.addLayout(self._z_pos_layout)

        # Snap + Poll checkboxes
        checks_row = QHBoxLayout()
        checks_row.setSpacing(15)
        self._snap_cb = QCheckBox("Snap on Click", self)
        self._snap_cb.setToolTip("Snap image after each move")
        self._snap_cb.setChecked(True)
        self._poll_cb = QCheckBox("Poll", self)
        self._poll_cb.setToolTip("Poll stage position periodically")
        self._poll_cb.setChecked(True)
        checks_row.addWidget(self._snap_cb)
        checks_row.addWidget(self._poll_cb)
        checks_row.addStretch()
        root.addLayout(checks_row)

    def _rebuild_z_widgets(self, z_devs: list[str]) -> None:
        """Remove and recreate per-device Z buttons and position spinboxes.

        Regular Z stages are rendered left-to-right; rotation (dial) stages
        always appear to the right of all regular stages.
        """
        self._z_btns_map.clear()
        self._z_spin_map.clear()
        self._rot_spin_map.clear()
        self._z_dial_map.clear()
        _clear_layout(self._z_btns_layout)
        _clear_layout(self._z_pos_layout)

        regular = [d for d in z_devs if not _is_rotation_stage(self._mmc, d)]
        rotation = [d for d in z_devs if _is_rotation_stage(self._mmc, d)]

        self._z_btns_layout.addStretch()
        for dev in regular:
            col = QVBoxLayout()
            col.setSpacing(2)
            col.setContentsMargins(0, 0, 0, 0)

            dev_lbl = QLabel(dev, self)
            dev_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            font = dev_lbl.font()
            font.setPointSize(font.pointSize() - 1)
            dev_lbl.setFont(font)
            col.addWidget(dev_lbl)

            btns = _ZButtons(self)
            btns.moveRequested.connect(
                lambda direction, d=dev: self._on_z_move(d, direction)
            )
            btns.wheelScrolled.connect(
                lambda direction, d=dev: self._on_z_move(
                    d, direction, max_step=MAX_WHEEL_STEP
                )
            )
            col.addWidget(btns)
            self._z_btns_map[dev] = btns
            col_widget = QWidget(self)
            col_widget.setLayout(col)
            self._z_btns_layout.addWidget(
                col_widget, alignment=Qt.AlignmentFlag.AlignTop
            )

        for dev in rotation:
            col = QVBoxLayout()
            col.setSpacing(2)
            col.setContentsMargins(0, 0, 0, 0)

            dev_lbl = QLabel(dev, self)
            dev_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            font = dev_lbl.font()
            font.setPointSize(font.pointSize() - 1)
            dev_lbl.setFont(font)
            col.addWidget(dev_lbl)

            dial = _DialWidget(_units_per_rev(self._mmc, dev), self)
            dial.stepRequested.connect(lambda delta, d=dev: self._on_dial_step(d, delta))
            col.addWidget(dial)
            self._z_dial_map[dev] = dial
            col_widget = QWidget(self)
            col_widget.setLayout(col)
            self._z_btns_layout.addWidget(
                col_widget, alignment=Qt.AlignmentFlag.AlignCenter
            )

        self._z_btns_layout.addStretch()

        self._z_pos_layout.addStretch()
        z_color = QColor("#6090e0")
        rot_color = QColor("#e09060")

        for dev in regular:
            col = QVBoxLayout()
            col.setSpacing(2)
            col.setContentsMargins(0, 0, 0, 0)

            lbl = QLabel(dev, self)
            lbl.setFont(_mono_font())
            _set_label_color(lbl, z_color)
            col.addWidget(lbl)

            spin = _PositionSpinBox(self)
            spin.goToRequested.connect(lambda v, d=dev: self._on_go_to(d, v))
            self._z_spin_map[dev] = spin
            col.addWidget(spin)
            self._z_pos_layout.addLayout(col)

        for dev in rotation:
            col = QVBoxLayout()
            col.setSpacing(2)
            col.setContentsMargins(0, 0, 0, 0)

            lbl = QLabel(dev, self)
            lbl.setFont(_mono_font())
            _set_label_color(lbl, rot_color)
            col.addWidget(lbl)

            rot_spin = _PositionSpinBox(self, suffix="°")
            rot_spin.setRange(0, 360)
            rot_spin.goToRequested.connect(
                lambda v, d=dev: self._on_go_to_absolute(d, v)
            )
            self._rot_spin_map[dev] = rot_spin
            col.addWidget(rot_spin)
            self._z_pos_layout.addLayout(col)

        self._z_pos_layout.addStretch()

    # ── Signal wiring ────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self._mmc.events.systemConfigurationLoaded.connect(self._on_cfg_loaded)
        self._mmc.events.stagePositionChanged.connect(self._on_z_pos_changed)
        self._poll_cb.toggled.connect(self._on_poll_toggled)
        self._stop_btn.clicked.connect(self._stop_all)

    # ── Configuration loaded ─────────────────────────────────────────

    def _on_cfg_loaded(self) -> None:
        z_devs = [str(d) for d in self._mmc.getLoadedDevicesOfType(DeviceType.Stage)]
        self._rebuild_z_widgets(z_devs)
        self._setup_accumulators(z_devs)
        self._update_positions()

    def _setup_accumulators(self, z_devs: list[str]) -> None:
        self._disconnect_accumulators()
        for dev in z_devs:
            if _is_rotation_stage(self._mmc, dev):
                continue  # rotation stages use direct setPosition, no accumulator needed
            with suppress(Exception):
                accum = QStageMoveAccumulator.for_device(dev, self._mmc)
                accum.moveFinished.connect(
                    lambda d=dev: self._update_single_position(d)
                )
                self._z_accums[dev] = accum

    def _disconnect_accumulators(self) -> None:
        for accum in self._z_accums.values():
            with suppress(TypeError, RuntimeError):
                accum.moveFinished.disconnect()
        self._z_accums.clear()

    # ── Movement ─────────────────────────────────────────────────────

    def _on_z_move(self, device: str, direction: int, max_step: float = 0) -> None:
        accum = self._z_accums.get(device)
        if not accum:
            return
        step = self._step_bar.current_step()
        if max_step > 0:
            step = min(step, max_step)
        dz = step * direction * (-1 if self._invert_z else 1)
        accum.snap_on_finish = self._snap_cb.isChecked()
        accum.move_relative(dz)

    def _on_go_to(self, device: str, value: float) -> None:
        if accum := self._z_accums.get(device):
            accum.move_absolute(value)

    def _on_dial_step(self, device: str, delta: float) -> None:
        """Relative move from dial — uses setRelativePosition to avoid wrap ambiguity."""
        print(f"[stage] setRelativePosition({device!r}, {delta:+.2f})")
        try:
            self._mmc.setRelativePosition(device, delta)
        except Exception as e:
            print(f"[stage] setRelativePosition FAILED: {e}")

    def _on_go_to_absolute(self, device: str, value: float) -> None:
        """Absolute move (used by rot_spin lineedit — value in degrees)."""
        print(f"[stage] _on_go_to_absolute({device!r}, {value:.2f})")
        # Sync the dial tracking so the next wheel-tick delta is correct
        if dial := self._z_dial_map.get(device):
            dial.sync_to(value)
        try:
            self._mmc.setPosition(device, value)
            print(f"[stage] setPosition OK")
        except Exception as e:
            print(f"[stage] setPosition FAILED: {e}")

    def _stop_all(self) -> None:
        # Cancel any pending debounced dial move before stopping
        for dial in self._z_dial_map.values():
            dial.cancel_pending()
        for dev in self._mmc.getLoadedDevicesOfType(DeviceType.Stage):
            try:
                self._mmc.stop(dev)
            except Exception:
                pass

    # ── Position updates ─────────────────────────────────────────────

    def _update_positions(self) -> None:
        for dev in self._z_spin_map:
            self._update_single_position(dev)
        for dev in self._z_dial_map:
            self._update_single_position(dev)

    def _update_single_position(self, device: str) -> None:
        try:
            pos = self._mmc.getPosition(device)
        except Exception:
            return
        if spin := self._z_spin_map.get(device):
            spin.set_core_value(pos)
        if dial := self._z_dial_map.get(device):
            dial.set_angle(pos)
        if rot_spin := self._rot_spin_map.get(device):
            rot_spin.set_core_value(pos % 360)

    def _on_polled_position(self, positions: list[tuple[str, float]]) -> None:
        for dev, z in positions:
            if spin := self._z_spin_map.get(dev):
                spin.set_core_value(z)
            if dial := self._z_dial_map.get(dev):
                dial.set_angle(z)
            if rot_spin := self._rot_spin_map.get(dev):
                rot_spin.set_core_value(z % 360)

    def _on_z_pos_changed(self, device: str, z: float) -> None:
        if spin := self._z_spin_map.get(device):
            spin.set_core_value(z)
        if dial := self._z_dial_map.get(device):
            dial.set_angle(z)
        if rot_spin := self._rot_spin_map.get(device):
            rot_spin.set_core_value(z % 360)

    # ── Polling ──────────────────────────────────────────────────────

    def _on_poll_toggled(self, checked: bool) -> None:
        if checked:
            self._stage_poller._running = True
            self._stage_poller.start()
        else:
            self._stage_poller.stop()
            self._stage_poller.wait()

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event is None:
            return
        if event.key() == Qt.Key.Key_Escape:
            self._stop_all()
        else:
            super().keyPressEvent(event)

    # ── Context menu ─────────────────────────────────────────────────

    def contextMenuEvent(self, event: QContextMenuEvent | None) -> None:
        if event is None:
            return
        menu = QMenu(self)
        invert_z = menu.addAction("Invert Z")
        if invert_z:
            invert_z.setCheckable(True)
            invert_z.setChecked(self._invert_z)
            invert_z.toggled.connect(self._set_invert_z)
        menu.exec(event.globalPos())

    def _set_invert_z(self, v: bool) -> None:
        self._invert_z = v
