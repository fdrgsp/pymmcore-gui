"""Minimal debug widget for Thorlabs Kinesis rotation stage.

Run with:
    python x.py [device_name] [config_file]

    device_name  : name of the Kinesis stage device (default: "ThorlabsKinesis")
    config_file  : path to a MicroManager config file (optional)

If no config is given the script tries to use whatever is already loaded in
the running CMMCorePlus instance (useful when launched from within the GUI).
"""

from __future__ import annotations

import sys

from pymmcore_plus import CMMCorePlus, DeviceType
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QDial,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import signals_blocked

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_pos(mmc: CMMCorePlus, device: str) -> float:
    return mmc.getPosition(device)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class KinesisDebugWidget(QWidget):
    """Minimal, self-contained rotation-stage widget for debugging."""

    def __init__(
        self,
        device: str,
        mmc: CMMCorePlus | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._device = device
        self._mmc = mmc or CMMCorePlus.instance()
        self._moving = False           # guard flag to prevent re-entrant moves
        self._pending_angle: int | None = None   # debounce: last requested angle
        self._last_sent_angle: float = 0.0       # cumulative, not wrapped to 0-360

        self.setWindowTitle(f"Kinesis debug – {device}")
        self._build_ui()
        self._connect_events()

        # Sync dial to current stage position on startup
        self._sync_dial_from_stage()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- position readout ---
        self._pos_label = QLabel("position: —")
        self._pos_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._pos_label)

        # --- dial (0..359 degrees, wrapping) ---
        self._dial = QDial()
        self._dial.setWrapping(True)
        self._dial.setMinimum(0)
        self._dial.setMaximum(359)
        self._dial.setNotchesVisible(True)
        self._dial.setMinimumSize(150, 150)
        # Only move stage when the user *releases* the dial or after a short
        # debounce – this prevents flooding the stage with commands while the
        # mouse wheel is still spinning.
        self._dial.valueChanged.connect(self._on_dial_value_changed)
        layout.addWidget(self._dial, alignment=Qt.AlignmentFlag.AlignCenter)

        # --- step size + manual go ---
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step (°):"))
        self._step = QDoubleSpinBox()
        self._step.setRange(0.01, 360)
        self._step.setValue(10)
        step_row.addWidget(self._step)

        btn_ccw = QPushButton("◀ CCW")
        btn_cw  = QPushButton("CW ▶")
        btn_ccw.clicked.connect(lambda: self._step_stage(-self._step.value()))
        btn_cw.clicked.connect(lambda:  self._step_stage(+self._step.value()))
        step_row.addWidget(btn_ccw)
        step_row.addWidget(btn_cw)
        layout.addLayout(step_row)

        # --- go to absolute angle ---
        abs_row = QHBoxLayout()
        abs_row.addWidget(QLabel("Go to (°):"))
        self._abs_spin = QDoubleSpinBox()
        self._abs_spin.setRange(0, 360)
        self._abs_spin.setValue(0)
        abs_row.addWidget(self._abs_spin)
        btn_go = QPushButton("Go")
        btn_go.clicked.connect(self._on_go_clicked)
        abs_row.addWidget(btn_go)
        layout.addLayout(abs_row)

        # --- poll timer to keep readout fresh ---
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(300)
        self._poll_timer.timeout.connect(self._refresh_pos_label)
        self._poll_timer.start()

        # --- debounce timer: send move only after user stops scrolling ---
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(150)   # ms quiet time before move
        self._debounce_timer.timeout.connect(self._send_debounced_move)

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------

    def _connect_events(self) -> None:
        self._mmc.events.stagePositionChanged.connect(self._on_stage_moved)

    def _disconnect_events(self) -> None:
        self._mmc.events.stagePositionChanged.disconnect(self._on_stage_moved)

    def closeEvent(self, event: object) -> None:  # type: ignore[override]
        self._disconnect_events()
        super().closeEvent(event)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Stage <-> UI synchronisation
    # ------------------------------------------------------------------

    def _sync_dial_from_stage(self) -> None:
        """Read stage position and update dial WITHOUT triggering a move."""
        try:
            pos = _get_pos(self._mmc, self._device)
        except Exception as exc:
            print(f"[x.py] could not read position: {exc}")
            return
        angle = int(round(pos)) % 360
        self._last_sent_angle = pos           # init cumulative from real position
        with signals_blocked(self._dial):
            self._dial.setValue(angle)
        self._refresh_pos_label()

    def _refresh_pos_label(self) -> None:
        try:
            pos = _get_pos(self._mmc, self._device)
        except Exception:
            return
        self._pos_label.setText(f"position: {pos:.3f} °")

    # ------------------------------------------------------------------
    # Dial → stage
    # ------------------------------------------------------------------

    def _on_dial_value_changed(self, angle: int) -> None:
        """Called on every tick while the mouse wheel is spinning.

        We store the requested angle and (re)start a debounce timer so that
        the actual move command is only sent once the user pauses.
        """
        self._pending_angle = angle
        self._debounce_timer.start()   # restart / extend the timeout
        print(f"[x.py] dial → {angle}°  (waiting for debounce)")

    def _send_debounced_move(self) -> None:
        if self._pending_angle is None:
            return
        dial_angle = self._pending_angle
        self._pending_angle = None

        # Compute shortest-path delta from last sent angle (handles 359→0 wrap)
        prev = self._last_sent_angle % 360
        delta = dial_angle - prev
        if delta > 180:
            delta -= 360
        elif delta < -180:
            delta += 360

        target = self._last_sent_angle + delta
        print(f"[x.py] setPosition({target:.2f})  (delta {delta:+.2f})")
        self._last_sent_angle = target
        self._move_to_angle(target)

    def _step_stage(self, delta: float) -> None:
        target = self._last_sent_angle + delta
        self._last_sent_angle = target
        print(f"[x.py] step {delta:+.2f}° → {target:.2f}°")
        self._move_to_angle(target)

    def _on_go_clicked(self) -> None:
        target = self._abs_spin.value()
        self._last_sent_angle = target   # explicit absolute go resets tracking
        print(f"[x.py] go to {target:.2f}°")
        self._move_to_angle(target)

    def _move_to_angle(self, angle: float) -> None:
        """Send setPosition and sync the dial, guarded against re-entry."""
        if self._moving:
            print("[x.py] _move_to_angle: stage already moving, skipping")
            return
        self._moving = True
        try:
            self._mmc.setPosition(self._device, angle)
        except Exception as exc:
            print(f"[x.py] setPosition failed: {exc}")
        finally:
            self._moving = False
        # Sync dial to the angle we *requested* (wrapped to 0-359 for the dial)
        with signals_blocked(self._dial):
            self._dial.setValue(int(round(angle)) % 360)

    # ------------------------------------------------------------------
    # Stage → UI
    # ------------------------------------------------------------------

    def _on_stage_moved(self, device: str, pos: float) -> None:
        if device != self._device:
            return
        print(f"[x.py] stagePositionChanged: {device} → {pos:.3f}")
        # Update dial WITHOUT triggering another move
        with signals_blocked(self._dial):
            self._dial.setValue(int(round(pos)) % 360)
        self._pos_label.setText(f"position: {pos:.3f} °")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    device_name = sys.argv[1] if len(sys.argv) > 1 else "KBD101_28252107"
    config_file = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\Admin\Desktop\mm\r.cfg"

    app = QApplication.instance() or QApplication(sys.argv)

    mmc = CMMCorePlus.instance()
    if config_file:
        print(f"[x.py] loading config: {config_file}")
        mmc.loadSystemConfiguration(config_file)

    # Check the device exists
    loaded = list(mmc.getLoadedDevicesOfType(DeviceType.Stage))
    print(f"[x.py] loaded Stage devices: {loaded}")
    if device_name not in loaded:
        print(
            f"[x.py] WARNING: '{device_name}' not in loaded Stage devices.\n"
            f"       Pass the correct name as argv[1], e.g.:\n"
            f"         python x.py MyRotationStage my_config.cfg"
        )

    w = KinesisDebugWidget(device=device_name, mmc=mmc)
    w.resize(300, 380)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
