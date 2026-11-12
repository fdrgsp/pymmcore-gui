"""Dialog choosing a color camera's white-balance gains and balancing them."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext, suppress
from typing import TYPE_CHECKING

from pymmcore_plus import DeviceType
from superqt.utils import create_worker, signals_blocked

from pymmcore_gui._qt.QtCore import Qt, Signal
from pymmcore_gui._qt.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from pymmcore_gui._settings import Settings
from pymmcore_gui._white_balance import (
    Channel,
    GainProperty,
    WhiteBalanceResult,
    candidate_properties,
    guess_gain_properties,
    one_push_white_balance,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pymmcore_plus import CMMCorePlus
    from superqt.utils import FunctionWorker


def _property_label(prop: GainProperty) -> str:
    if prop.has_limits:
        return f"{prop.name}  (range {prop.lower:g} to {prop.upper:g})"
    return f"{prop.name}  (no limits)"


def describe_result(result: WhiteBalanceResult) -> str:
    values = ", ".join(f"{name} = {value:g}" for name, value in result.values.items())
    ratios = ", ".join(f"{c}/green {r:.2f}" for c, r in result.ratios.items())
    if result.converged:
        return f"Balanced: {values}\n({ratios})"
    limited = (
        f"\n{', '.join(result.limited)} reached its limit." if result.limited else ""
    )
    return (
        f"Could not fully balance the image ({ratios}).{limited}"
        f"\nThe closest values were kept: {values}"
    )


class WhiteBalanceDialog(QDialog):
    """Choose per-camera gains and balance them in a worker thread.

    `snap_context` lets the preview exclude calibration frames from its album.
    It is created, entered and exited on the worker thread; it must not touch widgets.
    """

    balanced = Signal(object)
    balanceFailed = Signal(str)

    def __init__(
        self,
        mmcore: CMMCorePlus,
        parent: QWidget | None = None,
        *,
        snap_context: Callable[[], AbstractContextManager[object]] = nullcontext,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("White Balance")
        self._mmc = mmcore
        self._snap_context = snap_context
        self._props: dict[str, GainProperty] = {}
        self._worker: FunctionWorker[WhiteBalanceResult] | None = None

        hint = QLabel(
            "Point the camera at an empty, evenly lit background, then click "
            "Balance: the red and blue gains are adjusted until it is gray."
        )
        hint.setWordWrap(True)

        self.camera_combo = QComboBox()
        self.red_combo = QComboBox()
        self.blue_combo = QComboBox()
        self._combos: dict[Channel, QComboBox] = {
            "red": self.red_combo,
            "blue": self.blue_combo,
        }
        form = QFormLayout()
        form.addRow("Camera:", self.camera_combo)
        form.addRow("Red gain:", self.red_combo)
        form.addRow("Blue gain:", self.blue_combo)

        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.balance_btn = QPushButton("Balance")
        self.balance_btn.setDefault(True)
        close_btn = QPushButton("Close")
        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self.balance_btn)
        buttons.addWidget(close_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(hint)
        layout.addLayout(form)
        layout.addWidget(self.result_label)
        layout.addLayout(buttons)

        self.camera_combo.currentTextChanged.connect(self._populate_properties)
        for combo in self._combos.values():
            combo.currentIndexChanged.connect(self._update_balance_enabled)
        self.balance_btn.clicked.connect(self.balance)
        close_btn.clicked.connect(self.close)
        mmcore.events.systemConfigurationLoaded.connect(self._populate_cameras)
        self.destroyed.connect(self._disconnect)
        self._populate_cameras()

    def _disconnect(self) -> None:
        with suppress(TypeError, RuntimeError):
            self._mmc.events.systemConfigurationLoaded.disconnect(
                self._populate_cameras
            )

    def _populate_cameras(self) -> None:
        cameras = list(self._mmc.getLoadedDevicesOfType(DeviceType.Camera))
        with signals_blocked(self.camera_combo):
            self.camera_combo.clear()
            self.camera_combo.addItems(cameras)
            self.camera_combo.setCurrentText(self._mmc.getCameraDevice())
        self._populate_properties()

    def _populate_properties(self) -> None:
        camera = self.camera_combo.currentText()
        props = candidate_properties(self._mmc, camera) if camera else []
        self._props = {p.name: p for p in props}
        saved = Settings.instance().white_balance_properties(camera)
        guess = guess_gain_properties(props)
        for channel, combo in self._combos.items():
            with signals_blocked(combo):
                combo.clear()
                combo.addItem("(none)", None)
                for prop in props:
                    combo.addItem(_property_label(prop), prop.name)
                name = saved.get(channel) if saved is not None else None
                if saved is None or (name is not None and name not in self._props):
                    name = g.name if (g := guess.get(channel)) else None
                combo.setCurrentIndex(max(combo.findData(name), 0) if name else 0)
        self.result_label.clear()
        self._update_balance_enabled()

    def _update_balance_enabled(self) -> None:
        running = self._worker is not None
        gains = self.selected_gains()
        distinct = len({p.name for p in gains.values()}) == len(gains)
        self.balance_btn.setEnabled(not running and bool(gains) and distinct)
        for combo in (self.camera_combo, *self._combos.values()):
            combo.setEnabled(not running)

    def selected_gains(self) -> dict[Channel, GainProperty]:
        """The properties chosen for red and blue."""
        gains: dict[Channel, GainProperty] = {}
        for channel, combo in self._combos.items():
            name = combo.currentData()
            if name is not None and (prop := self._props.get(name)) is not None:
                gains[channel] = prop
        return gains

    def balance(self) -> bool:
        """Start balancing; emit `balanced` or `balanceFailed` with the outcome."""
        core = self._mmc
        camera = self.camera_combo.currentText()
        if self._worker is not None or not (gains := self.selected_gains()):
            return False
        if len({p.name for p in gains.values()}) != len(gains):
            self.result_label.setText("Choose different properties for red and blue.")
            return False
        if core.mda.is_running():
            self.result_label.setText("White balance is not available during an MDA.")
            return False
        Settings.instance().remember_white_balance_properties(
            camera, {c: p.name for c, p in gains.items()}
        )
        was_live = core.isSequenceRunning()

        def _run() -> WhiteBalanceResult:
            with self._snap_context():
                return one_push_white_balance(core, gains, camera=camera)

        def _finished(*_: object) -> None:
            self._worker = None
            self._update_balance_enabled()
            if was_live and not core.mda.is_running():
                core.startContinuousSequenceAcquisition()

        worker = create_worker(
            _run,
            _start_thread=False,
            _connect={
                "returned": self._on_balanced,
                "errored": self._on_failed,
                "finished": _finished,
            },
        )
        if was_live:
            core.stopSequenceAcquisition()
        self._worker = worker
        self._update_balance_enabled()
        self.result_label.setText("Balancing…")
        worker.start()
        return True

    def _on_balanced(self, result: WhiteBalanceResult) -> None:
        self.result_label.setText(describe_result(result))
        self.balanced.emit(result)

    def _on_failed(self, error: Exception) -> None:
        message = f"White balance failed: {error}"
        self.result_label.setText(message)
        self.balanceFailed.emit(message)
