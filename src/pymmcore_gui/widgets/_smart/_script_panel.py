"""Load a Smart Microscopy script and choose how it runs."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from superqt.iconify import QIconifyIcon

from pymmcore_gui._qt.QtCore import QEvent, Qt, Signal
from pymmcore_gui._qt.QtGui import QFont, QPalette
from pymmcore_gui._qt.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from pymmcore_gui._smart._controller import SmartRunConfig
from pymmcore_gui._smart._loader import AnalyzeFilter, ScriptError, inspect_script
from pymmcore_gui._theme import qcolor, theme
from pymmcore_gui.widgets._smart._params_form import ParamsForm

if TYPE_CHECKING:
    from pymmcore_gui._smart._controller import OnError
    from pymmcore_gui._smart._loader import ScriptSpec
    from pymmcore_gui.smart._api import ExecutionMode, Origin, SyncMode

EXECUTION_HELP: Final = {
    "thread": (
        "Runs in a background thread of this application: starts instantly and "
        "sees each frame without copying. A crash in the script's libraries "
        "takes the application down, and a hung analysis cannot be interrupted."
    ),
    "process": (
        "Runs in a separate Python process: takes a few seconds to start and "
        "each frame is copied to it, but a crash or hang cannot affect the "
        "application or the acquisition, and the process can be stopped."
    ),
}
SYNC_HELP: Final = {
    "blocking": (
        "Wait for each analysis before acquiring the next event, so every "
        "acquisition sees the latest decision."
    ),
    "async": (
        "Keep acquiring the base events while frames are analyzed; requested "
        "events are inserted as results arrive."
    ),
}


class ScriptPanel(QWidget):
    """Shows the loaded script and every setting that shapes the run."""

    scriptChanged = Signal(object)
    """The newly loaded `ScriptSpec`, or None when loading failed."""
    settingsChanged = Signal()
    """Any user-editable setting changed (for persistence)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._spec: ScriptSpec | None = None
        self._path: Path | None = None
        self._modified = False

        # ── header: name, status, path, description, error ──────────
        self._name = QLabel("No script loaded")
        font = QFont(self._name.font())
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 1.15)
        self._name.setFont(font)
        self._status_icon = QLabel()
        self._status_text = QLabel()
        status = QWidget()
        status_row = QHBoxLayout(status)
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(4)
        status_row.addWidget(self._status_icon)
        status_row.addWidget(self._status_text)
        header = QHBoxLayout()
        header.addWidget(self._name, 1)
        header.addWidget(status)

        self._path_label = QLabel()
        self._path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._path_label.setWordWrap(True)
        self._description = QLabel(
            "Load a Python script that defines analyze(image, frame, ctx), or "
            "start from a template."
        )
        self._description.setWordWrap(True)
        self._error = QLabel()
        self._error.setWordWrap(True)
        self._error.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._error.hide()

        # ── parameters ──────────────────────────────────────────────
        self.params = ParamsForm()
        self.params.valuesChanged.connect(self.settingsChanged)
        self._reset_params = QPushButton("Reset to defaults")
        self._reset_params.setProperty("variant", "subtle")
        self._reset_params.clicked.connect(self.params.reset)
        params_box = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_box)
        params_layout.addWidget(self.params)
        params_layout.addWidget(
            self._reset_params, alignment=Qt.AlignmentFlag.AlignRight
        )

        # ── execution ───────────────────────────────────────────────
        self._execution = _RadioChoice(
            {"thread": "Thread", "process": "Process"}, EXECUTION_HELP
        )
        self._sync = _RadioChoice({"blocking": "Blocking", "async": "Async"}, SYNC_HELP)
        self._on_error = QComboBox()
        self._on_error.addItem("Stop the run", "stop")
        self._on_error.addItem("Skip the frame and continue", "skip")
        self._on_error.setToolTip("What to do when analyze() raises an exception.")
        self._timeout = QDoubleSpinBox()
        self._timeout.setRange(0, 3600)
        self._timeout.setDecimals(1)
        self._timeout.setSuffix(" s")
        self._timeout.setSpecialValueText("None")
        self._timeout.setToolTip(
            "Blocking mode only: stop the run if one analysis takes longer."
        )
        self._max_events = QSpinBox()
        self._max_events.setRange(1, 10_000_000)
        self._max_events.setValue(10_000)
        self._max_events.setToolTip(
            "Safety limit: the run stops after this many events in total."
        )
        exec_box = QGroupBox("Execution")
        exec_form = QFormLayout(exec_box)
        exec_form.addRow("Run analysis in", self._execution)
        exec_form.addRow("Timing", self._sync)
        exec_form.addRow("On error", self._on_error)
        exec_form.addRow("Analysis timeout", self._timeout)
        exec_form.addRow("Max events", self._max_events)

        # ── which frames are analyzed ───────────────────────────────
        self._channels = QLineEdit()
        self._channels.setPlaceholderText("All channels")
        self._channels.setToolTip(
            "Comma-separated channel presets to analyze; empty analyzes all."
        )
        self._every_nth = QSpinBox()
        self._every_nth.setRange(1, 1_000_000)
        self._every_nth.setPrefix("every ")
        self._every_nth.setToolTip("Analyze every Nth frame (by frame number).")
        self._origin_base = QCheckBox("Base acquisition")
        self._origin_analysis = QCheckBox("Requested by analysis")
        origins = QWidget()
        origins_row = QHBoxLayout(origins)
        origins_row.setContentsMargins(0, 0, 0, 0)
        origins_row.addWidget(self._origin_base)
        origins_row.addWidget(self._origin_analysis)
        origins_row.addStretch()
        filter_box = QGroupBox("Frames to analyze")
        filter_form = QFormLayout(filter_box)
        filter_form.addRow("Channels", self._channels)
        filter_form.addRow("Frequency", self._every_nth)
        filter_form.addRow("From", origins)

        for signal in (
            self._execution.changed,
            self._sync.changed,
            self._on_error.currentIndexChanged,
            self._timeout.valueChanged,
            self._max_events.valueChanged,
            self._channels.textChanged,
            self._every_nth.valueChanged,
            self._origin_base.toggled,
            self._origin_analysis.toggled,
        ):
            signal.connect(self.settingsChanged)

        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self._path_label)
        layout.addWidget(self._description)
        layout.addWidget(self._error)
        layout.addWidget(params_box)
        layout.addWidget(exec_box)
        layout.addWidget(filter_box)
        layout.addStretch()

        self._settings_boxes = (params_box, exec_box, filter_box)
        self._apply_filter(AnalyzeFilter())
        self._update_status()

    # ------------------------------------------------------------ public API

    @property
    def spec(self) -> ScriptSpec | None:
        """The loaded, valid script (None if nothing valid is loaded)."""
        return self._spec

    @property
    def path(self) -> Path | None:
        """The last path loaded, valid or not."""
        return self._path

    def load(self, path: str | Path, saved: dict[str, Any] | None = None) -> bool:
        """Load the script at *path*; return whether it is valid.

        *saved* holds settings remembered for this script (see `settings`):
        they override the script's own defaults where still applicable.
        """
        self._path = Path(path).expanduser().resolve()
        self._modified = False
        try:
            spec = inspect_script(self._path)
        except ScriptError as e:
            self._spec = None
            self._show_error(e)
            self.scriptChanged.emit(None)
            return False
        self._spec = spec
        self._show_spec(spec, saved or {})
        self.scriptChanged.emit(spec)
        return True

    def reload(self) -> bool:
        """Re-read the current script, keeping the current settings."""
        if self._path is None:
            return False
        return self.load(self._path, self.settings())

    def set_modified(self, modified: bool) -> None:
        """Flag that the file changed on disk since it was loaded."""
        self._modified = modified
        self._update_status()

    def settings(self) -> dict[str, Any]:
        """Everything the user chose for this script, as plain JSON values."""
        return {
            "params": self.params.values() if self._spec else {},
            "execution": self._execution.value(),
            "sync": self._sync.value(),
            "on_error": self._on_error.currentData(),
            "analysis_timeout_s": self._timeout.value(),
            "max_total_events": self._max_events.value(),
            "filter": self._filter().to_dict(),
        }

    def run_config(self) -> SmartRunConfig:
        """The settings of the next run. Raises RuntimeError without a valid script."""
        spec = self._spec
        if spec is None:
            raise RuntimeError("No valid script is loaded.")
        timeout = self._timeout.value()
        return SmartRunConfig(
            spec=spec,
            params=spec.resolve_params(self.params.values()),
            execution=cast("ExecutionMode", self._execution.value()),
            sync=cast("SyncMode", self._sync.value()),
            filter=self._filter(),
            on_error=cast("OnError", self._on_error.currentData()),
            analysis_timeout_s=timeout or None,
            max_total_events=self._max_events.value(),
        )

    def set_locked(self, locked: bool) -> None:
        """Freeze the settings while a run uses them."""
        for box in self._settings_boxes:
            box.setEnabled(not locked)

    # --------------------------------------------------------------- display

    def _show_spec(self, spec: ScriptSpec, saved: dict[str, Any]) -> None:
        self._name.setText(spec.name)
        self._path_label.setText(str(spec.path))
        self._path_label.setToolTip(str(spec.path))
        self._description.setText(spec.description or "")
        self._description.setVisible(bool(spec.description))
        self._error.hide()

        self.params.set_params(spec.params, spec.resolve_params(saved.get("params")))
        self._reset_params.setVisible(bool(spec.params))
        self._execution.set_value(saved.get("execution", spec.execution))
        self._sync.set_value(saved.get("sync", spec.sync))
        self._on_error.setCurrentIndex(
            max(self._on_error.findData(saved.get("on_error", "stop")), 0)
        )
        self._timeout.setValue(float(saved.get("analysis_timeout_s") or 0))
        self._max_events.setValue(int(saved.get("max_total_events") or 10_000))
        self._apply_filter(_filter_from(saved.get("filter")) or spec.filter)
        for box in self._settings_boxes:
            box.setVisible(True)
        self._update_status()

    def _show_error(self, error: ScriptError) -> None:
        name = self._path.name if self._path else "Script"
        self._name.setText(name)
        self._path_label.setText(str(self._path or ""))
        self._description.hide()
        self._error.setText(str(error))
        self._error.show()
        for box in self._settings_boxes:
            box.setVisible(False)
        self._update_status()

    def _update_status(self) -> None:
        t = theme()
        if self._path is None:
            icon, color, text = None, t.text_secondary, ""
        elif self._spec is None:
            icon, color, text = "mdi:alert-circle", t.status_red, "Error"
        elif self._modified:
            icon, color, text = (
                "mdi:pencil-circle",
                t.status_amber,
                "Modified — changes apply at the next run",
            )
        else:
            icon, color, text = "mdi:check-circle", t.status_green, "Ready"
        qc = qcolor(color)
        size = self._status_text.fontMetrics().height()
        if icon is None:
            self._status_icon.clear()
        else:
            self._status_icon.setPixmap(
                QIconifyIcon(icon, color=qc.name()).pixmap(size, size)
            )
        self._status_text.setText(text)
        for label in (self._status_text, self._error):
            pal = label.palette()
            pal.setColor(
                QPalette.ColorRole.WindowText,
                qcolor(t.status_red) if label is self._error else qc,
            )
            label.setPalette(pal)

    def changeEvent(self, a0: QEvent | None) -> None:
        # Theme switches re-polish every widget: re-derive the status colors.
        if a0 is not None and a0.type() == QEvent.Type.StyleChange:
            self._update_status()
        super().changeEvent(a0)

    # ---------------------------------------------------------------- filter

    def _filter(self) -> AnalyzeFilter:
        channels = tuple(
            c.strip() for c in self._channels.text().split(",") if c.strip()
        )
        origins: set[Origin] = set()
        if self._origin_base.isChecked():
            origins.add("base")
        if self._origin_analysis.isChecked():
            origins.add("analysis")
        return AnalyzeFilter(
            channels=channels or None,
            every_nth=self._every_nth.value(),
            origins=frozenset(origins),
        )

    def _apply_filter(self, value: AnalyzeFilter) -> None:
        self._channels.setText(", ".join(value.channels or ()))
        self._every_nth.setValue(value.every_nth)
        self._origin_base.setChecked("base" in value.origins)
        self._origin_analysis.setChecked("analysis" in value.origins)


def _filter_from(data: object) -> AnalyzeFilter | None:
    """An `AnalyzeFilter` from its saved `to_dict` form, or None if unusable."""
    if not isinstance(data, dict):
        return None
    try:
        channels = data.get("channels")
        origins = frozenset(data.get("origins") or ("base", "analysis"))
        if not origins <= {"base", "analysis"}:
            return None
        return AnalyzeFilter(
            channels=tuple(channels) if channels else None,
            every_nth=max(1, int(data.get("every_nth", 1))),
            origins=cast("frozenset[Origin]", origins),
        )
    except (TypeError, ValueError):
        return None


class _RadioChoice(QWidget):
    """A row of radio buttons with per-choice help shown as tooltips."""

    changed = Signal(str)

    def __init__(
        self, labels: dict[str, str], help_text: dict[str, str] | None = None
    ) -> None:
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self._group = QButtonGroup(self)
        self._buttons: dict[str, QRadioButton] = {}
        for key, label in labels.items():
            button = QRadioButton(label)
            button.setObjectName(f"choice_{key}")
            if help_text and key in help_text:
                button.setToolTip(help_text[key])
            self._group.addButton(button)
            self._buttons[key] = button
            row.addWidget(button)
            button.toggled.connect(partial(self._on_toggled, key))
        row.addStretch()
        next(iter(self._buttons.values())).setChecked(True)

    def _on_toggled(self, key: str, checked: bool) -> None:
        if checked:
            self.changed.emit(key)

    def value(self) -> str:
        for key, button in self._buttons.items():
            if button.isChecked():
                return key
        return next(iter(self._buttons))

    def set_value(self, key: str) -> None:
        if (button := self._buttons.get(key)) is not None:
            button.setChecked(True)

    def button(self, key: str) -> QRadioButton:
        return self._buttons[key]
