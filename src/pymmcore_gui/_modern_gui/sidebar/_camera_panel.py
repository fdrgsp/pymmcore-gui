"""Camera settings panel for the sidebar."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_plus import DeviceType
from pymmcore_widgets import CameraRoiWidget, PropertyWidget
from superqt.utils import signals_blocked

from pymmcore_gui._modern_gui._theme import mono_font, qcolor, theme, ui_font
from pymmcore_gui._modern_gui._utils import current_core
from pymmcore_gui._qt.QtCore import QSize, Qt, Signal
from pymmcore_gui._qt.QtGui import QFont
from pymmcore_gui._qt.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QWidget,
)

from ._collapsible_panel import CollapsiblePanel

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus


class CameraSettingsWidget(QWidget):
    """Camera settings form: exposure, gain, binning, ROI."""

    exposureChanged = Signal(float)
    binningChanged = Signal(int)
    cameraChangeRequested = Signal()

    _GAIN_ROW = 2  # form row index where gain widget goes

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._form = QFormLayout(self)
        self._form.setContentsMargins(0, 0, 0, 0)
        self._form.setHorizontalSpacing(8)
        self._form.setVerticalSpacing(6)

        # Device info row
        device_row = QHBoxLayout()
        device_row.setContentsMargins(0, 0, 0, 0)
        device_row.setSpacing(4)
        self._device_label = QLabel("\u2014")
        self._device_label.setFont(mono_font(8))
        device_row.addWidget(self._device_label, 1)
        self._change_btn = QPushButton("\u2026")
        self._change_btn.setToolTip("Change camera device")
        self._change_btn.setFixedSize(theme().scaled(22), theme().scaled(22))
        self._change_btn.setProperty("variant", "ghost")
        self._change_btn.clicked.connect(self.cameraChangeRequested)
        device_row.addWidget(self._change_btn)
        self._form.addRow(device_row)

        # Exposure
        self._exposure = QDoubleSpinBox()
        self._exposure.setRange(0.001, 999999.0)
        self._exposure.setSuffix(" ms")
        self._exposure.setDecimals(1)
        self._exposure.setKeyboardTracking(False)
        self._exposure.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._exposure.valueChanged.connect(self.exposureChanged)
        self._form.addRow("Exposure", self._exposure)

        # Gain — placeholder label, replaced by PropertyWidget when core connects
        self._gain_placeholder = QLabel("\u2014")
        self._form.addRow("Gain", self._gain_placeholder)

        # Binning
        self._binning = QComboBox()
        self._binning.currentIndexChanged.connect(
            lambda: self.binningChanged.emit(self.binning())
        )
        self._form.addRow("Binning", self._binning)

        # Readout speed
        self._readout = QComboBox()
        self._readout.addItems(["Med (95MHz)", "Fast (200MHz)"])
        self._form.addRow("Readout", self._readout)

        # ROI
        roi_row = QHBoxLayout()
        roi_row.setContentsMargins(0, 0, 0, 0)
        roi_row.setSpacing(4)
        self._roi_label = QLabel("\u2014")
        self._roi_label.setFont(mono_font(9))
        roi_row.addWidget(self._roi_label, 1)
        from superqt.iconify import QIconifyIcon

        self._roi_btn = QPushButton()
        t = theme()
        icon_sz = t.scaled(14)
        self._roi_btn.setIcon(
            QIconifyIcon("mdi:crop", color=qcolor(t.text_secondary).name())
        )
        self._roi_btn.setIconSize(QSize(icon_sz, icon_sz))
        self._roi_btn.setToolTip("Set camera ROI")
        self._roi_btn.setFixedSize(t.scaled(22), t.scaled(22))
        self._roi_btn.setProperty("variant", "ghost")
        roi_row.addWidget(self._roi_btn)
        self._form.addRow("ROI", roi_row)

        self._apply_fonts()

    def _apply_fonts(self) -> None:
        label_font = ui_font(9, QFont.Weight.Medium)
        for i in range(self._form.rowCount()):
            item = self._form.itemAt(i, QFormLayout.ItemRole.LabelRole)
            if item and (w := item.widget()):
                w.setFont(label_font)

    def set_gain_widget(self, widget: QWidget) -> None:
        """Replace the gain placeholder with a real widget."""
        self._form.removeRow(self._GAIN_ROW)
        self._form.insertRow(self._GAIN_ROW, "Gain", widget)
        self._gain_placeholder = widget
        self._apply_fonts()

    def set_device_info(self, label: str, library: str, name: str) -> None:
        self._device_label.setText(f"{label} \u2013 {library} \u2013 {name}")

    def set_exposure(self, value: float) -> None:
        with signals_blocked(self._exposure):
            self._exposure.setValue(value)

    def set_binning_options(self, values: list[int], current: int) -> None:
        with signals_blocked(self._binning):
            self._binning.clear()
            for v in values:
                self._binning.addItem(f"{v}x{v}", v)
            idx = self._binning.findData(current)
            if idx >= 0:
                self._binning.setCurrentIndex(idx)

    def binning(self) -> int:
        return self._binning.currentData() or 1

    def set_roi_text(self, text: str) -> None:
        self._roi_label.setText(text)


def CollapsibleCameraPanel(parent: QWidget | None = None) -> CollapsiblePanel:
    """Create a Camera panel wrapped in a collapsible header."""
    panel = CollapsiblePanel(
        title="Camera",
        summary="",
        parent=parent,
    )

    content = CameraSettingsWidget(parent=panel)
    panel.body_layout.addWidget(content)

    core = current_core(parent)
    if core is not None:
        _connect_to_core(content, panel, core)

    return panel


def _show_camera_picker(widget: CameraSettingsWidget, core: CMMCorePlus) -> None:
    """Show a dialog to pick a different camera device."""
    cameras = [
        d
        for d in core.getLoadedDevicesOfType(DeviceType.CameraDevice)
        if d != core.getCameraDevice()
    ]
    if not cameras:
        return

    current = core.getCameraDevice()
    all_cameras = [current, *cameras] if current else cameras
    choice, ok = QInputDialog.getItem(
        widget, "Select Camera", "Camera device:", all_cameras, 0, False
    )
    if ok and choice and choice != current:
        core.setCameraDevice(choice)


def _shrink_labeled_slider_spinbox(
    prop_widget: QWidget, core: CMMCorePlus, device: str, prop: str
) -> None:
    """Resize the spinbox inside a PropertyWidget's LabeledSlider to fit the range."""
    inner = getattr(prop_widget, "inner_widget", None)
    spinbox = getattr(inner, "_spinbox", None) if inner else None
    if spinbox is None:
        return
    upper = core.getPropertyUpperLimit(device, prop)
    sample = f"{upper:.0f}"
    fm = spinbox.fontMetrics()
    # pad for cursor / margins
    spinbox.setFixedWidth(fm.horizontalAdvance(sample) + 16)


def _connect_to_core(
    widget: CameraSettingsWidget, panel: CollapsiblePanel, core: CMMCorePlus
) -> None:
    """Wire widget signals to core and core events back to widget."""

    def _update_summary() -> None:
        exp = core.getExposure()
        binn = core.getBinning()
        panel.header.summary = f"{exp:.0f} ms \u00b7 {binn}x{binn}"

    def _load_state() -> None:
        cam = core.getCameraDevice()
        if not cam:
            return

        widget.set_device_info(
            cam, core.getDeviceLibrary(cam), core.getDeviceName(cam)
        )
        widget.set_exposure(core.getExposure())

        allowed = core.getAllowedBinningValues()
        current = core.getBinning()
        if allowed:
            widget.set_binning_options(list(allowed), current)

        w, h = core.getImageWidth(), core.getImageHeight()
        depth = core.getBytesPerPixel() * 8
        widget.set_roi_text(f"{w} \u00d7 {h} \u00b7 {depth}-bit")

        _update_summary()

    # Device info
    cam = core.getCameraDevice()
    if cam:
        widget.set_device_info(
            cam, core.getDeviceLibrary(cam), core.getDeviceName(cam)
        )

    # Camera change button
    widget.cameraChangeRequested.connect(
        lambda: _show_camera_picker(widget, core)
    )

    # ROI button — opens CameraRoiWidget in a floating window
    def _open_roi_widget() -> None:
        roi_wdg = CameraRoiWidget(mmcore=core, parent=widget)
        roi_wdg.setWindowFlags(Qt.WindowType.Tool)
        roi_wdg.setWindowTitle("Camera ROI")
        roi_wdg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        roi_wdg.show()

    widget._roi_btn.clicked.connect(_open_roi_widget)

    # Gain — use PropertyWidget if camera has a Gain property
    if cam and core.hasProperty(cam, "Gain"):
        gain_widget = PropertyWidget(cam, "Gain", parent=widget, mmcore=core)
        _shrink_labeled_slider_spinbox(gain_widget, core, cam, "Gain")
        widget.set_gain_widget(gain_widget)

    # Widget -> Core
    widget.exposureChanged.connect(core.setExposure)
    widget.binningChanged.connect(core.setBinning)

    # Core -> Widget
    core.events.exposureChanged.connect(
        lambda _cam, exp: widget.set_exposure(exp)
    )
    core.events.exposureChanged.connect(lambda *_: _update_summary())
    core.events.roiSet.connect(lambda *_: _load_state())
    core.events.systemConfigurationLoaded.connect(_load_state)

    _load_state()
