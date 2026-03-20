"""Master widget script for testing stylesheets.

Assembles all key visual sub-components from pymmcore-widgets into a single
window and grabs it to a PNG.

Usage:
    uv run python scripts/master_widget.py [output.png]
"""

from __future__ import annotations

import sys

import useq
from pymmcore_plus import DeviceType, PropertyType
from pymmcore_widgets._icons import StandardIcon
from pymmcore_widgets._models import (
    ConfigGroup,
    ConfigPreset,
    Device,
    DevicePropertySetting,
)
from pymmcore_widgets.config_presets._views._config_groups_editor import (
    ConfigGroupsEditor,
)
from pymmcore_widgets.config_presets._views._device_property_selector import (
    _DeviceButtonToolbar,
)
from pymmcore_widgets.control._stage_widget import StageMovementButtons
from pymmcore_widgets.device_properties._property_widget import (
    FloatSpinBox,
    IntSpinBox,
    LabeledSlider,
)
from pymmcore_widgets.mda._save_widget import SaveGroupBox
from pymmcore_widgets.useq_widgets import (
    ChannelTable,
    GridPlanWidget,
    PositionTable,
    TimePlanWidget,
    ZPlanWidget,
)
from pymmcore_widgets.useq_widgets._well_plate_widget import WellPlateWidget
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pymmcore_gui._modern_gui._theme._dark import DARK_THEME

TIGHT = QSizePolicy.Policy.Maximum
EXPAND = QSizePolicy.Policy.Preferred


def _section(title: str, widget: QWidget, *, tight: bool = False) -> QGroupBox:
    """Wrap a widget in a labeled QGroupBox."""
    box = QGroupBox(title)
    layout = QVBoxLayout(box)
    layout.setContentsMargins(4, 4, 4, 4)
    layout.addWidget(widget)
    if tight:
        box.setSizePolicy(QSizePolicy(EXPAND, TIGHT))
    return box


def _build_channel_table() -> ChannelTable:
    tbl = ChannelTable(rows=0)
    tbl.setChannelGroups({"Channel": ["DAPI", "FITC", "Cy5"]})
    tbl.setValue(
        [
            useq.Channel(config="DAPI", exposure=100, do_stack=True),
            useq.Channel(config="FITC", exposure=50, acquire_every=2, do_stack=False),
            useq.Channel(config="Cy5", exposure=200, z_offset=1.5),
        ]
    )
    tbl.setMaximumHeight(130)
    return tbl


def _build_time_plan() -> TimePlanWidget:
    tbl = TimePlanWidget(rows=0)
    tbl.setValue(useq.TIntervalLoops(interval=2, loops=10))
    tbl.setMaximumHeight(90)
    return tbl


def _build_position_table() -> PositionTable:
    tbl = PositionTable(rows=0)
    tbl.setValue(
        [
            useq.Position(name="Pos0", x=100.0, y=200.0, z=50.0),
            useq.Position(name="Pos1", x=-50.0, y=300.0, z=55.0),
        ]
    )
    return tbl


def _build_z_plan() -> ZPlanWidget:
    w = ZPlanWidget()
    w.setValue(useq.ZTopBottom(top=10, bottom=-10, step=1))
    return w


def _build_grid_plan() -> GridPlanWidget:
    w = GridPlanWidget()
    w.setValue(useq.GridRowsColumns(rows=3, columns=4, overlap=(10, 10)))
    return w


def _build_save_group() -> SaveGroupBox:
    w = SaveGroupBox("Save Acquisition")
    w.setChecked(True)
    return w


def _build_stage_buttons() -> StageMovementButtons:
    w = StageMovementButtons(levels=2, show_x=True)
    w.setSizePolicy(QSizePolicy(TIGHT, TIGHT))
    return w


def _build_well_plate() -> WellPlateWidget:
    w = WellPlateWidget()
    w.setValue(useq.WellPlate.from_str("96-well"))
    w.setFixedSize(320, 180)
    return w


def _build_property_widgets() -> QWidget:
    """Build a column of all PropertyWidget variant types."""
    container = QWidget()
    layout = QGridLayout(container)
    layout.setContentsMargins(2, 2, 2, 2)
    layout.setVerticalSpacing(2)

    row = 0
    for label, widget in [
        ("IntSpinBox:", _make_int_spin()),
        ("FloatSpinBox:", _make_float_spin()),
        ("Slider (int):", _make_slider(False)),
        ("Slider (float):", _make_slider(True)),
        ("ComboBox:", _make_combo()),
        ("CheckBox:", _make_checkbox()),
        ("LineEdit:", QLineEdit("sample text")),
        ("ReadOnly:", _make_readonly_label()),
    ]:
        layout.addWidget(QLabel(label), row, 0)
        layout.addWidget(widget, row, 1)
        row += 1

    return container


def _make_int_spin() -> IntSpinBox:
    w = IntSpinBox()
    w.setRange(0, 1000)
    w.setValue(42)
    return w


def _make_float_spin() -> FloatSpinBox:
    w = FloatSpinBox()
    w.setRange(0, 100)
    w.setValue(3.14)
    return w


def _make_slider(is_float: bool) -> LabeledSlider:
    w = LabeledSlider(is_float=is_float)
    if is_float:
        w.setRange(0.0, 1.0)
        w.setValue(0.5)
    else:
        w.setRange(0, 255)
        w.setValue(128)
    return w


def _make_combo() -> QComboBox:
    w = QComboBox()
    w.addItems(["Option A", "Option B", "Option C"])
    return w


def _make_checkbox() -> QCheckBox:
    w = QCheckBox("Enabled")
    w.setChecked(True)
    return w


def _make_readonly_label() -> QLabel:
    w = QLabel("Read-only value")
    w.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
    return w


def _build_config_editor() -> ConfigGroupsEditor:
    """Build a ConfigGroupsEditor with sample data (no core needed)."""
    editor = ConfigGroupsEditor()

    camera = Device(
        label="Camera", name="DCam", library="DemoCamera", type=DeviceType.Camera
    )
    objective = Device(
        label="Objective",
        name="DObjective",
        library="DemoCamera",
        type=DeviceType.State,
    )

    channel_group = ConfigGroup(
        name="Channel",
        is_channel_group=True,
        presets={
            "DAPI": ConfigPreset(
                name="DAPI",
                settings=[
                    DevicePropertySetting(
                        device=camera,
                        property_name="Exposure",
                        value="100",
                        property_type=PropertyType.Float,
                        limits=(1.0, 10000.0),
                    ),
                ],
            ),
            "FITC": ConfigPreset(
                name="FITC",
                settings=[
                    DevicePropertySetting(
                        device=camera,
                        property_name="Exposure",
                        value="50",
                        property_type=PropertyType.Float,
                        limits=(1.0, 10000.0),
                    ),
                ],
            ),
        },
    )
    objective_group = ConfigGroup(
        name="Objective",
        presets={
            "10X": ConfigPreset(
                name="10X",
                settings=[
                    DevicePropertySetting(
                        device=objective,
                        property_name="Label",
                        value="10X",
                        allowed_values=("4X", "10X", "20X", "40X"),
                    ),
                ],
            ),
            "40X": ConfigPreset(
                name="40X",
                settings=[
                    DevicePropertySetting(
                        device=objective,
                        property_name="Label",
                        value="40X",
                        allowed_values=("4X", "10X", "20X", "40X"),
                    ),
                ],
            ),
        },
    )

    editor.setData([channel_group, objective_group])
    return editor


def _build_misc_qt_widgets() -> QWidget:
    """Standard Qt widgets not covered by custom classes."""
    from qtpy.QtWidgets import (
        QDial,
        QRadioButton,
        QTabWidget,
        QTextEdit,
        QToolButton,
    )

    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(2, 2, 2, 2)
    layout.setSpacing(4)

    # Buttons row
    btn_row = QHBoxLayout()
    for text, enabled in [("Run", True), ("Pause", False), ("Cancel", True)]:
        btn = QPushButton(text)
        btn.setEnabled(enabled)
        btn_row.addWidget(btn)
    tb = QToolButton()
    tb.setText("Tool")
    tb.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
    btn_row.addWidget(tb)
    layout.addLayout(btn_row)

    # ProgressBar
    pb = QProgressBar()
    pb.setValue(65)
    layout.addWidget(pb)

    # Slider + Dial
    slider_row = QHBoxLayout()
    sl = QSlider(Qt.Orientation.Horizontal)
    sl.setRange(0, 100)
    sl.setValue(50)
    slider_row.addWidget(sl, 1)
    dial = QDial()
    dial.setRange(0, 100)
    dial.setValue(35)
    dial.setFixedSize(50, 50)
    slider_row.addWidget(dial)
    layout.addLayout(slider_row)

    # Spin boxes
    spin_row = QHBoxLayout()
    for label, cls, val in [
        ("QSpinBox:", QSpinBox, 10),
        ("QDblSpin:", QDoubleSpinBox, 3.14),
    ]:
        spin_row.addWidget(QLabel(label))
        sb = cls()
        sb.setRange(0, 100)
        sb.setValue(val)
        spin_row.addWidget(sb)
    layout.addLayout(spin_row)

    # ComboBox + CheckBoxes
    combo_row = QHBoxLayout()
    combo = QComboBox()
    combo.addItems(["Item 1", "Item 2", "Item 3"])
    combo_row.addWidget(QLabel("Combo:"))
    combo_row.addWidget(combo)
    chk_on = QCheckBox("On")
    chk_on.setChecked(True)
    combo_row.addWidget(chk_on)
    chk_off = QCheckBox("Off")
    combo_row.addWidget(chk_off)
    layout.addLayout(combo_row)

    # Radio buttons + LineEdit
    radio_row = QHBoxLayout()
    for text, checked in [("Radio A", True), ("Radio B", False)]:
        rb = QRadioButton(text)
        rb.setChecked(checked)
        radio_row.addWidget(rb)
    le = QLineEdit("editable text")
    radio_row.addWidget(le)
    layout.addLayout(radio_row)

    # Mini tab widget
    tabs = QTabWidget()
    tabs.setMaximumHeight(60)
    tab1 = QLabel("Tab 1 content")
    tab1.setAlignment(Qt.AlignmentFlag.AlignCenter)
    tab2 = QLabel("Tab 2 content")
    tab2.setAlignment(Qt.AlignmentFlag.AlignCenter)
    tabs.addTab(tab1, "Tab A")
    tabs.addTab(tab2, "Tab B")
    layout.addWidget(tabs)

    # TextEdit
    te = QTextEdit()
    te.setPlainText("Multi-line\ntext area")
    te.setMaximumHeight(40)
    layout.addWidget(te)

    return container


def _build_device_toolbar() -> _DeviceButtonToolbar:
    return _DeviceButtonToolbar()


def _build_property_table_mock() -> QWidget:
    """Build a mock DevicePropertyTable showing device/property type icons."""
    from qtpy.QtWidgets import (
        QAbstractScrollArea,
        QHeaderView,
        QTableWidget,
        QTableWidgetItem,
    )

    # Sample properties: (device_label, prop_name, device_type, prop_type, value_widget)
    rows = [
        ("Camera", "Exposure", DeviceType.Camera, PropertyType.Float, ("float", 100.0)),
        (
            "Camera",
            "Binning",
            DeviceType.Camera,
            PropertyType.Integer,
            ("choice", ["1", "2", "4"]),
        ),
        (
            "Camera",
            "PixelType",
            DeviceType.Camera,
            PropertyType.String,
            ("choice", ["8bit", "16bit", "32bit"]),
        ),
        (
            "Camera",
            "TransposeXY",
            DeviceType.Camera,
            PropertyType.Integer,
            ("bool", True),
        ),
        (
            "Objective",
            "Label",
            DeviceType.State,
            PropertyType.String,
            ("choice", ["4X", "10X", "20X", "40X"]),
        ),
        ("Objective", "State", DeviceType.State, PropertyType.Integer, ("int", 1)),
        ("XYStage", "SpeedX", DeviceType.XYStage, PropertyType.Float, ("float", 10.0)),
        ("ZStage", "Position", DeviceType.Stage, PropertyType.Float, ("float", 50.0)),
        ("Shutter", "State", DeviceType.Shutter, PropertyType.Integer, ("bool", False)),
        (
            "AutoFocus",
            "Offset",
            DeviceType.AutoFocus,
            PropertyType.Float,
            ("float", 0.5),
        ),
        (
            "FilterWheel",
            "State",
            DeviceType.State,
            PropertyType.Enum,
            ("choice", ["Empty", "Red", "Green", "Blue"]),
        ),
        (
            "Core",
            "Initialize",
            DeviceType.Core,
            PropertyType.Integer,
            ("readonly", "1"),
        ),
        ("Hub", "Port", DeviceType.Hub, PropertyType.String, ("readonly", "COM3")),
    ]

    table = QTableWidget(len(rows), 3)
    table.setHorizontalHeaderLabels(["Device-Property", "Type", "Value"])
    table.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
    table.horizontalHeader().setStretchLastSection(True)
    table.horizontalHeader().setSectionResizeMode(
        0, QHeaderView.ResizeMode.ResizeToContents
    )
    table.horizontalHeader().setSectionResizeMode(
        1, QHeaderView.ResizeMode.ResizeToContents
    )
    vh = table.verticalHeader()
    vh.setSectionResizeMode(vh.ResizeMode.Fixed)
    vh.setDefaultSectionSize(24)
    vh.setVisible(False)
    table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

    for i, (dev, prop, dev_type, prop_type, val_info) in enumerate(rows):
        # Device-Property column with device icon
        item = QTableWidgetItem(f"{dev}-{prop}")
        dev_icon = StandardIcon.for_device_type(dev_type)
        item.setIcon(dev_icon.icon())
        table.setItem(i, 0, item)

        # Property type column with type icon
        allowed = val_info[1] if val_info[0] == "choice" else ()
        prop_icon = StandardIcon.for_property_type(prop_type, allowed)
        type_item = QTableWidgetItem(prop_type.name)
        type_item.setIcon(prop_icon.icon())
        table.setItem(i, 1, type_item)

        # Value column with appropriate widget
        kind = val_info[0]
        if kind == "float":
            w = FloatSpinBox()
            w.setRange(-1e6, 1e6)
            w.setValue(val_info[1])
            table.setCellWidget(i, 2, w)
        elif kind == "int":
            w = IntSpinBox()
            w.setRange(0, 100)
            w.setValue(val_info[1])
            table.setCellWidget(i, 2, w)
        elif kind == "choice":
            w = QComboBox()
            w.addItems(val_info[1])
            table.setCellWidget(i, 2, w)
        elif kind == "bool":
            w = QCheckBox()
            w.setChecked(val_info[1])
            table.setCellWidget(i, 2, w)
        elif kind == "readonly":
            lbl = QLabel(str(val_info[1]))
            lbl.setStyleSheet("QLabel { background: #AAA; padding: 2px; }")
            table.setCellWidget(i, 2, lbl)

    return table


def build_master_widget() -> QWidget:
    """Assemble all component sections into a master widget."""
    master = QWidget()
    outer = QVBoxLayout(master)
    outer.setSpacing(4)
    outer.setContentsMargins(6, 6, 6, 6)

    title = QLabel("pymmcore-widgets — Style Test")
    title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    font = title.font()
    font.setPointSize(14)
    font.setBold(True)
    title.setFont(font)
    title.setSizePolicy(QSizePolicy(EXPAND, TIGHT))
    outer.addWidget(title)

    # --- Row 1: tables ---
    r1 = QHBoxLayout()
    r1.setSpacing(4)
    r1.addWidget(_section("ChannelTable", _build_channel_table()), 3)
    r1.addWidget(_section("TimePlanWidget", _build_time_plan()), 2)
    r1.addWidget(_section("SaveGroupBox", _build_save_group(), tight=True), 2)
    outer.addLayout(r1)

    # --- Row 2: plans + positions ---
    r2 = QHBoxLayout()
    r2.setSpacing(4)
    r2.addWidget(_section("ZPlanWidget", _build_z_plan(), tight=True), 2)
    r2.addWidget(_section("GridPlanWidget", _build_grid_plan(), tight=True), 2)
    r2.addWidget(_section("PositionTable", _build_position_table()), 3)
    outer.addLayout(r2)

    # --- Row 3: stage, well plate, property variants, misc ---
    r3 = QHBoxLayout()
    r3.setSpacing(4)
    r3.addWidget(_section("StageMovementButtons", _build_stage_buttons(), tight=True))
    r3.addWidget(_section("WellPlateWidget", _build_well_plate(), tight=True))
    r3.addWidget(_section("Property Widgets", _build_property_widgets()), 2)
    r3.addWidget(_section("Misc Qt Primitives", _build_misc_qt_widgets(), tight=True))
    outer.addLayout(r3)

    # --- Row 4: property table + device toolbar + config editor ---
    r4 = QHBoxLayout()
    r4.setSpacing(4)
    left_col = QVBoxLayout()
    left_col.setSpacing(4)
    left_col.addWidget(
        _section("DeviceButtonToolbar", _build_device_toolbar(), tight=True)
    )
    left_col.addWidget(
        _section("DevicePropertyTable (mock)", _build_property_table_mock())
    )
    r4.addLayout(left_col, 2)
    r4.addWidget(_section("ConfigGroupsEditor", _build_config_editor()), 3)
    outer.addLayout(r4)

    return master


def main() -> None:
    """Execute the master widget and save to PNG."""
    from pymmcore_gui._modern_gui._theme import set_theme

    app = QApplication.instance() or QApplication(sys.argv)
    set_theme(DARK_THEME)
    output_path = sys.argv[1] if len(sys.argv) > 1 else "master_widget.png"

    widget = build_master_widget()
    widget.show()
    app.processEvents()

    # Let the layout settle, then resize to sizeHint
    widget.adjustSize()
    app.processEvents()

    pixmap = widget.grab()
    pixmap.save(output_path)
    print(f"Saved to {output_path} ({pixmap.width()}x{pixmap.height()})")

    if "--show" in sys.argv:
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
