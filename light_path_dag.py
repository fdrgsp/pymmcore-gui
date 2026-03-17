# ruff: noqa
# pylint: disable
"""Light Path DAG Editor — PyQt6 Implementation.

Dataclass-backed graph model rendered via QGraphicsScene.
Custom QGraphicsItem subclasses for device nodes, beam edges,
insert buttons, and the specimen marker.

Layout algorithm:
  - Dichroic at origin (0, 0)
  - Excitation traces upward (negative rows)
  - Shared traces downward (positive rows) to specimen
  - Emission traces rightward (positive columns)
  - Multiple sources → separate columns (fan-in)
  - Beam splitter children → stacked rows (fan-out)
  - No dichroic (trans/lightsheet) → specimen is pivot

Requires: PyQt6

Usage:
    python light_path_dag.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum, auto

from PyQt6.QtCore import (
    QObject,
    QPointF,
    QRectF,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QFontMetricsF,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPolygonF,
)
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSceneHoverEvent,
    QGraphicsSceneMouseEvent,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QStyleFactory,
    QVBoxLayout,
    QWidget,
)

# ═══════════════════════════════════════════════════════════════
# Design Tokens
# ═══════════════════════════════════════════════════════════════


class Clr:
    BG_DEEPEST = QColor(0x12, 0x12, 0x12)
    BG_BASE = QColor(0x1E, 0x1E, 0x1E)
    BG_RAISED = QColor(0x25, 0x25, 0x25)
    BG_SURFACE = QColor(0x2D, 0x2D, 0x2D)
    BG_HOVER = QColor(0x35, 0x35, 0x35)
    BG_ACTIVE = QColor(0x40, 0x40, 0x40)

    TEXT_PRIMARY = QColor(0xE0, 0xE0, 0xE0)
    TEXT_SECONDARY = QColor(0xA0, 0xA0, 0xA0)
    TEXT_DISABLED = QColor(0x70, 0x70, 0x70)

    BORDER_SUBTLE = QColor(0x33, 0x33, 0x33)
    BORDER_DEFAULT = QColor(0x44, 0x44, 0x44)

    ACCENT = QColor(0x4A, 0x9E, 0xFF)
    ACCENT_MUTED = QColor(0x4A, 0x9E, 0xFF, 0x26)

    GREEN = QColor(0x4C, 0xAF, 0x50)
    RED = QColor(0xEF, 0x53, 0x50)
    AMBER = QColor(0xFF, 0xA7, 0x26)

    EX = QColor(0x4A, 0x9E, 0xFF)
    EM = QColor(0x4C, 0xAF, 0x50)
    SHARED = QColor(0xFF, 0xA7, 0x26)


def _ui_font(size: float = 10, weight: int = QFont.Weight.Normal) -> QFont:
    f = QFont()
    f.setPointSizeF(size)
    f.setWeight(weight)
    return f


def _mono_font(size: float = 10, weight: int = QFont.Weight.Normal) -> QFont:
    for fam in ("JetBrains Mono", "SF Mono", "Cascadia Code", "Consolas"):
        if fam in QFontDatabase.families():
            f = QFont(fam)
            f.setPointSizeF(size)
            f.setWeight(weight)
            return f
    f = QFont()
    f.setStyleHint(QFont.StyleHint.Monospace)
    f.setPointSizeF(size)
    f.setWeight(weight)
    return f


# ═══════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════


class BeamType(Enum):
    EX = auto()
    EM = auto()
    SHARED = auto()

    @property
    def color(self) -> QColor:
        return {
            BeamType.EX: Clr.EX,
            BeamType.EM: Clr.EM,
            BeamType.SHARED: Clr.SHARED,
        }[self]


DEVICE_TYPES: dict[str, dict] = {
    "source": {"label": "Light Source", "removable": False},
    "laser": {"label": "Laser", "removable": False},
    "exfilter": {"label": "Ex Filter", "removable": True},
    "phaseplate": {"label": "Phase Plate", "removable": True},
    "dichroic": {"label": "Dichroic", "removable": False},
    "condenser": {"label": "Condenser", "removable": True},
    "objective": {"label": "Objective", "removable": False},
    "illumobjective": {"label": "Illum. Obj.", "removable": False},
    "scanmirror": {"label": "Scan Mirrors", "removable": True},
    "emfilter": {"label": "Em Filter", "removable": True},
    "beamsplitter": {"label": "Beam Splitter", "removable": False},
    "optivar": {"label": "Tube Lens", "removable": True},
    "port": {"label": "Port", "removable": True},
    "detector": {"label": "Detector", "removable": False},
    "pmt": {"label": "PMT", "removable": False},
    "specimen": {"label": "Specimen", "removable": False},
}

DEVICE_POSITIONS: dict[str, list[str]] = {
    "source": ["LED 365nm", "LED 470nm", "LED 555nm", "LED 630nm", "Halogen"],
    "laser": ["488nm", "561nm", "640nm", "405nm", "730nm STED"],
    "exfilter": [
        "AT350/50x",
        "ET470/40x",
        "ET545/30x",
        "ET620/60x",
        "Cleanup 488/10",
        "Empty",
    ],
    "phaseplate": ["Vortex 0-2π", "Top-hat", "None"],
    "dichroic": ["T400lp", "T495lpxr", "T660lpxr", "Quad-band", "Mirror"],
    "condenser": ["Phase 1", "DIC", "Darkfield", "Brightfield"],
    "objective": [
        "4×/0.13",
        "10×/0.30",
        "20×/0.50",
        "40×Oil 1.30",
        "63×Oil 1.40",
        "100×Oil 1.45",
    ],
    "illumobjective": ["5×/0.16", "10×/0.30", "20×/0.50"],
    "scanmirror": ["Galvo XY", "Resonant+Galvo"],
    "emfilter": [
        "ET460/50m",
        "ET525/50m",
        "ET605/70m",
        "ET700/75m",
        "Notch 488",
        "Notch 775",
        "Empty",
    ],
    "beamsplitter": ["50/50", "80/20", "565LP dichroic", "Bypass"],
    "optivar": ["1.0×", "1.5×", "2.0×"],
    "port": ["Eye 100%", "Left 100%", "Right 100%"],
    "detector": ["Orca Flash", "Orca Fusion"],
    "pmt": ["GaAsP PMT", "Multi-alkali", "HyD"],
}

INSERTABLE_EX = ["exfilter", "phaseplate", "scanmirror"]
INSERTABLE_EM = ["emfilter", "optivar", "port"]
INSERTABLE_SHARED = ["scanmirror"]


@dataclass
class LPNode:
    id: str
    device_type: str
    position: str = ""
    passthru: bool = False

    @property
    def label(self) -> str:
        return DEVICE_TYPES.get(self.device_type, {}).get("label", self.device_type)

    @property
    def removable(self) -> bool:
        return DEVICE_TYPES.get(self.device_type, {}).get("removable", False)

    @property
    def is_specimen(self) -> bool:
        return self.device_type == "specimen"


@dataclass
class LPEdge:
    source_id: str
    target_id: str
    beam: BeamType


@dataclass
class LPConfig:
    """A complete light path configuration (one channel)."""

    name: str
    color: QColor
    exposure: str
    nodes: list[LPNode] = field(default_factory=list)
    edges: list[LPEdge] = field(default_factory=list)

    _uid_counter: int = field(default=0, repr=False)

    def new_id(self) -> str:
        self._uid_counter += 1
        return f"n{self._uid_counter:04d}"

    def node_by_id(self, nid: str) -> LPNode | None:
        return next((n for n in self.nodes if n.id == nid), None)

    def insert_on_edge(self, edge: LPEdge, device_type: str) -> LPNode:
        """Splice a new node onto an existing edge."""
        new_node = LPNode(
            id=self.new_id(),
            device_type=device_type,
            position=DEVICE_POSITIONS.get(device_type, [""])[0],
        )
        self.nodes.append(new_node)
        idx = self.edges.index(edge)
        self.edges[idx] = LPEdge(edge.source_id, new_node.id, edge.beam)
        self.edges.insert(idx + 1, LPEdge(new_node.id, edge.target_id, edge.beam))
        return new_node

    def remove_node(self, node_id: str) -> None:
        """Remove a node, reconnecting edges through it."""
        in_edges = [e for e in self.edges if e.target_id == node_id]
        out_edges = [e for e in self.edges if e.source_id == node_id]
        self.edges = [
            e for e in self.edges if e.source_id != node_id and e.target_id != node_id
        ]
        for ie in in_edges:
            for oe in out_edges:
                self.edges.append(LPEdge(ie.source_id, oe.target_id, ie.beam))
        self.nodes = [n for n in self.nodes if n.id != node_id]

    def edges_for(self, node_id: str) -> list[LPEdge]:
        return [
            e for e in self.edges if e.source_id == node_id or e.target_id == node_id
        ]

    def beam_role(self, node_id: str) -> BeamType:
        """Determine the visual role of a node based on connected edge beams."""
        beams = {e.beam for e in self.edges_for(node_id)}
        if BeamType.SHARED in beams:
            return BeamType.SHARED
        if BeamType.EM in beams and BeamType.EX not in beams:
            return BeamType.EM
        if BeamType.EM in beams and BeamType.EX in beams:
            return BeamType.SHARED
        return BeamType.EX


# ═══════════════════════════════════════════════════════════════
# Layout Engine
# ═══════════════════════════════════════════════════════════════

NODE_W = 120
NODE_H = 50
GAP_X = 52
GAP_Y = 28
CELL_W = NODE_W + GAP_X
CELL_H = NODE_H + GAP_Y


@dataclass
class LayoutPos:
    col: float = 0
    row: float = 0

    @property
    def x(self) -> float:
        return self.col * CELL_W

    @property
    def y(self) -> float:
        return self.row * CELL_H


def compute_layout(cfg: LPConfig) -> dict[str, LayoutPos]:
    """Assign (col, row) grid positions to each node."""
    pos: dict[str, LayoutPos] = {n.id: LayoutPos() for n in cfg.nodes}

    dic = next((n for n in cfg.nodes if n.device_type == "dichroic"), None)
    spec = next((n for n in cfg.nodes if n.device_type == "specimen"), None)

    if dic:
        _layout_epi(cfg, pos, dic.id, spec.id if spec else None)
    elif spec:
        _layout_linear(cfg, pos, spec.id)

    # Normalize so minimum col/row is 0
    if pos:
        min_c = min(p.col for p in pos.values())
        min_r = min(p.row for p in pos.values())
        for p in pos.values():
            p.col -= min_c
            p.row -= min_r

    return pos


def _layout_epi(
    cfg: LPConfig,
    pos: dict[str, LayoutPos],
    dic_id: str,
    spec_id: str | None,
) -> None:
    pos[dic_id] = LayoutPos(0, 0)

    # Excitation upstream
    ex_inputs = [
        e.source_id
        for e in cfg.edges
        if e.target_id == dic_id and e.beam == BeamType.EX
    ]
    for i, src_id in enumerate(ex_inputs):
        col = 0.0 if len(ex_inputs) == 1 else (i - (len(ex_inputs) - 1) / 2)
        _trace_up(cfg, pos, src_id, dic_id, col, -1)

    # Shared downstream
    _trace_down_shared(cfg, pos, dic_id, spec_id)

    # Emission rightward
    em_outputs = [
        e.target_id
        for e in cfg.edges
        if e.source_id == dic_id and e.beam == BeamType.EM
    ]
    em_col = 1.0
    for tid in em_outputs:
        _trace_right(cfg, pos, tid, em_col, 0)
        em_col += _subtree_width(cfg, tid)

    # Trans detection from specimen
    if spec_id:
        trans_out = [
            e.target_id
            for e in cfg.edges
            if e.source_id == spec_id and e.beam == BeamType.EM
        ]
        for i, tid in enumerate(trans_out):
            _trace_down_from(cfg, pos, tid, em_col + i, pos[spec_id].row + 1)


def _trace_up(
    cfg: LPConfig,
    pos: dict[str, LayoutPos],
    node_id: str,
    stop_id: str,
    col: float,
    start_row: float,
) -> None:
    chain: list[str] = []
    cur = node_id
    while cur and cur != stop_id:
        chain.append(cur)
        parents = [
            e.source_id
            for e in cfg.edges
            if e.target_id == cur and e.beam == BeamType.EX
        ]
        cur = parents[0] if parents else None
    chain.reverse()
    for i, nid in enumerate(chain):
        pos[nid] = LayoutPos(col, start_row - (len(chain) - 1 - i))


def _trace_down_shared(
    cfg: LPConfig,
    pos: dict[str, LayoutPos],
    from_id: str,
    to_id: str | None,
) -> None:
    cur = from_id
    row_offset = 1
    while cur:
        children = [
            e.target_id
            for e in cfg.edges
            if e.source_id == cur and e.beam == BeamType.SHARED
        ]
        if not children:
            break
        nxt = children[0]
        pos[nxt] = LayoutPos(0, pos[from_id].row + row_offset)
        row_offset += 1
        if nxt == to_id:
            break
        cur = nxt


def _trace_right(
    cfg: LPConfig,
    pos: dict[str, LayoutPos],
    node_id: str,
    col: float,
    row: float,
) -> None:
    pos[node_id] = LayoutPos(col, row)
    children = [
        e.target_id
        for e in cfg.edges
        if e.source_id == node_id and e.beam == BeamType.EM
    ]
    if len(children) == 1:
        _trace_right(cfg, pos, children[0], col + 1, row)
    elif len(children) > 1:
        for i, cid in enumerate(children):
            _trace_right(cfg, pos, cid, col + 1, row + i)


def _trace_down_from(
    cfg: LPConfig,
    pos: dict[str, LayoutPos],
    node_id: str,
    col: float,
    row: float,
) -> None:
    pos[node_id] = LayoutPos(col, row)
    children = [
        e.target_id
        for e in cfg.edges
        if e.source_id == node_id and e.beam == BeamType.EM
    ]
    for i, cid in enumerate(children):
        _trace_down_from(cfg, pos, cid, col, row + 1 + i)


def _subtree_width(cfg: LPConfig, node_id: str) -> float:
    children = [
        e.target_id
        for e in cfg.edges
        if e.source_id == node_id and e.beam == BeamType.EM
    ]
    if len(children) <= 1:
        return 1
    return sum(_subtree_width(cfg, c) for c in children)


def _layout_linear(cfg: LPConfig, pos: dict[str, LayoutPos], spec_id: str) -> None:
    # Trace ex chain up from specimen
    ex_chain: list[str] = []
    cur = spec_id
    while True:
        parents = [
            e.source_id
            for e in cfg.edges
            if e.target_id == cur and e.beam == BeamType.EX
        ]
        if not parents:
            break
        ex_chain.insert(0, parents[0])
        cur = parents[0]

    pos[spec_id] = LayoutPos(0, len(ex_chain))
    for i, nid in enumerate(ex_chain):
        pos[nid] = LayoutPos(0, i)

    em_start = [
        e.target_id
        for e in cfg.edges
        if e.source_id == spec_id and e.beam == BeamType.EM
    ]
    for i, tid in enumerate(em_start):
        _trace_right(cfg, pos, tid, 1, pos[spec_id].row + i)


# ═══════════════════════════════════════════════════════════════
# QGraphicsItems
# ═══════════════════════════════════════════════════════════════


class DeviceNodeItem(QGraphicsObject):
    """A single device node in the light path graph."""

    clicked = pyqtSignal()
    remove_requested = pyqtSignal()
    passthru_toggled = pyqtSignal()

    WIDTH = NODE_W
    HEIGHT = NODE_H

    def __init__(
        self,
        node: LPNode,
        beam_role: BeamType,
        parent: QGraphicsItem | None = None,
    ) -> None:
        super().__init__(parent)
        self.node = node
        self.beam_role = beam_role
        self._hovered = False
        self._selected = False

        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)

    def boundingRect(self) -> QRectF:
        return QRectF(-2, -8, self.WIDTH + 4, self.HEIGHT + 18)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(0, 0, self.WIDTH, self.HEIGHT)
        radius = 6.0

        # ── Background ──
        if self.node.passthru:
            bg = Clr.BG_RAISED if not self._hovered else Clr.BG_HOVER
            bg.setAlpha(100)
        else:
            bg = Clr.BG_HOVER if self._hovered else Clr.BG_RAISED
        painter.setBrush(QBrush(bg))

        # ── Border ──
        if self._selected:
            pen = QPen(Clr.ACCENT, 1.5)
        elif self.node.passthru:
            pen = QPen(Clr.BORDER_DEFAULT, 1.0, Qt.PenStyle.DashLine)
        elif self._hovered:
            pen = QPen(Clr.TEXT_DISABLED, 1.0)
        else:
            pen = QPen(Clr.BORDER_DEFAULT, 1.0)
        painter.setPen(pen)

        path = QPainterPath()
        path.addRoundedRect(r, radius, radius)
        painter.drawPath(path)

        # ── Top color accent ──
        accent_color = self.beam_role.color
        if self.node.passthru:
            accent_color = QColor(accent_color)
            accent_color.setAlpha(80)
        top_r = QRectF(0, 0, self.WIDTH, 2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(accent_color))
        # Clip to rounded top
        clip = QPainterPath()
        clip.addRoundedRect(r, radius, radius)
        painter.setClipPath(clip)
        painter.drawRect(top_r)
        painter.setClipping(False)

        # ── Type label ──
        type_font = _ui_font(6, QFont.Weight.Medium)
        painter.setFont(type_font)
        tc = QColor(accent_color)
        if self.node.passthru:
            tc.setAlpha(120)
        painter.setPen(tc)
        type_rect = QRectF(0, 8, self.WIDTH, 12)
        painter.drawText(
            type_rect,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            self.node.label.upper(),
        )

        # ── Name ──
        name_font = _ui_font(8, QFont.Weight.DemiBold)
        painter.setFont(name_font)
        name_color = QColor(Clr.TEXT_PRIMARY)
        if self.node.passthru:
            name_color.setAlpha(100)
        painter.setPen(name_color)
        name_rect = QRectF(4, 22, self.WIDTH - 8, 20)
        name_text = self.node.position or "—"
        fm = QFontMetricsF(name_font)
        elided = fm.elidedText(
            name_text, Qt.TextElideMode.ElideRight, name_rect.width()
        )

        if self.node.passthru:
            # Strikethrough
            painter.drawText(
                name_rect,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                elided,
            )
            text_width = fm.horizontalAdvance(elided)
            text_x = name_rect.center().x() - text_width / 2
            text_y = name_rect.top() + fm.ascent() / 2 + 4
            line_pen = QPen(Clr.TEXT_DISABLED, 1)
            painter.setPen(line_pen)
            painter.drawLine(
                QPointF(text_x, text_y),
                QPointF(text_x + text_width, text_y),
            )
        else:
            painter.drawText(
                name_rect,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                elided,
            )

        # ── Pass-through badge ──
        if self.node.passthru:
            badge_font = _ui_font(5.5, QFont.Weight.DemiBold)
            painter.setFont(badge_font)
            painter.setPen(Clr.TEXT_DISABLED)
            painter.drawText(
                QRectF(0, -8, self.WIDTH, 10),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                "PASS-THROUGH",
            )

        # ── Remove button (on hover, if removable) ──
        if self._hovered and self.node.removable:
            rm_x, rm_y, rm_r = self.WIDTH - 2, -4, 7
            painter.setPen(QPen(Clr.BORDER_DEFAULT, 1))
            painter.setBrush(QBrush(Clr.BG_ACTIVE))
            painter.drawEllipse(QPointF(rm_x, rm_y), rm_r, rm_r)
            painter.setPen(Clr.TEXT_DISABLED)
            painter.setFont(_ui_font(7, QFont.Weight.Bold))
            painter.drawText(
                QRectF(rm_x - rm_r, rm_y - rm_r, rm_r * 2, rm_r * 2),
                Qt.AlignmentFlag.AlignCenter,
                "✕",
            )

        # ── Pass-through toggle (on hover) ──
        if self._hovered and not self.node.is_specimen:
            pt_text = "● Activate" if self.node.passthru else "○ Pass-thru"
            pt_font = _ui_font(5.5)
            painter.setFont(pt_font)
            pt_fm = QFontMetricsF(pt_font)
            pt_w = pt_fm.horizontalAdvance(pt_text) + 12
            pt_h = 13
            pt_x = (self.WIDTH - pt_w) / 2
            pt_y = self.HEIGHT + 1

            painter.setPen(QPen(Clr.BORDER_DEFAULT, 1))
            painter.setBrush(QBrush(Clr.BG_SURFACE))
            painter.drawRoundedRect(QRectF(pt_x, pt_y, pt_w, pt_h), 6, 6)
            painter.setPen(Clr.TEXT_DISABLED)
            painter.drawText(
                QRectF(pt_x, pt_y, pt_w, pt_h),
                Qt.AlignmentFlag.AlignCenter,
                pt_text,
            )

    # ── Hit testing ──

    def _rm_hit(self, pos: QPointF) -> bool:
        if not self.node.removable:
            return False
        rm_center = QPointF(self.WIDTH - 2, -4)
        return (pos - rm_center).manhattanLength() < 10

    def _pt_hit(self, pos: QPointF) -> bool:
        pt_rect = QRectF(0, self.HEIGHT + 1, self.WIDTH, 14)
        return pt_rect.contains(pos)

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self._hovered = True
        self.update()

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self._hovered = False
        self.update()

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._rm_hit(event.pos()):
                self.remove_requested.emit()
            elif self._pt_hit(event.pos()):
                self.passthru_toggled.emit()
            else:
                self.clicked.emit()
        super().mousePressEvent(event)

    @property
    def selected(self) -> bool:
        return self._selected

    @selected.setter
    def selected(self, val: bool) -> None:
        self._selected = val
        self.update()


class SpecimenItem(QGraphicsObject):
    """The specimen marker — dashed circle + label."""

    SIZE = 40

    def __init__(self, parent: QGraphicsItem | None = None) -> None:
        super().__init__(parent)

    def boundingRect(self) -> QRectF:
        return QRectF(-4, -4, self.SIZE + 8, self.SIZE + 20)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.SIZE / 2
        cx, cy = r, r

        # Dashed circle
        pen = QPen(Clr.TEXT_DISABLED, 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(255, 255, 255, 6)))
        painter.drawEllipse(QPointF(cx, cy), r, r)

        # Emoji
        painter.setFont(_ui_font(14))
        painter.setPen(Clr.TEXT_DISABLED)
        painter.drawText(
            QRectF(0, 0, self.SIZE, self.SIZE),
            Qt.AlignmentFlag.AlignCenter,
            "🐛",
        )

        # Label
        painter.setFont(_ui_font(6, QFont.Weight.DemiBold))
        painter.setPen(Clr.TEXT_DISABLED)
        painter.drawText(
            QRectF(0, self.SIZE + 2, self.SIZE, 12),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            "SPECIMEN",
        )


class BeamEdgeItem(QGraphicsObject):
    """A beam line connecting two nodes, with an arrow at the midpoint."""

    def __init__(
        self,
        edge: LPEdge,
        p1: QPointF,
        p2: QPointF,
        is_passthru: bool = False,
        parent: QGraphicsItem | None = None,
    ) -> None:
        super().__init__(parent)
        self.edge = edge
        self.p1 = p1
        self.p2 = p2
        self.is_passthru = is_passthru
        self.setZValue(-1)

    def boundingRect(self) -> QRectF:
        return QRectF(
            min(self.p1.x(), self.p2.x()) - 8,
            min(self.p1.y(), self.p2.y()) - 8,
            abs(self.p2.x() - self.p1.x()) + 16,
            abs(self.p2.y() - self.p1.y()) + 16,
        )

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = self.edge.beam.color
        opacity = 0.15 if self.is_passthru else 0.3

        # Glow
        glow_pen = QPen(QColor(color), 10)
        glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        glow_color = QColor(color)
        glow_color.setAlphaF(0.04)
        glow_pen.setColor(glow_color)
        painter.setPen(glow_pen)
        painter.drawLine(self.p1, self.p2)

        # Line
        line_color = QColor(color)
        line_color.setAlphaF(opacity)
        line_pen = QPen(line_color, 2.5)
        line_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        if self.is_passthru:
            line_pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(line_pen)
        painter.drawLine(self.p1, self.p2)

        # Arrow at midpoint
        import math

        mx = (self.p1.x() + self.p2.x()) / 2
        my = (self.p1.y() + self.p2.y()) / 2
        angle = math.atan2(self.p2.y() - self.p1.y(), self.p2.x() - self.p1.x())
        s = 5
        arrow = QPolygonF(
            [
                QPointF(mx + math.cos(angle) * s, my + math.sin(angle) * s),
                QPointF(mx + math.cos(angle + 2.5) * s, my + math.sin(angle + 2.5) * s),
                QPointF(mx + math.cos(angle - 2.5) * s, my + math.sin(angle - 2.5) * s),
            ]
        )
        arrow_color = QColor(color)
        arrow_color.setAlphaF(opacity)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(arrow_color))
        painter.drawPolygon(arrow)


class InsertButton(QGraphicsObject):
    """A + button that appears on edge hover."""

    clicked = pyqtSignal()
    RADIUS = 10

    def __init__(
        self,
        center: QPointF,
        parent: QGraphicsItem | None = None,
    ) -> None:
        super().__init__(parent)
        self.center = center
        self.setPos(center)
        self._hovered = False
        self._visible = False

        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setZValue(20)

    def boundingRect(self) -> QRectF:
        r = self.RADIUS
        return QRectF(-r - 2, -r - 2, r * 2 + 4, r * 2 + 4)

    def show_button(self) -> None:
        self._visible = True
        self.update()

    def hide_button(self) -> None:
        self._visible = False
        self._hovered = False
        self.update()

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if not self._visible and not self._hovered:
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.RADIUS

        if self._hovered:
            painter.setPen(QPen(Clr.ACCENT, 1))
            painter.setBrush(QBrush(Clr.ACCENT))
            text_color = QColor(255, 255, 255)
        else:
            painter.setPen(QPen(Clr.BORDER_DEFAULT, 1))
            painter.setBrush(QBrush(Clr.BG_SURFACE))
            text_color = Clr.TEXT_DISABLED

        painter.drawEllipse(QPointF(0, 0), r, r)

        painter.setPen(text_color)
        painter.setFont(_ui_font(11, QFont.Weight.Bold))
        painter.drawText(
            QRectF(-r, -r, r * 2, r * 2),
            Qt.AlignmentFlag.AlignCenter,
            "+",
        )

    def hoverEnterEvent(self, event) -> None:
        self._hovered = True
        self._visible = True
        self.update()

    def hoverLeaveEvent(self, event) -> None:
        self._hovered = False
        self._visible = False
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()


class EdgeHoverZone(QGraphicsRectItem):
    """Invisible wide rect over an edge that triggers the insert button."""

    def __init__(
        self,
        p1: QPointF,
        p2: QPointF,
        insert_btn: InsertButton,
        parent: QGraphicsItem | None = None,
    ) -> None:
        # Build a fat rect along the edge
        import math

        dx, dy = p2.x() - p1.x(), p2.y() - p1.y()
        length = math.hypot(dx, dy)
        if length < 1:
            length = 1
        _nx, _ny = -dy / length * 10, dx / length * 10  # normal, 10px thick

        super().__init__(parent)
        self.setRect(
            min(p1.x(), p2.x()) - 10,
            min(p1.y(), p2.y()) - 10,
            abs(dx) + 20,
            abs(dy) + 20,
        )
        self.setBrush(QBrush(Qt.GlobalColor.transparent))
        self.setPen(QPen(Qt.PenStyle.NoPen))
        self.setAcceptHoverEvents(True)
        self.setZValue(5)
        self._btn = insert_btn

    def hoverEnterEvent(self, event) -> None:
        self._btn.show_button()

    def hoverLeaveEvent(self, event) -> None:
        # Small delay so user can reach the button
        QTimer.singleShot(150, self._maybe_hide)

    def _maybe_hide(self) -> None:
        if not self._btn._hovered:
            self._btn.hide_button()


# ═══════════════════════════════════════════════════════════════
# Position Picker Popup
# ═══════════════════════════════════════════════════════════════


class PositionPicker(QWidget):
    """Popup list for selecting device positions."""

    position_selected = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint
        )
        self.setFixedWidth(200)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(2)

        self._title = QLabel()
        self._title.setFont(_ui_font(7, QFont.Weight.Medium))
        pal = self._title.palette()
        pal.setColor(QPalette.ColorRole.WindowText, Clr.TEXT_DISABLED)
        self._title.setPalette(pal)
        layout.addWidget(self._title)

        self._list = QListWidget()
        self._list.setFont(_mono_font(9))
        self._list.setFrameShape(QListWidget.Shape.NoFrame)
        self._list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._list)

        # Style the popup background via palette
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, Clr.BG_RAISED)
        self.setPalette(pal)
        self.setAutoFillBackground(True)

    def show_for(
        self,
        title: str,
        positions: list[str],
        current: str,
        global_pos: QPointF,
    ) -> None:
        self._title.setText(title.upper())
        self._list.clear()
        for i, p in enumerate(positions):
            item = QListWidgetItem(f"  {i + 1}  {p}")
            item.setData(Qt.ItemDataRole.UserRole, p)
            self._list.addItem(item)
            if p == current:
                item.setSelected(True)

        self._list.setFixedHeight(min(len(positions) * 28 + 4, 250))
        self.adjustSize()

        # Position near the click
        x, y = int(global_pos.x()) + 8, int(global_pos.y())
        screen_rect = QApplication.primaryScreen().availableGeometry()
        if x + self.width() > screen_rect.right():
            x = int(global_pos.x()) - self.width() - 8
        if y + self.height() > screen_rect.bottom():
            y = screen_rect.bottom() - self.height() - 8
        self.move(x, y)
        self.show()

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        pos = item.data(Qt.ItemDataRole.UserRole)
        self.position_selected.emit(pos)
        self.hide()


# ═══════════════════════════════════════════════════════════════
# Insert Device Picker Popup
# ═══════════════════════════════════════════════════════════════


class InsertDevicePicker(QWidget):
    """Popup for choosing which device type to insert."""

    device_selected = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint
        )
        self.setFixedWidth(180)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(2)

        title = QLabel("INSERT DEVICE")
        title.setFont(_ui_font(7, QFont.Weight.Medium))
        pal = title.palette()
        pal.setColor(QPalette.ColorRole.WindowText, Clr.TEXT_DISABLED)
        title.setPalette(pal)
        layout.addWidget(title)

        self._list = QListWidget()
        self._list.setFont(_ui_font(9))
        self._list.setFrameShape(QListWidget.Shape.NoFrame)
        self._list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._list)

        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, Clr.BG_RAISED)
        self.setPalette(pal)
        self.setAutoFillBackground(True)

    def show_for(self, device_types: list[str], global_pos: QPointF) -> None:
        self._list.clear()
        for dt in device_types:
            label = DEVICE_TYPES.get(dt, {}).get("label", dt)
            item = QListWidgetItem(f"  {label}")
            item.setData(Qt.ItemDataRole.UserRole, dt)
            self._list.addItem(item)
        self._list.setFixedHeight(min(len(device_types) * 28 + 4, 200))
        self.adjustSize()

        x, y = int(global_pos.x()) + 8, int(global_pos.y()) - 20
        screen_rect = QApplication.primaryScreen().availableGeometry()
        if x + self.width() > screen_rect.right():
            x = int(global_pos.x()) - self.width() - 8
        self.move(x, y)
        self.show()

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        dt = item.data(Qt.ItemDataRole.UserRole)
        self.device_selected.emit(dt)
        self.hide()


# ═══════════════════════════════════════════════════════════════
# The Scene + View
# ═══════════════════════════════════════════════════════════════


class LightPathScene(QGraphicsScene):
    """Renders a LPConfig as an interactive DAG."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.setBackgroundBrush(QBrush(Clr.BG_BASE))

    def rebuild(self, config: LPConfig) -> None:
        self.clear()
        self._config = config
        positions = compute_layout(config)

        pad = 30.0
        node_center: dict[str, QPointF] = {}

        # ── Place nodes ──
        for node in config.nodes:
            lp = positions.get(node.id, LayoutPos())
            x = lp.x + pad
            y = lp.y + pad

            if node.is_specimen:
                item = SpecimenItem()
                # Center the specimen on the cell
                item.setPos(x + (NODE_W - SpecimenItem.SIZE) / 2, y)
                self.addItem(item)
                node_center[node.id] = QPointF(
                    x + NODE_W / 2, y + SpecimenItem.SIZE / 2
                )
            else:
                role = config.beam_role(node.id)
                item = DeviceNodeItem(node, role)
                item.setPos(x, y)
                self.addItem(item)
                node_center[node.id] = QPointF(x + NODE_W / 2, y + NODE_H / 2)

                # Connect signals
                item.clicked.connect(
                    lambda n=node, it=item: self._on_node_clicked(n, it)
                )
                item.remove_requested.connect(lambda n=node: self._on_remove(n))
                item.passthru_toggled.connect(
                    lambda n=node: self._on_passthru_toggle(n)
                )

        # ── Draw edges ──
        for edge in config.edges:
            p1 = node_center.get(edge.source_id)
            p2 = node_center.get(edge.target_id)
            if not p1 or not p2:
                continue

            src_node = config.node_by_id(edge.source_id)
            tgt_node = config.node_by_id(edge.target_id)
            is_pt = (src_node and src_node.passthru) or (tgt_node and tgt_node.passthru)

            edge_item = BeamEdgeItem(edge, p1, p2, is_pt)
            self.addItem(edge_item)

            # Insert button at midpoint (skip edges to/from specimen)
            if not (src_node and src_node.is_specimen) and not (
                tgt_node and tgt_node.is_specimen
            ):
                mid = (p1 + p2) / 2
                ins_btn = InsertButton(mid)
                ins_btn.clicked.connect(lambda e=edge: self._on_insert_clicked(e))
                self.addItem(ins_btn)

                hover_zone = EdgeHoverZone(p1, p2, ins_btn)
                self.addItem(hover_zone)

        # Fit scene rect
        sr = self.itemsBoundingRect().adjusted(-20, -20, 40, 40)
        self.setSceneRect(sr)

    # ── Interaction handlers ──

    def _on_node_clicked(self, node: LPNode, item: DeviceNodeItem) -> None:
        positions = DEVICE_POSITIONS.get(node.device_type, [])
        if not positions:
            return

        # Deselect all
        for it in self.items():
            if isinstance(it, DeviceNodeItem):
                it.selected = False
        item.selected = True

        # Show picker
        view = self.views()[0] if self.views() else None
        if not view:
            return

        global_pos = view.mapToGlobal(
            view.mapFromScene(item.pos() + QPointF(NODE_W + 4, 0))
        )

        picker = PositionPicker(view)
        picker.show_for(node.label, positions, node.position, QPointF(global_pos))

        def on_selected(pos: str) -> None:
            node.position = pos
            node.passthru = False
            self.rebuild(self._config)

        picker.position_selected.connect(on_selected)

    def _on_remove(self, node: LPNode) -> None:
        self._config.remove_node(node.id)
        self.rebuild(self._config)

    def _on_passthru_toggle(self, node: LPNode) -> None:
        node.passthru = not node.passthru
        self.rebuild(self._config)

    def _on_insert_clicked(self, edge: LPEdge) -> None:
        if edge.beam == BeamType.EX:
            types = INSERTABLE_EX
        elif edge.beam == BeamType.EM:
            types = INSERTABLE_EM
        else:
            types = INSERTABLE_SHARED

        view = self.views()[0] if self.views() else None
        if not view:
            return

        p1_id, p2_id = edge.source_id, edge.target_id
        # Get approximate screen position of edge midpoint
        pos_map = compute_layout(self._config)
        lp1 = pos_map.get(p1_id, LayoutPos())
        lp2 = pos_map.get(p2_id, LayoutPos())
        mid_scene = QPointF(
            (lp1.x + lp2.x) / 2 + 30 + NODE_W / 2,
            (lp1.y + lp2.y) / 2 + 30 + NODE_H / 2,
        )
        global_pos = view.mapToGlobal(view.mapFromScene(mid_scene))

        picker = InsertDevicePicker(view)
        picker.show_for(types, QPointF(global_pos))

        def on_selected(device_type: str) -> None:
            self._config.insert_on_edge(edge, device_type)
            self.rebuild(self._config)

        picker.device_selected.connect(on_selected)


class LightPathView(QGraphicsView):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = LightPathScene(self)
        self.setScene(self._scene)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

        # Background
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Base, Clr.BG_BASE)
        self.setPalette(pal)

    def set_config(self, config: LPConfig) -> None:
        self._config = config
        self._scene.rebuild(config)
        # Fit in view with some padding
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_config"):
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


# ═══════════════════════════════════════════════════════════════
# Config Tab Bar
# ═══════════════════════════════════════════════════════════════


class ConfigTabBar(QWidget):
    config_selected = pyqtSignal(str)

    def __init__(
        self, configs: dict[str, LPConfig], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._configs = configs
        self._active = ""
        self._buttons: dict[str, QPushButton] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(4)

        for key, cfg in configs.items():
            btn = QPushButton(cfg.name)
            btn.setFont(_ui_font(9, QFont.Weight.Medium))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedHeight(28)
            btn.clicked.connect(lambda checked, k=key: self._select(k))
            layout.addWidget(btn)
            self._buttons[key] = btn

        layout.addStretch()

    def _select(self, key: str) -> None:
        self._active = key
        self.config_selected.emit(key)
        self.update()

    def set_active(self, key: str) -> None:
        self._active = key
        self.update()

    def paintEvent(self, event) -> None:
        # Draw background
        p = QPainter(self)
        p.fillRect(self.rect(), Clr.BG_RAISED)
        p.setPen(QPen(Clr.BORDER_SUBTLE, 1))
        p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        p.end()

        # Update button appearance via palette
        for key, btn in self._buttons.items():
            self._configs[key]
            pal = btn.palette()
            if key == self._active:
                pal.setColor(QPalette.ColorRole.Button, Clr.BG_RAISED)
                pal.setColor(QPalette.ColorRole.ButtonText, Clr.TEXT_PRIMARY)
            else:
                pal.setColor(QPalette.ColorRole.Button, QColor(0, 0, 0, 0))
                pal.setColor(QPalette.ColorRole.ButtonText, Clr.TEXT_SECONDARY)
            btn.setPalette(pal)

        super().paintEvent(event)


# ═══════════════════════════════════════════════════════════════
# Sample Data
# ═══════════════════════════════════════════════════════════════


def make_sample_configs() -> dict[str, LPConfig]:
    configs = {}

    # GFP
    gfp = LPConfig(name="GFP", color=QColor("#00CC66"), exposure="100 ms")
    gfp.nodes = [
        LPNode("src", "source", "LED 470nm"),
        LPNode("exf", "exfilter", "ET470/40x"),
        LPNode("dic", "dichroic", "T495lpxr"),
        LPNode("obj", "objective", "40×Oil 1.30"),
        LPNode("spec", "specimen"),
        LPNode("emf", "emfilter", "ET525/50m"),
        LPNode("det", "detector", "Orca Flash"),
    ]
    gfp.edges = [
        LPEdge("src", "exf", BeamType.EX),
        LPEdge("exf", "dic", BeamType.EX),
        LPEdge("dic", "obj", BeamType.SHARED),
        LPEdge("obj", "spec", BeamType.SHARED),
        LPEdge("dic", "emf", BeamType.EM),
        LPEdge("emf", "det", BeamType.EM),
    ]
    configs["gfp"] = gfp

    # DAPI
    dapi = LPConfig(name="DAPI", color=QColor("#4472C4"), exposure="50 ms")
    dapi.nodes = [
        LPNode("src", "source", "LED 365nm"),
        LPNode("exf", "exfilter", "AT350/50x"),
        LPNode("dic", "dichroic", "T400lp"),
        LPNode("obj", "objective", "40×Oil 1.30"),
        LPNode("spec", "specimen"),
        LPNode("emf", "emfilter", "ET460/50m"),
        LPNode("det", "detector", "Orca Flash"),
    ]
    dapi.edges = [
        LPEdge("src", "exf", BeamType.EX),
        LPEdge("exf", "dic", BeamType.EX),
        LPEdge("dic", "obj", BeamType.SHARED),
        LPEdge("obj", "spec", BeamType.SHARED),
        LPEdge("dic", "emf", BeamType.EM),
        LPEdge("emf", "det", BeamType.EM),
    ]
    configs["dapi"] = dapi

    # STED
    sted = LPConfig(name="STED", color=QColor("#FF7043"), exposure="10 ms")
    sted.nodes = [
        LPNode("las1", "laser", "488nm"),
        LPNode("exf1", "exfilter", "Cleanup 488/10"),
        LPNode("las2", "laser", "730nm STED"),
        LPNode("vort", "phaseplate", "Vortex 0-2π"),
        LPNode("dic", "dichroic", "Quad-band"),
        LPNode("scan", "scanmirror", "Galvo XY"),
        LPNode("obj", "objective", "100×Oil 1.45"),
        LPNode("spec", "specimen"),
        LPNode("emf", "emfilter", "ET525/50m"),
        LPNode("notch", "emfilter", "Notch 775"),
        LPNode("pmt1", "pmt", "GaAsP PMT"),
    ]
    sted.edges = [
        LPEdge("las1", "exf1", BeamType.EX),
        LPEdge("exf1", "dic", BeamType.EX),
        LPEdge("las2", "vort", BeamType.EX),
        LPEdge("vort", "dic", BeamType.EX),
        LPEdge("dic", "scan", BeamType.SHARED),
        LPEdge("scan", "obj", BeamType.SHARED),
        LPEdge("obj", "spec", BeamType.SHARED),
        LPEdge("dic", "emf", BeamType.EM),
        LPEdge("emf", "notch", BeamType.EM),
        LPEdge("notch", "pmt1", BeamType.EM),
    ]
    configs["sted"] = sted

    # Dual-Cam
    dc = LPConfig(name="Dual-Cam", color=QColor("#AB47BC"), exposure="50 ms")
    dc.nodes = [
        LPNode("src", "source", "LED 470nm"),
        LPNode("exf", "exfilter", "ET470/40x"),
        LPNode("dic", "dichroic", "Quad-band"),
        LPNode("obj", "objective", "63×Oil 1.40"),
        LPNode("spec", "specimen"),
        LPNode("bs", "beamsplitter", "565LP dichroic"),
        LPNode("emfg", "emfilter", "ET525/50m"),
        LPNode("emfr", "emfilter", "ET700/75m"),
        LPNode("cam1", "detector", "Orca Flash"),
        LPNode("cam2", "detector", "Orca Fusion"),
    ]
    dc.edges = [
        LPEdge("src", "exf", BeamType.EX),
        LPEdge("exf", "dic", BeamType.EX),
        LPEdge("dic", "obj", BeamType.SHARED),
        LPEdge("obj", "spec", BeamType.SHARED),
        LPEdge("dic", "bs", BeamType.EM),
        LPEdge("bs", "emfg", BeamType.EM),
        LPEdge("bs", "emfr", BeamType.EM),
        LPEdge("emfg", "cam1", BeamType.EM),
        LPEdge("emfr", "cam2", BeamType.EM),
    ]
    configs["dualcam"] = dc

    # Brightfield
    bf = LPConfig(name="Brightfield", color=QColor("#C0C0C0"), exposure="10 ms")
    bf.nodes = [
        LPNode("src", "source", "Halogen"),
        LPNode("cond", "condenser", "Brightfield"),
        LPNode("spec", "specimen"),
        LPNode("obj", "objective", "40×Oil 1.30", passthru=True),
        LPNode("det", "detector", "Orca Flash"),
    ]
    bf.edges = [
        LPEdge("src", "cond", BeamType.EX),
        LPEdge("cond", "spec", BeamType.EX),
        LPEdge("spec", "obj", BeamType.EM),
        LPEdge("obj", "det", BeamType.EM),
    ]
    configs["bf"] = bf

    return configs


# ═══════════════════════════════════════════════════════════════
# Main Window
# ═══════════════════════════════════════════════════════════════


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Light Path — DAG Editor")
        self.resize(900, 650)

        self._configs = make_sample_configs()
        self._active_key = "gfp"

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Tab bar
        self._tab_bar = ConfigTabBar(self._configs)
        self._tab_bar.config_selected.connect(self._on_config_selected)
        self._tab_bar.set_active(self._active_key)
        layout.addWidget(self._tab_bar)

        # Graph view
        self._view = LightPathView()
        layout.addWidget(self._view)

        # Initial load
        self._view.set_config(self._configs[self._active_key])

    def _on_config_selected(self, key: str) -> None:
        self._active_key = key
        self._tab_bar.set_active(key)
        self._view.set_config(self._configs[key])


# ═══════════════════════════════════════════════════════════════
# Palette + Entry Point
# ═══════════════════════════════════════════════════════════════


def make_dark_palette() -> QPalette:
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, Clr.BG_BASE)
    p.setColor(QPalette.ColorRole.WindowText, Clr.TEXT_PRIMARY)
    p.setColor(QPalette.ColorRole.Base, Clr.BG_DEEPEST)
    p.setColor(QPalette.ColorRole.AlternateBase, Clr.BG_RAISED)
    p.setColor(QPalette.ColorRole.Button, Clr.BG_SURFACE)
    p.setColor(QPalette.ColorRole.ButtonText, Clr.TEXT_PRIMARY)
    p.setColor(QPalette.ColorRole.Highlight, Clr.ACCENT)
    p.setColor(QPalette.ColorRole.HighlightedText, Clr.TEXT_PRIMARY)
    p.setColor(QPalette.ColorRole.ToolTipBase, Clr.BG_RAISED)
    p.setColor(QPalette.ColorRole.ToolTipText, Clr.TEXT_PRIMARY)
    p.setColor(QPalette.ColorRole.PlaceholderText, Clr.TEXT_DISABLED)
    p.setColor(QPalette.ColorRole.Mid, Clr.BORDER_DEFAULT)
    p.setColor(QPalette.ColorRole.Dark, Clr.BG_DEEPEST)
    p.setColor(QPalette.ColorRole.Shadow, QColor(0, 0, 0))
    p.setColor(QPalette.ColorRole.Light, Clr.BG_HOVER)
    p.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.WindowText,
        Clr.TEXT_DISABLED,
    )
    p.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.ButtonText,
        Clr.TEXT_DISABLED,
    )
    return p


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setPalette(make_dark_palette())
    app.setFont(_ui_font(10))

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
