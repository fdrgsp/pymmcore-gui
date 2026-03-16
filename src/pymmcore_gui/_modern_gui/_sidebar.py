from __future__ import annotations

from pymmcore_gui._qt.QtCore import (  # type: ignore[attr-defined]
    QEasingCurve,
    QEvent,
    QMimeData,
    QObject,
    QPoint,
    QPropertyAnimation,
    QRect,
    Qt,
    pyqtProperty,  # pyright: ignore
)
from pymmcore_gui._qt.QtGui import (
    QColor,
    QDrag,
    QDragEnterEvent,
    QDragLeaveEvent,
    QDragMoveEvent,
    QDropEvent,
    QEnterEvent,
    QFont,
    QFontMetrics,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPalette,
    QPen,
    QPixmap,
)
from pymmcore_gui._qt.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ._enums import DeviceStatus
from ._theme import ROW_HEIGHT, Sp, mono_font, qcolor, theme, ui_font

RADIUS_LG = 6
SIDEBAR_W = 300

SP_PREFERRED = QSizePolicy.Policy.Preferred
SP_MAXIMUM = QSizePolicy.Policy.Maximum


class CollapsiblePanelHeader(QWidget):
    """
    Custom-painted header bar for a collapsible panel.

    Draws: chevron (▶ / ▼), status dot, title, summary text.
    All via QPainter — no child widgets, no QSS.
    """

    HEADER_HEIGHT = ROW_HEIGHT

    def __init__(
        self,
        title: str,
        summary: str = "",
        status: DeviceStatus = DeviceStatus.CONNECTED,
        show_status_dot: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._summary = summary
        self._status = status
        self._show_dot = show_status_dot
        self._expanded = False
        self._hovered = False
        self._chevron_angle = 0.0  # 0 = collapsed (right), 90 = expanded (down)

        self.setFixedHeight(self.HEADER_HEIGHT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

        # Chevron animation
        self._chevron_anim = QPropertyAnimation(self, b"chevronAngle")
        self._chevron_anim.setDuration(200)
        self._chevron_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    # -- Properties for animation --

    def _get_chevron_angle(self) -> float:
        return self._chevron_angle

    def _set_chevron_angle(self, val: float) -> None:
        self._chevron_angle = val
        self.update()

    chevronAngle = pyqtProperty(float, _get_chevron_angle, _set_chevron_angle)

    # -- Public API --

    @property
    def expanded(self) -> bool:
        """Whether the chevron points down (expanded) or right (collapsed)."""
        return self._expanded

    @expanded.setter
    def expanded(self, val: bool) -> None:
        self._expanded = val
        self._chevron_anim.stop()
        self._chevron_anim.setStartValue(self._chevron_angle)
        self._chevron_anim.setEndValue(90.0 if val else 0.0)
        self._chevron_anim.start()

    @property
    def summary(self) -> str:
        """Short text shown to the right of the title when collapsed."""
        return self._summary

    @summary.setter
    def summary(self, val: str) -> None:
        self._summary = val
        self.update()

    @property
    def status(self) -> DeviceStatus:
        """Device status controlling the colored indicator dot."""
        return self._status

    @status.setter
    def status(self, val: DeviceStatus) -> None:
        self._status = val
        self.update()

    # -- Events --

    def enterEvent(self, event: QEnterEvent | None) -> None:
        self._hovered = True
        self.update()

    def leaveEvent(self, a0: QEvent | None) -> None:
        self._hovered = False
        self.update()

    # -- Painting --

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()

        # Background
        t = theme()
        bg = qcolor(t.bg_raised if self._hovered else t.bg_base)
        p.fillRect(0, 0, w, h, bg)

        # Bottom border
        p.setPen(QPen(qcolor(t.border_subtle), 1))
        p.drawLine(0, h - 1, w, h - 1)

        x = Sp.SM  # running x cursor

        # ── Chevron ──
        p.save()
        p.translate(x + 6, h / 2)
        p.rotate(self._chevron_angle)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qcolor(t.text_disabled))
        # Small right-pointing triangle
        tri = [
            (-3, -4),
            (4, 0),
            (-3, 4),
        ]
        from PyQt6.QtCore import QPointF
        from PyQt6.QtGui import QPolygonF

        poly = QPolygonF([QPointF(tx, ty) for tx, ty in tri])
        p.drawPolygon(poly)
        p.restore()
        x += 12 + Sp.XS

        # ── Status dot ──
        if self._show_dot:
            dot_color = qcolor(
                {
                    DeviceStatus.CONNECTED: t.status_green,
                    DeviceStatus.DISCONNECTED: t.status_red,
                    DeviceStatus.BUSY: t.status_amber,
                    DeviceStatus.ERROR: t.status_red,
                }.get(self._status, t.text_disabled)
            )

            dot_r = 3.5
            dot_cx = x + dot_r
            dot_cy = h / 2

            # Glow
            p.setPen(Qt.PenStyle.NoPen)
            glow = QColor(dot_color)
            glow.setAlphaF(0.3)
            p.setBrush(glow)
            p.drawEllipse(QPointF(dot_cx, dot_cy), dot_r + 2, dot_r + 2)

            # Dot
            p.setBrush(dot_color)
            p.drawEllipse(QPointF(dot_cx, dot_cy), dot_r, dot_r)

            x += int(dot_r * 2) + Sp.XS

        # ── Title ──
        title_font = ui_font(10, QFont.Weight.DemiBold)
        p.setFont(title_font)
        p.setPen(qcolor(t.text_primary))
        title_rect = QRect(x, 0, w - x - Sp.SM, h)
        fm = QFontMetrics(title_font)
        title_width = fm.horizontalAdvance(self._title)
        p.drawText(title_rect, Qt.AlignmentFlag.AlignVCenter, self._title)

        # ── Summary (shown when collapsed, fades based on chevron angle) ──
        if self._summary:
            # Opacity: 1 when chevron=0 (collapsed), 0 when chevron=90 (expanded)
            opacity = 1.0 - (self._chevron_angle / 90.0)
            if opacity > 0.01:
                summary_font = mono_font(8)
                p.setFont(summary_font)
                summary_color = qcolor(t.text_secondary)
                summary_color.setAlphaF(opacity)
                p.setPen(summary_color)

                summary_x = x + title_width + Sp.SM
                summary_rect = QRect(summary_x, 0, w - summary_x - Sp.SM, h)
                p.drawText(
                    summary_rect,
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                    self._summary,
                )

        p.end()


# ═══════════════════════════════════════════════════════════════════
# CollapsiblePanel — header + animated body
# ═══════════════════════════════════════════════════════════════════


class CollapsiblePanel(QWidget):
    """A collapsible panel with an animated body."""

    HEADER_H = CollapsiblePanelHeader.HEADER_HEIGHT

    def __init__(
        self,
        title: str,
        summary: str = "",
        status: DeviceStatus = DeviceStatus.CONNECTED,
        show_status_dot: bool = True,
        expanded: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._expanded = expanded

        # Main layout — zero margins, zero spacing
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        self._header = CollapsiblePanelHeader(
            title=title,
            summary=summary,
            status=status,
            show_status_dot=show_status_dot,
        )
        self._header.installEventFilter(self)
        layout.addWidget(self._header)

        # Body container
        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(Sp.SM, Sp.XS, Sp.SM, Sp.SM)
        self._body_layout.setSpacing(Sp.SM)
        layout.addWidget(self._body)

        # Animation targets the panel's own fixedHeight.
        # This avoids touching body constraints during animation, sidestepping
        # layout-propagation glitches through the sizeHint chain.
        self._anim = QPropertyAnimation(self, b"panelHeight")
        self._anim.setDuration(250)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.finished.connect(self._on_anim_finished)

        if not expanded:
            self._body.setMaximumHeight(0)

        # Set initial state
        if expanded:
            self._header.expanded = True

        # Drag state
        self._drag_start_pos: QPoint | None = None
        self._drag_started = False
        self._drag_highlighted = False

        self.setSizePolicy(SP_PREFERRED, SP_MAXIMUM)

    # -- Animated property: panel's own fixed height --

    def _get_panel_height(self) -> int:
        return self.height()

    def _set_panel_height(self, h: int) -> None:
        self.setFixedHeight(h)
        # Force synchronous parent layout so the scroll area and sibling
        # panels update in the same frame. Safe because we animate the
        # panel's own fixedHeight — no body-sizeHint chain is involved.
        parent = self.parentWidget()
        if parent is not None and (lay := parent.layout()) is not None:
            lay.activate()

    panelHeight = pyqtProperty(int, _get_panel_height, _set_panel_height)

    @property
    def body_layout(self) -> QVBoxLayout:
        """Layout to add content widgets to."""
        return self._body_layout

    @property
    def header(self) -> CollapsiblePanelHeader:
        """The clickable header widget."""
        return self._header

    @property
    def expanded(self) -> bool:
        """Whether the panel body is visible."""
        return self._expanded

    @expanded.setter
    def expanded(self, val: bool) -> None:
        if val == self._expanded:
            return
        self._expanded = val
        self._header.expanded = val
        self._animate(val)

    def toggle(self) -> None:
        """Toggle between expanded and collapsed."""
        self.expanded = not self._expanded

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        """Handle click-to-toggle and drag-to-reorder on the header."""
        if a0 is self._header and isinstance(a1, QMouseEvent):
            etype = a1.type()
            if etype == QEvent.Type.MouseButtonPress:
                self._drag_start_pos = a1.position().toPoint()
                self._drag_started = False
                return True
            if etype == QEvent.Type.MouseMove and self._drag_start_pos is not None:
                pos = a1.position().toPoint()
                if (
                    not self._drag_started
                    and (pos - self._drag_start_pos).manhattanLength()
                    > QApplication.startDragDistance()
                ):
                    self._drag_started = True
                    self._start_drag()
                return True
            if etype == QEvent.Type.MouseButtonRelease:
                if self._drag_start_pos is not None and not self._drag_started:
                    self.toggle()
                self._drag_start_pos = None
                self._drag_started = False
                return True
        return super().eventFilter(a0, a1)

    def _start_drag(self) -> None:
        """Initiate a QDrag for reordering this panel."""
        content = self.parentWidget()
        if isinstance(content, SidebarContent):
            content._drag_source = self

        drag = QDrag(self)
        mime = QMimeData()
        mime.setData("application/x-panel-drag", b"")
        drag.setMimeData(mime)

        # Semi-transparent grab of the header as the drag pixmap
        raw = self._header.grab()
        pixmap = QPixmap(raw.size())
        pixmap.fill(Qt.GlobalColor.transparent)
        p = QPainter(pixmap)
        p.setOpacity(0.7)
        p.drawPixmap(0, 0, raw)
        p.end()
        drag.setPixmap(pixmap)
        if self._drag_start_pos is not None:
            drag.setHotSpot(self._drag_start_pos)

        drag.exec(Qt.DropAction.MoveAction)

        # Clean up
        self._drag_start_pos = None
        self._drag_started = False
        if isinstance(content, SidebarContent):
            content._drag_source = None

    def _animate(self, expanding: bool) -> None:
        self._anim.stop()
        current_h = self.height()

        if expanding:
            # Unconstrain body so we can measure the natural panel height
            self._body.setMaximumHeight(16777215)
            target = self.HEADER_H + self._body.sizeHint().height()
            # Lock at current collapsed height, then animate to target
            self.setFixedHeight(current_h)
            self._anim.setStartValue(current_h)
            self._anim.setEndValue(target)
        else:
            # Lock at current expanded height, then animate to header-only
            self.setFixedHeight(current_h)
            self._anim.setStartValue(current_h)
            self._anim.setEndValue(self.HEADER_H)

        self._anim.start()

    def _on_anim_finished(self) -> None:
        # Remove fixed-height constraint so the layout manages us normally
        self.setMinimumHeight(0)
        self.setMaximumHeight(16777215)
        if not self._expanded:
            self._body.setMaximumHeight(0)

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        """Draw bottom border and optional drag-hover highlight."""
        p = QPainter(self)
        if self._drag_highlighted:
            p.fillRect(self.rect(), QColor(255, 255, 255, 20))
        p.setPen(QPen(qcolor(theme().border_subtle), 1))
        p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        p.end()


class SidebarContent(QWidget):
    """Scroll content widget that supports drag-reorder of CollapsiblePanels."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._drag_source: CollapsiblePanel | None = None
        self._hover_panel: CollapsiblePanel | None = None
        self._drop_above = True

        # Indicator line (positioned absolutely, shown during drag)
        self._drop_line = QFrame(self)
        self._drop_line.setFixedHeight(3)
        self._drop_line.setAutoFillBackground(True)
        line_pal = self._drop_line.palette()
        line_pal.setColor(QPalette.ColorRole.Window, QColor(255, 255, 255))
        self._drop_line.setPalette(line_pal)
        self._drop_line.hide()

    def _panels(self) -> list[CollapsiblePanel]:
        """Return ordered list of CollapsiblePanel children."""
        lay = self.layout()
        if not lay:
            return []
        return [
            w
            for i in range(lay.count())
            if (item := lay.itemAt(i)) is not None
            and isinstance(w := item.widget(), CollapsiblePanel)
        ]

    def _panel_at_pos(self, pos: QPoint) -> CollapsiblePanel | None:
        """Find which panel contains the given position."""
        for panel in self._panels():
            if panel.geometry().contains(pos):
                return panel
        return None

    def _clear_hover(self) -> None:
        if self._hover_panel is not None:
            self._hover_panel._drag_highlighted = False
            self._hover_panel.update()
            self._hover_panel = None
        self._drop_line.hide()

    def dragEnterEvent(self, a0: QDragEnterEvent | None) -> None:
        if a0 is None:
            return
        mime = a0.mimeData()
        if mime is not None and mime.hasFormat("application/x-panel-drag"):
            a0.acceptProposedAction()

    def dragMoveEvent(self, a0: QDragMoveEvent | None) -> None:
        if a0 is None:
            return
        mime = a0.mimeData()
        if mime is None or not mime.hasFormat("application/x-panel-drag"):
            return
        a0.acceptProposedAction()

        pos = a0.position().toPoint()
        panel = self._panel_at_pos(pos)

        # Hovering over the source panel or empty space -> hide indicator
        if panel is None or panel is self._drag_source:
            self._clear_hover()
            return

        # Update highlight on the hovered panel
        if self._hover_panel is not panel:
            self._clear_hover()
            self._hover_panel = panel
            panel._drag_highlighted = True
            panel.update()

        # Top half -> line above, bottom half -> line below
        rect = panel.geometry()
        above = pos.y() < rect.top() + rect.height() // 2
        self._drop_above = above
        line_y = rect.top() - 1 if above else rect.bottom()
        self._drop_line.setGeometry(0, line_y, self.width(), 3)
        self._drop_line.raise_()
        self._drop_line.show()

    def dragLeaveEvent(self, a0: QDragLeaveEvent | None) -> None:
        self._clear_hover()

    def dropEvent(self, a0: QDropEvent | None) -> None:
        if a0 is None:
            return
        mime = a0.mimeData()
        if mime is None or not mime.hasFormat("application/x-panel-drag"):
            return
        a0.acceptProposedAction()

        source = self._drag_source
        target = self._hover_panel
        drop_above = self._drop_above
        self._clear_hover()

        if source is None or target is None or source is target:
            return

        lay = self.layout()
        if not isinstance(lay, QVBoxLayout):
            return

        panels = self._panels()
        src_idx = panels.index(source)
        tgt_idx = panels.index(target)

        insert_idx = tgt_idx if drop_above else tgt_idx + 1
        if src_idx < insert_idx:
            insert_idx -= 1
        if src_idx == insert_idx:
            return

        lay.removeWidget(source)
        lay.insertWidget(insert_idx, source)


class PlaceholderContent(QWidget):
    """Simple placeholder showing lines of text to represent panel content."""

    def __init__(
        self,
        lines: list[str],
        height: int = 80,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Sp.XXS)
        for line in lines:
            lbl = QLabel(line)
            lbl.setFont(mono_font(8))
            # Use palette for color — no stylesheet
            pal = lbl.palette()
            pal.setColor(QPalette.ColorRole.WindowText, qcolor(theme().text_secondary))
            lbl.setPalette(pal)
            layout.addWidget(lbl)


class Sidebar(QWidget):
    """Scrollable sidebar containing collapsible panels."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(SIDEBAR_W)

        # Scroll content
        content = SidebarContent()
        self._layout = QVBoxLayout(content)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)

        # Outer layout holds the scroll area (1px right margin for border)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 1, 0)
        outer.setSpacing(0)
        outer.addWidget(scroll)

        # ── Positioning ──
        pos = CollapsiblePanel(
            title="Positioning",
            summary="74.4, -147.6, 12.3",
            expanded=True,
        )
        pos.body_layout.addWidget(
            PlaceholderContent(
                [
                    "X:    74.40  μm",
                    "Y:  -147.60  μm",
                    "Z:    12.34  μm",
                    "",
                    "Step: [0.1] [1] [10] [100] [1k]",
                    "",
                    "┌─── Joystick ───┐",
                    "│                │",
                    "│       +        │",
                    "│                │",
                    "└────────────────┘",
                ]
            )
        )
        self._layout.addWidget(pos)

        # ── Objective ──
        obj = CollapsiblePanel(
            title="Objective",
            summary="40x Oil 1.30",
        )
        obj.body_layout.addWidget(
            PlaceholderContent(
                [
                    "[4x] [10x] [20x] [40x Oil] [63x] [100x]",
                    "Pixel size: 0.162 μm",
                ]
            )
        )
        self._layout.addWidget(obj)

        # ── Camera ──
        cam = CollapsiblePanel(
            title="Camera",
            summary="100 ms \u00b7 1x1",
        )
        cam.body_layout.addWidget(
            PlaceholderContent(
                [
                    "Exposure:  100 ms",
                    "Gain:      1.0",
                    "Binning:   [1x1] [2x2] [4x4]",
                    "Format:    2048 x 2048 \u00b7 16 bit",
                ]
            )
        )
        self._layout.addWidget(cam)

        # ── Channels ──
        ch = CollapsiblePanel(
            title="Channels",
            summary="DAPI · GFP · Cy5",
        )
        ch.body_layout.addWidget(
            PlaceholderContent(
                [
                    "■ DAPI    50 ms   👁",
                    "■ GFP    100 ms   👁",
                    "■ Cy5    200 ms   👁",
                ]
            )
        )
        self._layout.addWidget(ch)

        # ── Histogram ──
        hist = CollapsiblePanel(
            title="Histogram",
            summary="0 - 4095",
            show_status_dot=False,
        )
        hist.body_layout.addWidget(
            PlaceholderContent(
                [
                    "┌──── histogram ────┐",
                    "│▓▓░░░░░░░░░░░░░░░░│",
                    "└──────────────────-┘",
                    "[⚡ Auto] [↺ Reset] [㏒] [◐]",
                ]
            )
        )
        self._layout.addWidget(hist)

        # ── Acquisition ──
        acq = CollapsiblePanel(
            title="Acquisition",
            summary="Zx50 \u00b7 Tx100",
            show_status_dot=False,
        )
        acq.body_layout.addWidget(
            PlaceholderContent(
                [
                    "☑ Z-Stack      50 sl · 0.5 μm",
                    "☑ Time Series  100 x 30 s",
                    "☐ Tile Scan    —",
                    "☐ Positions    —",
                    "",
                    "Frames:    15,000",
                    "Duration:  50 min",
                    "Storage:   11.2 GB",
                ]
            )
        )
        self._layout.addWidget(acq)

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        """Draw right border."""
        p = QPainter(self)
        p.setPen(QPen(qcolor(theme().border_subtle), 1))
        p.drawLine(self.width() - 1, 0, self.width() - 1, self.height())
        p.end()
