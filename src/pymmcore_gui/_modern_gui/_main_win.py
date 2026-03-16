"""
Collapsible Panel Sidebar — Qt Implementation.

Scrollable left panel with collapsible sections, matching the HTML mockup.
No QSS. Only QPalette + QProxyStyle(Fusion) + custom QWidget painting.

Usage:
    python collapsible_panels.py
"""

from __future__ import annotations

import sys

from pymmcore_plus import CMMCorePlus

from pymmcore_gui._modern_gui._sidebar import Sidebar
from pymmcore_gui._qt.QtGui import (
    QColor,
    QPalette,
)
from pymmcore_gui._qt.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QSizePolicy,
    QWidget,
)

from ._theme import MicroscopeStyle, make_dark_palette, ui_font

# ═══════════════════════════════════════════════════════════════════
# Placeholder content widgets (just labels for now)
# ═══════════════════════════════════════════════════════════════════

SP_EXPANDING = QSizePolicy.Policy.Expanding


class MainWindow(QMainWindow):
    """Top-level window with sidebar + viewport."""

    def __init__(self, *, mmcore: CMMCorePlus | None = None) -> None:
        super().__init__()

        # MOVE ME
        if isinstance(qapp := QApplication.instance(), QApplication):
            qapp.setStyle(MicroscopeStyle())
            qapp.setPalette(make_dark_palette())
            qapp.setFont(ui_font(10))

        self._mmc = mmcore or CMMCorePlus.instance()
        self.setWindowTitle("Microscope Control — Panel Mockup")
        self.resize(900, 700)

        central = QWidget()
        self.setCentralWidget(central)

        # Viewport placeholder
        viewport = QWidget()
        viewport.setSizePolicy(SP_EXPANDING, SP_EXPANDING)

        # Paint the viewport area black
        pal = viewport.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(0, 0, 0))
        viewport.setPalette(pal)
        viewport.setAutoFillBackground(True)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(Sidebar())
        layout.addWidget(viewport)

    @property
    def mmcore(self) -> CMMCorePlus | None:
        """Access to the microscope core, if provided."""
        return self._mmc


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    """Launch the mockup application."""
    app = QApplication(sys.argv)

    # Apply our style and palette
    app.setStyle(MicroscopeStyle())
    app.setPalette(make_dark_palette())

    # Global font
    app.setFont(ui_font(10))

    win = MainWindow()
    win.show()
    win.resize(900, 500)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
