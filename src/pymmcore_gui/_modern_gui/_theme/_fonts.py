from __future__ import annotations

from pymmcore_gui._qt.QtGui import QFont, QFontDatabase


def ui_font(size_pt: float = 10, weight: int = QFont.Weight.Normal) -> QFont:
    """System UI font."""
    f = QFont()
    f.setPointSizeF(size_pt)
    f.setWeight(weight)
    return f


DEFAULT_FAMILIES = [
    "Fira Code",
    "Source Code Pro",
    "JetBrains Mono",
    "SF Mono",
    "Cascadia Code",
    "Fira Code",
    "Consolas",
]


def mono_font(size_pt: float = 10, weight: int = QFont.Weight.Normal) -> QFont:
    """Monospace font."""
    for fam in DEFAULT_FAMILIES:
        if fam in QFontDatabase.families():
            f = QFont(fam)
            f.setPointSizeF(size_pt)
            f.setWeight(weight)
            return f
    f = QFont()
    f.setStyleHint(QFont.StyleHint.Monospace)
    f.setPointSizeF(size_pt)
    f.setWeight(weight)
    return f
