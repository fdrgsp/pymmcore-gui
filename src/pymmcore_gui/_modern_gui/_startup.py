"""Launch dialog: pick a layout and a configuration before the window opens.

Replaces the old "load the last-used config?" message box, which asked about
one thing (the config) and silently decided the other (the layout). Both are
now explicit, remembered, and — crucially — validated: a config file that has
since moved or been deleted is dropped from the list rather than offered and
then failing to load.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pymmcore_gui._layouts import DEFAULT_LAYOUT_NAME, available_layouts
from pymmcore_gui._qt.QtCore import Qt
from pymmcore_gui._qt.QtGui import QPixmap
from pymmcore_gui._qt.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from pymmcore_gui._settings import Settings

from ._theme import theme

if TYPE_CHECKING:
    from collections.abc import Sequence

DEMO_CONFIG: Final = "MMConfig_demo.cfg"
"""Resolved by pymmcore-plus against the Micro-Manager install it finds."""

_DEMO_LABEL: Final = "Demo configuration"
_BROWSE_LABEL: Final = "Browse…"
_BROWSE_ROLE: Final = "__browse__"
_LOGO: Final = Path(__file__).parent.parent / "resources" / "logo_trans.png"
_LOGO_HEIGHT: Final = 96
_MIN_WIDTH: Final = 400
_CFG_FILTER: Final = "Micro-Manager config (*.cfg);;All files (*)"


@dataclass(frozen=True)
class StartupChoice:
    """What the user picked in :class:`StartupDialog`."""

    layout: str = DEFAULT_LAYOUT_NAME
    """A name from ``_layouts.available_layouts()``."""
    config: str | None = None
    """Config to load: a file path, the demo config's name, or None for neither."""


class StartupDialog(QDialog):
    """Logo over two aligned ``label + combo`` fields, and Start / Quit.

    Deliberately owns no application state: it reads the saved layouts and
    recent configs when constructed and hands back a :class:`StartupChoice`.
    Persisting that choice is the caller's job (see ``_app.create_mmgui``).
    """

    def __init__(
        self, parent: QWidget | None = None, *, preselect_layout: str | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("pymmcore-gui")
        self.setModal(True)

        t = theme()
        settings = Settings.instance()

        logo = QLabel()
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if (pixmap := QPixmap(str(_LOGO))) and not pixmap.isNull():
            logo.setPixmap(
                pixmap.scaledToHeight(
                    t.scaled(_LOGO_HEIGHT),
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        self._layout_combo = QComboBox()
        self._layout_combo.addItems(available_layouts())
        # An explicit `-l` wins over what was remembered from last time.
        self._select_layout(preselect_layout or settings.modern_window.last_layout)

        self._config_combo = QComboBox()
        self._fill_config_combo(settings.existing_recent_configs())
        # Browsing isn't a value -- it opens a file dialog and then becomes
        # one. Reverting on cancel needs the index we came from.
        self._last_config_index = self._config_combo.currentIndex()
        self._config_combo.activated.connect(self._on_config_activated)

        # A form layout is what keeps the two labels aligned with each other
        # (and is what the Hardware page's panes already use).
        form = QFormLayout()
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(t.sp_sm)
        form.setVerticalSpacing(t.sp_xs)
        form.addRow("Layout:", self._layout_combo)
        form.addRow("Configuration:", self._config_combo)

        # Both buttons are parented to the dialog at construction, before
        # setDefault: QPushButton.setDefault only registers with the dialog it
        # can find by walking up its parents, so calling it on a parentless
        # button silently does nothing. Qt then falls back to making the first
        # autoDefault button the default -- which is Quit, since it sits left
        # of Start -- and Enter quit the application instead of starting it.
        self._start_btn = QPushButton("Start", self)
        self._start_btn.setProperty("variant", "primary")
        self._start_btn.setDefault(True)
        self._start_btn.clicked.connect(self.accept)
        quit_btn = QPushButton("Quit", self)
        quit_btn.setProperty("variant", "subtle")
        # Belt and braces: without this, Enter *while Quit itself has focus*
        # still activates it, default button or not.
        quit_btn.setAutoDefault(False)
        quit_btn.clicked.connect(self.reject)

        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(quit_btn)
        buttons.addWidget(self._start_btn)

        # Wide enough that a long config file name isn't elided down to
        # nothing on a first launch, where nothing else stretches the dialog.
        self.setMinimumWidth(t.scaled(_MIN_WIDTH))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(t.sp_lg, t.sp_lg, t.sp_lg, t.sp_lg)
        outer.setSpacing(t.sp_lg)
        outer.addWidget(logo)
        outer.addLayout(form)
        outer.addLayout(buttons)

        # Start on the first field rather than letting Qt pick, so Enter
        # reaches the default button (Start) from a predictable place on
        # every platform and binding.
        self._layout_combo.setFocus()

    # ── value ─────────────────────────────────────────────────────

    def value(self) -> StartupChoice:
        """Return the current selection."""
        return StartupChoice(
            layout=self._layout_combo.currentText() or DEFAULT_LAYOUT_NAME,
            config=self._config_combo.currentData(),
        )

    # ── layout field ──────────────────────────────────────────────

    def _select_layout(self, name: str | None) -> None:
        """Preselect *name*, or the first offered layout if it's gone.

        ``available_layouts`` puts "Last session" first when there is one, so
        the fallback is "whatever you had last time" rather than a bare
        default -- quitting never silently discards an arrangement.
        """
        if name and (index := self._layout_combo.findText(name)) >= 0:
            self._layout_combo.setCurrentIndex(index)
        else:
            self._layout_combo.setCurrentIndex(0)

    # ── config field ──────────────────────────────────────────────

    def _fill_config_combo(self, recent: Sequence[Path]) -> None:
        combo = self._config_combo
        combo.addItem(_DEMO_LABEL, DEMO_CONFIG)
        for path in recent:
            combo.addItem(path.name, str(path))
            combo.setItemData(combo.count() - 1, str(path), Qt.ItemDataRole.ToolTipRole)
        combo.insertSeparator(combo.count())
        combo.addItem(_BROWSE_LABEL, _BROWSE_ROLE)
        # The most recent config is the likely answer; demo is the fallback
        # for a first launch, when it's the only real entry anyway.
        combo.setCurrentIndex(1 if recent else 0)

    def _on_config_activated(self, index: int) -> None:
        combo = self._config_combo
        if combo.itemData(index) != _BROWSE_ROLE:
            self._last_config_index = index
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a hardware configuration", "", _CFG_FILTER
        )
        if not path:
            combo.setCurrentIndex(self._last_config_index)
            return
        # Insert above the separator so Browse… stays last, and reuse an
        # existing row for a file that's already listed.
        if (existing := combo.findData(path)) >= 0:
            combo.setCurrentIndex(existing)
        else:
            insert_at = combo.count() - 2
            combo.insertItem(insert_at, Path(path).name, path)
            combo.setItemData(insert_at, path, Qt.ItemDataRole.ToolTipRole)
            combo.setCurrentIndex(insert_at)
        self._last_config_index = combo.currentIndex()
