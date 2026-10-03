"""The Smart Microscopy tab: event-driven acquisitions steered by a Python script."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from pymmcore_plus.smart import ScriptError, SmartRunError, dry_run, inspect_script
from superqt.iconify import QIconifyIcon

from pymmcore_gui._array_viewer import set_source_icon
from pymmcore_gui._ndv_viewers import AcquireViewersManager
from pymmcore_gui._qt.QtAds import CDockManager
from pymmcore_gui._qt.QtCore import QEvent, QFileSystemWatcher, QSize, Qt, QUrl, Signal
from pymmcore_gui._qt.QtGui import QDesktopServices
from pymmcore_gui._qt.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QWidget,
)
from pymmcore_gui._run_owner import RunOwner, RunOwnership
from pymmcore_gui._settings import Settings
from pymmcore_gui._theme import qcolor, theme
from pymmcore_gui.widgets._smart._bridge import SmartController
from pymmcore_gui.widgets._smart._mda import SmartMDAWidget
from pymmcore_gui.widgets._smart._monitor import SmartMonitor
from pymmcore_gui.widgets._smart._script_panel import ScriptPanel
from pymmcore_gui.widgets._tab_page import TabPage
from pymmcore_gui.widgets._toolbars import toolbar_separator

if TYPE_CHECKING:
    import ndv
    import numpy as np
    import useq
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.mda import SingleOutput
    from pymmcore_plus.smart import HookResult

TEMPLATES_DIR: Final = Path(__file__).parents[2] / "resources" / "smart_templates"

_MDA_COLUMN_WIDTH: Final = 560
_SCRIPT_COLUMN_WIDTH: Final = 420


def open_in_editor(path: Path) -> None:
    """Open *path* in a text editor -- never with the default ``.py`` handler.

    The default application for ``.py`` files can be an interpreter launcher
    (macOS "Python Launcher", Windows ``py.exe``) that would *run* the script.
    """
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-t", str(path)])
    elif sys.platform == "win32":
        try:
            os.startfile(str(path), "edit")  # type: ignore[attr-defined]
        except OSError:
            subprocess.Popen(["notepad.exe", str(path)])
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def list_templates() -> list[tuple[str, Path]]:
    """(name, path) of each bundled script template, sorted by name."""
    templates = []
    for path in sorted(TEMPLATES_DIR.glob("*.py")):
        try:
            templates.append((inspect_script(path).name, path))
        except ScriptError:  # pragma: no cover - templates are tested
            continue
    return sorted(templates)


class SmartMicroscopyPage(TabPage):
    """Base acquisition, analysis script, live view and run monitor.

    Layout::

        toolbar
        ┌──────────────┬──────────────┬──────────────────────────┐
        │ base MDA     │ script &     │ live viewer of the run   │
        │ (Run, Pause, │ run settings ├──────────────────────────┤
        │  Cancel)     │              │ monitor: frames + log    │
        └──────────────┴──────────────┴──────────────────────────┘
    """

    mdaRunningChanged = Signal(bool)
    """Relays the MDA lock, like ``AcquirePage.mdaRunningChanged``."""
    analysisError = Signal(str)
    """A user-facing error from the analysis side (for a toast)."""

    _prepared = Signal(object)  # worker thread -> GUI thread, see _start_run
    _tested = Signal(object)  # worker thread -> GUI thread, see _test_on_image

    def __init__(
        self,
        mmcore: CMMCorePlus,
        parent: QWidget | None = None,
        *,
        run_ownership: RunOwnership | None = None,
    ) -> None:
        super().__init__(parent)
        self._mmc = mmcore
        # Without a window-wide RunOwnership (page used on its own), a private
        # one still lets this page claim its runs for its own viewers.
        self._ownership = run_ownership or RunOwnership(mmcore, self)
        self._locked = False
        self._starting = False
        self._base_sequence: useq.MDASequence | None = None

        self.controller = SmartController(mmcore, self)
        self.controller.runFinished.connect(self._on_run_finished)
        self.controller.analysisError.connect(self._on_analysis_error)
        self._prepared.connect(self._on_prepared)
        self._tested.connect(self._on_tested)

        # ── left: base acquisition ──────────────────────────────────
        self.mda = SmartMDAWidget(mmcore)
        self.mda.set_launcher(self._start_run)
        self.mda.mdaLockChanged.connect(self.set_mda_lock)
        self.left.add_widget(self.mda, stretch=1)

        # ── middle: script and run settings ─────────────────────────
        self.script_panel = ScriptPanel()
        self.script_panel.scriptChanged.connect(self._on_script_changed)
        self.script_panel.settingsChanged.connect(self._remember_script)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.script_panel)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        # Wrap instead of scrolling sideways: a sideways scroll hid the
        # status badge and clipped the description.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(320)

        # ── right: live viewer above the monitor ────────────────────
        self._viewer_dock_manager = CDockManager()
        self.viewers = AcquireViewersManager(
            self._viewer_dock_manager,
            mmcore,
            parent=self,
            accepts_run=partial(self._ownership.accepts, RunOwner.SMART),
            title_prefix=self._viewer_title_prefix,
        )
        self.viewers.mdaViewerCreated.connect(self._on_viewer_created)
        self.viewers.mdaViewerClosed.connect(self._on_viewer_closed)
        self._viewer_placeholder = QLabel(
            "The live view of a smart run appears here when it starts."
        )
        self._viewer_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._viewer_placeholder.setEnabled(False)
        self._viewer_placeholder.setMinimumHeight(240)
        self._viewer_stack = QStackedWidget()
        self._viewer_stack.addWidget(self._viewer_placeholder)
        self._viewer_stack.addWidget(self._viewer_dock_manager)

        self.monitor = SmartMonitor()
        self.monitor.bind(self.controller)

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self._viewer_stack)
        right.addWidget(self.monitor)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)
        right.setCollapsible(0, False)
        # Explicit, or the (small) placeholder's size hint decides and the
        # viewer opens as a thin strip above a tall monitor.
        right.setSizes([600, 400])
        self.monitor.setMinimumHeight(160)

        self._content_split = QSplitter(Qt.Orientation.Horizontal)
        self._content_split.addWidget(scroll)
        self._content_split.addWidget(right)
        self._content_split.setStretchFactor(0, 0)
        self._content_split.setStretchFactor(1, 1)
        self._content_split.setSizes([_SCRIPT_COLUMN_WIDTH, 900])
        self._content_split.setCollapsible(0, False)
        self.add_content_widget(self._content_split)
        # The editor is unusable when squeezed; the viewer's minimum width
        # must be taken from the right-hand columns instead.
        self.left.setMinimumWidth(480)
        self._h_split.setCollapsible(0, False)
        self._h_split.setSizes([_MDA_COLUMN_WIDTH, 1300])

        # ── toolbar ─────────────────────────────────────────────────
        self._load_btn = self._tool_button(
            "Load script…", "material-symbols:folder-open-outline-rounded"
        )
        self._load_btn.clicked.connect(self.prompt_load_script)
        self._recent_menu = QMenu(self._load_btn)
        self._recent_menu.aboutToShow.connect(self._fill_recent_menu)
        self._load_btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._load_btn.customContextMenuRequested.connect(
            lambda pos: self._recent_menu.exec(self._load_btn.mapToGlobal(pos))
        )
        self._reload_btn = self._tool_button(
            "Reload", "material-symbols:refresh-rounded"
        )
        self._reload_btn.setToolTip("Re-read the script from disk")
        self._reload_btn.clicked.connect(self.reload_script)
        self._template_btn = self._tool_button(
            "New from template", "material-symbols:note-add-outline-rounded"
        )
        template_menu = QMenu(self._template_btn)
        for name, path in list_templates():
            action = template_menu.addAction(name)
            if action is not None:
                action.triggered.connect(partial(self.new_from_template, path))
        self._template_btn.setMenu(template_menu)
        self._editor_btn = self._tool_button(
            "Open in editor", "material-symbols:edit-document-outline-rounded"
        )
        self._editor_btn.clicked.connect(self._open_script_in_editor)
        self._test_btn = self._tool_button(
            "Test on last image", "material-symbols:science-outline-rounded"
        )
        self._test_btn.setToolTip(
            "Run analyze() once on the most recent image (snap one first). "
            "Shows what it would request; nothing is acquired."
        )
        self._test_btn.clicked.connect(self.test_on_last_image)
        self._stop_btn = self._tool_button(
            "Stop after current", "material-symbols:stop-circle-outline-rounded"
        )
        self._stop_btn.setToolTip(
            "Finish the run after the event being acquired; nothing else that is "
            "queued is acquired."
        )
        self._stop_btn.clicked.connect(self.controller.request_stop)

        for widget in (
            self._load_btn,
            self._reload_btn,
            self._template_btn,
            self._editor_btn,
        ):
            self.toolbar.add_widget(widget)
        self.toolbar.add_widget(toolbar_separator())
        self.toolbar.add_widget(self._test_btn)
        self.toolbar.add_stretch()
        self.toolbar.add_widget(self._stop_btn)
        self._apply_toolbar_icons()

        # ── script file watching ────────────────────────────────────
        self._watcher = QFileSystemWatcher(self)
        self._watcher.fileChanged.connect(self._on_script_file_changed)

        self._update_controls()
        self._restore_last_script()

    # ------------------------------------------------------------ public API

    @property
    def run_ownership(self) -> RunOwnership:
        return self._ownership

    def load_script(self, path: str | Path) -> bool:
        """Load *path* with the settings remembered for it; return validity."""
        saved = Settings.instance().smart_microscopy.settings_for(path)
        ok = self.script_panel.load(path, saved)
        self._watch(self.script_panel.path)
        self._remember_script()
        return ok

    def reload_script(self) -> bool:
        ok = self.script_panel.reload()
        self._watch(self.script_panel.path)
        return ok

    def prompt_load_script(self) -> None:
        start = str(self.script_panel.path.parent) if self.script_panel.path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load analysis script", start, "Python scripts (*.py)"
        )
        if path:
            self.load_script(path)

    def new_from_template(self, template: Path) -> Path | None:
        """Copy *template* to a location the user picks, load and open it."""
        start = self.script_panel.path.parent if self.script_panel.path else Path.home()
        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Save new script as",
            str(start / template.name),
            "Python scripts (*.py)",
        )
        if not dest:
            return None
        dest_path = Path(dest)
        if dest_path.suffix != ".py":
            dest_path = dest_path.with_suffix(".py")
        shutil.copyfile(template, dest_path)
        if self.load_script(dest_path):
            open_in_editor(dest_path)
        return dest_path

    def set_mda_lock(self, locked: bool) -> None:
        """Freeze the settings while any acquisition owns the hardware."""
        if locked == self._locked:
            return
        self._locked = locked
        self._update_controls()
        self.mdaRunningChanged.emit(locked)

    def cancel_acquisition(self) -> None:
        """Cancel the running acquisition, with the editor's usual feedback."""
        if self.controller.is_active():
            self.controller.cancel()
        self.mda.cancel_acquisition()

    def shutdown(self) -> None:
        """Stop any run and the analysis worker (the window is closing)."""
        self.controller.shutdown()
        self._remember_script()

    # ----------------------------------------------------------- running

    def _start_run(
        self, sequence: useq.MDASequence, output: SingleOutput | None
    ) -> None:
        """Launcher for the MDA editor's Run button.

        Starting the analysis worker can take seconds (process mode), so it
        happens on a background thread behind the editor's progress overlay;
        `_on_prepared` then starts the acquisition on the GUI thread.
        """
        if self._starting or self.controller.is_active():
            self.mda.launch_failed()
            return
        if not self.reload_script() or self.script_panel.spec is None:
            self.mda.launch_failed()
            QMessageBox.warning(
                self,
                "No valid script",
                "Load a valid analysis script before starting a smart run.",
            )
            return
        if next(iter(sequence), None) is None:
            self.mda.launch_failed()
            QMessageBox.warning(
                self,
                "Nothing to acquire",
                "The base acquisition contains no events. Enable at least one "
                "channel (or position) so there is a first frame to analyze.",
            )
            return

        config = self.script_panel.run_config()
        self._starting = True
        self._update_controls()
        self.mda.show_busy(
            "Starting analysis process…"
            if config.execution == "process"
            else "Loading analysis script…"
        )
        self._remember_script()

        def _prepare() -> None:
            try:
                self.controller.prepare(sequence, config, output=output)
                outcome: object = None
            except Exception as e:  # reported on the GUI thread
                outcome = e
            self._prepared.emit((outcome, sequence))

        threading.Thread(target=_prepare, name="smart-prepare", daemon=True).start()

    def _on_prepared(self, payload: tuple[Any, ...]) -> None:
        error, sequence = payload
        self._starting = False
        self.mda.hide_busy()
        if error is None:
            try:
                self._ownership.claim(RunOwner.SMART)
                self._base_sequence = sequence
                self.controller.start()
            except (SmartRunError, RuntimeError) as e:
                self._ownership.release()
                self.controller.abandon()
                error = e
        if error is not None:
            self.mda.launch_failed()
            self._update_controls()
            QMessageBox.critical(self, "Smart run not started", str(error))
            return
        self._update_controls()

    def _on_run_finished(self, summary: dict[str, Any]) -> None:
        self._update_controls()
        window = self.window()
        status = window.statusBar() if isinstance(window, QMainWindow) else None
        if status is not None:
            text = str(summary.get("status", "")).replace("_", " ")
            status.showMessage(
                f"Smart run {text}: {summary.get('frames', 0)} frames. "
                f"Records in {summary.get('run_dir')}",
                10_000,
            )

    def _on_analysis_error(self, message: str, fatal: bool) -> None:
        first_line = message.strip().splitlines()[0] if message.strip() else message
        self.analysisError.emit(
            ("Analysis worker failed: " if fatal else "Analysis error: ") + first_line
        )

    # --------------------------------------------------------- test on image

    def test_on_last_image(self) -> None:
        """Run ``analyze`` once on the latest image; acquire nothing."""
        if self.script_panel.spec is None and not self.reload_script():
            QMessageBox.warning(self, "No valid script", "Load a valid script first.")
            return
        image = self._last_image()
        if image is None:
            QMessageBox.information(
                self,
                "No image yet",
                "Snap an image first (Snap on the Acquire tab), then test again.",
            )
            return
        self.reload_script()
        config = self.script_panel.run_config()
        self._test_btn.setEnabled(False)
        self.mda.show_busy("Testing the script…")
        core = self._mmc

        def _run() -> None:
            # Never raises: the outcome is shown on the GUI thread.
            try:
                outcome: HookResult | str = dry_run(config, image, core=core)
            except Exception as e:
                outcome = f"The test did not complete: {e}"
            self._tested.emit(outcome)

        threading.Thread(target=_run, name="smart-test", daemon=True).start()

    def _on_tested(self, outcome: HookResult | str) -> None:
        self.mda.hide_busy()
        self._update_controls()
        box = QMessageBox(self)
        box.setWindowTitle("Test on last image")
        if isinstance(outcome, str):
            box.setIcon(QMessageBox.Icon.Critical)
            box.setText(outcome)
        elif not outcome.ok:
            box.setIcon(QMessageBox.Icon.Critical)
            box.setText("analyze() raised an exception.")
            box.setDetailedText(outcome.error or "")
        else:
            box.setIcon(QMessageBox.Icon.Information)
            box.setText(_describe_test_result(outcome))
            response = outcome.response
            if response is not None and response.events:
                box.setDetailedText(
                    "\n".join(
                        e.model_dump_json(exclude={"sequence"}, exclude_none=True)
                        for e in response.events
                    )
                )
        box.exec()

    def _last_image(self) -> np.ndarray | None:
        for getter in (self._mmc.getImage, self._mmc.getLastImage):
            with suppress(Exception):
                image = getter()
                if image is not None and image.size:
                    return image
        return None

    # ------------------------------------------------------------- viewers

    def _viewer_title_prefix(self) -> str:
        spec = self.controller.config.spec if self.controller.config else None
        return f"Smart {spec.name}" if spec else "Smart"

    def _on_viewer_created(self, viewer: ndv.ArrayViewer) -> None:
        self._viewer_stack.setCurrentWidget(self._viewer_dock_manager)
        base = self._base_sequence
        if base is not None:
            # "Re-use MDA…" loads this run's *base* sequence back into the
            # editor (the runner itself only saw an event iterator).
            viewer.mda_sequence = base  # type: ignore[attr-defined]
            viewer._reuse_mda_callback = partial(self.mda.setValue, base)  # type: ignore[attr-defined]

    def _on_viewer_closed(self, *_: object) -> None:
        if not self.viewers._records:
            self._viewer_stack.setCurrentWidget(self._viewer_placeholder)

    # --------------------------------------------------------------- script

    def _on_script_changed(self, spec: object) -> None:
        self._update_controls()

    def _remember_script(self) -> None:
        if (path := self.script_panel.path) is None or self.script_panel.spec is None:
            return
        settings = Settings.instance()
        settings.smart_microscopy.remember_script(path, self.script_panel.settings())
        settings.flush()

    def _restore_last_script(self) -> None:
        last = Settings.instance().smart_microscopy.last_script
        if last is not None and last.is_file():
            self.load_script(last)

    def _fill_recent_menu(self) -> None:
        self._recent_menu.clear()
        recent = [
            p
            for p in Settings.instance().smart_microscopy.recent_scripts
            if p.is_file()
        ]
        if not recent:
            action = self._recent_menu.addAction("No recent scripts")
            if action is not None:
                action.setEnabled(False)
            return
        for path in recent:
            action = self._recent_menu.addAction(path.name)
            if action is not None:
                action.setToolTip(str(path))
                action.triggered.connect(partial(self.load_script, path))

    def _open_script_in_editor(self) -> None:
        if (path := self.script_panel.path) is not None:
            open_in_editor(path)

    def _watch(self, path: Path | None) -> None:
        if files := self._watcher.files():
            self._watcher.removePaths(files)
        if path is not None and path.is_file():
            self._watcher.addPath(str(path))

    def _on_script_file_changed(self, path: str) -> None:
        # Editors that save atomically replace the file, dropping the watch.
        if Path(path).is_file() and path not in self._watcher.files():
            self._watcher.addPath(path)
        self.script_panel.set_modified(True)

    # -------------------------------------------------------------- chrome

    def _update_controls(self) -> None:
        busy = self._locked or self._starting
        smart_running = self.controller.is_active()
        has_path = self.script_panel.path is not None
        for widget in (self._load_btn, self._template_btn):
            widget.setEnabled(not busy)
        self._reload_btn.setEnabled(not busy and has_path)
        self._editor_btn.setEnabled(has_path)
        self._test_btn.setEnabled(not busy and self.script_panel.spec is not None)
        self._stop_btn.setEnabled(smart_running and self._locked)
        self.script_panel.set_locked(busy)

    def _tool_button(self, text: str, icon: str) -> QPushButton:
        button = QPushButton(text)
        button.setProperty("variant", "subtle")
        button.setProperty("_smart_icon", icon)
        return button

    def _apply_toolbar_icons(self) -> None:
        color = qcolor(theme().text_secondary).name()
        size = theme().scaled(16)
        for button in self.toolbar.findChildren(QPushButton):
            if icon := button.property("_smart_icon"):
                set_source_icon(button, QIconifyIcon(icon, color=color))
                button.setIconSize(QSize(size, size))

    def changeEvent(self, a0: QEvent | None) -> None:
        if a0 is not None and a0.type() == QEvent.Type.StyleChange:
            self._apply_toolbar_icons()
        super().changeEvent(a0)


def _describe_test_result(result: HookResult) -> str:
    lines = [f"analyze() finished in {result.duration_ms:.1f} ms."]
    response = result.response
    if response is None or (not response.events and not response.stop):
        lines.append("It requested nothing (the acquisition would continue unchanged).")
    else:
        if response.events:
            lines.append(
                f"It requested {len(response.events)} event(s), "  # type: ignore[arg-type]
                f"priority '{response.priority}', timing '{response.timing}'."
            )
        if response.drop_base:
            lines.append("It would discard the remaining base events.")
        if response.stop:
            lines.append("It would stop the run.")
    if result.records:
        lines.append(
            "Recorded: " + ", ".join(f"{k}={v}" for k, v in result.records.items())
        )
    if result.logs:
        lines.append("Log:\n" + "\n".join(f"[{lvl}] {msg}" for lvl, msg in result.logs))
    return "\n".join(lines)
