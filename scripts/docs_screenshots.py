"""Regenerate the screenshots in docs/images from the running app.

    uv run python scripts/docs_screenshots.py

It starts the real app with Micro-Manager's demo devices, drives it, and saves each
widget with `QWidget.grab`, so nothing else on your screen can end up in an image.

Settings go to a temporary folder, so your own settings and layouts are not touched,
and the app starts on its default layout whatever you have arranged.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from pyconify import _cache as pyconify_cache
from pymmcore_plus import CMMCorePlus, find_micromanager

if TYPE_CHECKING:
    from collections.abc import Callable

    from qtpy.QtGui import QImage
    from qtpy.QtWidgets import QWidget

OUTPUT = Path(__file__).parent.parent / "docs" / "images"
WINDOW_SIZE = (1800, 1120)
# wide enough to read on a high-resolution screen, without making huge files
MAX_IMAGE_WIDTH = 1800
STEP_MS = 900
TIMEOUT_S = 180
WAIT = object()  # returned by a step that isn't done yet, to run it again


def main() -> None:
    if not (mm_path := find_micromanager()):
        sys.exit("Install Micro-Manager first: mmcore install")
    # Exits rather than waits, should a dialog open that nobody is there to close.
    watchdog = threading.Timer(TIMEOUT_S, lambda: os._exit(3))
    watchdog.daemon = True
    watchdog.start()

    # Before the home folder moves: device adapters were found above, and icons must
    # keep coming from the cache already there. pyconify reads its environment variable
    # on import, too late for setting that. Each empty cache would fetch every icon
    # again, which soon gets the API to refuse (a "?" in place of each icon).
    pyconify_cache.PYCONIFY_CACHE = str(pyconify_cache.get_cache_directory())
    real_home = str(Path.home())
    home = tempfile.mkdtemp(prefix="pymm-docs-")
    for name in ("HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "XDG_DATA_HOME"):
        os.environ[name] = home

    # After moving the home folder: the settings path is decided on import.
    import useq
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QApplication, QTableWidget

    from pymmcore_gui import _app, _settings
    from pymmcore_gui._channel_settings_store import channel_settings_store
    from pymmcore_gui._theme import DARK_THEME, set_theme
    from pymmcore_gui.widgets._acquire import AcquirePage
    from pymmcore_gui.widgets._panels import PanelKey
    from pymmcore_gui.widgets._preferences import PreferencesDialog
    from pymmcore_gui.widgets._startup import StartupDialog
    from pymmcore_gui.widgets._toolbars import SnapButton
    from pymmcore_gui.widgets.image_preview._snap_album import SnapAlbumPopup

    if home not in str(_settings.SETTINGS_FILE_NAME):
        sys.exit(f"Refusing to touch {_settings.SETTINGS_FILE_NAME}")

    # In the order the app does it: a core made before Qt uses other signals.
    app = _app._QAPP = _app.MMQApplication(sys.argv[:1])
    core = CMMCorePlus.instance()
    core.setDeviceAdapterSearchPaths([mm_path])

    OUTPUT.mkdir(parents=True, exist_ok=True)

    set_theme(DARK_THEME)
    startup = StartupDialog()
    startup.setModal(False)
    startup.show()

    win = _app.create_mmgui(
        mm_config=str(Path(mm_path) / "MMConfig_demo.cfg"),
        mmcore=core,
        install_sentry=False,
        install_sys_excepthook=False,
        exec_app=False,
    )
    page = win.findChild(AcquirePage)
    assert page is not None
    mda = page.mda_widget
    tabs = win._mode_tabs  # type: ignore[attr-defined]
    labels = win.TAB_LABELS  # type: ignore[attr-defined]

    def show_tab(label: str) -> None:
        tabs._select(labels.index(label))

    def save(image: QImage, name: str) -> None:
        if image.width() > MAX_IMAGE_WIDTH:
            image = image.scaledToWidth(MAX_IMAGE_WIDTH, _smooth())
        if not image.save(str(OUTPUT / f"{name}.png")):
            raise OSError(f"Could not write {name}.png")
        print(f"{name}.png  {image.width()} x {image.height()}", flush=True)

    def capture(widget: QWidget, name: str, *, fit: bool = False) -> None:
        """Save `widget`, or with `fit` only as much of it as its contents need."""
        rect = widget.rect()
        if fit:
            rect.setHeight(min(rect.height(), widget.sizeHint().height()))
        save(widget.grab(rect).toImage(), name)

    def hide_home_folder() -> None:
        for table in win._installation.findChildren(QTableWidget):  # type: ignore[attr-defined]
            for row in range(table.rowCount()):
                for column in range(table.columnCount()):
                    item = table.item(row, column)
                    if item is not None and item.text().startswith(real_home):
                        item.setText("~" + item.text()[len(real_home) :])

    def show_mda_result() -> None:
        for dock in page.viewers._records:
            dock.setAsCurrentTab()

    def snap(preset: str, exposure: float) -> None:
        core.setConfig("Channel", preset)
        core.waitForConfig("Channel", preset)
        store = channel_settings_store(core)
        settings = store.get("Channel", preset)
        assert settings is not None
        store.set("Channel", preset, {**settings, "exposure": exposure})
        snap_button = page.findChild(SnapButton)
        assert snap_button is not None
        snap_button.click()

    def set_up_acquisition() -> None:
        mda.setValue(
            useq.MDASequence(
                channels=[
                    {"config": "DAPI", "exposure": 20},
                    {"config": "FITC", "exposure": 50},
                ],
                z_plan={"range": 4, "step": 1},
                time_plan={"interval": 0.2, "loops": 3},
            )
        )
        for axis in ("c", "z", "t"):
            section = mda.tabs.section(axis)
            section.set_checked(True)
            section.set_expanded(axis != "t")

    def show_grid() -> None:
        for axis in ("z", "c"):
            mda.tabs.section(axis).set_expanded(False)
        section = mda.tabs.section("g")
        section.set_checked(True)
        section.set_expanded(True)

    def show_positions() -> None:
        mda.tabs.section("g").set_expanded(False)
        mda.tabs.section("g").set_checked(False)
        section = mda.tabs.section("p")
        section.set_checked(True)
        section.set_expanded(True)

    def capture_album() -> None:
        # Never shown: a popup on the screen would wait for someone to close it.
        popup = SnapAlbumPopup(page.viewers._snap_album)
        popup.resize(300, 260)
        capture(popup, "snap-album")
        popup.deleteLater()

    def capture_preferences() -> None:
        dialog = PreferencesDialog(page, win)
        dialog.setModal(False)
        # shows its placeholder, rather than this machine's temp folder
        dialog._scratch_dir.clear()
        dialog.show()
        state["preferences"] = dialog

    state: dict[str, QWidget] = {}
    steps: list[Callable[[], object]] = [
        lambda: None,
        lambda: capture(startup, "startup-dialog"),
        startup.close,
        lambda: win.resize(*WINDOW_SIZE),
        lambda: show_tab("Installation"),
        lambda: None,
        lambda: None,
        hide_home_folder,
        lambda: capture(win, "installation"),
        lambda: show_tab("Hardware Setup"),
        lambda: None,
        lambda: capture(win, "hardware-setup"),
        lambda: show_tab("Configurations"),
        lambda: None,
        lambda: capture(win, "configurations"),
        lambda: win._configurations._tabs.setCurrentIndex(1),  # type: ignore[attr-defined]
        lambda: None,
        lambda: capture(win, "pixel-configuration"),
        lambda: win._configurations._tabs.setCurrentIndex(0),  # type: ignore[attr-defined]
        lambda: show_tab("Acquire"),
        lambda: None,
        lambda: snap("DAPI", 20),
        lambda: snap("FITC", 50),
        lambda: snap("Rhodamine", 10),
        lambda: None,
        lambda: capture(win, "acquire"),
        lambda: capture(page.panel_widget(PanelKey.CHANNELS), "channels-panel"),
        capture_album,
        capture_preferences,
        lambda: None,
        lambda: capture(state["preferences"], "preferences"),
        lambda: state["preferences"].close(),
        set_up_acquisition,
        lambda: None,
        lambda: capture(page.panel_widget(PanelKey.MDA), "mda-panel"),
        show_grid,
        lambda: None,
        lambda: capture(page.panel_widget(PanelKey.MDA), "mda-grid"),
        show_positions,
        lambda: None,
        lambda: capture(page.panel_widget(PanelKey.MDA), "mda-positions"),
        lambda: (mda.tabs.section("p").set_expanded(False), mda.tabs.section("p").set_checked(False)),
        lambda: mda.control_btns.run_btn.click(),
        lambda: WAIT if core.mda.is_running() else None,
        show_mda_result,
        lambda: None,
        lambda: capture(win, "acquisition"),
        lambda: page.open_panel(PanelKey.STAGES),
        lambda: None,
        lambda: capture(page.panel_widget(PanelKey.STAGES), "stages-panel", fit=True),
        lambda: os._exit(0),
    ]

    def run(index: int = 0) -> None:
        next_index = index if steps[index]() is WAIT else index + 1
        QTimer.singleShot(STEP_MS, lambda: run(next_index))

    QTimer.singleShot(1500, run)
    QApplication.instance().exec()  # type: ignore[union-attr]


def _smooth() -> object:
    from qtpy.QtCore import Qt

    return Qt.TransformationMode.SmoothTransformation


if __name__ == "__main__":
    main()
