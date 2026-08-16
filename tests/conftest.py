from __future__ import annotations

import os

from pymmcore_gui._settings import Settings

# This is a temporary fix due to a `DeprecationWarning` from the `qtconsole` package:
# """DeprecationWarning: Jupyter is migrating its paths to use standard platformdirs
# given by the platformdirs library.  To remove this warning and
# see the appropriate new directories, set the environment variable
# `JUPYTER_PLATFORM_DIRS=1` and then run `jupyter --paths`.
# The use of platformdirs will be the default in `jupyter_core` v6"""
os.environ["JUPYTER_PLATFORM_DIRS"] = "1"

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from pymmcore_plus import CMMCorePlus, configure_logging
from pymmcore_plus.core import _mmcore_plus

from pymmcore_gui import _app

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest import FixtureRequest

    from pymmcore_gui._qt.QtWidgets import QApplication

TEST_CONFIG = str(Path(__file__).parent / "test_config.cfg")

configure_logging(stderr_level="CRITICAL")


@pytest.fixture(scope="session")
def qapp_cls() -> type[QApplication]:
    return _app.MMQApplication


@pytest.fixture(scope="session", autouse=True)
def _flush_deletion_queue(qapp: QApplication) -> Iterator[None]:
    # Drain deleteLater() objects before interpreter shutdown; PySide6/Windows
    # crashes (exit code 1) if they're flushed after extension modules are torn down.
    yield
    for _ in range(5):
        qapp.processEvents()


# to create a new CMMCorePlus() for every test
@pytest.fixture(autouse=True)
def mmcore() -> Iterator[CMMCorePlus]:
    # Clear the singleton so the new instance auto-registers via __init__
    _mmcore_plus._instance = None
    mmc = CMMCorePlus()
    mmc.loadSystemConfiguration(TEST_CONFIG)
    yield mmc
    mmc.waitForSystem()
    _mmcore_plus._instance = None


# fresh default settings for every test
@pytest.fixture(autouse=True)
def settings() -> Iterator[Settings]:
    settings = Settings()
    with patch("pymmcore_gui._settings._GLOBAL_SETTINGS", settings):
        yield settings


# empty, throw-away layouts directory for every test
@pytest.fixture(autouse=True)
def layouts_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect saved layouts away from the real user data directory.

    Unlike ``Settings``, layouts are plain files that ``_layouts`` writes
    eagerly -- without this, running the tests would litter (and read back)
    the developer's actual ``USER_DATA_DIR/layouts``.
    """
    directory = tmp_path / "layouts"
    monkeypatch.setenv("MMGUI_LAYOUTS_DIR", str(directory))
    return directory


# install the _modern_gui theme/style before every test that has a QApplication
@pytest.fixture(autouse=True)
def _init_gui_theme() -> None:
    """Guarantee the ``_modern_gui`` theme is initialized for every test.

    ``_modern_gui`` widgets — and the shared ``MemoryMDAWidget``/``ThemedStageExplorer``
    used by the legacy main window — call ``theme()`` during construction, which
    raises ``RuntimeError`` unless ``set_theme()`` has run (it installs the
    ``MicroscopeStyle`` and the process-wide scaled-theme view). Only
    ``test_new_gui.py`` calls it explicitly, so a test that builds those widgets
    from another file (e.g. ``test_main_window.py``) used to pass or fail purely
    on collection order. Initializing here removes that dependency and gives
    every test a deterministic dark-theme baseline; tests that need a specific
    theme still call ``set_theme`` themselves.

    This deliberately does *not* depend on the ``qapp`` fixture: forcing a
    ``QApplication`` into existence would break tests that assert none exists yet
    (e.g. ``test_app.test_main_app``). Any test that builds widgets pulls in the
    session-scoped ``qapp`` first, so it is already present when this runs.
    """
    from pymmcore_gui._qt.QtWidgets import QApplication

    if QApplication.instance() is not None:
        from pymmcore_gui._modern_gui._theme import DARK_THEME, reset_zoom, set_theme

        set_theme(DARK_THEME)
        # set_theme() re-applies the *current* zoom step, it doesn't reset it --
        # so a test that calls set_zoom()/zoom_in() etc. would otherwise leak
        # its zoom level into whichever test runs next.
        reset_zoom()


@pytest.fixture()
def check_leaks(request: FixtureRequest, qapp: QApplication) -> Iterator[None]:
    """Run after each test to ensure no widgets have been left around.

    When this test fails, it means that a widget being tested has an issue closing
    cleanly. Perhaps a strong reference has leaked somewhere.  Look for
    `functools.partial(self._method)` or `lambda: self._method` being used in that
    widget's code.
    """
    nbefore = len(qapp.topLevelWidgets())
    failures_before = request.session.testsfailed
    yield
    # if the test failed, don't worry about checking widgets
    if request.session.testsfailed - failures_before:
        return

    remaining = qapp.topLevelWidgets()

    if len(remaining) > nbefore:
        print()
        for r in remaining:
            print(r, r.parent())
        test = f"{request.node.path.name}::{request.node.originalname}"
        raise AssertionError(f"topLevelWidgets remaining after {test!r}: {remaining}")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Explicitly flush Qt state before interpreter shutdown begins.

    pytest-qt's own ``qapp`` fixture creates the QApplication but never tears
    it down (it's a plain ``return``, not a generator) -- destruction is left
    to whatever order CPython's interpreter-shutdown GC happens to run in.
    On windows-latest py3.13PySide6 that has produced a silent
    STATUS_ACCESS_VIOLATION well after pytest's own "N passed" summary, with
    no diagnostics at all (faulthandler is itself torn down by that point in
    shutdown). Closing windows and pumping events here, while still in a
    normal monitored execution context, flushes any pending deferred
    deletions/timers instead of leaving them for uncontrolled GC ordering.
    """
    from pymmcore_gui._qt.QtWidgets import QApplication

    app = QApplication.instance()
    if not isinstance(app, QApplication):
        return
    app.closeAllWindows()
    app.processEvents()


@pytest.fixture(autouse=True)
def _mock_pyconify(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Mock pyconify.svg_path to avoid network requests in tests."""
    svg_dir = tmp_path / "icons"
    svg_dir.mkdir()
    _counter = 0

    def mock_svg_path(*key: str, color: str | None = None, **kwargs: object) -> Path:
        nonlocal _counter
        fill = color or "currentColor"
        svg_content = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
            f'<rect width="24" height="24" fill="{fill}"/></svg>'
        )
        svg_file = svg_dir / f"icon_{_counter}.svg"
        _counter += 1
        svg_file.write_text(svg_content)
        return svg_file

    monkeypatch.setattr(
        "pymmcore_widgets.control._stage_widget.svg_path", mock_svg_path
    )
    monkeypatch.setattr("superqt.iconify.svg_path", mock_svg_path)
    yield
