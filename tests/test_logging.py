from __future__ import annotations

import logging
import warnings
from importlib.metadata import version
from typing import TYPE_CHECKING

import pytest
from pymmcore_plus import CMMCorePlus, configure_logging
from pymmcore_plus._logger import logger as pymm_logger

from pymmcore_gui import _logging
from pymmcore_gui._app import ndv_excepthook

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def app_log(tmp_path: Path) -> Iterator[Path]:
    # logging is process-wide: earlier tests may leave an app logger's level set
    loggers = [logging.getLogger(name) for name in _logging.VERBOSE_LOGGERS]
    levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.NOTSET)
    path = tmp_path / "app.log"
    assert _logging.setup_logging(path) == path
    yield path
    for logger, level in zip(loggers, levels, strict=True):
        logger.setLevel(level)
    root = logging.getLogger()
    while _logging._root_handlers:
        root.removeHandler(_logging._root_handlers.pop())
    logging.captureWarnings(False)
    configure_logging(stderr_level="CRITICAL")  # as conftest.py does


def test_the_whole_app_logs_to_the_file(app_log: Path) -> None:
    logging.getLogger("pymmcore_gui.some_widget").debug("gui detail")
    logging.getLogger("pymmcore_widgets.some_widget").debug("widgets detail")
    logging.getLogger("some_library").debug("library detail")
    logging.getLogger("some_library").info("library news")
    pymm_logger.info("core news")
    # what Python calls to show a warning (the tests turn them into errors)
    warnings.showwarning("a warning", UserWarning, __file__, 1)

    text = app_log.read_text()
    assert "pymmcore-gui" in text and "starting: python" in text
    assert f"pymmcore-plus {version('pymmcore-plus')}" in text
    assert "gui detail" in text
    assert "widgets detail" in text
    assert "library detail" not in text
    assert "library news" in text
    assert text.count("core news") == 1
    assert "UserWarning: a warning" in text


def test_log_drops_its_oldest_entries(app_log: Path) -> None:
    handler = _logging._file_handler()
    assert handler is not None
    assert handler.maxBytes == _logging.LOG_FILE_MB * 1_000_000
    assert handler.backupCount == _logging.LOG_BACKUPS

    handler.maxBytes = 500
    for i in range(200):
        logging.getLogger("pymmcore_gui").info("entry %d", i)
    files = sorted(app_log.parent.glob("app.log*"))
    assert len(files) == 1 + _logging.LOG_BACKUPS
    assert "entry 199" in app_log.read_text()
    assert "entry 0\n" not in "".join(f.read_text() for f in files)


def test_log_rotates_while_a_core_writes_to_it(
    app_log: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """MMCore writes its CoreLog to the same file, from C++."""
    core = CMMCorePlus()
    handler = _logging._file_handler()
    assert handler is not None
    handler.maxBytes = 2000
    for i in range(300):
        core.logMessage(f"core entry {i}")
        logging.getLogger("pymmcore_gui").info("gui entry %d", i)
    # only Python records rotate the file: this one lands after the last rotation
    core.logMessage("after rotating")
    core.setPrimaryLogFile("")  # flushes MMCore's asynchronous log

    assert len(list(app_log.parent.glob("app.log.*"))) == _logging.LOG_BACKUPS
    text = app_log.read_text()
    assert "gui entry 299" in text
    assert "after rotating" in text
    assert "Logging error" not in capsys.readouterr().err


def test_uncaught_exceptions_are_logged(
    app_log: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    try:
        raise ValueError("boom")
    except ValueError as e:
        ndv_excepthook(ValueError, e, e.__traceback__)

    text = app_log.read_text()
    assert f"[ERROR,{_logging.UNCAUGHT_LOGGER}] Uncaught exception" in text
    assert "ValueError: boom" in text
    # the excepthook prints its own traceback: the record is not echoed
    assert "Uncaught exception" not in capsys.readouterr().err
