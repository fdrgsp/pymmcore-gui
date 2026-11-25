"""Write everything the app logs to pymmcore-plus's rotating log file.

pymmcore-plus only writes its own logger there (MMCore's CoreLog goes to a file
next to it); records from the rest of the app would otherwise be lost.
"""

from __future__ import annotations

import logging
import platform
import sys
from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

from pymmcore_plus import configure_logging
from pymmcore_plus._logger import LOG_FILE, current_logfile
from pymmcore_plus._logger import logger as pymm_logger

from pymmcore_gui import __version__

if TYPE_CHECKING:
    from types import TracebackType

# start a new file at LOG_FILE_MB, dropping all but the LOG_BACKUPS newest older ones
LOG_FILE_MB = 10
LOG_BACKUPS = 5
# the app's own stack logs everything; other libraries from INFO
VERBOSE_LOGGERS = (
    "pymmcore_gui",
    "pymmcore-gui",
    "pymmcore_widgets",
    "useq",
    "ome_writers",
    "ndv",
)
# versions recorded when the app starts
PACKAGES = (
    "pymmcore-plus",
    "pymmcore",
    "pymmcore-nano",
    "pymmcore-widgets",
    "useq-schema",
    "ome-writers",
    "ndv",
    "PyQt6",
    "PySide6",
)
# Sentry's excepthook integration reports these already: it ignores this logger
UNCAUGHT_LOGGER = "pymmcore_gui.uncaught"

logger = logging.getLogger(__name__)
# what setup_logging added to the root logger, removed if it runs again
_root_handlers: list[logging.Handler] = []


def _not_pymmcore_plus(record: logging.LogRecord) -> bool:
    """Leave pymmcore-plus records to its own handlers, which write them already."""
    name = record.name
    return name != pymm_logger.name and not name.startswith(f"{pymm_logger.name}.")


class _AppLogHandler(logging.Handler):
    """Hand the app's records to pymmcore-plus's file handler, sharing its rotation."""

    def __init__(self, target: RotatingFileHandler) -> None:
        super().__init__()
        self.target = target
        self.addFilter(_not_pymmcore_plus)

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.INFO or record.name.startswith(VERBOSE_LOGGERS):
            self.target.handle(record)


def setup_logging(file: str | Path | None = LOG_FILE) -> Path | None:
    """Log the whole app to `file`, rotated, and record what is starting.

    Returns the log file, or None when logging to a file is turned off
    (`PYMM_LOG_FILE=0`).
    """
    configure_logging(file=file, file_rotation=LOG_FILE_MB, file_retention=LOG_BACKUPS)
    root = logging.getLogger()
    while _root_handlers:
        root.removeHandler(_root_handlers.pop())
    # Python warnings too, through the "py.warnings" logger
    logging.captureWarnings(True)

    # A handler on the root logger means Python no longer prints the app's
    # warnings to stderr by itself (logging.lastResort): keep doing so.
    if sys.stderr:
        stderr = logging.StreamHandler(sys.stderr)
        stderr.setLevel(logging.WARNING)
        stderr.addFilter(_not_pymmcore_plus)
        stderr.addFilter(lambda record: not getattr(record, "printed", False))
        _add_to_root(stderr)

    if (file_handler := _file_handler()) is None:
        return None
    _add_to_root(_AppLogHandler(file_handler))

    info = [f"python {platform.python_version()}", platform.platform()]
    for pkg in PACKAGES:
        with suppress(PackageNotFoundError):
            info.append(f"{pkg} {version(pkg)}")
    logger.info("pymmcore-gui %s starting: %s", __version__, ", ".join(info))
    return Path(file_handler.baseFilename)


def log_uncaught(
    exc_type: type[BaseException], exc_value: BaseException, tb: TracebackType | None
) -> None:
    """Log an exception the excepthook already printed: only the file gets it."""
    logging.getLogger(UNCAUGHT_LOGGER).error(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, tb),
        extra={"printed": True},
    )


def _add_to_root(handler: logging.Handler) -> None:
    _root_handlers.append(handler)
    logging.getLogger().addHandler(handler)


def _file_handler() -> RotatingFileHandler | None:
    return next(
        (h for h in pymm_logger.handlers if isinstance(h, RotatingFileHandler)), None
    )


def log_file() -> Path | None:
    """Return the file the app logs to, if any."""
    return current_logfile(pymm_logger)
