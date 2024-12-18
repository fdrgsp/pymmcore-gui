"""A Micro-Manager GUI based on pymmcore-widgets and pymmcore-plus."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pymmcore-gui")
except PackageNotFoundError:
    __version__ = "uninstalled"

from ._main_window import MicroManagerGUI

__all__ = ["MicroManagerGUI"]
