from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._main_win import MainWindow

__all__ = ["MainWindow"]


def __getattr__(name: str) -> object:
    # Import MainWindow lazily so that importing a leaf module such as
    # ``pymmcore_gui._modern_gui._theme`` does not eagerly pull in ``_main_win``
    # (and through it ``_acquire`` and the feature widgets). Those widgets import
    # ``_modern_gui._theme`` themselves, so an eager import here creates an import
    # cycle. ``importlib``/``getattr``-based resolution (e.g. the CLI's
    # ``"pymmcore_gui._modern_gui.MainWindow"``) still works through this hook.
    if name == "MainWindow":
        from ._main_win import MainWindow

        return MainWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
