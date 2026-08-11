"""Registry of dockable Acquire-page panels.

Adding a new panel -- from ``pymmcore_widgets`` or anywhere else -- is a
single :class:`PanelInfo` entry in :data:`PANELS`; ``AcquirePage`` builds its
docks, toolbar buttons, and layout persistence entirely from this list. This
module holds only data and factory functions: no ``QWidget`` subclasses, no
docking logic, no pydantic. See ``AcquirePage._create_panel`` for how a
``PanelInfo`` becomes a dock.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from pymmcore_gui._qt.QtAds import DockWidgetArea
from pymmcore_gui.actions.widget_actions import (
    create_exception_log,
    create_property_browser,
)

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

    from pymmcore_gui._qt.QtWidgets import QWidget

PanelFactory = Callable[["QWidget", "CMMCorePlus"], "QWidget"]


class PanelKey:
    """Stable string keys for each registered panel.

    Used as dock ``objectName`` suffixes (see :attr:`PanelInfo.dock_name`)
    and persisted verbatim in settings -- do not rename an existing value.
    """

    MDA: Final = "mda"
    PRESETS: Final = "presets"
    PROPERTIES: Final = "properties"
    CONSOLE: Final = "console"
    EXCEPTION_LOG: Final = "exception_log"


def _ignoring_core(fn: Callable[[QWidget], QWidget]) -> PanelFactory:
    """Adapt a ``widget_actions`` factory, which resolves the core itself."""

    def _create(parent: QWidget, _core: CMMCorePlus) -> QWidget:
        return fn(parent)

    return _create


def _create_mda(_parent: QWidget, core: CMMCorePlus) -> QWidget:
    from pymmcore_gui.widgets._mda_widget import MemoryMDAWidget

    return MemoryMDAWidget(mmcore=core)


def _create_presets(_parent: QWidget, core: CMMCorePlus) -> QWidget:
    from ._acquire_presets import AcquisitionPresetSelector

    return AcquisitionPresetSelector(mmcore=core)


def _create_console(_parent: QWidget, core: CMMCorePlus) -> QWidget:
    # local import: keeps IPython/qtconsole out of startup for users who
    # never open the console panel.
    from pymmcore_gui.widgets._mm_console import MMConsole

    return MMConsole(mmcore=core)


def _refresh_mda(widget: QWidget) -> None:
    with suppress(RuntimeError):
        widget.refresh_channel_table()  # type: ignore[attr-defined]


def _refresh_presets(widget: QWidget) -> None:
    with suppress(RuntimeError):
        widget.refresh()  # type: ignore[attr-defined]


def _refresh_property_browser(widget: QWidget) -> None:
    # PropertyBrowser exposes no public refresh; rebuild its table directly
    # (guarded, in case the internals change). The widget itself already
    # handles systemConfigurationLoaded.
    with suppress(RuntimeError):
        fn = getattr(getattr(widget, "_prop_table", None), "_rebuild_table", None)
        if callable(fn):
            with suppress(Exception):
                fn()


@dataclass(frozen=True)
class PanelInfo:
    """Metadata describing one dockable Acquire-page panel."""

    key: str
    title: str
    icon: str
    tooltip: str
    create: PanelFactory
    area: DockWidgetArea = DockWidgetArea.RightDockWidgetArea
    default_open: bool = False
    unstyle: bool = False
    refresh: Callable[[QWidget], None] | None = None
    always_visible: bool = False
    """If True, this panel's toolbar button can't be hidden from the customize menu."""

    @property
    def dock_name(self) -> str:
        """ADS ``objectName``. Must never change: ``restoreState`` matches by it."""
        return f"acquire_{self.key}"


PANELS: Final[tuple[PanelInfo, ...]] = (
    PanelInfo(
        key=PanelKey.MDA,
        title="MDA",
        icon="qlementine-icons:cube-16",
        tooltip="MDA — show or hide the multi-dimensional acquisition panel",
        create=_create_mda,
        area=DockWidgetArea.LeftDockWidgetArea,
        default_open=True,
        refresh=_refresh_mda,
        always_visible=True,
    ),
    PanelInfo(
        key=PanelKey.PRESETS,
        title="Groups and Presets",
        icon="mdi:format-list-group",
        tooltip="Groups and Presets — show or hide the group/preset panel",
        create=_create_presets,
        refresh=_refresh_presets,
    ),
    PanelInfo(
        key=PanelKey.PROPERTIES,
        title="Properties",
        icon="mdi-light:format-list-bulleted",
        tooltip="Properties — open the device property browser panel",
        create=_ignoring_core(create_property_browser),
        unstyle=True,
        refresh=_refresh_property_browser,
    ),
    PanelInfo(
        key=PanelKey.CONSOLE,
        title="Console",
        icon="iconoir:terminal",
        tooltip="Console — open an IPython console panel",
        create=_create_console,
    ),
    PanelInfo(
        key=PanelKey.EXCEPTION_LOG,
        title="Exception Log",
        icon="mdi-light:alert",
        tooltip="Exception Log — show or hide the exception log panel",
        create=_ignoring_core(create_exception_log),
        unstyle=True,
    ),
)

PANELS_BY_KEY: Final[Mapping[str, PanelInfo]] = {p.key: p for p in PANELS}
