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
    STAGE_EXPLORER: Final = "stage_explorer"
    PRESETS: Final = "presets"
    PROPERTIES: Final = "properties"
    STAGES: Final = "stages"
    CONSOLE: Final = "console"
    EXCEPTION_LOG: Final = "exception_log"


class StageKind:
    """Which stage-control widget flavor is docked under ``PanelKey.STAGES``.

    Both are kept intentionally -- see the comment on the ``STAGES`` entry in
    :data:`PANELS` for how ``AcquirePage`` lets the user pick between them.
    """

    XYZ: Final = "xyz"
    """XYZStageWidget -- follows the core's *default* XY stage and focus device."""
    PER_DEVICE: Final = "per_device"
    """StagesPanel -- add-on-demand, one StageWidget per chosen device."""


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


def _create_stage_explorer(parent: QWidget, core: CMMCorePlus) -> QWidget:
    from pymmcore_gui.widgets._stage_explorer import ThemedStageExplorer

    return ThemedStageExplorer(parent=parent, mmcore=core)


def create_stage_widget(parent: QWidget, core: CMMCorePlus, device: str) -> QWidget:
    """Build the panel content for one XY or Z stage device.

    Shared by every stage a user adds through the Stages panel's "Add Stage"
    picker (see ``_acquire_stages.StagesPanel``), so they all look alike.
    """
    from pymmcore_widgets import StageWidget

    widget = StageWidget(device=device, parent=parent, mmcore=core)
    widget.setMinimumSize(widget.minimumSizeHint())
    return widget


def _create_stages(parent: QWidget, core: CMMCorePlus) -> QWidget:
    from ._acquire_stages import StagesPanel

    return StagesPanel(parent=parent, mmcore=core)


def _create_stage_xyz(parent: QWidget, core: CMMCorePlus) -> QWidget:
    """Build a single widget controlling the core's *default* XY/Z stage.

    Unlike ``StageWidget`` (one instance per device, added on demand through
    the Stages panel), ``XYZStageWidget`` targets no fixed device -- it
    always follows whichever devices are currently the core's default XY
    stage and focus device.
    """
    from pymmcore_widgets import XYZStageWidget

    widget = XYZStageWidget(parent=parent, mmcore=core)
    widget.setMinimumSize(widget.minimumSizeHint())
    return widget


STAGE_KIND_FACTORIES: Final[Mapping[str, tuple[PanelFactory, bool]]] = {
    # kind: (factory, needs unstyle_widgets())
    #
    # XYZStageWidget's move/halt buttons are plain QPushButtons with no
    # "variant" set, so without unstyling they render in Qt's default style
    # -- unlike StagesPanel, which already runs the same button classes
    # through unstyle_widgets() itself, per added device (_acquire_stages.py)
    # -- and the hover highlight the app's QSS gives "subtle" buttons is easy
    # to miss under that default.
    StageKind.XYZ: (_create_stage_xyz, True),
    StageKind.PER_DEVICE: (_create_stages, False),
}


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
    """Open in the built-in "Default" layout (see ``AcquirePage.reset_layout``)."""
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
        icon="proicons:cube",
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
        icon="hugeicons:property-new",
        tooltip="Groups and Presets — show or hide the group/preset panel",
        create=_create_presets,
        default_open=True,
        refresh=_refresh_presets,
    ),
    PanelInfo(
        key=PanelKey.PROPERTIES,
        title="Properties",
        icon="hugeicons:property-edit",
        tooltip="Properties — open the device property browser panel",
        create=_ignoring_core(create_property_browser),
        unstyle=True,
        refresh=_refresh_property_browser,
    ),
    PanelInfo(
        key=PanelKey.STAGES,
        title="Stages",
        icon="mdi:arrow-all",
        tooltip=(
            "Stages — control the default XY/Z stage "
            "(right-click for per-device controls)"
        ),
        # AcquirePage overrides construction for this one key (see
        # AcquirePage._stage_widget_for / STAGE_KIND_FACTORIES): the user
        # picks between StagesPanel (per-device, add on demand) and
        # XYZStageWidget (follows the core's default XY/Z devices) from the
        # Stages button's right-click menu. This default is only the
        # fallback/documentation of the initial kind (see StageKind).
        create=_create_stage_xyz,
    ),
    PanelInfo(
        key=PanelKey.STAGE_EXPLORER,
        title="Stage Explorer",
        icon="material-symbols:map-search-outline-rounded",
        tooltip="Stage Explorer — show or hide the stage exploration panel",
        create=_create_stage_explorer,
    ),
    PanelInfo(
        key=PanelKey.CONSOLE,
        title="Console",
        icon="griddy-icons:console",
        tooltip="Console — open an IPython console panel",
        create=_create_console,
    ),
    PanelInfo(
        key=PanelKey.EXCEPTION_LOG,
        title="Exception Log",
        icon="si:alert-line",
        tooltip="Exception Log — show or hide the exception log panel",
        create=_ignoring_core(create_exception_log),
        unstyle=True,
    ),
)

PANELS_BY_KEY: Final[Mapping[str, PanelInfo]] = {p.key: p for p in PANELS}
