"""Channel table exposing every writable numeric property with limits."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_plus import CMMCorePlus, PropertyType
from pymmcore_widgets.mda import (
    ChannelProperty,
    CollapsibleCoreMDATabs,
    CoreConnectedChannelTable,
)
from pymmcore_widgets.useq_widgets import ComboColumn
from superqt.utils import signals_blocked

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from pymmcore_gui._qt.QtWidgets import QWidget


_PROPERTY_SEPARATOR = " · "


def _property_label(device: str, prop: str) -> str:
    """Return the stable user-facing identifier for a device property."""
    return f"{device}{_PROPERTY_SEPARATOR}{prop}"


class RangedPropertyChannelTable(CoreConnectedChannelTable):
    """Core channel table with one arbitrary ranged property per channel.

    The upstream table discovers only config groups that wrap one ranged numeric
    property. This variant presents the underlying Micro-Manager properties
    directly. Every initialized, writable integer/float property with limits is
    available; no device-name or property-name heuristics are applied.

    The existing upstream storage keys and methods are deliberately retained so
    previously saved channel-property metadata remains compatible.
    """

    def __init__(
        self,
        rows: int = 0,
        mmcore: CMMCorePlus | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(rows, mmcore, parent)
        self.show_light_source.setText("Show Property")
        self.show_light_source.setToolTip(
            "Set one writable numeric device property with limits per channel.\n"
            "The selected Property and Value are applied during MDA, Snap, and Live."
        )
        self._rename_property_headers()

    def rangedProperties(self) -> Mapping[str, tuple[str, str]]:
        """Return ``{display label: (device, property)}`` for available choices."""
        return self.lightSources()

    def _find_light_sources(self) -> dict[str, tuple[str, str]]:
        """Return every initialized, writable numeric property with limits."""
        properties = self._mmc.iterProperties(
            property_type=(PropertyType.Integer, PropertyType.Float),
            has_limits=True,
            is_read_only=False,
            as_object=False,
        )
        pairs = sorted(
            (
                (str(device), str(prop))
                for device, prop in properties
                if not self._mmc.isPropertyPreInit(device, prop)
            ),
            key=lambda pair: (pair[0].casefold(), pair[1].casefold()),
        )
        return {_property_label(device, prop): (device, prop) for device, prop in pairs}

    def _update_light_sources(self) -> None:
        """Rebuild the property column from the core's ranged properties."""
        self._light_sources = self._find_light_sources()

        table = self.table()
        property_col = table.indexOf(self._light_source_column)
        if property_col < 0:  # pragma: no cover
            return

        with signals_blocked(self):
            table.removeColumn(property_col)
            self._light_source_column = ComboColumn(
                key=self.LIGHT_SOURCE.key,
                header="Property",
                default="",
                allowed_values=("", *self._light_sources),
            )
            table.addColumn(self._light_source_column, property_col)
            self._apply_light_source_visibility()
            self._sync_intensity_widgets(force=True)
        self._rename_property_headers()
        self.valueChanged.emit()

    def setChannelProperties(self, value: Iterable[ChannelProperty]) -> None:
        """Restore properties by device/property, including older saved metadata."""
        labels_by_property = {
            dev_prop: label for label, dev_prop in self._light_sources.items()
        }
        normalized: list[ChannelProperty] = []
        for entry in value:
            label = labels_by_property.get(
                (entry["device"], entry["property"]),
                entry["group"],
            )
            normalized.append(
                ChannelProperty(
                    channel_index=entry["channel_index"],
                    config=entry["config"],
                    group=label,
                    device=entry["device"],
                    property=entry["property"],
                    value=entry["value"],
                )
            )
        super().setChannelProperties(normalized)

    def _rename_property_headers(self) -> None:
        table = self.table()
        property_col = table.indexOf(self._light_source_column)
        value_col = table.indexOf(self.INTENSITY)
        if property_col >= 0 and (item := table.horizontalHeaderItem(property_col)):
            item.setText("Property")
        if value_col >= 0 and (item := table.horizontalHeaderItem(value_col)):
            item.setText("Value")


class RangedPropertyCollapsibleCoreMDATabs(CollapsibleCoreMDATabs):
    """Collapsible MDA tabs using :class:`RangedPropertyChannelTable`."""

    def create_subwidgets(self) -> None:
        super().create_subwidgets()
        inherited_channels = self.channels
        self.channels = RangedPropertyChannelTable(1, self._mmc)
        inherited_channels.deleteLater()


__all__ = [
    "RangedPropertyChannelTable",
    "RangedPropertyCollapsibleCoreMDATabs",
]
