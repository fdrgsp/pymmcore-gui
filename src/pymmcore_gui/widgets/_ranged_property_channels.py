"""Channel table exposing every writable numeric property with limits."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_plus import PropertyType
from pymmcore_widgets.mda import (
    ChannelProperty,
    CollapsibleCoreMDATabs,
    CoreConnectedChannelTable,
)
from pymmcore_widgets.useq_widgets import ComboColumn
from superqt.utils import signals_blocked

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


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
                header=self.LIGHT_SOURCE.header,
                default="",
                allowed_values=("", *self._light_sources),
            )
            table.addColumn(self._light_source_column, property_col)
            self._apply_light_source_visibility()
            self._sync_intensity_widgets(force=True)
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
