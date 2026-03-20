"""Objectives panel for the sidebar."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_widgets import ObjectivesWidget

from pymmcore_gui._modern_gui._utils import current_core

from ._collapsible_panel import CollapsiblePanel

if TYPE_CHECKING:
    from pymmcore_gui._qt.QtWidgets import QWidget


def CollapsibleObjectivesPanel(parent: QWidget | None = None) -> CollapsiblePanel:
    """Create an Objectives panel wrapped in a collapsible header."""
    panel = CollapsiblePanel(
        title="Objective",
        summary="",
        parent=parent,
    )

    core = current_core(parent)
    content = ObjectivesWidget(parent=panel, mmcore=core)
    if layout := content.layout():
        layout.takeAt(0)  # Remove the label, we have it in the header

    # Update summary when objective changes
    if core is not None and (devices := core.guessObjectiveDevices()):

        @core.events.propertyChanged.connect
        def _on_prop_changed(device: str, property: str, value: str) -> None:
            if device in devices and property == "Label":
                panel.header.summary = value

        if label := core.getProperty("Objective", "Label"):
            panel.header.summary = label

    panel.body_layout.addWidget(content)
    return panel
