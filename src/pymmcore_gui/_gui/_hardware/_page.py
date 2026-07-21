"""Hardware Setup tab: available devices | installed devices | setup & properties.

Edits a :class:`pymmcore_plus.model.Microscope` config model (rather than
mutating the running system), mirroring the Micro-Manager hardware
configuration wizard.
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from pymmcore_plus import CMMCorePlus, DeviceType
from pymmcore_plus.model import Device, Microscope

from pymmcore_gui._gui._tab_page import TabPage
from pymmcore_gui._qt.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QPushButton,
    QWidget,
)

from ._panes import AvailableDevicesPane, InstalledDevicesPane
from ._peripherals import PeripheralsDialog
from ._setup_pane import DeviceSetupPane

if TYPE_CHECKING:
    from pymmcore_plus.model import AvailableDevice, Property

CFG_FILTER = "Micro-Manager config (*.cfg);;All files (*)"


class HardwareSetupPage(TabPage):
    """Three-pane hardware configuration editor."""

    def __init__(
        self, mmcore: CMMCorePlus | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._core = mmcore or CMMCorePlus.instance()
        self._model = Microscope()
        # device loaded into core but not yet initialized/committed to the model
        self._pending: Device | None = None
        self._selected_available: AvailableDevice | None = None
        # NOTE: tracked here rather than via Microscope.is_dirty(), which
        # currently reports True even immediately after mark_clean().
        self._dirty = False

        self._available = AvailableDevicesPane()
        self._installed = InstalledDevicesPane()
        self._setup = DeviceSetupPane()

        # available devices | setup & properties | installed devices
        self.left.add_widget(self._available, 1)
        self.add_content_widget(self._setup)
        self.right.add_widget(self._installed, 1)
        self.bottom.hide()

        for text, slot in (
            ("New", self.new_config),
            ("Load…", self.load_config),
            ("Save", self.save_config),
            ("Save As…", self.save_config_as),
            ("Reload from core", self.reload_model),
        ):
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            self.toolbar.add_widget(btn)
        self.toolbar.add_stretch()

        self._available.deviceSelected.connect(self._on_available_selected)
        self._available.addRequested.connect(self._on_add_shortcut)
        self._installed.deviceSelected.connect(self._on_installed_selected)
        self._installed.removeRequested.connect(self._on_remove_requested)
        self._setup.addRequested.connect(self._begin_add)
        self._setup.addConfirmed.connect(self._finish_add)
        self._setup.addCancelled.connect(self._cancel_add)
        self._setup.propertyChanged.connect(self._on_property_changed)
        self._setup.delayChanged.connect(self._on_delay_changed)

        # A configuration may be loaded into the core *after* this page is
        # built (create_mmgui constructs the window before prompting for a
        # config), so track the core rather than only reading it once.
        self._core.events.systemConfigurationLoaded.connect(
            self._on_system_config_loaded
        )

        self.reload_model()
        self._seed_hardware_sizes()

    def _on_system_config_loaded(self) -> None:
        """Repopulate when a config is loaded into the core from anywhere."""
        # the widget may already be torn down on the C++ side
        with suppress(RuntimeError):
            self.reload_model()

    # ── model ─────────────────────────────────────────────────────

    @property
    def model(self) -> Microscope:
        """The configuration model being edited."""
        return self._model

    def reload_model(self) -> None:
        """Rebuild the model from the current state of the core."""
        self._cancel_add()
        self._model = Microscope.create_from_core(self._core)
        self._refresh_all()
        self._dirty = False

    # ── config files ──────────────────────────────────────────────

    def new_config(self) -> None:
        """Discard the current configuration and start from scratch."""
        if not self._confirm_discard("Start a new configuration?"):
            return
        self._cancel_add()
        with suppress(Exception):
            self._core.unloadAllDevices()
        self._model = Microscope()
        self._refresh_all()
        self._dirty = False

    def load_config(self) -> None:
        """Load a .cfg file into the model and into the core."""
        if not self._confirm_discard("Load a different configuration?"):
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load hardware configuration", "", CFG_FILTER
        )
        if not path:
            return

        self._cancel_add()
        try:
            model = Microscope.create_from_config(path)
        except Exception as e:
            self._warn(f"Failed to read {path}:\n\n{e}")
            return

        with suppress(Exception):
            self._core.unloadAllDevices()
        errors: dict[str, str] = {}

        def _on_fail(d: Device | Property, e: BaseException) -> None:
            errors.setdefault(d.name, str(e))

        model.initialize(self._core, on_fail=_on_fail)
        self._model = model
        self._refresh_all()
        self._dirty = False
        if errors:
            listing = "\n".join(f"  • {n}: {m}" for n, m in errors.items())
            self._warn(f"Some devices failed to initialize:\n\n{listing}")

    def save_config(self) -> None:
        """Save to the model's current file, prompting if it has none."""
        if not self._model.config_file:
            self.save_config_as()
            return
        self._save_to(self._model.config_file)

    def save_config_as(self) -> None:
        """Save the configuration to a chosen .cfg file."""
        start = self._model.config_file or "MMConfig.cfg"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save hardware configuration", start, CFG_FILTER
        )
        if path:
            self._save_to(path)

    def _save_to(self, path: str) -> None:
        try:
            self._model.save(path)
        except Exception as e:
            self._warn(f"Failed to save {path}:\n\n{e}")
            return
        self._model.config_file = str(path)
        self._dirty = False
        self._status(f"Saved {Path(path).name}")

    def _confirm_discard(self, question: str) -> bool:
        """Ask before throwing away unsaved edits."""
        if not self._dirty:
            return True
        reply = QMessageBox.question(
            self,
            "Unsaved changes",
            f"The configuration has unsaved changes.\n\n{question}",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
        )
        return reply == QMessageBox.StandardButton.Discard

    # ── refresh ───────────────────────────────────────────────────

    def _refresh_all(self) -> None:
        """Re-read available devices and repopulate every pane."""
        with suppress(Exception):
            self._model.load_available_devices(self._core)
        self._available.set_devices(self._model.available_devices)
        self._installed.set_devices(self._model.devices)
        self._setup.show_empty()

    # ── selection ─────────────────────────────────────────────────

    def _on_available_selected(self, dev: AvailableDevice | None) -> None:
        self._selected_available = dev
        if self._pending is not None:
            return  # keep the pending setup form on screen
        if dev is None:
            self._setup.show_empty()
        else:
            self._setup.show_available(dev, self._suggest_label(dev))

    def _on_installed_selected(self, dev: Device | None) -> None:
        if self._pending is not None or dev is None:
            return
        self._setup.show_installed(dev, self._port_device_for(dev))

    def _port_device_for(self, dev: Device) -> Device | None:
        """Return the serial device backing `dev`'s "Port" property, if any."""
        if not (port := dev.port):
            return None
        return next(
            (d for d in self._model.available_serial_devices if d.name == port), None
        )

    # ── adding ────────────────────────────────────────────────────

    def _on_add_shortcut(self, dev: AvailableDevice) -> None:
        """Add button / double-click in the available list."""
        self._selected_available = dev
        self._begin_add(self._suggest_label(dev))

    def _begin_add(self, label: str) -> None:
        """Load the selected device so its pre-init properties can be set."""
        if (av := self._selected_available) is None:
            return
        label = label.strip()
        if not label:
            self._warn("Please provide a device label.")
            return
        if any(d.name == label for d in self._model.devices):
            self._warn(f"A device labelled {label!r} already exists.")
            return

        self._cancel_add()
        dev = Device(
            name=label,
            library=av.library,
            adapter_name=av.adapter_name,
            device_type=av.device_type,
            description=av.description,
        )
        try:
            dev.load(self._core)
            dev.update_from_core(self._core)
        except Exception as e:
            with suppress(Exception):
                dev.unload(self._core)
            self._warn(f"Failed to load {label!r}:\n\n{e}")
            return

        self._pending = dev
        if any(p.is_pre_init for p in dev.properties):
            self._setup.show_pending(dev)
        else:
            self._finish_add()

    def _finish_add(self) -> None:
        """Initialize the pending device and commit it to the model."""
        if (dev := self._pending) is None:
            return
        try:
            dev.initialize(self._core, apply_pre_init=True)
        except Exception as e:
            self._warn(f"Failed to initialize {dev.name!r}:\n\n{e}")
            return

        self._model.devices.append(dev)
        self._pending = None
        self._dirty = True
        # a new hub exposes child peripherals, so refresh availability first
        with suppress(Exception):
            self._model.load_available_devices(self._core)
        self._available.set_devices(self._model.available_devices)
        if dev.device_type is DeviceType.Hub:
            self._add_peripherals(dev)
        self._installed.set_devices(self._model.devices)
        self._setup.show_installed(dev, self._port_device_for(dev))

    def _add_peripherals(self, hub: Device) -> None:
        """Offer to add the peripherals belonging to a freshly added hub."""
        dlg = PeripheralsDialog(hub, self._model, self)
        if not dlg.has_peripherals() or not dlg.exec():
            return

        failures: dict[str, str] = {}
        for child in dlg.selected_peripherals():
            try:
                child.load(self._core)
                # applies parent_label via core.setParentLabel
                child.apply_to_core(self._core)
                child.initialize(self._core, apply_pre_init=True)
            except Exception as e:
                failures[child.name] = str(e)
                with suppress(Exception):
                    child.unload(self._core)
                continue
            self._model.devices.append(child)
            self._dirty = True

        with suppress(Exception):
            self._model.load_available_devices(self._core)
        self._available.set_devices(self._model.available_devices)
        if failures:
            listing = "\n".join(f"  • {n}: {m}" for n, m in failures.items())
            self._warn(f"Some peripherals could not be added:\n\n{listing}")

    def _cancel_add(self) -> None:
        """Unload the pending device, if any."""
        if (dev := self._pending) is None:
            return
        self._pending = None
        with suppress(Exception):
            dev.unload(self._core)
        if self._selected_available is not None:
            self._setup.show_available(
                self._selected_available, self._suggest_label(self._selected_available)
            )
        else:
            self._setup.show_empty()

    # ── removing ──────────────────────────────────────────────────

    def _on_remove_requested(self, dev: Device) -> None:
        if dev not in self._model.devices:
            return  # pragma: no cover
        # removing a hub orphans its peripherals, so drop them too
        self._dirty = True
        doomed = [dev, *(d for d in self._model.devices if d.parent_label == dev.name)]
        for victim in doomed:
            with suppress(ValueError):
                self._model.devices.remove(victim)
            with suppress(Exception):
                victim.unload(self._core)

        with suppress(Exception):
            self._model.load_available_devices(self._core)
        self._available.set_devices(self._model.available_devices)
        self._installed.set_devices(self._model.devices)
        self._setup.show_empty()

    # ── properties ────────────────────────────────────────────────

    def _on_property_changed(self, prop: Property, value: str) -> None:
        prop.value = value
        self._dirty = True
        if self._pending is not None:
            return  # applied when the pending device is initialized

        device = next(
            (d for d in self._model.devices if d.name == prop.device_name), None
        )
        try:
            if prop.is_pre_init and device is not None and device.initialized:
                # pre-init values only take effect on a fresh init, so reload
                device.initialize(self._core, reload=True, apply_pre_init=True)
            else:
                prop.apply_to_core(self._core)
        except Exception as e:
            self._warn(f"Failed to set {prop.name!r}:\n\n{e}")

    def _on_delay_changed(self, dev: Device, delay_ms: float) -> None:
        dev.delay_ms = delay_ms
        self._dirty = True
        with suppress(Exception):
            self._core.setDeviceDelayMs(dev.name, delay_ms)

    # ── helpers ───────────────────────────────────────────────────

    def _suggest_label(self, dev: AvailableDevice) -> str:
        """Propose a unique label for a device about to be added."""
        taken = {d.name for d in self._model.devices}
        taken.update(self._core.getLoadedDevices())
        if (base := dev.adapter_name) not in taken:
            return base
        i = 2
        while f"{base}-{i}" in taken:
            i += 1
        return f"{base}-{i}"

    def _warn(self, message: str) -> None:
        QMessageBox.warning(self, "Hardware Setup", message)

    def _status(self, message: str) -> None:
        """Show a transient message in the main window's status bar."""
        win = self.window()
        if (bar := getattr(win, "statusBar", None)) and callable(bar):
            with suppress(Exception):
                bar().showMessage(message, 5000)

    def _seed_hardware_sizes(self) -> None:
        """Give the three panes a sensible initial split."""
        self._h_split.setSizes([400, 520, 400])
