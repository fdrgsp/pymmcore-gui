"""Hardware Setup tab: available devices | installed devices | setup & properties.

Edits a :class:`pymmcore_plus.model.Microscope` config model (rather than
mutating the running system), mirroring the Micro-Manager hardware
configuration wizard.
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, cast

from pymmcore_plus import CMMCorePlus, DeviceType, Keyword
from pymmcore_plus.model import Device, Microscope

from pymmcore_gui._gui._busy import BusyOverlay, busy
from pymmcore_gui._gui._tab_page import TabPage
from pymmcore_gui._qt.QtCore import Qt
from pymmcore_gui._qt.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QPushButton,
    QSplitter,
    QWidget,
)

from ._panes import AvailableDevicesPane, InstalledDevicesPane
from ._peripherals import PeripheralsDialog
from ._setup_pane import DeviceSetupPane

if TYPE_CHECKING:
    from pymmcore_plus.model import AvailableDevice, Property

    from pymmcore_gui._qt.QtGui import QResizeEvent
    from pymmcore_gui._qt.QtWidgets import QStatusBar

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
        # serial device loaded on behalf of a pending device's "Port"
        self._pending_port: Device | None = None
        self._selected_available: AvailableDevice | None = None
        # NOTE: tracked here rather than via Microscope.is_dirty(), which
        # currently reports True even immediately after mark_clean().
        self._dirty = False
        # True while new_config()/load_config() is rebuilding the core, so this
        # page's own systemConfigurationLoaded handler doesn't redundantly
        # rebuild a model it's already in the middle of constructing.
        self._loading = False

        self._available = AvailableDevicesPane()
        self._installed = InstalledDevicesPane()
        self._setup = DeviceSetupPane()
        self._overlay = BusyOverlay(self)

        # available devices | (installed devices over device settings)
        self.left.add_widget(self._available, 1)
        self._detail_split = QSplitter(Qt.Orientation.Vertical)
        self._detail_split.addWidget(self._installed)
        self._detail_split.addWidget(self._setup)
        self._detail_split.setStretchFactor(0, 1)
        self._detail_split.setStretchFactor(1, 1)
        self.add_content_widget(self._detail_split)
        self.right.hide()
        self.bottom.hide()

        for text, slot in (
            ("New", self.new_config),
            ("Load…", self.load_config),
            ("Save…", self.save_config),
        ):
            btn = QPushButton(text)
            btn.setProperty("variant", "primary")
            btn.clicked.connect(slot)
            self.toolbar.add_widget(btn)
        self.toolbar.add_stretch()

        self._available.deviceSelected.connect(self._on_available_selected)
        self._installed.deviceSelected.connect(self._on_installed_selected)
        self._installed.removeRequested.connect(self._on_remove_requested)
        self._setup.addRequested.connect(self._begin_add)
        self._setup.addConfirmed.connect(self._finish_add)
        self._setup.addCancelled.connect(self._cancel_add)
        self._setup.propertyChanged.connect(self._on_property_changed)
        self._setup.delayChanged.connect(self._on_delay_changed)
        self._setup.renameRequested.connect(self._rename_device)
        self._setup.stateLabelChanged.connect(self._on_state_label_changed)
        self._setup.portSelected.connect(self._on_port_selected)

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
        if self._loading:
            # new_config()/load_config() already owns the current rebuild
            # (and will emit this same event again once it's truly done).
            return
        # the widget may already be torn down on the C++ side
        with suppress(RuntimeError):
            self._reload("Loading configuration…")

    # ── model ─────────────────────────────────────────────────────

    @property
    def model(self) -> Microscope:
        """The configuration model being edited."""
        return self._model

    def reload_model(self) -> None:
        """Rebuild the model from the current state of the core."""
        # NOTE: no arguments — this is wired to QPushButton.clicked, which would
        # otherwise pass its `checked` bool as the first parameter.
        self._reload("Scanning device adapters…")

    def _reload(self, message: str) -> None:
        with busy(self._overlay, message):
            self._cancel_add()
            self._model = Microscope.create_from_core(self._core)
            # create_from_core doesn't carry over the file the core was loaded
            # from, which would otherwise leave Save with no target.
            with suppress(Exception):
                self._model.config_file = self._core.systemConfigurationFile() or ""
            self._refresh_all()
        self._dirty = False

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        super().resizeEvent(a0)
        self._overlay.setGeometry(self.rect())

    # ── config files ──────────────────────────────────────────────

    def new_config(self) -> None:
        """Discard the current configuration and start from scratch."""
        if not self._confirm_discard("Start a new configuration?"):
            return
        self._cancel_add()
        self._loading = True
        try:
            with busy(self._overlay, "Starting a new configuration…"):
                with suppress(Exception):
                    self._core.unloadAllDevices()
                # unloadAllDevices() only clears devices — config groups and
                # pixel-size configs are independent of any device and would
                # otherwise survive, now dangling on devices that no longer
                # exist.
                for group_name in list(self._core.getAvailableConfigGroups()):
                    with suppress(Exception):
                        self._core.deleteConfigGroup(group_name)
                for pixel_config_name in list(
                    self._core.getAvailablePixelSizeConfigs()
                ):
                    with suppress(Exception):
                        self._core.deletePixelSizeConfig(pixel_config_name)
                self._model = Microscope()
                self._refresh_all()
            self._dirty = False
        finally:
            self._loading = False
        self._core.events.systemConfigurationLoaded.emit()

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
        self._loading = True
        try:
            with busy(self._overlay, f"Loading {Path(path).name}…"):
                # MMCore's native loader implements the complete .cfg
                # semantics, including System/Startup, Core device roles,
                # and commands newer than pymmcore-plus's editable model
                # parser.  Once the live system is correct, mirror it into
                # the model used by this page.
                self._core.loadSystemConfiguration(path)
                self._model = Microscope.create_from_core(self._core)
                self._model.config_file = path
                self._refresh_all()
            self._dirty = False
        except Exception as e:
            self._warn(f"Failed to load {path}:\n\n{e}")
        finally:
            self._loading = False

    def is_dirty(self) -> bool:
        """Whether the hardware configuration has unsaved edits."""
        return self._dirty

    def save_config(self) -> bool:
        """Save the configuration, always asking where.

        Defaults to the file the config came from, so overwriting it is a
        deliberate confirmation rather than a silent write. Returns True if the
        file was written, False if cancelled or on error.
        """
        start = self._model.config_file or "MMConfig.cfg"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save hardware configuration", start, CFG_FILTER
        )
        if not path:
            return False
        return self._save_to(path)

    def _save_to(self, path: str) -> bool:
        # Config groups and pixel-size configs are edited on the Configurations
        # tab, which commits them to the live core. Pull those into the model so
        # a saved .cfg captures hardware, groups and pixel sizes together.
        with suppress(Exception):
            self._model.update_config_groups_from_core(self._core)
        with suppress(Exception):
            self._model.update_pixel_sizes_from_core(self._core)
        try:
            self._model.save(path)
        except Exception as e:
            self._warn(f"Failed to save {path}:\n\n{e}")
            return False
        self._model.config_file = str(path)
        self._dirty = False
        self._status(f"Saved {Path(path).name}")
        return True

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
            # Only one of the two lists represents what the setup pane is
            # showing at any given time -- clear the other so its own
            # (stale) selection highlight doesn't imply otherwise.
            self._installed.clear_selection()
            self._setup.show_available(dev, self._suggest_label(dev))

    def _on_installed_selected(self, dev: Device | None) -> None:
        if self._pending is not None or dev is None:
            return
        self._available.clear_selection()
        self._setup.show_installed(dev, self._port_device_for(dev))

    def _port_device_for(self, dev: Device) -> Device | None:
        """Return the serial device backing `dev`'s "Port" property, if any."""
        if not (port := dev.port):
            return None
        return next((d for d in self._model.devices if d.name == port), None)

    # ── adding ────────────────────────────────────────────────────

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
            self._setup.show_pending(dev, self._serial_choices())
        else:
            self._finish_add()

    def _serial_choices(self) -> list[Device]:
        """Serial devices offered for a "Port" property."""
        return list(self._model.available_serial_devices)

    def _on_port_selected(self, name: str, library: str) -> None:
        """Load the chosen serial device so its settings can be configured.

        MMCore only accepts a *loaded* serial device as a "Port" value, so the
        port is loaded up front and initialized before the device that uses it.
        """
        if (dev := self._pending) is None:
            return
        self._unload_pending_port()

        if name and library:
            port = Device(
                name=name,
                library=library,
                adapter_name=name,
                device_type=DeviceType.Serial,
            )
            try:
                if name not in self._core.getLoadedDevices():
                    port.load(self._core)
                port.update_from_core(self._core)
                self._pending_port = port
            except Exception as e:
                self._warn(f"Failed to open port {name!r}:\n\n{e}")

        with suppress(Exception):
            dev.get_property(Keyword.Port).value = name

        self._setup.show_pending(dev, self._serial_choices(), self._pending_port)

    def _unload_pending_port(self) -> None:
        """Drop a provisionally-loaded port device."""
        if (port := self._pending_port) is None:
            return
        self._pending_port = None
        if port not in self._model.devices:
            with suppress(Exception):
                port.unload(self._core)

    def _finish_add(self) -> None:
        """Initialize the pending device and commit it to the model."""
        if (dev := self._pending) is None:
            return
        # The serial port must be configured and initialized BEFORE the device
        # that uses it, or initialization can fail or hang.
        if (port := self._pending_port) is not None:
            try:
                for prop in port.properties:
                    if not prop.is_read_only:
                        prop.apply_to_core(self._core, then_update=False)
                self._core.initializeDevice(port.name)
                port.initialized = True
            except Exception as e:
                self._warn(f"Failed to initialize port {port.name!r}:\n\n{e}")
                return
            if port not in self._model.devices:
                self._model.devices.append(port)
            self._pending_port = None

        try:
            # reload first: a previous failed attempt can leave the device
            # half-initialized in the core, so each try starts from a clean
            # load (this is what the Java DeviceSetupDlg does too).
            dev.initialize(self._core, reload=True, apply_pre_init=True)
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
        self._unload_pending_port()
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

    def _rename_device(self, dev: Device, new_name: str) -> None:
        """Relabel an installed device.

        MMCore identifies devices by label, so (as the Java wizard does) the
        device is unloaded and reloaded under the new name, then references to
        the old name elsewhere in the model are repaired.
        """
        new_name = new_name.strip()
        old_name = dev.name
        if not new_name or new_name == old_name:
            return

        taken = {d.name for d in self._model.devices if d is not dev}
        if new_name in taken or new_name in self._core.getLoadedDevices():
            self._warn(f"A device labelled {new_name!r} already exists.")
            self._setup.show_installed(dev, self._port_device_for(dev))
            return

        try:
            with suppress(Exception):
                dev.unload(self._core)
            dev.name = new_name
            dev.load(self._core)
            if dev.parent_label:
                self._core.setParentLabel(dev.name, dev.parent_label)
            dev.initialize(self._core, apply_pre_init=True)
        except Exception as e:
            dev.name = old_name  # roll the model back to match the core
            self._warn(f"Failed to rename {old_name!r}:\n\n{e}")
            self._reload("Reloading after failed rename…")
            return

        # properties carry their owning device's name
        for prop in dev.properties:
            prop.device_name = new_name
        # peripherals point at their hub, and port users at their serial device
        for other in self._model.devices:
            if other.parent_label == old_name:
                other.parent_label = new_name
                with suppress(Exception):
                    self._core.setParentLabel(other.name, new_name)
            if other.port == old_name:
                with suppress(Exception):
                    other.get_property(Keyword.Port).value = new_name

        self._dirty = True
        self._installed.set_devices(self._model.devices)
        self._setup.show_installed(dev, self._port_device_for(dev))
        self._status(f"Renamed {old_name} to {new_name}")

    def _on_delay_changed(self, dev: Device, delay_ms: float) -> None:
        dev.delay_ms = delay_ms
        self._dirty = True
        with suppress(Exception):
            self._core.setDeviceDelayMs(dev.name, delay_ms)

    def _on_state_label_changed(self, dev: Device, state: int, label: str) -> None:
        """Rename one position of a state device (filter wheel, turret, ...)."""
        label = label.strip()
        if not label or (state < len(dev.labels) and dev.labels[state] == label):
            return
        old_labels = dev.labels
        dev.set_label(state, label)
        try:
            self._core.defineStateLabel(dev.name, state, label)
        except Exception as e:
            dev.labels = old_labels  # roll the model back to match the core
            self._warn(f"Failed to set label for state {state}:\n\n{e}")
            self._setup.show_installed(dev, self._port_device_for(dev))
            return
        self._dirty = True
        self._status(f"Renamed state {state} of {dev.name} to {label!r}")

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
                cast("QStatusBar", bar()).showMessage(message, 5000)

    def _seed_hardware_sizes(self) -> None:
        """Split the page in half: available devices | installed-over-settings.

        Equal stretch factors keep the two halves balanced as the window
        resizes (TabPage defaults the center to grow and the docks to stay
        fixed, which would otherwise skew the split). The hidden right dock
        gets 0.
        """
        self._h_split.setStretchFactor(0, 1)  # available devices
        self._h_split.setStretchFactor(1, 1)  # installed-over-settings column
        self._h_split.setStretchFactor(2, 0)  # hidden right dock
        self._h_split.setSizes([500, 500, 0])
        self._detail_split.setSizes([500, 500])
