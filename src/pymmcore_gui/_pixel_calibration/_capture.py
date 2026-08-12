from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

from ._models import PixelCalibrationError


@dataclass(frozen=True)
class CalibrationCaptureSettings:
    """Temporary optical and illumination state used for calibration images."""

    resolution_settings: tuple[tuple[str, str, str], ...]
    channel_group: str
    channel_config: str
    exposure_ms: float
    light_properties: tuple[tuple[str, str, float], ...] = ()
    camera: str = ""


class CaptureCore(Protocol):
    """MMCore surface used to apply and restore capture settings."""

    def getConfigData(self, group: str, config: str) -> Any: ...

    def getProperty(self, device: str, prop: str) -> str: ...

    def setProperty(self, device: str, prop: str, value: str | float) -> None: ...

    def waitForDevice(self, device: str) -> None: ...

    def setConfig(self, group: str, config: str) -> None: ...

    def waitForConfig(self, group: str, config: str) -> None: ...

    def getExposure(self) -> float: ...

    def setExposure(self, exposure_ms: float) -> None: ...

    def getCameraDevice(self) -> str: ...

    def setCameraDevice(self, camera: str) -> None: ...

    def getCurrentPixelSizeConfig(self) -> str: ...


def _config_settings(
    core: CaptureCore, group: str, config: str
) -> tuple[tuple[str, str, str], ...]:
    if not group or not config:
        return ()
    return tuple(
        (str(device), str(prop), str(value))
        for device, prop, value in core.getConfigData(group, config)
    )


class CaptureStateTransaction:
    """Apply capture state and restore every affected property exactly once."""

    def __init__(
        self,
        core: Any,
        settings: CalibrationCaptureSettings,
        *,
        resolution_id: str,
    ) -> None:
        # Generated CMMCorePlus overloads use adapter-specific parameter names,
        # so static structural matching is narrower than the runtime API.
        self._core = cast("CaptureCore", core)
        self._settings = settings
        self._resolution_id = resolution_id
        self._snapshot: dict[tuple[str, str], str] = {}
        self._property_order: list[tuple[str, str]] = []
        self._old_exposure: float | None = None
        self._old_camera: str | None = None
        self._applied = False

    def _changed_properties(self) -> tuple[tuple[str, str, str | float], ...]:
        selected = self._settings
        values: dict[tuple[str, str], str | float] = {}
        for device, prop, resolution_value in selected.resolution_settings:
            values[(device, prop)] = resolution_value
        for device, prop, config_value in _config_settings(
            self._core, selected.channel_group, selected.channel_config
        ):
            values[(device, prop)] = config_value
        for device, prop, light_value in selected.light_properties:
            values[(device, prop)] = light_value
        return tuple((device, prop, value) for (device, prop), value in values.items())

    def apply(self) -> None:
        """Snapshot affected state, apply settings, and verify the resolution match."""
        if self._applied:
            raise RuntimeError("Capture settings transaction is already active")
        changed = self._changed_properties()
        self._property_order = [(device, prop) for device, prop, _ in changed]
        self._snapshot = {
            (device, prop): str(self._core.getProperty(device, prop))
            for device, prop in self._property_order
        }
        self._old_camera = str(self._core.getCameraDevice())
        self._applied = True

        camera = self._settings.camera or self._old_camera
        if camera and camera != self._old_camera:
            self._core.setCameraDevice(camera)
            self._core.waitForDevice(camera)
        self._old_exposure = float(self._core.getExposure())

        # The resolution settings go first. The channel is applied afterwards so
        # any accidental overlap is detected by the final resolution-ID check.
        for device, prop, resolution_value in self._settings.resolution_settings:
            self._core.setProperty(device, prop, resolution_value)
            self._core.waitForDevice(device)

        group = self._settings.channel_group
        config = self._settings.channel_config
        if group and config:
            self._core.setConfig(group, config)
            self._core.waitForConfig(group, config)

        # A Core-Camera property may occur in a configuration. The explicit
        # camera selected by the user remains authoritative; the resolution-ID
        # verification below rejects incompatible resolution bindings.
        if camera and str(self._core.getCameraDevice()) != camera:
            self._core.setCameraDevice(camera)
            self._core.waitForDevice(camera)

        self._core.setExposure(self._settings.exposure_ms)
        if camera := str(self._core.getCameraDevice()):
            self._core.waitForDevice(camera)

        for device, prop, light_value in self._settings.light_properties:
            self._core.setProperty(device, prop, light_value)
            self._core.waitForDevice(device)

        current = str(self._core.getCurrentPixelSizeConfig())
        if current != self._resolution_id:
            raise PixelCalibrationError(
                f"Capture settings do not match resolution {self._resolution_id!r}; "
                f"MMCore currently matches {current or '<none>'!r}"
            )

    def restore(self) -> None:
        """Restore the exact pre-transaction properties and exposure."""
        if not self._applied:
            return
        errors: list[BaseException] = []
        for device, prop in reversed(self._property_order):
            try:
                self._core.setProperty(device, prop, self._snapshot[(device, prop)])
                self._core.waitForDevice(device)
            except BaseException as error:
                errors.append(error)
        if self._old_exposure is not None:
            try:
                camera = self._settings.camera or self._old_camera or ""
                if camera and str(self._core.getCameraDevice()) != camera:
                    self._core.setCameraDevice(camera)
                    self._core.waitForDevice(camera)
                self._core.setExposure(self._old_exposure)
                if camera := str(self._core.getCameraDevice()):
                    self._core.waitForDevice(camera)
            except BaseException as error:
                errors.append(error)
        if self._old_camera is not None:
            try:
                if str(self._core.getCameraDevice()) != self._old_camera:
                    self._core.setCameraDevice(self._old_camera)
                    if self._old_camera:
                        self._core.waitForDevice(self._old_camera)
            except BaseException as error:
                errors.append(error)
        self._applied = False
        if errors:
            details = "; ".join(str(error) for error in errors)
            raise PixelCalibrationError(
                f"Failed to restore calibration capture settings: {details}"
            )


__all__ = ["CalibrationCaptureSettings", "CaptureStateTransaction"]
