from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from ndv.models import ChannelMode
from pymmcore_plus import PropertyType

import pymmcore_gui.widgets._white_balance_dialog as dialog_module
from pymmcore_gui._white_balance import (
    TOLERANCE,
    Channel,
    GainProperty,
    WhiteBalanceError,
    WhiteBalanceResult,
    candidate_properties,
    channel_means,
    gain_property,
    guess_gain_properties,
    one_push_white_balance,
)
from pymmcore_gui.widgets._white_balance_dialog import WhiteBalanceDialog
from pymmcore_gui.widgets.image_preview._ndv_preview import NDVPreview

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot

    from pymmcore_gui._settings import Settings
    from pymmcore_gui.widgets.image_preview._snap_album import SnappedImage

PropSpec = tuple[Any, PropertyType, tuple[float, float] | None, bool]


class FakeColorCamera:
    """A color camera whose channel means follow its white-balance properties."""

    def __init__(
        self,
        props: dict[str, PropSpec],
        gain: Callable[[str, float], float],
        base: tuple[float, float, float] = (1200.0, 2000.0, 900.0),
    ) -> None:
        self.props = props
        self.values = {k: v[0] for k, v in props.items()}
        self.gain = gain
        self.base = base
        self.snaps = 0
        self.camera = "Cam"
        self.snapped_with: set[str] = set()

    def getCameraDevice(self) -> str:
        return self.camera

    def setCameraDevice(self, camera: str) -> None:
        self.camera = camera

    def getDevicePropertyNames(self, _dev: str) -> tuple[str, ...]:
        return tuple(self.props)

    def isPropertyReadOnly(self, _dev: str, name: str) -> bool:
        return self.props[name][3]

    def getPropertyType(self, _dev: str, name: str) -> PropertyType:
        return self.props[name][1]

    def hasPropertyLimits(self, _dev: str, name: str) -> bool:
        return self.props[name][2] is not None

    def getPropertyLowerLimit(self, _dev: str, name: str) -> float:
        return self.props[name][2][0]  # type: ignore[index]

    def getPropertyUpperLimit(self, _dev: str, name: str) -> float:
        return self.props[name][2][1]  # type: ignore[index]

    def getProperty(self, _dev: str, name: str) -> str:
        return str(self.values[name])

    def setProperty(self, _dev: str, name: str, value: float) -> None:
        self.values[name] = value

    def getImageBitDepth(self) -> int:
        return 12

    def waitForDevice(self, _dev: str) -> None:
        pass

    def snap(self) -> np.ndarray:
        self.snaps += 1
        self.snapped_with.add(self.camera)
        r, g, b = self.base
        r *= self.gain("red", float(self.values["WB Red"]))
        b *= self.gain("blue", float(self.values["WB Blue"]))
        rng = np.random.default_rng(self.snaps)
        img = np.stack([np.full((32, 32), v) for v in (r, g, b)], axis=-1)
        img = img + rng.normal(0, 5, img.shape)
        out: np.ndarray = np.clip(img, 0, 4095).astype(np.uint16)
        return out


def _camera(
    value: float,
    limits: tuple[float, float] | None,
    ptype: PropertyType,
    gain: Callable[[str, float], float],
    **kwargs: Any,
) -> FakeColorCamera:
    props: dict[str, PropSpec] = {
        "WB Red": (value, ptype, limits, False),
        "WB Blue": (value, ptype, limits, False),
        "WhiteBalance Auto": ("Off", PropertyType.String, None, False),
        "Exposure": (10.0, PropertyType.Float, (0, 1000), False),
        "Temperature": (20.0, PropertyType.Float, (0, 50), True),
    }
    return FakeColorCamera(props, gain, **kwargs)


def _gains(cam: FakeColorCamera) -> dict[Channel, GainProperty]:
    return guess_gain_properties(candidate_properties(cam, "Cam"))  # type: ignore[arg-type]


def _assert_balanced(cam: FakeColorCamera, result: WhiteBalanceResult) -> None:
    assert result.converged
    r, g, b = channel_means(cam.snap(), 4095)
    assert r / g == pytest.approx(1, abs=TOLERANCE + 0.01)
    assert b / g == pytest.approx(1, abs=TOLERANCE + 0.01)


def test_candidates_are_writable_numbers_and_the_guess_reads_their_names() -> None:
    cam = _camera(100, (1, 1000), PropertyType.Integer, lambda _c, v: v)
    names = [p.name for p in candidate_properties(cam, "Cam")]  # type: ignore[arg-type]
    assert names == ["WB Red", "WB Blue", "Exposure"]
    assert _gains(cam) == {
        "red": GainProperty("WB Red", 1, 1000, integer=True),
        "blue": GainProperty("WB Blue", 1, 1000, integer=True),
    }


def test_a_property_without_limits_is_searched_within_a_factor_of_ten() -> None:
    cam = _camera(50.0, None, PropertyType.Float, lambda _c, v: v)
    prop = gain_property(cam, "Cam", "WB Red")  # type: ignore[arg-type]
    assert prop == GainProperty("WB Red", 5, 500, integer=False, has_limits=False)


def test_proportional_integer_gain_converges() -> None:
    cam = _camera(100, (1, 1000), PropertyType.Integer, lambda _c, v: v / 100)
    result = one_push_white_balance(cam, _gains(cam))  # type: ignore[arg-type]
    _assert_balanced(cam, result)
    assert result.values["WB Red"] == round(100 * 2000 / 1200)


def test_offset_gain_converges_with_secant_steps() -> None:
    # zero means "no correction" and the gain is not proportional to the value
    cam = _camera(0.0, (-100, 100), PropertyType.Float, lambda _c, v: 2 ** (v / 50))
    result = one_push_white_balance(cam, _gains(cam))  # type: ignore[arg-type]
    _assert_balanced(cam, result)


def test_only_the_chosen_gain_is_balanced() -> None:
    cam = _camera(100, (1, 1000), PropertyType.Integer, lambda _c, v: v / 100)
    blue = _gains(cam)["blue"]
    result = one_push_white_balance(cam, {"blue": blue})  # type: ignore[arg-type]
    assert result.converged
    assert result.values == {"WB Blue": round(100 * 2000 / 900)}
    assert cam.values["WB Red"] == 100


def test_duplicate_gain_properties_are_rejected() -> None:
    cam = _camera(100, (1, 1000), PropertyType.Integer, lambda _c, v: v / 100)
    red = _gains(cam)["red"]
    with pytest.raises(WhiteBalanceError, match="different properties"):
        one_push_white_balance(cam, {"red": red, "blue": red})  # type: ignore[arg-type]
    assert cam.snaps == 0
    assert cam.values["WB Red"] == 100


def test_unbounded_gains_are_refreshed_before_balancing() -> None:
    cam = _camera(1, None, PropertyType.Float, lambda _c, v: v / 100)
    gains = _gains(cam)
    cam.values.update({"WB Red": 100, "WB Blue": 100})
    result = one_push_white_balance(cam, gains)  # type: ignore[arg-type]
    _assert_balanced(cam, result)


def test_snap_failure_restores_gains_and_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cam = _camera(100, (1, 1000), PropertyType.Integer, lambda _c, v: v / 100)
    cam.camera = "Mono"
    snap = cam.snap

    def fail_after_first_snap() -> np.ndarray:
        if cam.snaps:
            raise RuntimeError("snap failed")
        return snap()

    monkeypatch.setattr(cam, "snap", fail_after_first_snap)
    with pytest.raises(RuntimeError, match="snap failed"):
        one_push_white_balance(cam, _gains(cam), camera="Cam")  # type: ignore[arg-type]
    assert cam.values["WB Red"] == cam.values["WB Blue"] == 100
    assert cam.camera == "Mono"


def test_gain_changes_settle_before_snapping(monkeypatch: pytest.MonkeyPatch) -> None:
    cam = _camera(100, (1, 1000), PropertyType.Integer, lambda _c, v: v / 100)
    snap, set_property = cam.snap, cam.setProperty
    pending = False

    def set_gain(dev: str, name: str, value: float) -> None:
        nonlocal pending
        pending = True
        set_property(dev, name, value)

    def wait(_dev: str) -> None:
        nonlocal pending
        pending = False

    def snap_when_ready() -> np.ndarray:
        assert not pending
        return snap()

    monkeypatch.setattr(cam, "setProperty", set_gain)
    monkeypatch.setattr(cam, "waitForDevice", wait)
    monkeypatch.setattr(cam, "snap", snap_when_ready)
    result = one_push_white_balance(cam, _gains(cam))  # type: ignore[arg-type]
    _assert_balanced(cam, result)


def test_iteration_limit_does_not_apply_unmeasured_gains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cam = _camera(100, (1, 1000), PropertyType.Integer, lambda _c, _v: 1)
    set_property = cam.setProperty
    writes: list[tuple[int, float]] = []

    def record_write(dev: str, name: str, value: float) -> None:
        writes.append((cam.snaps, value))
        set_property(dev, name, value)

    monkeypatch.setattr("pymmcore_gui._white_balance.MAX_ITERATIONS", 2)
    monkeypatch.setattr(cam, "setProperty", record_write)
    result = one_push_white_balance(cam, _gains(cam))  # type: ignore[arg-type]
    assert result.iterations == cam.snaps == 2
    assert all(iteration == 1 or value == 100 for iteration, value in writes)


def test_gain_at_its_limit_keeps_the_closest_values() -> None:
    cam = _camera(
        100,
        (1, 150),
        PropertyType.Integer,
        lambda _c, v: v / 100,
        base=(1200, 2000, 1900),  # red needs 167 (> 150), blue 105
    )
    result = one_push_white_balance(cam, _gains(cam))  # type: ignore[arg-type]
    assert not result.converged
    assert result.limited == ["WB Red"]
    assert cam.values["WB Red"] == 150
    assert result.ratios["blue"] == pytest.approx(1, abs=TOLERANCE)


def test_restoring_best_values_is_covered_by_error_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cam = _camera(100, (1, 1000), PropertyType.Integer, lambda _c, _v: 1)
    monkeypatch.setattr(
        cam, "snap", lambda: np.broadcast_to(np.array(cam.base), (8, 8, 3))
    )
    set_property = cam.setProperty
    failed = False

    def fail_once_on_restore(dev: str, name: str, value: float) -> None:
        nonlocal failed
        if value == 100 and not failed:
            failed = True
            raise RuntimeError("restore failed")
        set_property(dev, name, value)

    monkeypatch.setattr(cam, "setProperty", fail_once_on_restore)
    with pytest.raises(RuntimeError, match="restore failed"):
        one_push_white_balance(cam, _gains(cam))  # type: ignore[arg-type]
    assert cam.values["WB Red"] == cam.values["WB Blue"] == 100


def test_integer_rounding_is_not_reported_as_a_property_limit() -> None:
    cam = _camera(
        1, (0, 100), PropertyType.Integer, lambda _c, v: v, base=(1940, 2000, 1900)
    )
    result = one_push_white_balance(cam, {"red": _gains(cam)["red"]})  # type: ignore[arg-type]
    assert not result.converged
    assert result.limited == []


def test_saturated_frame_fails_and_restores_the_gains() -> None:
    cam = _camera(
        100,
        (1, 1000),
        PropertyType.Integer,
        lambda _c, v: v / 100,
        base=(4095, 4095, 4095),
    )
    with pytest.raises(WhiteBalanceError, match="too bright"):
        one_push_white_balance(cam, _gains(cam))  # type: ignore[arg-type]
    assert cam.values["WB Red"] == 100


def test_dark_frame_fails() -> None:
    with pytest.raises(WhiteBalanceError, match="too dark"):
        channel_means(np.full((8, 8, 3), 10, np.uint16), 4095)


@pytest.mark.parametrize("shape", [(8, 8), (0, 8, 3)])
def test_invalid_frame_shape_fails(shape: tuple[int, ...]) -> None:
    with pytest.raises(WhiteBalanceError, match="needs an RGB image"):
        channel_means(np.zeros(shape, np.uint16), 4095)


def test_another_camera_is_current_only_while_balancing() -> None:
    cam = _camera(100, (1, 1000), PropertyType.Integer, lambda _c, v: v / 100)
    cam.camera = "Mono"
    one_push_white_balance(cam, _gains(cam), camera="Cam")  # type: ignore[arg-type]
    assert cam.snapped_with == {"Cam"}
    assert cam.camera == "Mono"


def test_dialog_remembers_the_chosen_properties_and_pauses_live(
    mmcore: CMMCorePlus,
    qtbot: QtBot,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contexts: list[str] = []

    @contextmanager
    def snap_context() -> Iterator[None]:
        contexts.append("enter")
        yield
        contexts.append("exit")

    dialog = WhiteBalanceDialog(mmcore, snap_context=snap_context)
    qtbot.addWidget(dialog)
    assert dialog.camera_combo.currentText() == mmcore.getCameraDevice()
    # the demo camera has no white-balance properties to guess
    assert dialog.red_combo.currentData() is None
    assert not dialog.balance_btn.isEnabled()

    dialog.red_combo.setCurrentIndex(dialog.red_combo.findData("Gain"))
    dialog.blue_combo.setCurrentIndex(dialog.blue_combo.findData("Gain"))
    assert not dialog.balance_btn.isEnabled()
    assert not dialog.balance()
    dialog.blue_combo.setCurrentIndex(dialog.blue_combo.findData("Offset"))
    assert dialog.balance_btn.isEnabled()

    calls: list[tuple[set[str], str | None, bool]] = []

    def fake_balance(
        core: CMMCorePlus, gains: Mapping[str, GainProperty], *, camera: str | None
    ) -> WhiteBalanceResult:
        calls.append(
            ({p.name for p in gains.values()}, camera, core.isSequenceRunning())
        )
        return WhiteBalanceResult({"Gain": 2}, {"red": 1.0, "blue": 1.0}, True, 1)

    monkeypatch.setattr(dialog_module, "one_push_white_balance", fake_balance)
    mmcore.startContinuousSequenceAcquisition()
    with qtbot.waitSignal(dialog.balanced):
        assert dialog.balance()
        assert not dialog.balance_btn.isEnabled()
        assert not dialog.red_combo.isEnabled()
    qtbot.waitUntil(lambda: dialog._worker is None)
    assert mmcore.isSequenceRunning()
    assert dialog.balance_btn.isEnabled()
    mmcore.stopSequenceAcquisition()

    assert calls == [({"Gain", "Offset"}, "Camera", False)]
    assert contexts == ["enter", "exit"]
    assert "Balanced: Gain = 2" in dialog.result_label.text()
    assert settings.white_balance_properties("Camera") == {
        "red": "Gain",
        "blue": "Offset",
    }

    again = WhiteBalanceDialog(mmcore)
    qtbot.addWidget(again)
    assert again.red_combo.currentData() == "Gain"
    assert again.blue_combo.currentData() == "Offset"


def test_dialog_reports_a_failure(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    dialog = WhiteBalanceDialog(mmcore)
    qtbot.addWidget(dialog)
    dialog.red_combo.setCurrentIndex(dialog.red_combo.findData("Gain"))
    # the demo camera is monochrome by default
    with qtbot.waitSignal(dialog.balanceFailed):
        assert dialog.balance()
    assert "needs an RGB image" in dialog.result_label.text()
    qtbot.waitUntil(lambda: dialog._worker is None)


@pytest.mark.parametrize(
    ("saved", "expected"),
    [
        (None, ("WB Red", "WB Blue")),
        ({"red": "Old Red", "blue": "WB Blue"}, ("WB Red", "WB Blue")),
        ({"red": "WB Red"}, ("WB Red", None)),
        ({}, (None, None)),
    ],
)
def test_dialog_falls_back_for_missing_saved_properties(
    mmcore: CMMCorePlus,
    qtbot: QtBot,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    saved: dict[str, str] | None,
    expected: tuple[str | None, str | None],
) -> None:
    props = [GainProperty(f"WB {c}", 1, 1000, integer=True) for c in ("Red", "Blue")]
    monkeypatch.setattr(dialog_module, "candidate_properties", lambda *_: props)
    if saved is not None:
        settings.remember_white_balance_properties(mmcore.getCameraDevice(), saved)
    dialog = WhiteBalanceDialog(mmcore)
    qtbot.addWidget(dialog)
    assert (dialog.red_combo.currentData(), dialog.blue_combo.currentData()) == expected


def test_snap_context_failure_resumes_live(mmcore: CMMCorePlus, qtbot: QtBot) -> None:
    @contextmanager
    def fail_context() -> Iterator[None]:
        raise RuntimeError("context failed")
        yield

    dialog = WhiteBalanceDialog(mmcore, snap_context=fail_context)
    qtbot.addWidget(dialog)
    dialog.red_combo.setCurrentIndex(dialog.red_combo.findData("Gain"))
    mmcore.startContinuousSequenceAcquisition()
    try:
        with qtbot.waitSignal(dialog.balanceFailed):
            assert dialog.balance()
        qtbot.waitUntil(lambda: dialog._worker is None)
        assert mmcore.isSequenceRunning()
        assert "context failed" in dialog.result_label.text()
    finally:
        mmcore.stopSequenceAcquisition()


def test_color_camera_preview_uses_rgb_mode_with_a_white_balance_button(
    mmcore: CMMCorePlus, qtbot: QtBot
) -> None:
    preview = NDVPreview(mmcore)
    qtbot.addWidget(preview)
    preview.show()
    mmcore.setProperty("Camera", "PixelType", "32bitRGB")
    mmcore.snapImage()
    qtbot.waitUntil(lambda: preview._white_balance_btn is not None)

    assert preview.viewer.display_model.channel_mode is ChannelMode.RGBA
    btn = preview._white_balance_btn
    assert btn is not None
    assert btn.property("variant") == "subtle"
    row = btn.parentWidget()
    layout = row._lut_layout  # type: ignore[union-attr]
    assert layout.indexOf(btn) == layout.indexOf(row.auto_clim) + 1  # type: ignore[union-attr]

    btn.click()
    dialog = preview._white_balance_dialog
    assert dialog is not None and dialog.isVisible()

    recorded: list[SnappedImage] = []
    preview.snapRecorded.connect(recorded.append)
    with preview._unrecorded_snaps():
        with preview._unrecorded_snaps():
            mmcore.snapImage()
        mmcore.snapImage()
    mmcore.snapImage()
    qtbot.wait(0)
    assert len(recorded) == 1
    dialog.close()


@pytest.mark.parametrize("missing_control", ["_lut_layout", "auto_clim"])
def test_rgb_preview_survives_missing_white_balance_controls(
    mmcore: CMMCorePlus,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    missing_control: str,
) -> None:
    add_button = NDVPreview._add_white_balance_button

    def remove_control(self: NDVPreview, row: Any) -> None:
        with monkeypatch.context() as patch:
            patch.delattr(row, missing_control)
            add_button(self, row)

    monkeypatch.setattr(NDVPreview, "_add_white_balance_button", remove_control)
    preview = NDVPreview(mmcore)
    qtbot.addWidget(preview)
    preview.show()
    mmcore.setProperty("Camera", "PixelType", "32bitRGB")
    mmcore.snapImage()
    qtbot.waitUntil(lambda: "RGB" in preview.viewer._lut_controllers)

    assert preview.viewer._lut_controllers["RGB"].handles
    assert preview._white_balance_btn is None
    assert "Could not add the white-balance button" in caplog.text
