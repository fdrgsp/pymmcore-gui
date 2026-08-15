from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import patch

from pymmcore_plus import __version__
from typer.testing import CliRunner

from pymmcore_gui import _settings
from pymmcore_gui._cli import app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()
subrun = subprocess.run


def test_show_version() -> None:
    """show version should work."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "pymmcore-plus" in result.stdout
    assert __version__ in result.stdout


def test_settings(tmp_path: Path) -> None:
    """show version should work."""
    with patch.object(_settings, "reset_to_defaults") as mock_reset:
        result = runner.invoke(app, ["settings", "--reset"])
        assert result.exit_code == 0
        mock_reset.assert_called_once()


def test_default_command_forwards_config_to_standard_gui(tmp_path: Path) -> None:
    config = tmp_path / "startup.cfg"
    config.touch()
    argv = ["mmgui", "-c", str(config)]

    with (
        patch("sys.argv", argv),
        patch("pymmcore_gui.create_mmgui") as mock_create,
    ):
        result = runner.invoke(app, argv[1:])

    assert result.exit_code == 0
    assert mock_create.call_args.kwargs["mm_config"] == config.resolve()
    # by default the standard (dock-based) MicroManagerGUI is used
    assert mock_create.call_args.kwargs["window_cls"] is None


def test_modern_flag_uses_modern_gui() -> None:
    with patch("pymmcore_gui.create_mmgui") as mock_create:
        result = runner.invoke(app, ["run", "--modern"])

    assert result.exit_code == 0
    assert (
        mock_create.call_args.kwargs["window_cls"]
        == "pymmcore_gui._modern_gui.MainWindow"
    )


def test_layout_flag_is_forwarded(tmp_path: Path) -> None:
    with patch("pymmcore_gui.create_mmgui") as mock_create:
        result = runner.invoke(app, ["run", "--modern", "-l", "My rig"])

    assert result.exit_code == 0
    assert mock_create.call_args.kwargs["layout"] == "My rig"


def test_unknown_layout_warns_but_still_launches() -> None:
    """A stale `-l` should open the app on the default layout, not refuse to start."""
    with patch("pymmcore_gui.create_mmgui") as mock_create:
        result = runner.invoke(app, ["run", "--modern", "-l", "nope"])

    assert result.exit_code == 0
    assert "No layout named 'nope'" in result.stdout
    mock_create.assert_called_once()


def test_layouts_command_lists_the_built_in_layout() -> None:
    result = runner.invoke(app, ["layouts"])
    assert result.exit_code == 0
    assert "Default" in result.stdout
