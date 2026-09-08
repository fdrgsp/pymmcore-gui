@echo off
title pymmcore-gui
cd /d "%~dp0.."

echo Updating to the latest "cite" version...
git checkout cite
git pull
uv sync -U

uv run mmgui
if errorlevel 1 (
    echo.
    echo pymmcore-gui exited with an error - see above for details.
    pause
)
