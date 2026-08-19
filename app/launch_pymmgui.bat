@echo off
title pymmcore-gui
cd /d "%~dp0.."

uv run mmgui
if errorlevel 1 (
    echo.
    echo pymmcore-gui exited with an error - see above for details.
    pause
)
