@echo off
title pymmcore-gui
cd /d "%~dp0.."

set attempts=0

:run
uv run mmgui
if not errorlevel 1 goto done

set /a attempts+=1
if %attempts% geq 2 goto fail

echo.
echo mmgui failed to start - this is usually antivirus briefly locking the
echo executable right after install. Retrying in 3 seconds...
timeout /t 3 /nobreak >nul
goto run

:fail
echo.
echo pymmcore-gui exited with an error - see above for details.
pause

:done
