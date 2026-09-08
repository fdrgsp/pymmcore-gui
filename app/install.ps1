# pymmcore-gui installer (source-based, works around the packaged .exe
# crashing on machines running CrowdStrike Falcon - see the "cite" branch
# discussion for the crash analysis).
#
# Usage (from a fresh PowerShell window, no admin rights required):
#
#   irm https://raw.githubusercontent.com/fdrgsp/pymmcore-gui/cite/app/install.ps1 | iex
#
# What it does:
#   1. Installs `uv` (the Python package/venv manager) if it isn't already
#      on PATH, via astral's official installer.
#   2. Deletes %LOCALAPPDATA%\pymmcore-gui-app if it exists, then clones a
#      fresh copy of this repo's `cite` branch there and `uv sync`s it.
#   3. Creates a "pymmcore-gui" shortcut on the Desktop, with the app's
#      logo, that runs the GUI from source via `uv run` -- no PyInstaller
#      build involved, so nothing here is flagged/broken by endpoint
#      security software the way the packaged .exe is.
#
# Safe to re-run: every run starts from a fully deleted install directory,
# so there's never any local/stale state to fight with -- just delete and
# reinstall. The launcher itself does not self-update; re-run this script
# whenever you want the latest version.

$ErrorActionPreference = "Stop"

$RepoUrl = "https://github.com/fdrgsp/pymmcore-gui.git"
$Branch = "cite"
$InstallDir = "$env:LOCALAPPDATA\pymmcore-gui-app"
$ShortcutPath = "$env:USERPROFILE\Desktop\pymmcore-gui.lnk"

function Write-Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

# --- 1. uv -------------------------------------------------------------

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Step "Installing uv..."
    irm https://astral.sh/uv/install.ps1 | iex
    # the installer updates the persistent PATH, but this session doesn't
    # see it yet -- pick up its default install location for the rest of
    # this run.
    $uvDir = "$env:USERPROFILE\.local\bin"
    if (Test-Path "$uvDir\uv.exe") {
        $env:Path = "$uvDir;$env:Path"
    }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv installation did not put 'uv' on PATH. Open a new PowerShell window and re-run this script."
    }
} else {
    Write-Host "uv is already installed."
}

# --- 2. git --------------------------------------------------------------

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Step "git not found -- attempting to install via winget..."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id Git.Git -e --source winget --accept-source-agreements --accept-package-agreements
        $env:Path = "$env:ProgramFiles\Git\cmd;$env:Path"
    }
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "git is required. Install it from https://git-scm.com/downloads and re-run this script."
    }
}

# --- 3. clone a fresh copy of the repo ------------------------------------
# Always starts by deleting InstallDir outright rather than resetting it in
# place: no local edits, stray files, or stale venv ever survive from a
# previous run, so there's nothing for this script to ever get stuck on.

if (Test-Path $InstallDir) {
    Write-Step "Removing existing install at $InstallDir..."
    Remove-Item -Recurse -Force $InstallDir
}

Write-Step "Cloning pymmcore-gui ('$Branch') to $InstallDir..."
git clone --branch $Branch $RepoUrl $InstallDir
git -C $InstallDir checkout $Branch

# --- 4. sync environment --------------------------------------------------

Write-Step "Syncing the Python environment..."
uv sync --project $InstallDir -U

# --- 5. desktop shortcut ---------------------------------------------------

Write-Step "Creating Desktop shortcut..."

$batPath = "$InstallDir\app\launch_pymmgui.bat"
$iconPath = "$InstallDir\src\pymmcore_gui\resources\icon.ico"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($ShortcutPath)
$shortcut.TargetPath = $batPath
$shortcut.WorkingDirectory = $InstallDir
$shortcut.IconLocation = $iconPath
$shortcut.WindowStyle = 1
$shortcut.Description = "pymmcore-gui (run from source via uv)"
$shortcut.Save()

Write-Step "Done!"
Write-Host "Double-click 'pymmcore-gui' on your Desktop to launch." -ForegroundColor Green
