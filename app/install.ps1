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
#   2. Clones this repo (or updates it, if already installed) to
#      %LOCALAPPDATA%\pymmcore-gui.
#   3. Creates a "pymmcore-gui" shortcut on the Desktop, with the app's
#      logo, that runs the GUI from source via `uv run` -- no PyInstaller
#      build involved, so nothing here is flagged/broken by endpoint
#      security software the way the packaged .exe is.
#
# Safe to re-run: it just pulls the latest changes and rewrites the
# shortcut.

$ErrorActionPreference = "Stop"

$RepoUrl = "https://github.com/fdrgsp/pymmcore-gui.git"
$Branch = "cite"
$InstallDir = "$env:LOCALAPPDATA\pymmcore-gui"
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

# --- 3. clone or update the repo -----------------------------------------
# Always ends with InstallDir being a pristine checkout of $Branch: any
# local edits, stray files, or a half-finished/non-git directory from a
# previous run are wiped rather than fought with. Re-running this script
# is the supported way to reset a broken or hand-modified install.

if (Test-Path "$InstallDir\.git") {
    Write-Step "Resetting existing install at $InstallDir to a clean copy of '$Branch'..."
    git -C $InstallDir fetch origin $Branch
    git -C $InstallDir checkout $Branch
    git -C $InstallDir reset --hard "origin/$Branch"
    git -C $InstallDir clean -fdx
} else {
    if (Test-Path $InstallDir) {
        Write-Step "Removing non-git contents of $InstallDir..."
        Remove-Item -Recurse -Force $InstallDir
    }
    Write-Step "Cloning pymmcore-gui to $InstallDir..."
    git clone --branch $Branch $RepoUrl $InstallDir
}

# --- 4. sync environment --------------------------------------------------
# Keeps .venv in step with whatever code was just checked out above --
# without this, a reset/clone can leave the environment stale (e.g. missing
# console scripts like `mmgui`) until the next `uv run` happens to notice.

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
