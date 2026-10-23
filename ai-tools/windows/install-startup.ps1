# Adds (or with -Remove, deletes) a Startup-folder shortcut that launches the
# AI assistant tray icon at login. The tray starts the daemon itself.
#
#   powershell -ExecutionPolicy Bypass -File ai-tools\windows\install-startup.ps1
#   powershell -ExecutionPolicy Bypass -File ai-tools\windows\install-startup.ps1 -Remove
#
# Run `python ai-tools\ai_assistant.py --tray` from a terminal once first, so
# the .venv bootstrap and pip installs happen where you can see them.
param(
    [string]$Python = "",
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$shortcut = Join-Path ([Environment]::GetFolderPath("Startup")) "AI Assistant.lnk"

if ($Remove) {
    if (Test-Path $shortcut) { Remove-Item $shortcut; Write-Host "Removed $shortcut" }
    else { Write-Host "No startup shortcut found." }
    return
}

$script = (Resolve-Path (Join-Path $PSScriptRoot "..\ai_assistant.py")).Path
$venvPythonw = Join-Path $PSScriptRoot "..\.venv\Scripts\pythonw.exe"

if (-not $Python) {
    if (Test-Path $venvPythonw) {
        $Python = (Resolve-Path $venvPythonw).Path
    } else {
        $cmd = Get-Command pythonw.exe -ErrorAction SilentlyContinue
        if (-not $cmd) { throw "pythonw.exe not found on PATH; pass -Python C:\path\to\pythonw.exe" }
        $Python = $cmd.Source
    }
}

$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut($shortcut)
$link.TargetPath = $Python
$link.Arguments = "`"$script`" --tray"
$link.WorkingDirectory = Split-Path $script
$link.Description = "AI Assistant tray icon"
$link.Save()
Write-Host "Created $shortcut -> $Python `"$script`" --tray"
