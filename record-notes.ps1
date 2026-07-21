# Windows launcher for voice-notes. Runs the app in the project venv.
#   .\record-notes.ps1              # start the localhost web UI
#   .\record-notes.ps1 --import file.m4a
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
  Write-Host "venv not found. Run: powershell -ExecutionPolicy Bypass -File install.ps1" -ForegroundColor Yellow
  exit 1
}
$env:PYTHONUNBUFFERED = "1"
& $py "record_notes.py" @args
