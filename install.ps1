# One-command installer for voice-notes (Windows / PowerShell).
#
#   powershell -ExecutionPolicy Bypass -File install.ps1
#
# Sets up a fully-local, offline-capable install:
#   1. a Python venv + dependencies
#   2. pre-downloads the local Whisper transcription model
#   3. installs Ollama and pulls a small local LLM for the summaries
#   4. writes an offline-friendly config
#
# Override models:  $env:VOICE_NOTES_WHISPER_MODEL="small"; $env:VOICE_NOTES_LLM="llama3.2:3b"
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$WhisperModel = if ($env:VOICE_NOTES_WHISPER_MODEL) { $env:VOICE_NOTES_WHISPER_MODEL } else { "large-v3-turbo" }
$LlmModel     = if ($env:VOICE_NOTES_LLM) { $env:VOICE_NOTES_LLM } else { "qwen2.5:3b" }
$Config       = Join-Path $env:APPDATA "voice-notes\config.toml"

function Say($m)  { Write-Host "==> $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "!! $m" -ForegroundColor Yellow }

# --- 1. Python + venv --------------------------------------------------------
Say "Checking Python (need 3.11+)…"
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command python3 -ErrorAction SilentlyContinue }
if (-not $py) { Warn "Python 3.11+ not found. Install from https://python.org"; exit 1 }

Say "Creating virtualenv (.venv) and installing dependencies…"
& $py.Path -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --quiet --upgrade pip
& .\.venv\Scripts\python.exe -m pip install --quiet -r requirements.txt

# --- 2. Pre-download the Whisper model --------------------------------------
Say "Downloading local transcription model '$WhisperModel' (one-time)…"
& .\.venv\Scripts\python.exe -c "import sys; from faster_whisper import WhisperModel; WhisperModel('$WhisperModel', device='cpu', compute_type='int8'); print('  transcription model ready.')"

# --- 3. Ollama + a small local summary model --------------------------------
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
  Say "Installing Ollama (local LLM runtime)…"
  if (Get-Command winget -ErrorAction SilentlyContinue) {
    winget install --silent --accept-package-agreements --accept-source-agreements Ollama.Ollama
  } else {
    Warn "winget not found. Install Ollama from https://ollama.com/download then re-run."
    exit 1
  }
}

Say "Starting Ollama and pulling small summary model '$LlmModel'…"
try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:11434/api/tags | Out-Null }
catch {
  Start-Process -WindowStyle Hidden ollama -ArgumentList "serve"
  for ($i=0; $i -lt 30; $i++) {
    try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:11434/api/tags | Out-Null; break }
    catch { Start-Sleep 1 }
  }
}
ollama pull $LlmModel

# --- 4. Offline-friendly config ---------------------------------------------
Say "Writing config ($Config)…"
New-Item -ItemType Directory -Force -Path (Split-Path $Config) | Out-Null
if (-not (Test-Path $Config)) {
  & .\.venv\Scripts\python.exe -c "import record_notes, pathlib; p=pathlib.Path(r'$Config'); p.write_text(record_notes.DEFAULT_CONFIG)"
  (Get-Content $Config) -replace 'ollama_model = "qwen2.5:7b"', "ollama_model = ""$LlmModel""" | Set-Content $Config
  Write-Host "  config uses local Ollama model: $LlmModel"
} else {
  Warn "Existing config kept as-is: $Config"
}

Say "Done. Everything runs locally."
Write-Host ""
Write-Host "  Start the web UI:   .\record-notes.ps1"
Write-Host "  Then open:          http://127.0.0.1:8765"
