#!/usr/bin/env bash
# One-command installer for voice-notes (macOS / Linux).
#
#   ./install.sh
#
# Sets up everything for a fully-local, offline-capable install:
#   1. a Python venv + dependencies
#   2. pre-downloads the local Whisper transcription model
#   3. installs Ollama and pulls a small local LLM for the summaries
#   4. writes an offline-friendly config
#
# After it finishes both transcription AND summaries run locally — no cloud,
# no Mac Studio required. Override the models with env vars:
#   VOICE_NOTES_WHISPER_MODEL=small  VOICE_NOTES_LLM=llama3.2:3b  ./install.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd -P)"
cd "$HERE"

WHISPER_MODEL="${VOICE_NOTES_WHISPER_MODEL:-large-v3-turbo}"   # try 'small' on low-end machines
LLM_MODEL="${VOICE_NOTES_LLM:-qwen2.5:3b}"                     # small local summary model
CONFIG="$HOME/.config/voice-notes/config.toml"

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }

# --- 1. Python + venv --------------------------------------------------------
say "Checking Python (need 3.11+)…"
PY="$(command -v python3 || command -v python || true)"
[ -n "$PY" ] || { warn "Python 3.11+ not found. Install it from https://python.org"; exit 1; }
"$PY" - <<'PYEOF' || { echo "Python 3.11+ required." >&2; exit 1; }
import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)
PYEOF

say "Creating virtualenv (.venv) and installing dependencies…"
"$PY" -m venv .venv
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -r requirements.txt

# --- 2. Pre-download the Whisper model --------------------------------------
say "Downloading local transcription model '$WHISPER_MODEL' (one-time)…"
./.venv/bin/python - "$WHISPER_MODEL" <<'PYEOF'
import sys
from faster_whisper import WhisperModel
WhisperModel(sys.argv[1], device="cpu", compute_type="int8")
print("  transcription model ready.")
PYEOF

# --- 3. Ollama + a small local summary model --------------------------------
if ! command -v ollama >/dev/null 2>&1; then
  say "Installing Ollama (local LLM runtime)…"
  if [ "$(uname)" = "Darwin" ]; then
    if command -v brew >/dev/null 2>&1; then
      brew install ollama
    else
      warn "Homebrew not found. Install Ollama from https://ollama.com/download then re-run."
      exit 1
    fi
  else
    curl -fsSL https://ollama.com/install.sh | sh
  fi
fi

say "Starting Ollama and pulling small summary model '$LLM_MODEL'…"
if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  nohup ollama serve >/tmp/voice-notes-ollama.log 2>&1 &
  for _ in $(seq 1 30); do
    curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
  done
fi
ollama pull "$LLM_MODEL"

# --- 4. Offline-friendly config ---------------------------------------------
say "Writing config ($CONFIG)…"
mkdir -p "$(dirname "$CONFIG")"
# Let the app create defaults if absent, then point the summary backend at the
# local Ollama model we just pulled (never overwrite a user's existing config).
if [ ! -f "$CONFIG" ]; then
  ./.venv/bin/python -c "import record_notes" 2>/dev/null || true
  ./.venv/bin/python - "$CONFIG" "$LLM_MODEL" <<'PYEOF'
import sys, pathlib
cfg, model = pathlib.Path(sys.argv[1]), sys.argv[2]
import record_notes
cfg.parent.mkdir(parents=True, exist_ok=True)
if not cfg.exists():
    cfg.write_text(record_notes.DEFAULT_CONFIG)
text = cfg.read_text().replace('ollama_model = "qwen2.5:7b"', f'ollama_model = "{model}"')
cfg.write_text(text)
print("  config uses local Ollama model:", model)
PYEOF
else
  warn "Existing config kept as-is: $CONFIG (summary backend not changed)."
fi

say "Done ✅  Everything runs locally."
echo
echo "  Start the web UI:   ./record-notes"
echo "  Then open:          http://127.0.0.1:8765"
echo
echo "  (Keep 'ollama serve' running for local summaries; on macOS the Ollama"
echo "   app or 'brew services start ollama' does that automatically.)"
