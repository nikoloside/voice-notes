# Migration From type-by-voice

This folder separates the notes pipeline from `type-by-voice`.

## Old locations

`type-by-voice` stored notes by default under:

```text
~/.local/share/voice-term/notes/
```

The new default is:

```text
~/.local/share/voice-notes/sessions/
```

To keep old sessions visible in the new UI, copy the timestamped session
folders from the old root to the new root. Do not copy `_uploads` unless you
need the original uploaded source files.

## Config mapping

Old `[notes]` keys map to the new config like this:

| Old key | New key |
| --- | --- |
| `notes.dir` | `recording.dir` |
| `notes.capture_system` | `recording.capture_system` |
| `notes.keep_audio` | `recording.keep_audio` |
| `notes.summarizer` | `summary.backend` |
| `notes.ollama_model` | `summary.ollama_model` |
| `notes.ollama_url` | `summary.ollama_url` |
| `notes.port` | `server.port` |
| `notes.open_browser` | `server.open_browser` |

`notes.key` has no replacement because `voice-notes` intentionally does not
own global hotkeys.

## Source split

The reusable notes implementation was copied from:

```text
type-by-voice/voice_notes.py
```

The new standalone wrapper copies the model loading and transcription bridge
from `voice_term.py` into `record_notes.py`, without bringing over dictation,
tray, hotkey, or paste behavior.
