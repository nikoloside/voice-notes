# Architecture

`voice-notes` is the extracted recording and notes pipeline from
`type-by-voice`.

## Boundary

Included here:

- Long-form microphone recording
- Optional speaker/system audio capture
- Audio file upload/import
- macOS Voice Memos import
- Local Whisper transcription
- Local summary generation
- Localhost session browser and download UI

Excluded here:

- Global dictation hotkeys
- Clipboard paste or direct typing into another app
- Tray indicator and desktop launcher integration

## Main files

- `record_notes.py`: standalone application entry point, model loading, mic
  stream, CLI, and server controller.
- `voice_notes.py`: session model, audio decoding, Voice Memos listing,
  summarization, localhost UI, and file layout.
- `record-notes`: venv launcher that also exposes pip-installed CUDA/cuDNN
  libraries to `faster-whisper`.

## Data layout

Default root:

```text
~/.local/share/voice-notes/sessions/
```

Session directories:

```text
20260713-163500/
  meta.json
  transcript.md
  notes.md
  summary.md
  chunks.json
  audio.wav
```

Uploads are first stored under `_uploads/`, then processed into a normal
session directory.

## Runtime flow

Recording:

```text
MicRecorder -> NotesSession.feed_mic()
  + optional SystemAudioTap
  -> silence-based chunks
  -> faster-whisper
  -> transcript.md
  -> notes.md running readable notes
  -> Summarizer
  -> summary.md
```

Import:

```text
audio file -> decode_audio()
  -> split_for_transcription()
  -> for each chunk:
       skip if chunks.json already marks it done
       faster-whisper with timestamps and previous transcript context
       transcript.md
       notes.md core notes and value/todo/checkpoints
       chunks.json checkpoint
  -> Summarizer
  -> summary.md
```

The localhost server is unauthenticated and binds to `127.0.0.1` by default.
Keep it on loopback unless you intentionally add authentication.
