#!/usr/bin/env python3
"""Standalone recorder/importer for local transcript and summary notes.

This is the extracted notes side of type-by-voice:
record or import audio, transcribe it locally with faster-whisper, and write
transcript.md, summary.md, audio.wav, and meta.json into a session folder.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import signal
import sys
import threading
import time
import webbrowser
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

CONFIG_PATH = Path.home() / ".config" / "voice-notes" / "config.toml"

DEFAULT_CONFIG = """\
# voice-notes configuration

[model]
# faster-whisper model. large-v3-turbo is fast and multilingual.
name = "large-v3-turbo"
# auto -> CUDA if available, else CPU. Or force "cuda" / "cpu".
device = "auto"
# auto -> float16 on GPU, int8 on CPU. Or "float16" / "int8" / "int8_float16".
compute_type = "auto"
# Default to Chinese for this separated notes workflow. Use "auto" only when
# one recording intentionally mixes languages.
language = "zh"
auto_languages = ["zh", "en", "ja"]
auto_language_threshold = 0.45
initial_prompt = "以下是一段中文语音记录，请使用简体中文忠实转写。"

[transcription]
# Accuracy-oriented defaults. Longer chunks preserve more Chinese context while
# still updating progress regularly.
chunk_seconds = 30.0
chunk_radius = 5.0
live_max_chunk_seconds = 24.0
live_min_chunk_seconds = 4.0
beam_size = 8
condition_on_previous_text = true
vad_filter = true

[audio]
sample_rate = 16000
# Input device: name substring or numeric index. Empty = system default.
device = ""

[recording]
# Where session folders are written.
dir = "~/.local/share/voice-notes/sessions"
# Also capture speaker/system audio when the platform supports it.
capture_system = true
# Keep the mixed/imported audio as audio.wav.
keep_audio = true

[summary]
# auto = OpenAI-compatible server (openai_url, e.g. LM Studio) if configured
# and reachable, else local Ollama, else built-in extractive fallback.
# Other values: "openai", "ollama", "builtin".
backend = "auto"
ollama_model = "qwen2.5:7b"
ollama_url = "http://127.0.0.1:11434"
# OpenAI-compatible server, e.g. LM Studio on another machine via Tailscale:
# openai_url = "http://100.x.x.x:1234/v1". Empty url = disabled.
# Empty model = use the first model the server reports.
openai_url = ""
openai_model = ""
openai_api_key = "lm-studio"

[server]
host = "127.0.0.1"
port = 8765
open_browser = true
"""


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(DEFAULT_CONFIG)
        print(f"[init] Created default config at {CONFIG_PATH}")
    with CONFIG_PATH.open("rb") as f:
        cfg = tomllib.load(f)

    base = tomllib.loads(DEFAULT_CONFIG)
    for section, values in base.items():
        cfg.setdefault(section, {})
        for key, value in values.items():
            cfg[section].setdefault(key, value)
    return cfg


class MicRecorder:
    """Small mic stream that feeds every block to an active NotesSession."""

    def __init__(self, sample_rate: int, device):
        import numpy as np
        import sounddevice as sd

        self._np = np
        self.sample_rate = sample_rate
        self.device = device
        self.tap = None
        self.level = 0.0
        self.stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device if device not in ("", None) else None,
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status):
        tap = self.tap
        if tap is not None:
            np = self._np
            block = indata.copy().astype(np.float32, copy=False).flatten()
            self.level = float(np.sqrt(np.mean(np.square(block), dtype=np.float64)))
            try:
                tap(block)
            except Exception:
                pass
        else:
            self.level = 0.0

    def start(self):
        self.stream.start()

    def close(self):
        self.tap = None
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


class VoiceNotesApp:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.notes_dir = Path(str(cfg["recording"]["dir"])).expanduser()
        self.notes_dir.mkdir(parents=True, exist_ok=True)

        self.language = str(cfg["model"]["language"]).lower()
        self.auto_languages = [
            str(l).lower() for l in cfg["model"].get("auto_languages", [])
        ]
        self.initial_prompt = str(cfg["model"].get("initial_prompt", "")).strip()
        self.auto_language_threshold = float(
            cfg["model"].get("auto_language_threshold", 0.45)
        )
        self.tcfg = cfg.get("transcription", {})
        self._last_language: str | None = None
        if self.language == "auto":
            self.language = None

        self.model = None
        self._ready = threading.Event()
        self._load_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._stop = threading.Event()
        self._model_error = ""

        self.recorder: MicRecorder | None = None
        self.notes_session = None
        self.import_sessions = {}
        self.notes_server = None

    def _audio_device(self):
        device_cfg = self.cfg["audio"]["device"]
        if isinstance(device_cfg, str) and device_cfg.isdigit():
            return int(device_cfg)
        return device_cfg

    def ensure_recorder(self) -> bool:
        if self.recorder is not None:
            return True
        try:
            rec = MicRecorder(int(self.cfg["audio"]["sample_rate"]), self._audio_device())
            rec.start()
        except Exception as e:
            print(f"[error] Could not open microphone: {e}")
            return False
        self.recorder = rec
        return True

    def load_model(self) -> bool:
        if self._ready.is_set():
            return True
        with self._load_lock:
            if self._ready.is_set():
                return True
            try:
                self.model = self._load_model()
            except Exception as e:
                self._model_error = str(e)
                print(f"[error] Model load failed: {e}")
                return False
            self._ready.set()
            print("[ready] Model loaded.")
            return True

    def _load_model(self):
        _preload_cuda_libs()
        from faster_whisper import WhisperModel

        name = str(self.cfg["model"]["name"])
        device = str(self.cfg["model"]["device"]).lower()
        compute_type = str(self.cfg["model"]["compute_type"]).lower()

        if device == "auto":
            device = "cuda" if _cuda_available() else "cpu"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        print(f"[model] Loading '{name}' on {device} ({compute_type}).")
        try:
            return WhisperModel(name, device=device, compute_type=compute_type)
        except Exception as e:
            if device == "cuda":
                print(f"[warn] CUDA load failed ({e}); falling back to CPU/int8.")
                return WhisperModel(name, device="cpu", compute_type="int8")
            raise

    def _pick_language(self, audio) -> str | None:
        if self.language is not None:
            return self.language
        if not self.auto_languages:
            return None
        try:
            _, _, probs = self.model.detect_language(audio=audio)
        except Exception as e:
            print(f"[warn] Language detection failed ({e}); letting Whisper decide.")
            return None
        ranked = {lang: p for lang, p in probs}
        best = max(self.auto_languages, key=lambda lang: ranked.get(lang, 0.0))
        conf = ranked.get(best, 0.0)
        if conf < self.auto_language_threshold and self._last_language:
            print(
                f"[lang] keep {self._last_language} "
                f"(best {best} {conf:.2f} < {self.auto_language_threshold:.2f})"
            )
            return self._last_language
        self._last_language = best
        print(f"[lang] {best} ({conf:.2f})")
        return best

    def set_language(self, language: str | None):
        lang = (language or "").strip().lower()
        if not lang:
            return
        self.language = None if lang == "auto" else lang
        self._last_language = None
        print(f"[language] {'auto' if self.language is None else self.language}")

    def _transcribe_kwargs(self, audio, context: str = "") -> dict:
        language = self._pick_language(audio)
        prompt = self.initial_prompt if language == "zh" else ""
        context = " ".join((context or "").split())
        if context:
            label = {
                "zh": "前文上下文：",
                "ja": "前文の文脈：",
                "en": "Previous context: ",
            }.get(language, "Context: ")
            prompt = (prompt + "\n" if prompt else "") + f"{label}{context[-500:]}"
        kwargs = {
            "language": language,
            "beam_size": int(self.tcfg.get("beam_size", 8)),
            "vad_filter": bool(self.tcfg.get("vad_filter", True)),
            "condition_on_previous_text": bool(
                self.tcfg.get("condition_on_previous_text", True)
            ),
        }
        if prompt:
            kwargs["initial_prompt"] = prompt
        return kwargs

    def _notes_transcribe(self, audio, context: str = "") -> str:
        with self._model_lock:
            segments, _ = self.model.transcribe(
                audio, **self._transcribe_kwargs(audio, context)
            )
            return _clean_transcript_text("".join(seg.text for seg in segments))

    def _notes_transcribe_segments(self, audio, context: str = "") -> list:
        with self._model_lock:
            segments, _ = self.model.transcribe(
                audio, **self._transcribe_kwargs(audio, context)
            )
            return [
                (seg.start, text)
                for seg in segments
                if (text := _clean_transcript_text(seg.text))
            ]

    def _notes_summarizer(self):
        from voice_notes import Summarizer

        scfg = self.cfg["summary"]
        return Summarizer(
            backend=str(scfg["backend"]),
            ollama_model=str(scfg["ollama_model"]),
            ollama_url=str(scfg["ollama_url"]),
            openai_url=str(scfg["openai_url"]),
            openai_model=str(scfg["openai_model"]),
            openai_api_key=str(scfg["openai_api_key"]),
        )

    def create_import_session(self, path: Path, title: str, source: str = "upload",
                              session_id: str | None = None,
                              resume: bool = False):
        if not self._ready.is_set():
            return None
        from voice_notes import ImportSession

        session = ImportSession(
            base_dir=self.notes_dir,
            sample_rate=int(self.cfg["audio"]["sample_rate"]),
            src_path=path,
            title=title,
            transcribe_segments_fn=self._notes_transcribe_segments,
            summarizer=self._notes_summarizer(),
            source=source,
            on_done=self._on_notes_done,
            chunk_target=float(self.tcfg.get("chunk_seconds", 30.0)),
            chunk_radius=float(self.tcfg.get("chunk_radius", 5.0)),
            session_id=session_id,
            resume=resume,
        )
        self.import_sessions[session.id] = session
        print(f"[import] {session.name} -> {self._notes_url(session)}")
        return session

    def start_import(self, path, title: str, source: str = "upload"):
        session = self.create_import_session(Path(path), title, source)
        return session.id if session else None

    def resume_import(self, sid: str):
        session = self.import_sessions.get(sid)
        if session and getattr(session, "_thread", None) and session._thread.is_alive():
            return sid
        meta_path = self.notes_dir / sid / "meta.json"
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            return None
        source_path = meta.get("source_path")
        path = Path(source_path) if source_path else self.notes_dir / sid / "audio.wav"
        if not path.exists():
            return None
        session = self.create_import_session(
            path,
            meta.get("title") or path.stem,
            meta.get("source") or "upload",
            session_id=sid,
            resume=True,
        )
        return session.id if session else None

    def start_notes_session(self):
        if self.notes_session:
            return None
        if not self._ready.is_set():
            return None
        if not self.ensure_recorder():
            return None

        from voice_notes import NotesSession

        rcfg = self.cfg["recording"]
        session = NotesSession(
            base_dir=self.notes_dir,
            sample_rate=int(self.cfg["audio"]["sample_rate"]),
            transcribe_fn=self._notes_transcribe,
            summarizer=self._notes_summarizer(),
            capture_system=bool(rcfg["capture_system"]),
            keep_audio=bool(rcfg["keep_audio"]),
            on_done=self._on_notes_done,
            min_chunk_sec=float(self.tcfg.get("live_min_chunk_seconds", 4.0)),
            max_chunk_sec=float(self.tcfg.get("live_max_chunk_seconds", 24.0)),
        )
        self.notes_session = session
        self.recorder.tap = session.feed_mic
        src = "mic + speakers" if session.system_audio else "mic only"
        print(f"[record] Started ({src}) -> {self._notes_url(session)}")
        if self.cfg["server"].get("open_browser", True) and self.notes_server:
            webbrowser.open(self._notes_url(session))
        return session.id

    def stop_notes_session(self):
        session = self.notes_session
        if not session:
            return None
        self.notes_session = None
        if self.recorder:
            self.recorder.tap = None
        session.stop()
        print("[record] Stopped; generating transcript and summary...")
        return session

    def active(self):
        return self.notes_session

    def _on_notes_done(self, session):
        if getattr(session, "status", "") == "error":
            print(f"[done] Failed: {getattr(session, 'error', '')}")
        else:
            print(f"[done] Summary ready: {session.dir / 'summary.md'}")

    def _notes_url(self, session=None) -> str:
        if self.notes_server and self.notes_server.url:
            return f"{self.notes_server.url}/s/{session.id}" if session else self.notes_server.url
        return str(session.dir if session else self.notes_dir)

    def start_server(self):
        from types import SimpleNamespace
        from voice_notes import NotesServer

        scfg = self.cfg["server"]
        self.notes_server = NotesServer(
            self.notes_dir,
            SimpleNamespace(
                start=self.start_notes_session,
                stop=self.stop_notes_session,
                active=self.active,
                import_file=self.start_import,
                resume_import=self.resume_import,
                set_language=self.set_language,
            ),
            host=str(scfg["host"]),
            port=int(scfg["port"]),
        )
        return self.notes_server.start()

    def shutdown(self):
        if self._stop.is_set():
            return
        self._stop.set()
        session = self.stop_notes_session()
        if session:
            session.wait()
        if self.recorder:
            self.recorder.close()
        if self.notes_server:
            self.notes_server.stop()


def _preload_cuda_libs():
    """Load pip-installed CUDA/cuDNN libs before faster-whisper opens them."""
    import ctypes
    import importlib.util

    spec = importlib.util.find_spec("nvidia")
    if not spec or not spec.submodule_search_locations:
        return
    base = list(spec.submodule_search_locations)[0]
    patterns = [
        "cublas/lib/libcublasLt.so*",
        "cublas/lib/libcublas.so*",
        "cuda_nvrtc/lib/libnvrtc*.so*",
        "cudnn/lib/libcudnn_*.so*",
        "cudnn/lib/libcudnn.so*",
    ]
    pending = []
    for pat in patterns:
        pending.extend(sorted(glob.glob(os.path.join(base, pat))))
    pending = list(dict.fromkeys(pending))
    for _ in range(3):
        if not pending:
            break
        still = []
        for path in pending:
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                still.append(path)
        pending = still


def _cuda_available() -> bool:
    try:
        from ctranslate2 import get_cuda_device_count

        return get_cuda_device_count() > 0
    except Exception:
        return shutil.which("nvidia-smi") is not None


_COMMON_WHISPER_HALLUCINATIONS = {
    "ill see you next time",
    "i ll see you next time",
    "see you next time",
    "thank you for watching",
    "thanks for watching",
    "thank you",
}


def _clean_transcript_text(text: str) -> str:
    text = " ".join((text or "").split())
    ascii_norm = re.sub(r"[^a-z]+", " ", text.lower()).strip()
    if ascii_norm in _COMMON_WHISPER_HALLUCINATIONS:
        return ""
    return text


def apply_cli_overrides(cfg: dict, args) -> dict:
    if args.data_dir:
        cfg["recording"]["dir"] = args.data_dir
    if args.host:
        cfg["server"]["host"] = args.host
    if args.port:
        cfg["server"]["port"] = args.port
    if args.no_browser:
        cfg["server"]["open_browser"] = False
    if args.language:
        cfg["model"]["language"] = args.language
    return cfg


def run_server(args) -> int:
    app = VoiceNotesApp(apply_cli_overrides(load_config(), args))

    def stop(*_):
        app.shutdown()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    url = app.start_server()
    if not url:
        return 1
    print(f"[server] Open {url}")
    if app.cfg["server"].get("open_browser", True):
        webbrowser.open(url)

    threading.Thread(target=app.load_model, daemon=True).start()
    while not app._stop.wait(0.2):
        pass
    return 0


def run_record(args) -> int:
    app = VoiceNotesApp(apply_cli_overrides(load_config(), args))
    if app.cfg["server"].get("open_browser", True):
        app.start_server()
    if not app.load_model():
        return 1
    sid = app.start_notes_session()
    if not sid:
        app.shutdown()
        return 1
    try:
        input("Recording. Press Enter to stop.\n")
    except KeyboardInterrupt:
        pass
    session = app.stop_notes_session()
    if session:
        session.wait()
    app.shutdown()
    return 0


def run_import(args) -> int:
    app = VoiceNotesApp(apply_cli_overrides(load_config(), args))
    path = Path(args.import_path).expanduser()
    if not path.exists():
        print(f"[error] File not found: {path}")
        return 1
    if not app.load_model():
        return 1
    session = app.create_import_session(
        path,
        args.title or path.stem,
        source="upload",
    )
    if not session:
        return 1
    session._thread.join()
    return 0 if session.status != "error" else 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record or import audio and turn it into local notes."
    )
    parser.add_argument("--list-devices", action="store_true", help="list audio devices")
    parser.add_argument("--record", action="store_true", help="record in the terminal")
    parser.add_argument("--import", dest="import_path", help="import an audio file")
    parser.add_argument("--title", help="title for --import output")
    parser.add_argument("--data-dir", help="override the session data directory")
    parser.add_argument("--language", help='transcription language, e.g. "zh", "en", "ja", or "auto"')
    parser.add_argument("--host", help="override server host")
    parser.add_argument("--port", type=int, help="override server port")
    parser.add_argument("--no-browser", action="store_true", help="do not open browser")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return 0
    if args.import_path:
        return run_import(args)
    if args.record:
        return run_record(args)
    return run_server(args)


if __name__ == "__main__":
    raise SystemExit(main())
