#!/usr/bin/env python3
"""voice-term notes: continuous session recording -> live transcript -> summary.md.

A "notes session" records continuously (microphone + what your speakers are
playing, mixed), transcribes it locally chunk by chunk with the same
faster-whisper model the push-to-talk path uses, and when you stop it, runs a
local summarization pipeline that writes notes + a summary as markdown.

Everything is viewable live on a small localhost web page:

    http://127.0.0.1:8765          all sessions (+ start/stop buttons)
    http://127.0.0.1:8765/s/<id>   one session: live transcript -> summary.md

Summarization stays local too: Ollama if it's running (best quality), else a
built-in extractive fallback (zero dependencies). See README.md.
"""
from __future__ import annotations

import inspect
import json
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import request as urlrequest
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from graph_build import aggregate_graph, entity_names_by_type


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Cheap linear resample — plenty for 16 kHz speech transcription."""
    if src_sr == dst_sr or len(audio) == 0:
        return audio.astype(np.float32, copy=False)
    n = max(1, int(round(len(audio) * dst_sr / src_sr)))
    return np.interp(np.linspace(0.0, len(audio) - 1.0, n),
                     np.arange(len(audio)), audio).astype(np.float32)


# --------------------------------------------------------------------------- #
# System (speaker) audio capture
# --------------------------------------------------------------------------- #
class SystemAudioTap:
    """Capture what the speakers are playing.

    Linux: the default sink's monitor via `parec` (PulseAudio *and* PipeWire's
    pulse shim ship it), asking the server to resample to our rate/mono.
    macOS: a loopback input device (BlackHole / Loopback) if one exists —
    CoreAudio has no built-in monitor source. `brew install blackhole-2ch`,
    then set a Multi-Output Device (speakers + BlackHole) as the output.
    Best-effort: if neither is available we simply record mic-only.
    """

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.level = 0.0
        self.proc = None
        self._thread = None
        self._sd_stream = None

    @staticmethod
    def default_monitor() -> str | None:
        if not (shutil.which("pactl") and shutil.which("parec")):
            return None
        try:
            out = subprocess.run(
                ["pactl", "get-default-sink"],
                capture_output=True, text=True, timeout=3,
            ).stdout.strip()
        except Exception:
            return None
        return f"{out}.monitor" if out else None

    def _start_darwin(self, on_audio) -> bool:
        """macOS: read a loopback device (BlackHole/Loopback) via sounddevice."""
        try:
            import sounddevice as sd
        except ImportError:
            return False
        dev = None
        for i, d in enumerate(sd.query_devices()):
            if (d["max_input_channels"] > 0
                    and re.search(r"blackhole|loopback", d["name"], re.I)):
                dev = i
                break
        if dev is None:
            print("[notes] No loopback device found — speaker audio needs "
                  "BlackHole (`brew install blackhole-2ch`) plus a "
                  "Multi-Output Device (speakers + BlackHole) as the output. "
                  "Recording mic only.")
            return False
        info = sd.query_devices(dev)
        src_sr = int(info["default_samplerate"])
        ch = min(2, info["max_input_channels"])

        def cb(indata, frames, time_info, status):
            mono = indata.mean(axis=1) if indata.ndim > 1 else indata
            self.level = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))
            try:
                on_audio(_resample(mono, src_sr, self.sample_rate))
            except Exception:
                pass

        try:
            self._sd_stream = sd.InputStream(
                device=dev, channels=ch, samplerate=src_sr,
                dtype="float32", callback=cb)
            self._sd_stream.start()
        except Exception as e:
            print(f"[notes] Could not open loopback device ({e}); mic only.")
            self._sd_stream = None
            return False
        print(f"[notes] Speaker audio via '{info['name']}'.")
        return True

    def start(self, on_audio) -> bool:
        """Start streaming speaker audio to `on_audio(float32 ndarray)`."""
        if sys.platform == "darwin":
            return self._start_darwin(on_audio)
        monitor = self.default_monitor()
        if not monitor:
            return False
        try:
            self.proc = subprocess.Popen(
                ["parec", "--device", monitor,
                 "--rate", str(self.sample_rate), "--channels", "1",
                 "--format", "float32le", "--latency-msec", "60"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
        except Exception:
            self.proc = None
            return False
        self._thread = threading.Thread(
            target=self._reader, args=(on_audio,), daemon=True)
        self._thread.start()
        return True

    def _reader(self, on_audio):
        block = int(self.sample_rate * 0.2) * 4  # 0.2 s of float32
        while self.proc and self.proc.poll() is None:
            data = self.proc.stdout.read(block)
            if not data:
                break
            audio = np.frombuffer(data, dtype=np.float32)
            self.level = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
            try:
                on_audio(audio)
            except Exception:
                pass
        self.level = 0.0

    def stop(self):
        proc, self.proc = self.proc, None
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        stream, self._sd_stream = self._sd_stream, None
        if stream:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass


class _StreamBuf:
    """A tiny thread-safe FIFO of audio samples (for aligning mic vs system)."""

    def __init__(self):
        self._chunks: list[np.ndarray] = []
        self._n = 0
        self._lock = threading.Lock()

    def push(self, audio: np.ndarray):
        if len(audio) == 0:
            return
        with self._lock:
            self._chunks.append(audio)
            self._n += len(audio)

    @property
    def available(self) -> int:
        return self._n

    def pull(self, n: int) -> np.ndarray:
        """Return exactly n samples (zero-padded if fewer are buffered)."""
        out = []
        got = 0
        with self._lock:
            while got < n and self._chunks:
                c = self._chunks[0]
                take = min(len(c), n - got)
                out.append(c[:take])
                if take == len(c):
                    self._chunks.pop(0)
                else:
                    self._chunks[0] = c[take:]
                got += take
            self._n -= got
        if got < n:
            out.append(np.zeros(n - got, dtype=np.float32))
        return np.concatenate(out) if len(out) > 1 else out[0]


# --------------------------------------------------------------------------- #
# Decoding audio files (uploads / Voice Memos)
# --------------------------------------------------------------------------- #
def decode_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Decode any audio file to float32 mono at `sample_rate`.

    Tries ffmpeg, then afconvert (ships with macOS), then the wave module for
    plain PCM wav files.
    """
    if shutil.which("ffmpeg"):
        p = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path),
             "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "-"],
            capture_output=True)
        if p.returncode == 0 and p.stdout:
            return np.frombuffer(p.stdout, dtype=np.float32)
    if shutil.which("afconvert"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            p = subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", f"LEI16@{sample_rate}",
                 "-c", "1", str(path), str(tmp_path)], capture_output=True)
            if p.returncode == 0:
                return _read_wav(tmp_path, sample_rate)
        finally:
            tmp_path.unlink(missing_ok=True)
    if path.suffix.lower() == ".wav":
        return _read_wav(path, sample_rate)
    raise RuntimeError(
        f"Could not decode '{path.name}'. Install ffmpeg to import audio.")


def _read_wav(path: Path, sample_rate: int) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise RuntimeError(f"Unsupported wav sample width in {path.name}")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        ch, sr = w.getnchannels(), w.getframerate()
    audio = data.astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    return _resample(audio, sr, sample_rate)


def split_for_transcription(audio: np.ndarray, sr: int,
                            target: float = 15.0, radius: float = 3.0):
    """Yield (start_sec, chunk) pieces, cutting at the quietest 20 ms window
    near each ~target-second boundary so words aren't chopped mid-syllable."""
    pos = 0
    n = len(audio)
    win = int(sr * 0.02)
    while n - pos > int(sr * (target + radius)):
        lo = pos + int(sr * (target - radius))
        hi = pos + int(sr * (target + radius))
        seg = audio[lo:hi]
        # RMS of each 20 ms window; cut at the quietest one
        m = len(seg) // win * win
        rms = np.sqrt(np.mean(np.square(seg[:m].reshape(-1, win)), axis=1))
        cut = lo + int(np.argmin(rms)) * win + win // 2
        yield pos / sr, audio[pos:cut]
        pos = cut
    if n - pos > 0:
        yield pos / sr, audio[pos:]


# --------------------------------------------------------------------------- #
# Voice Memos (macOS) — list the user's recordings for import
# --------------------------------------------------------------------------- #
VOICE_MEMOS_DIR = (Path.home() / "Library" / "Group Containers"
                   / "group.com.apple.VoiceMemos.shared" / "Recordings")


def list_voice_memos() -> tuple[list[dict] | None, str | None]:
    """Return (items, error). items: [{path,title,date,duration}] newest first.

    Reads the CloudRecordings.db metadata when possible, else falls back to
    listing *.m4a files. The folder is TCC-protected: without Full Disk Access
    for the process, listing fails — we return a helpful error instead.
    """
    if sys.platform != "darwin":
        return None, "Voice Memos import is only available on macOS."
    if not VOICE_MEMOS_DIR.exists():
        return None, "No Voice Memos library found on this Mac."
    try:
        # NB: Path.glob would silently swallow the TCC PermissionError.
        files = {p.name: p for p in VOICE_MEMOS_DIR.iterdir()
                 if p.suffix.lower() == ".m4a"}
    except OSError:
        return None, ("Cannot read the Voice Memos library. Grant Full Disk "
                      "Access to the app/terminal running voice-term "
                      "(System Settings → Privacy & Security → Full Disk "
                      "Access), then reload.")

    items = []
    db = VOICE_MEMOS_DIR / "CloudRecordings.db"
    if db.exists():
        tmp = None
        try:
            import sqlite3
            # The Voice Memos app keeps recent edits (e.g. renames) in the
            # WAL file, and an immutable read-only open ignores the WAL —
            # titles would be stale. Copy db+wal+shm to a temp dir and open
            # the copy so SQLite replays the WAL.
            tmp = Path(tempfile.mkdtemp(prefix="voice-memos-db-"))
            for suffix in ("", "-wal", "-shm"):
                src = Path(str(db) + suffix)
                if src.exists():
                    shutil.copy2(src, tmp / src.name)
            con = sqlite3.connect(f"file:{tmp / db.name}?mode=ro", uri=True)
            cols = {r[1] for r in con.execute("PRAGMA table_info(ZCLOUDRECORDING)")}
            # ZENCRYPTEDTITLE holds the name shown in the Voice Memos app
            # (user-given or location default); ZCUSTOMLABEL is often just a
            # timestamp string, so it is only a fallback.
            title_cols = [c for c in ("ZENCRYPTEDTITLE", "ZCUSTOMLABEL")
                          if c in cols]
            if {"ZPATH", "ZDATE", "ZDURATION"} <= cols:
                sel = ", ".join(title_cols) or "ZPATH"
                q = (f"SELECT ZPATH, ZDATE, ZDURATION, {sel} "
                     f"FROM ZCLOUDRECORDING ORDER BY ZDATE DESC")
                for zpath, zdate, dur, *titles in con.execute(q):
                    if not zpath:
                        continue
                    p = files.get(Path(zpath).name)
                    if not p:
                        continue
                    title = next((str(t) for t in titles if t), Path(zpath).stem)
                    # Core Data epoch (2001-01-01) -> unix
                    ts = (zdate or 0) + 978307200
                    items.append({
                        "path": str(p),
                        "title": title,
                        "date": datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M"),
                        "duration": _fmt_ts(dur or 0),
                    })
            con.close()
        except Exception:
            items = []
        finally:
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)
    if not items:   # no db / unreadable db -> plain file listing
        for p in sorted(files.values(), key=lambda p: p.stat().st_mtime,
                        reverse=True):
            items.append({
                "path": str(p),
                "title": p.stem,
                "date": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                "duration": "",
            })
    return items, None


# --------------------------------------------------------------------------- #
# Summarization (local): Ollama if running, else built-in extractive fallback
# --------------------------------------------------------------------------- #
_SUMMARY_PROMPT = """\
You are given the transcript of a recorded session (a meeting, a call, a video
being played, or someone thinking out loud). Timestamps are [mm:ss].

Write the whole answer in the SAME language the transcript is mostly in.
Output ONLY markdown, with exactly these sections:

## Summary
A tight 3-6 sentence overview of what happened / what it was about.

## Notes
Structured bullet points of the key content: topics, decisions, numbers,
names, anything worth keeping. Group related points; keep timestamps for the
important ones like [12:34].

## Action Items
- [ ] checkbox list of concrete follow-ups, with owner if mentioned.
(If there are none, write "None".)

Transcript:
---
{transcript}
---
"""

_FULL_SUMMARY_PROMPT = """\
你是一个会议/语音笔记整理助手。下面是一场录音的完整口语转写（可能有转写错误和
口头语），时间戳格式是 [h:mm:ss]。请把它整理成一份详细的全文总结（Layer 2）。

要求：
- 用转写内容的主要语言输出（中文内容用中文）。
- 忽略口头语、重复和明显的转写错误；要提炼，不要逐句复述。
- 按主题归组，不要按时间流水罗列；每个主题用 ### 小标题 + 列表。
- 保留所有实质内容：判断、结论、数字、名字、日期、分歧和未决问题。
- 明确出现的行动项写成 - [ ] 待办；存疑或需要回看确认的点单独标出。
- 只输出 markdown 正文，不要输出解释。

转写内容：
---
{transcript}
---
"""

_MERGE_PROMPT = """\
你是一个会议/语音笔记整理助手。同一场录音太长，已按时间顺序分成几部分分别做了
详细总结。请把下面这些部分总结合并成一份完整的全文总结（Layer 2）。

要求：
- 用输入内容的主要语言输出。
- 合并重复主题，按主题归组，不要按部分罗列。
- 保留所有实质内容：判断、结论、数字、名字、日期、待办（- [ ]）和存疑点。
- 只输出 markdown 正文，不要输出解释。

各部分总结：
---
{parts}
---
"""

_ONE_PAGE_PROMPT = """\
你是一个会议纪要整理助手。下面是一场录音的详细全文总结。请把它再精简成
一页纸（One page）的正式纪要（Layer 3），让没参加的人一分钟内看懂。

要求：
- 用输入内容的主要语言输出。
- 大幅精简：只保留最重要的结论、判断和行动项，总量控制在一页以内。
- 只输出 markdown，结构如下：

## 一句话结论
（最核心的结论或方向，1-3 句）

## 核心要点
（最重要的判断和信息，按重要性排列，最多 8 条）

## Action Todo
（按优先级排列的行动项）
- [ ] ...

## Checkpoints / 待确认
- ...

详细总结：
---
{summary}
---
"""

_ENTITIES_PROMPT = """\
你是一个知识图谱抽取助手。下面是一场会议/录音的整理笔记。请从中抽取出知识实体
和它们之间的关系，用于把多场会议连成一张个人知识图谱。

实体类型只用这几种（英文小写）：
- person   人物（姓名、称呼）
- project  项目 / 产品 / 作品
- org      公司 / 机构 / 团队
- concept  概念 / 主题 / 技术 / 方法
- decision 明确的决策或结论
- todo     行动项 / 待办

要求：
- 用笔记的主要语言给 name（中文内容用中文），name 要短、可跨会议复用（如某个
  产品名、公司名、题材/IP 名），不要写成一句话。
- 同一实体只出一次。忽略无意义的口头语。
- desc 用一句话说明这个实体在本次会议里是什么 / 起什么作用。
- relations 里 from/to 必须是上面 entities 的 name；label 用几个字说明关系
  （如“考虑用作Demo IP”“负责”“依赖”）。没有明确关系就少写或不写。
- 只输出 JSON，不要输出任何解释或 markdown 代码围栏，格式：

{{"entities":[{{"name":"","type":"person","desc":""}}],
 "relations":[{{"from":"","to":"","label":""}}]}}

笔记：
---
{summary}
---
"""

_ALIASES_PROMPT = """\
下面是从多场会议里抽取出的知识实体名（每行 “名称 (类型)”）。有些是同一个
人 / 项目 / 机构 / 概念 / 主题的不同叫法（缩写、别名、同义、同一主题的不同措辞），
请把它们归并到一个规范名下。

要求：
- 只在确实指向同一个对象、或同一个具体主题时合并；不同的概念不要合并。
- 不要跨不相容类型合并（person 不要和 project/concept 合并；concept 之间、
  project 之间可以）。
- canonical 选最完整、最常用的写法（尽量用已有名称之一）。
- 只输出 JSON，只列需要合并的组（单独的不用列）：
{{"groups":[{{"canonical":"规范名","aliases":["别名1","别名2"]}}]}}

实体列表：
---
{names}
---
"""

_ACTION_RE = re.compile(
    r"(todo|action|next step|need to|should |must |will |follow.?up|deadline|"
    r"しましょう|してください|やります|やろう|予定|必要|宿題|"
    r"要|计划|安排|待办|需要|记得)",
    re.IGNORECASE,
)

_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿豈-﫿]+")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _strip_think(text: str) -> str:
    """Drop <think>…</think> blocks that reasoning models emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# Entity types the knowledge-graph extraction is allowed to use.
ENTITY_TYPES = ("person", "project", "org", "concept", "decision", "todo")


def _parse_graph_json(raw: str) -> dict:
    """Parse the entity-extraction model output into {entities, relations}.

    Tolerant of ```json fences and surrounding prose: grabs the outermost
    JSON object. Returns {} if nothing parseable."""
    if not raw:
        return {}
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(text[start:end + 1])
    except Exception:
        return {}
    ents, rels = [], []
    seen = set()
    for e in data.get("entities", []):
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "")).strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        etype = str(e.get("type", "concept")).strip().lower()
        if etype not in ENTITY_TYPES:
            etype = "concept"
        ents.append({"name": name, "type": etype,
                     "desc": str(e.get("desc", "")).strip()})
    valid = {e["name"].lower() for e in ents}
    for r in data.get("relations", []):
        if not isinstance(r, dict):
            continue
        a = str(r.get("from", "")).strip()
        b = str(r.get("to", "")).strip()
        if a.lower() in valid and b.lower() in valid and a.lower() != b.lower():
            rels.append({"from": a, "to": b,
                         "label": str(r.get("label", "")).strip()})
    return {"entities": ents, "relations": rels}


class Summarizer:
    """Turn a transcript into summary/notes markdown — on your own machines.

    backend:
      "auto"    -> OpenAI-compatible server (openai_url, e.g. LM Studio on a
                   LAN/Tailscale box) if configured and reachable, else local
                   Ollama, else built-in extractive fallback.
      "openai"  -> always the OpenAI-compatible server.
      "ollama"  -> always Ollama.
      "builtin" -> never use an LLM.
    """

    def __init__(self, backend: str = "auto",
                 ollama_model: str = "qwen2.5:7b",
                 ollama_url: str = "http://127.0.0.1:11434",
                 openai_url: str = "",
                 openai_model: str = "",
                 openai_api_key: str = "lm-studio"):
        self.backend = backend.lower()
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url.rstrip("/")
        self.openai_url = openai_url.rstrip("/")
        self.openai_model = openai_model
        self.openai_api_key = openai_api_key
        self._openai_model_cache = ""
        self._backend_cache: tuple[str | None, float] | None = None

    def _ollama_up(self) -> bool:
        try:
            with urlrequest.urlopen(self.ollama_url + "/api/tags", timeout=2):
                return True
        except Exception:
            return False

    def _openai_up(self) -> bool:
        if not self.openai_url:
            return False
        try:
            with urlrequest.urlopen(self.openai_url + "/models", timeout=3):
                return True
        except Exception:
            return False

    def _llm_backend(self) -> str | None:
        """Which backend to use right now (probe results cached ~30 s)."""
        if self.backend in ("openai", "ollama"):
            return self.backend
        if self.backend == "builtin":
            return None
        now = time.monotonic()
        if self._backend_cache and now - self._backend_cache[1] < 30:
            return self._backend_cache[0]
        resolved = ("openai" if self._openai_up()
                    else "ollama" if self._ollama_up() else None)
        self._backend_cache = (resolved, now)
        return resolved

    def available(self) -> bool:
        """True when an LLM backend can be used right now."""
        return self._llm_backend() is not None

    def _openai_first_model(self) -> str:
        if not self._openai_model_cache:
            with urlrequest.urlopen(self.openai_url + "/models",
                                    timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("id", "") for m in data.get("data", [])]
            self._openai_model_cache = models[0] if models else ""
        return self._openai_model_cache

    def _openai_generate(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.openai_model or self._openai_first_model(),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "stream": False,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.openai_api_key:
            headers["Authorization"] = f"Bearer {self.openai_api_key}"
        req = urlrequest.Request(
            self.openai_url + "/chat/completions", data=payload,
            headers=headers)
        with urlrequest.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = (data.get("choices") or [{}])[0].get(
            "message", {}).get("content", "") or ""
        return _strip_think(text)

    def _generate(self, prompt: str, num_ctx: int = 8192) -> str:
        backend = self._llm_backend()
        if backend == "openai":
            return self._openai_generate(prompt)
        if backend == "ollama":
            return self._ollama_generate(prompt, num_ctx=num_ctx)
        return ""

    def _ollama_generate(self, prompt: str, num_ctx: int = 8192) -> str:
        payload = json.dumps({
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_ctx": num_ctx},
        }).encode("utf-8")
        req = urlrequest.Request(
            self.ollama_url + "/api/generate", data=payload,
            headers={"Content-Type": "application/json"})
        with urlrequest.urlopen(req, timeout=600) as resp:
            text = json.loads(resp.read().decode("utf-8")).get("response", "")
        return _strip_think(text)

    # A single pass handles up to this many transcript characters; longer
    # transcripts are split into blocks of _BLOCK_CHARS, summarized, merged.
    # Sized to fit small server context windows (LM Studio defaults are often
    # only a few k tokens); a block that still overflows is auto-split further.
    _SINGLE_PASS_CHARS = 9000
    _BLOCK_CHARS = 8000
    _MIN_BLOCK_CHARS = 1500

    def _summarize_chunk(self, text: str, depth: int = 0) -> str:
        """Summarize one transcript block; if the server rejects it for being
        too long (HTTP 400 context overflow), split in half and merge."""
        try:
            return self._generate(
                _FULL_SUMMARY_PROMPT.format(transcript=text), num_ctx=16384)
        except Exception as e:
            too_long = "400" in str(e) or "context" in str(e).lower()
            if not (too_long and depth < 4 and len(text) > self._MIN_BLOCK_CHARS):
                raise
            mid = text.rfind("\n", 0, len(text) // 2)
            if mid <= 0:
                mid = len(text) // 2
            print(f"[notes] block too long, splitting ({len(text)} chars)")
            a = self._summarize_chunk(text[:mid], depth + 1)
            b = self._summarize_chunk(text[mid:], depth + 1)
            got = [p for p in (a, b) if p]
            if len(got) <= 1:
                return got[0] if got else ""
            return self._generate(
                _MERGE_PROMPT.format(parts="\n\n".join(got)), num_ctx=16384)

    def full_summary(self, transcript: str) -> str:
        """Layer 2: one detailed summary of the WHOLE transcript.

        Long transcripts are split into blocks internally (invisible in the
        output) so the model context is never overflowed."""
        transcript = transcript.strip()
        if not transcript:
            return ""
        if len(transcript) <= self._SINGLE_PASS_CHARS:
            return self._summarize_chunk(transcript)
        lines = transcript.splitlines()
        blocks, cur, cur_len = [], [], 0
        for line in lines:
            if cur and cur_len + len(line) > self._BLOCK_CHARS:
                blocks.append("\n".join(cur))
                cur, cur_len = [], 0
            cur.append(line)
            cur_len += len(line)
        if cur:
            blocks.append("\n".join(cur))
        parts = []
        for i, block in enumerate(blocks, 1):
            print(f"[notes] Layer 2: summarizing part {i}/{len(blocks)}...")
            part = self._summarize_chunk(block)
            if part:
                parts.append(f"（第 {i}/{len(blocks)} 部分）\n{part}")
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0].split("\n", 1)[-1]
        return self._generate(
            _MERGE_PROMPT.format(parts="\n\n".join(parts)), num_ctx=32768)

    def one_page(self, full_summary: str) -> str:
        """Layer 3: condense the Layer 2 full summary into one page."""
        if not full_summary.strip():
            return ""
        return self._generate(
            _ONE_PAGE_PROMPT.format(summary=full_summary), num_ctx=16384)

    def extract_entities(self, summary_md: str) -> dict:
        """Knowledge-graph pass: entities + relations from a session's notes.

        Returns {"entities": [...], "relations": [...]}; {} on failure."""
        if not summary_md.strip() or not self.available():
            return {}
        raw = self._generate(
            _ENTITIES_PROMPT.format(summary=summary_md), num_ctx=16384)
        return _parse_graph_json(raw)

    # --- built-in extractive fallback (no LLM, fully offline) --- #
    @staticmethod
    def _tokens(s: str) -> list[str]:
        toks = [w.lower() for w in _WORD_RE.findall(s)]
        for run in _CJK_RE.findall(s):  # CJK has no spaces; use bigrams
            toks.extend(run[i:i + 2] for i in range(len(run) - 1))
        return toks

    def _builtin(self, transcript: str) -> str:
        # Strip headings, timestamps and bullets; split into sentences.
        plain = re.sub(r"^#.*$", "", transcript, flags=re.MULTILINE)
        plain = re.sub(r"^\s*[-*]\s*(\*\*\[[\d:]+\]\*\*)?\s*", "", plain,
                       flags=re.MULTILINE)
        sents = [s.strip() for s in re.split(r"(?<=[。．！？!?.])\s+|\n+", plain)
                 if len(s.strip()) >= 8]
        if not sents:
            return "## Summary\n(No speech was captured.)\n"

        freq: dict[str, int] = {}
        for s in sents:
            for t in self._tokens(s):
                freq[t] = freq.get(t, 0) + 1
        def score(s: str) -> float:
            toks = self._tokens(s)
            return sum(freq.get(t, 0) for t in toks) / (len(toks) or 1) ** 0.5

        ranked = sorted(range(len(sents)), key=lambda i: score(sents[i]),
                        reverse=True)
        top = sorted(ranked[:5])                      # summary, original order
        notes = sorted(ranked[:min(12, len(sents))])  # a few more, as notes
        actions = [s for s in sents if _ACTION_RE.search(s)][:10]
        keywords = [w for w, _ in sorted(freq.items(), key=lambda kv: -kv[1])
                    if len(w) > 2][:8]

        md = ["## Summary", " ".join(sents[i] for i in top), ""]
        md += ["## Notes"]
        md += [f"- {sents[i]}" for i in notes]
        if keywords:
            md += ["", f"**Keywords:** {', '.join(keywords)}"]
        md += ["", "## Action Items"]
        md += [f"- [ ] {a}" for a in actions] if actions else ["None"]
        md += ["", "> _Built-in extractive summary (no local LLM found). "
                   "Run [Ollama](https://ollama.com) for real notes._"]
        return "\n".join(md)

    def summarize(self, transcript: str) -> str:
        if self.available():
            try:
                out = self._generate(
                    _SUMMARY_PROMPT.format(transcript=transcript),
                    num_ctx=16384)
                if out:
                    return out
            except Exception as e:
                print(f"[notes] LLM summarization failed ({e}); "
                      f"using built-in fallback.")
        return self._builtin(transcript)


def _clean_md_headings(text: str) -> str:
    """Collapse duplicated leading heading markers some models emit, e.g.
    '### ### 标题' -> '### 标题'. The duplicate markers must be
    whitespace-separated so a normal '## 标题' is left untouched."""
    return re.sub(r"^(#{1,6})[ \t]+(?:#{1,6}[ \t]+)+", r"\1 ", text,
                  flags=re.MULTILINE)


def _layered_summaries(summarizer: Summarizer, transcript: str) -> tuple[str, str]:
    """Run Layer 2 (full summary) then Layer 3 (one page) over a finished
    transcript. Returns ("", "") when no LLM backend is available."""
    if not summarizer.available():
        return "", ""
    try:
        layer2 = _clean_md_headings(summarizer.full_summary(transcript))
        layer3 = _clean_md_headings(summarizer.one_page(layer2)) if layer2 else ""
        return layer2, layer3
    except Exception as e:
        print(f"[notes] layered summarization failed ({e})")
        return "", ""


def build_entities(session_dir: Path, summarizer: Summarizer,
                   force: bool = False) -> dict:
    """Extract knowledge-graph entities for one session and cache them in
    entities.json. Reuses the cache unless force=True. Returns the graph dict
    (possibly {} when no LLM / no summary)."""
    out = session_dir / "entities.json"
    if out.exists() and not force:
        try:
            return json.loads(out.read_text())
        except Exception:
            pass
    summary = ""
    for name in ("summary.md", "notes.md"):
        try:
            summary = (session_dir / name).read_text()
            if summary.strip():
                break
        except OSError:
            continue
    graph = summarizer.extract_entities(summary)
    if graph.get("entities"):
        try:
            out.write_text(json.dumps(graph, ensure_ascii=False))
        except OSError:
            pass
    return graph


def build_entity_aliases(base_dir: Path, summarizer: Summarizer) -> dict:
    """LLM pass that clusters entity aliases/same-topic across all sessions
    into canonical names. Writes graph_aliases.json ({alias: canonical}) which
    aggregate_graph() then applies to merge nodes. Returns the alias map."""
    if not summarizer.available():
        return {}
    pairs = entity_names_by_type(base_dir)
    if len(pairs) < 3:
        return {}
    listing = "\n".join(f"- {name} ({etype})" for name, etype in pairs)
    raw = summarizer._generate(
        _ALIASES_PROMPT.format(names=listing), num_ctx=16384)
    # tolerant JSON extraction (fences / surrounding prose)
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        return {}
    try:
        obj = json.loads(text[s:e + 1])
    except Exception:
        return {}
    type_of = {name: etype for name, etype in pairs}
    valid = set(type_of)
    alias_map: dict[str, str] = {}
    for grp in obj.get("groups", []):
        canonical = str(grp.get("canonical", "")).strip()
        aliases = [str(a).strip() for a in grp.get("aliases", []) if str(a).strip()]
        if not canonical or not aliases:
            continue
        # Canonical type: the canonical's own type if known, else the most
        # common type among the group's members.
        member_types = [type_of[a] for a in aliases if a in type_of]
        if canonical in type_of:
            member_types.append(type_of[canonical])
        ctype = max(set(member_types), key=member_types.count) if member_types else ""
        for a in aliases:
            # Only merge names we actually have, same type as canonical, and
            # never a name into itself. Type-safety stops a named project/
            # person being folded into a generic concept.
            if a in valid and a != canonical and type_of.get(a) == ctype:
                alias_map[a] = canonical
    try:
        (base_dir / "graph_aliases.json").write_text(
            json.dumps(alias_map, ensure_ascii=False, indent=1))
    except OSError:
        pass
    return alias_map


def _norm_key(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).lower()


# --------------------------------------------------------------------------- #
# A notes session: mix -> chunk on silence -> transcribe -> summarize
# --------------------------------------------------------------------------- #
class NotesSession:
    SILENCE_RMS = 0.006     # below this the block counts as silence
    SILENCE_SEC = 0.9       # cut a chunk after this much silence...
    MIN_CHUNK_SEC = 3.0     # ...but only once it's at least this long
    MAX_CHUNK_SEC = 18.0    # hard cut so live transcript keeps flowing

    def __init__(self, base_dir: Path, sample_rate: int, transcribe_fn,
                 summarizer: Summarizer, capture_system: bool = True,
                 keep_audio: bool = True, on_done=None,
                 min_chunk_sec: float | None = None,
                 max_chunk_sec: float | None = None,
                 language: str = "zh"):
        self.id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = base_dir / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.transcribe_fn = transcribe_fn   # (float32 ndarray) -> str
        self.summarizer = summarizer
        self.language = language if language in ("zh", "ja", "en", "auto") else "zh"
        self.on_done = on_done               # called with self when summary.md exists
        if min_chunk_sec is not None:
            self.MIN_CHUNK_SEC = float(min_chunk_sec)
        if max_chunk_sec is not None:
            self.MAX_CHUNK_SEC = float(max_chunk_sec)
        self.started = time.time()
        self.status = "recording"            # recording -> summarizing -> done
        self.level = 0.0                     # mixed level (0..1) for the tray
        self.system_audio = False

        self._mic = _StreamBuf()
        self._sys = _StreamBuf()
        self._sys_tap = SystemAudioTap(sample_rate) if capture_system else None
        if self._sys_tap:
            self.system_audio = self._sys_tap.start(self._sys.push)
            if not self.system_audio:
                self._sys_tap = None
                print("[notes] Speaker capture unavailable (pactl/parec not "
                      "found?); recording mic only.")

        self._wav = None
        if keep_audio:
            self._wav = wave.open(str(self.dir / "audio.wav"), "wb")
            self._wav.setnchannels(1)
            self._wav.setsampwidth(2)
            self._wav.setframerate(sample_rate)

        self._chunk: list[np.ndarray] = []
        self._chunk_len = 0
        self._chunk_start = 0        # sample offset of current chunk
        self._consumed = 0           # total mixed samples processed
        self._transcribed_until = 0.0
        self._silence = 0.0
        self._segments: list[tuple[float, str]] = []   # (seconds, text)
        self._jobs: queue.Queue = queue.Queue()
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        self._llm_layer2 = ""
        self._llm_final = ""

        self._write_meta()
        self._write_transcript()
        self._write_notes()
        self._mixer_t = threading.Thread(target=self._mixer, daemon=True)
        self._worker_t = threading.Thread(target=self._worker, daemon=False)
        self._mixer_t.start()
        self._worker_t.start()

    # --- audio in --- #
    def feed_mic(self, audio: np.ndarray):
        """Called from the shared Recorder stream with every mic block."""
        self._mic.push(audio.astype(np.float32, copy=False).flatten())

    # --- mixing / chunking --- #
    def _mixer(self):
        while not self._stopping.is_set():
            time.sleep(0.1)
            self._drain()
        # final flush: take everything either side still holds
        self._drain(final=True)
        if self._sys_tap:
            self._sys_tap.stop()
        self._cut(force=True)
        if self._wav:
            try:
                self._wav.close()
            except Exception:
                pass
            self._wav = None
        self._jobs.put(None)   # sentinel: worker finishes queue then summarizes

    def _drain(self, final: bool = False):
        if self._sys_tap:
            n = (max if final else min)(self._mic.available, self._sys.available)
        else:
            n = self._mic.available
        if n <= 0:
            return
        mixed = self._mic.pull(n)
        if self._sys_tap:
            mixed = np.clip(mixed + self._sys.pull(n), -1.0, 1.0)

        if self._wav:
            try:
                self._wav.writeframes(
                    (mixed * 32767).astype(np.int16).tobytes())
            except Exception:
                pass

        sr = self.sample_rate
        rms = float(np.sqrt(np.mean(np.square(mixed), dtype=np.float64)))
        sys_lvl = self._sys_tap.level if self._sys_tap else 0.0
        self.level = min(1.0, max(rms, sys_lvl) * 9.0)
        self._silence = self._silence + n / sr if rms < self.SILENCE_RMS else 0.0

        self._chunk.append(mixed)
        self._chunk_len += n
        self._consumed += n

        dur = self._chunk_len / sr
        if dur >= self.MAX_CHUNK_SEC or (
                dur >= self.MIN_CHUNK_SEC and self._silence >= self.SILENCE_SEC):
            self._cut()

    def _cut(self, force: bool = False):
        if self._chunk_len == 0:
            return
        if not force and self._chunk_len < self.sample_rate * 1.0:
            return
        audio = np.concatenate(self._chunk)
        t0 = self._chunk_start / self.sample_rate
        self._chunk = []
        self._chunk_start = self._consumed
        self._chunk_len = 0
        self._silence = 0.0
        # Skip chunks that are essentially silence.
        if float(np.sqrt(np.mean(np.square(audio)))) < self.SILENCE_RMS * 0.8:
            return
        self._jobs.put((t0, audio))

    # --- transcription / summary --- #
    def _worker(self):
        while True:
            job = self._jobs.get()
            if job is None:
                break
            t0, audio = job
            try:
                text = _call_transcribe(self.transcribe_fn, audio, self._context_text())
            except Exception as e:
                print(f"[notes] chunk transcription failed: {e}")
                continue
            self._transcribed_until = max(
                self._transcribed_until,
                t0 + len(audio) / self.sample_rate,
            )
            if text:
                with self._lock:
                    self._segments.append((t0, text))
                self._write_transcript()
                self._write_notes()
            self._write_meta()
        self._finalize()

    def _finalize(self):
        self.status = "summarizing"
        self._write_meta()
        transcript = self.transcript_md()
        (self.dir / "transcript.md").write_text(transcript)
        self._write_notes()
        print("[notes] Summarizing...")
        self._llm_layer2, self._llm_final = _layered_summaries(
            self.summarizer, transcript)
        self._write_notes()
        try:
            body = _summary_body(self._llm_layer2, self._llm_final, self.language) \
                or self.summarizer.summarize(transcript)
        except Exception as e:
            body = f"## Summary\n(Summarization failed: {e})"
        dur = _fmt_ts(self._consumed / self.sample_rate)
        started = datetime.fromtimestamp(self.started).strftime("%Y-%m-%d %H:%M")
        head = (f"# Session notes — {started}\n\n"
                f"_{dur} recorded · mic{' + speakers' if self.system_audio else ''}"
                f" · transcribed & summarized locally_\n\n")
        (self.dir / "summary.md").write_text(head + body + "\n")
        self.status = "done"
        self._write_notes()
        self._write_meta()
        print(f"[notes] Done: {self.dir / 'summary.md'}")
        try:
            build_entities(self.dir, self.summarizer, force=True)
            build_entity_aliases(self.dir.parent, self.summarizer)
        except Exception as e:
            print(f"[notes] entity extraction failed ({e})")
        if self.on_done:
            try:
                self.on_done(self)
            except Exception:
                pass

    # --- files / views --- #
    def title(self) -> str:
        with self._lock:
            first = self._segments[0][1] if self._segments else ""
        return (first[:60] + "…") if len(first) > 60 else (first or "(no speech yet)")

    def _context_text(self) -> str:
        with self._lock:
            texts = [text for _, text in self._segments[-4:]]
        return " ".join(texts)[-500:]

    def _segments_snapshot(self) -> list[tuple[float, str]]:
        with self._lock:
            return list(self._segments)

    def transcript_md(self) -> str:
        started = datetime.fromtimestamp(self.started).strftime("%Y-%m-%d %H:%M")
        with self._lock:
            lines = [f"- **[{_fmt_ts(t)}]** {text}" for t, text in self._segments]
        return (f"# Transcript — {started}\n\n" + "\n".join(lines) + "\n")

    def notes_md(self) -> str:
        started = datetime.fromtimestamp(self.started).strftime("%Y-%m-%d %H:%M")
        segments = self._segments_snapshot()
        dur = _fmt_ts(self._consumed / self.sample_rate)
        source = "mic + speakers" if self.system_audio else "mic"
        return _running_notes_md(
            f"Session notes — {started}",
            segments,
            self.status,
            duration=dur,
            source=source,
            llm_layer2=self._llm_layer2,
            llm_final=self._llm_final,
            language=self.language,
        )

    def _write_transcript(self):
        try:
            (self.dir / "transcript.md").write_text(self.transcript_md())
        except OSError:
            pass

    def _write_notes(self):
        try:
            (self.dir / "notes.md").write_text(self.notes_md())
        except OSError:
            pass

    def _write_meta(self):
        meta = {
            "id": self.id,
            "started": datetime.fromtimestamp(self.started).isoformat(timespec="seconds"),
            "status": self.status,
            "duration": _fmt_ts(self._consumed / self.sample_rate),
            "recorded_seconds": round(self._consumed / self.sample_rate, 3),
            "transcribed_seconds": round(self._transcribed_until, 3),
            "progress_percent": _progress_percent(
                self._transcribed_until,
                self._consumed / self.sample_rate,
                self.status,
            ),
            "progress_label": _progress_label(
                self._transcribed_until,
                self._consumed / self.sample_rate,
                self.status,
                total_name="recorded",
            ),
            "title": self.title(),
            "system_audio": self.system_audio,
            "source": "live",
            "language": self.language,
        }
        try:
            (self.dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
        except OSError:
            pass

    def stop(self):
        """Stop recording; transcription drains and the summary is written
        asynchronously (on_done fires when summary.md exists)."""
        if self._stopping.is_set():
            return
        self._stopping.set()

    def wait(self, timeout: float | None = None):
        self._worker_t.join(timeout)


class ImportSession:
    """An uploaded file / Voice Memo pushed through the same pipeline:
    decode -> transcribe (chunked, real timestamps) -> notes + summary.md.
    Produces the same session-folder layout as NotesSession, so the web UI
    shows both kinds identically."""

    def __init__(self, base_dir: Path, sample_rate: int, src_path: Path,
                 title: str, transcribe_segments_fn, summarizer: Summarizer,
                 source: str = "upload", on_done=None,
                 chunk_target: float = 30.0, chunk_radius: float = 5.0,
                 session_id: str | None = None, resume: bool = False,
                 language: str = "zh"):
        self.id = session_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        while not session_id and (base_dir / self.id).exists():
            self.id += "0"
        self.dir = base_dir / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.src_path = Path(src_path)
        self.name = title or self.src_path.stem
        self.transcribe_segments_fn = transcribe_segments_fn  # audio -> [(s, text)]
        self.summarizer = summarizer
        self.source = source
        self.language = language if language in ("zh", "ja", "en", "auto") else "zh"
        self.on_done = on_done
        self.chunk_target = chunk_target
        self.chunk_radius = chunk_radius
        self.started = time.time()
        self.status = "transcribing"
        self.error = ""
        self._duration = 0.0
        self._processed_until = 0.0
        self._chunk_records: dict[str, dict] = {}
        self._segments: list[tuple[float, str]] = []
        self._lock = threading.Lock()
        self._llm_layer2 = ""
        self._llm_final = ""
        if resume:
            self._load_chunk_records()
            self._rebuild_segments_from_chunks()
            self._processed_until = max(
                (float(r.get("end", 0.0)) for r in self._chunk_records.values()
                 if r.get("status") == "done"),
                default=0.0,
            )
        self._write_meta()
        self._write_transcript()
        self._write_notes()
        self._thread = threading.Thread(target=self._run, daemon=False)
        self._thread.start()

    def _run(self):
        try:
            audio = decode_audio(self.src_path, self.sample_rate)
            self._duration = len(audio) / self.sample_rate
            self._write_meta()
            with wave.open(str(self.dir / "audio.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(self.sample_rate)
                w.writeframes((np.clip(audio, -1, 1) * 32767)
                              .astype(np.int16).tobytes())
            for index, (t0, chunk) in enumerate(split_for_transcription(
                    audio, self.sample_rate,
                    target=self.chunk_target, radius=self.chunk_radius)):
                end = min(self._duration, t0 + len(chunk) / self.sample_rate)
                existing = self._chunk_records.get(str(index))
                if (existing and existing.get("status") == "done"
                        and _chunk_record_matches(existing, t0, end)):
                    self._processed_until = max(self._processed_until, end)
                    self._rebuild_segments_from_chunks()
                    self._write_transcript()
                    self._write_notes()
                    self._write_meta()
                    continue

                segments = []
                for start, text in _call_transcribe_segments(
                        self.transcribe_segments_fn, chunk, self._context_text()):
                    if text:
                        segments.append((t0 + start, text))
                self._chunk_records[str(index)] = {
                    "index": index,
                    "start": t0,
                    "end": end,
                    "status": "done",
                    "segments": segments,
                    "text": "".join(text for _, text in segments),
                    "note": "".join(text for _, text in segments),
                }
                self._write_chunk_records()
                self._rebuild_segments_from_chunks()
                self._processed_until = max(self._processed_until, end)
                self._write_transcript()
                self._write_notes()
                self._write_meta()
            transcript = self.transcript_md()
            (self.dir / "transcript.md").write_text(transcript)
            self.status = "summarizing"
            self._processed_until = self._duration
            self._write_meta()
            self._write_notes()
            print(f"[notes] Summarizing '{self.name}'...")
            self._llm_layer2, self._llm_final = _layered_summaries(
                self.summarizer, transcript)
            self._write_notes()
            body = _summary_body(self._llm_layer2, self._llm_final, self.language) \
                or self.summarizer.summarize(transcript)
            started = datetime.fromtimestamp(self.started).strftime("%Y-%m-%d %H:%M")
            head = (f"# {self.name}\n\n"
                    f"_{_fmt_ts(self._duration)} audio · imported {started}"
                    f" · transcribed & summarized locally_\n\n")
            (self.dir / "summary.md").write_text(head + body + "\n")
            self.status = "done"
            self._processed_until = self._duration
            self._write_notes()
            try:
                build_entities(self.dir, self.summarizer, force=True)
                build_entity_aliases(self.dir.parent, self.summarizer)
            except Exception as e:
                print(f"[notes] entity extraction failed ({e})")
            print(f"[notes] Done: {self.dir / 'summary.md'}")
        except Exception as e:
            self.status = "error"
            self.error = str(e)
            self._write_notes()
            print(f"[notes] Import failed: {e}")
        self._write_meta()
        if self.on_done:
            try:
                self.on_done(self)
            except Exception:
                pass

    def title(self) -> str:
        return self.name

    def _context_text(self) -> str:
        with self._lock:
            texts = [text for _, text in self._segments[-4:]]
        return " ".join(texts)[-500:]

    def _segments_snapshot(self) -> list[tuple[float, str]]:
        with self._lock:
            return list(self._segments)

    def transcript_md(self) -> str:
        with self._lock:
            lines = [f"- **[{_fmt_ts(t)}]** {text}" for t, text in self._segments]
        return f"# Transcript — {self.name}\n\n" + "\n".join(lines) + "\n"

    def notes_md(self) -> str:
        segments = self._segments_snapshot()
        return _running_notes_md(
            self.name,
            segments,
            self.status,
            duration=_fmt_ts(self._duration),
            source=self.source,
            error=self.error,
            llm_layer2=self._llm_layer2,
            llm_final=self._llm_final,
            language=self.language,
        )

    def _write_transcript(self):
        try:
            (self.dir / "transcript.md").write_text(self.transcript_md())
        except OSError:
            pass

    def _write_notes(self):
        try:
            (self.dir / "notes.md").write_text(self.notes_md())
        except OSError:
            pass

    def _write_meta(self):
        meta = {
            "id": self.id,
            "started": datetime.fromtimestamp(self.started).isoformat(timespec="seconds"),
            "status": self.status,
            "duration": _fmt_ts(self._duration),
            "recorded_seconds": round(self._duration, 3),
            "transcribed_seconds": round(self._processed_until, 3),
            "progress_percent": _progress_percent(
                self._processed_until,
                self._duration,
                self.status,
            ),
            "progress_label": _progress_label(
                self._processed_until,
                self._duration,
                self.status,
                total_name="audio",
            ),
            "title": self.name,
            "system_audio": False,
            "source": self.source,
            "source_path": str(self.src_path),
            "chunk_count": len(self._chunk_records),
            "done_chunk_count": sum(
                1 for r in self._chunk_records.values()
                if r.get("status") == "done"
            ),
            "error": self.error,
            "language": self.language,
        }
        try:
            (self.dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
        except OSError:
            pass

    def _load_chunk_records(self):
        try:
            data = json.loads((self.dir / "chunks.json").read_text())
        except Exception:
            data = []
        self._chunk_records = {
            str(r.get("index")): r for r in data
            if isinstance(r, dict) and r.get("index") is not None
        }

    def _write_chunk_records(self):
        records = sorted(
            self._chunk_records.values(),
            key=lambda r: int(r.get("index", 0)),
        )
        try:
            (self.dir / "chunks.json").write_text(
                json.dumps(records, ensure_ascii=False, indent=2)
            )
        except OSError:
            pass

    def _rebuild_segments_from_chunks(self):
        records = sorted(
            self._chunk_records.values(),
            key=lambda r: int(r.get("index", 0)),
        )
        segments = []
        for record in records:
            for t, text in record.get("segments", []):
                if text:
                    segments.append((float(t), str(text)))
        with self._lock:
            self._segments = segments


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _accepts_context(fn) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return True
    return len(params) >= 2


def _call_transcribe(fn, audio: np.ndarray, context: str) -> str:
    if _accepts_context(fn):
        return fn(audio, context)
    return fn(audio)


def _call_transcribe_segments(fn, audio: np.ndarray, context: str) -> list:
    if _accepts_context(fn):
        return fn(audio, context)
    return fn(audio)


def _chunk_record_matches(record: dict, start: float, end: float,
                          tolerance: float = 0.25) -> bool:
    try:
        old_start = float(record.get("start", -1.0))
        old_end = float(record.get("end", -1.0))
    except (TypeError, ValueError):
        return False
    return abs(old_start - start) <= tolerance and abs(old_end - end) <= tolerance


def _progress_percent(done: float, total: float, status: str) -> int:
    if status == "done":
        return 100
    if total <= 0:
        return 0
    return max(0, min(100, int(round(done / total * 100))))


def _progress_label(done: float, total: float, status: str,
                    total_name: str = "audio") -> str:
    if status == "done":
        return f"Complete · {_fmt_ts(total)} {total_name}"
    if status == "summarizing":
        return f"Transcript complete · summarizing {_fmt_ts(total)} {total_name}"
    if total <= 0:
        return "Preparing audio..."
    return f"Transcribed {_fmt_ts(done)} / {_fmt_ts(total)} {total_name}"


def _meta_with_progress_defaults(meta: dict) -> dict:
    if "progress_percent" in meta and "progress_label" in meta:
        return meta
    status = meta.get("status", "")
    duration = meta.get("duration", "")
    if status == "done":
        meta.setdefault("progress_percent", 100)
        meta.setdefault("progress_label", f"Complete · {duration}")
    elif status == "summarizing":
        meta.setdefault("progress_percent", 100)
        meta.setdefault("progress_label", f"Transcript complete · summarizing {duration}")
    else:
        meta.setdefault("progress_percent", 0)
        meta.setdefault("progress_label", "Preparing audio...")
    return meta


def _readable_text(text: str) -> str:
    text = " ".join((text or "").split())
    replacements = [
        (r"(呃|嗯|啊|哦|诶|哎|等一下|抱歉|我靠)[,，、\s]*", ""),
        (r"(^|[。！？!?，,、\s])(就是|然后)[,，、\s]*", r"\1"),
        (r"(不是){2,}", "不是"),
        (r"(对){2,}", "对"),
        (r"(好){2,}", "好"),
        (r"\s+", " "),
    ]
    for pat, repl in replacements:
        text = re.sub(pat, repl, text)
    return text.strip(" ，,。")


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?])\s+|[。！？!?]\s*|\n+", text)
    return [p.strip(" ，,。") for p in parts if len(p.strip()) >= 4]


def _clauses_from_text(text: str) -> list[str]:
    parts = re.split(r"[。！？!?；;]\s*|(?<=[，,])\s*", text)
    out = []
    for part in parts:
        clean = _readable_text(part)
        if len(clean) >= 4 and clean not in out:
            out.append(clean)
    return out


def _sections_from_segments(segments: list[tuple[float, str]],
                            max_chars: int = 1400,
                            max_seconds: float = 150.0) -> list[dict]:
    sections = []
    current = []
    start = None
    current_len = 0
    for t, text in segments:
        clean = _readable_text(text)
        if not clean:
            continue
        if start is None:
            start = t
        if current and (current_len + len(clean) > max_chars
                        or t - start >= max_seconds):
            sections.append({"start": start, "end": current[-1][0], "items": current})
            start = t
            current = []
            current_len = 0
        current.append((t, clean))
        current_len += len(clean)
    if current and start is not None:
        sections.append({"start": start, "end": current[-1][0], "items": current})
    return sections


def _keywords_from_text(text: str, limit: int = 8) -> list[str]:
    toks = Summarizer._tokens(text)
    stop = {
        "这个", "那个", "就是", "然后", "不是", "一个", "可能", "还是", "觉得",
        "看到", "一下", "现在", "整体", "the", "and", "you", "that", "this",
    }
    counts: dict[str, int] = {}
    for tok in toks:
        if len(tok) < 2 or tok in stop:
            continue
        counts[tok] = counts.get(tok, 0) + 1
    return [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:limit]]


_POINT_CUE_RE = re.compile(
    r"问题|原因|目标|重点|核心|需要|应该|必须|可以|希望|觉得|感觉|整体|"
    r"体验|内容|逻辑|转换|总结|摘要|checkpoint|todo|待办|确认|回看|"
    r"不懂|不知道|不确定|可能|好像|什么意思|为什么",
    re.IGNORECASE,
)

_TODO_RE = re.compile(
    r"需要|应该|必须|接下来|之后|重新|补|修改|改成|优化|做一个|"
    r"确认|回看|停止|重启|重新生成|继续|"
    r"todo|to do|next|follow",
    re.IGNORECASE,
)

_CHECKPOINT_RE = re.compile(
    r"checkpoint|检查点|确认|回看|不懂|不知道|不确定|可能|好像|什么意思|"
    r"精度|问题|原因|是否|是不是|看一下",
    re.IGNORECASE,
)


def _trim_point(text: str, limit: int = 92) -> str:
    text = _readable_text(text)
    if len(text) <= limit:
        return text
    return text[:limit - 3].rstrip("，,、 ") + "..."


def _score_clause(clause: str, keywords: list[str]) -> float:
    score = min(len(clause), 80) / 80.0
    if _POINT_CUE_RE.search(clause):
        score += 1.0
    for kw in keywords:
        if kw in clause:
            score += 0.3
    if re.search(r"这个|那个|它|这段|整体|核心|主要", clause):
        score += 0.2
    return score


def _logical_points(text: str, limit: int = 4) -> list[str]:
    clauses = _clauses_from_text(text)
    if not clauses:
        return []
    keywords = _keywords_from_text(text, limit=6)
    ranked = sorted(
        enumerate(clauses),
        key=lambda item: _score_clause(item[1], keywords),
        reverse=True,
    )
    picked = sorted(ranked[:limit], key=lambda item: item[0])
    points = []
    for _, clause in picked:
        point = _trim_point(clause)
        if point and point not in points:
            points.append(point)
    return points


def _section_topic(text: str, points: list[str]) -> str:
    if points:
        lead = "；".join(points[:2])
        return f"本段核心是在说明：{_trim_point(lead, 150)}"
    keywords = _keywords_from_text(text, limit=5)
    if keywords:
        return f"这一段主要围绕相关主题展开，暂时只能提取到关键词：{', '.join(keywords[:4])}。"
    return "这一段还没有足够内容形成明确主题。"


def _section_note(section: dict) -> dict:
    start = float(section["start"])
    items = section["items"]
    text = "。".join(part for _, part in items)
    sentences = _split_sentences(text)
    if not sentences and text:
        sentences = [text]
    points = _logical_points(text, limit=4)
    overview = _section_topic(text, points)
    todos = [
        _trim_point(s) for s in sentences
        if _TODO_RE.search(s) and len(_readable_text(s)) >= 6
    ][:4]
    checkpoints = [
        _trim_point(s) for s in sentences
        if _CHECKPOINT_RE.search(s) and len(_readable_text(s)) >= 6
    ][:4]
    valuable = points[:3] or [_trim_point(s) for s in sentences[:2]]
    uncertain = [
        s for s in sentences
        if re.search(r"不懂|不知道|不确定|可能|好像|这个吗|什么意思|是这个吗", s)
    ][:3]
    return {
        "start": start,
        "overview": overview or "这一段还没有足够内容形成摘要。",
        "points": points,
        "valuable": valuable,
        "todos": todos,
        "checkpoints": checkpoints,
        "uncertain": uncertain,
        "keywords": _keywords_from_text(text, limit=5),
    }


def _section_notes(sections: list[dict]) -> list[dict]:
    return [_section_note(section) for section in sections]


def _final_note_from_sections(notes: list[dict], status: str, language: str = "zh") -> list[str]:
    labels = {
        "ja": ("Layer 2の分割要約を待っています。", "現在のノート案", "重要な内容", "明確なToDoはありません。", "確認事項はありません。", "処理中です。後続チャンクの完了後に更新されます。"),
        "en": ("Waiting for the Layer 2 section summaries.", "Current note draft", "Valuable content", "No explicit to-dos.", "No explicit checkpoints.", "Processing continues; this will update after later chunks finish."),
        "zh": ("等待 Layer 2 形成分段摘要后生成完整 note。", "当前完整 Note 草稿", "有价值内容", "暂无明确待办。", "暂无明确检查点。", "还在处理中，后续 chunk 完成后这里会继续合并更新。"),
    }.get(language) or ("等待 Layer 2 形成分段摘要后生成完整 note。", "当前完整 Note 草稿", "有价值内容", "暂无明确待办。", "暂无明确检查点。", "还在处理中，后续 chunk 完成后这里会继续合并更新。")
    if not notes:
        return [labels[0]]
    lines = []
    lines.append("### " + labels[1])
    lines.append("")
    lines.append("**" + labels[2] + "**")
    valuable_items = []
    for note in notes[:8]:
        valuable_items.extend(note.get("valuable", [])[:2])
    for item in valuable_items[:12]:
        lines.append(f"- {item}")

    todos = []
    checkpoints = []
    for note in notes:
        todos.extend(note.get("todos", []))
        checkpoints.extend(note.get("checkpoints", []))
        checkpoints.extend(note.get("uncertain", []))

    lines += ["", "**To do**"]
    if todos:
        lines += [f"- [ ] {item}" for item in _dedupe(todos)[:10]]
    else:
        lines.append("- " + labels[3])

    lines += ["", "**Checkpoints**"]
    if checkpoints:
        lines += [f"- {item}" for item in _dedupe(checkpoints)[:10]]
    else:
        lines.append("- " + labels[4])

    if status not in ("done", "error"):
        lines += ["", "> " + labels[5]]
    return lines


def _dedupe(items: list[str]) -> list[str]:
    out = []
    for item in items:
        clean = _trim_point(item, 120)
        if clean and clean not in out:
            out.append(clean)
    return out


def _summary_body(layer2: str, layer3: str, language: str = "zh") -> str:
    """summary.md body: the one-page note first, full summary after it."""
    if not layer2 and not layer3:
        return ""
    parts = []
    if layer3:
        parts.append(layer3)
    if layer2:
        heading = {"ja": "# 全文要約（Layer 2）", "en": "# Full summary (Layer 2)"}.get(language, "# 详细总结（Layer 2）")
        parts += ["", "---", "", heading, "", layer2]
    return "\n".join(parts).strip()


_NOTES_TEXT = {
    "zh": {
        "running": "运行中的笔记", "layer1": "原始转写片段", "layer2": "全文总结", "layer3": "一页纸",
        "status_recording": "正在录音和转写", "status_transcribing": "正在逐段转写",
        "status_summarizing": "正在生成最终摘要", "status_done": "已完成", "status_error": "失败",
        "wait_first": "等待第一段语音转写完成。", "summarizing_l2": "Layer 1 已完成，正在用本地 LLM 生成全文总结…",
        "wait_l2": "等待 Layer 1 全部转写完成后，一次性生成全文总结。", "overview": "主要讲什么",
        "points": "核心逻辑点", "heuristic": "_（规则抽取版；启动 Ollama 后重跑可得到 LLM 全文总结。）_",
        "empty": "（没有可总结的内容。）", "summarizing_l3": "等 Layer 2 全文总结完成后，再精简成一页纸。",
        "wait_l3": "等待 Layer 2 完成后生成一页纸总结。",
    },
    "ja": {
        "running": "作成中のノート", "layer1": "生の文字起こし", "layer2": "全文要約", "layer3": "1ページ要約",
        "status_recording": "録音・文字起こし中", "status_transcribing": "文字起こし中",
        "status_summarizing": "最終要約を作成中", "status_done": "完了", "status_error": "エラー",
        "wait_first": "最初の音声の文字起こしを待っています。", "summarizing_l2": "Layer 1が完了しました。ローカルLLMで全文要約を作成しています…",
        "wait_l2": "Layer 1の文字起こしがすべて完了すると、全文要約を作成します。", "overview": "主な内容",
        "points": "要点", "heuristic": "_（ルールベースの抽出版です。Ollamaを起動して再実行すると、LLMによる全文要約を生成できます。）_",
        "empty": "（要約できる内容がありません。）", "summarizing_l3": "Layer 2の全文要約が完了すると、1ページにまとめます。",
        "wait_l3": "Layer 2の完了後に1ページ要約を作成します。",
    },
    "en": {
        "running": "Running notes", "layer1": "Raw transcript chunks", "layer2": "Full summary", "layer3": "One page",
        "status_recording": "Recording and transcribing", "status_transcribing": "Transcribing",
        "status_summarizing": "Creating final summary", "status_done": "Done", "status_error": "Error",
        "wait_first": "Waiting for the first audio transcript.", "summarizing_l2": "Layer 1 is complete. Creating the full summary with the local LLM…",
        "wait_l2": "The full summary will be created after all of Layer 1 is transcribed.", "overview": "Overview",
        "points": "Key points", "heuristic": "_(Rule-based extraction; rerun with Ollama for an LLM full summary.)_",
        "empty": "(There is no content to summarize.)", "summarizing_l3": "The one-page note will be created after the Layer 2 summary.",
        "wait_l3": "The one-page summary will be created after Layer 2 is complete.",
    },
}

def _running_notes_md(title: str, segments: list[tuple[float, str]], status: str,
                      duration: str = "", source: str = "", error: str = "",
                      llm_layer2: str = "", llm_final: str = "",
                      language: str = "zh") -> str:
    strings = _NOTES_TEXT.get(language, _NOTES_TEXT["zh"])
    status_label = strings.get("status_" + status, status)
    meta = " · ".join(x for x in (duration, source, status_label) if x)
    lines = [f"# {strings['running']} — {title}", f"_{meta}_", ""]
    if error:
        lines += [f"> {error}", ""]

    lines += ["## Layer 1 · " + strings['layer1']]
    if segments:
        lines += [f"- **[{_fmt_ts(t)}]** {text}" for t, text in segments]
    else:
        lines += [strings['wait_first']]

    lines += ["", "## Layer 2 · " + strings['layer2']]
    if llm_layer2:
        lines.append(llm_layer2)
    elif status == "summarizing":
        lines.append(strings['summarizing_l2'])
    elif status not in ("done", "error"):
        lines.append(strings['wait_l2'])
    else:
        # No LLM available: fall back to the heuristic section digest.
        sections = _sections_from_segments(segments)
        notes = _section_notes(sections)
        for i, note in enumerate(notes, 1):
            lines.append(f"### {i}. {_fmt_ts(note['start'])}")
            lines.append(f"- **{strings['overview']}:** {note['overview']}")
            if note["points"]:
                lines.append(f"- **{strings['points']}:**")
                lines += [f"  {idx}. {point}"
                          for idx, point in enumerate(note["points"], 1)]
        if notes:
            lines.append(strings['heuristic'])
        else:
            lines.append(strings['empty'])

    lines += ["", "## Layer 3 · " + strings['layer3']]
    if llm_final:
        lines.append(llm_final)
    elif status == "summarizing":
        lines.append(strings['summarizing_l3'])
    elif status not in ("done", "error"):
        lines.append(strings['wait_l3'])
    else:
        sections = _sections_from_segments(segments)
        lines += _final_note_from_sections(_section_notes(sections), status, language)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Localhost web UI
# --------------------------------------------------------------------------- #
_PAGE_CSS = """
:root{--bg:#0f1115;--fg:#e6e8ee;--mut:#8a90a2;--card:#171a21;--line:#262b36;
--acc:#6c8cff;--rec:#f43f5e;--ok:#34d399;--amb:#fbbf24}
@media(prefers-color-scheme:light){:root{--bg:#f7f8fa;--fg:#1a1d24;--mut:#69707f;
--card:#fff;--line:#e4e7ee;--acc:#4463d8}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.6 -apple-system,"Segoe UI","Noto Sans CJK JP",sans-serif}
.wrap{max-width:860px;margin:0 auto;padding:28px 20px 60px}
a{color:var(--acc);text-decoration:none}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:26px 0 8px}
.mut{color:var(--mut);font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:18px 20px;margin:14px 0}
.pill{display:inline-flex;align-items:center;gap:7px;font-size:13px;
padding:3px 12px;border-radius:99px;border:1px solid var(--line)}
.dot{width:9px;height:9px;border-radius:50%}
.rec .dot{background:var(--rec);animation:pulse 1.2s infinite}
.sum .dot{background:var(--amb);animation:pulse 1.2s infinite}
.done .dot{background:var(--ok)}
.err .dot{background:var(--rec)}
@keyframes pulse{50%{opacity:.35}}
button{background:var(--acc);color:#fff;border:0;border-radius:9px;
padding:9px 18px;font-size:14px;cursor:pointer}
button.stop{background:var(--rec)}button:disabled{opacity:.5;cursor:default}
button.b2{background:transparent;color:var(--acc);border:1px solid var(--line)}
select{background:var(--card);color:var(--fg);border:1px solid var(--line);
border-radius:8px;padding:8px 10px;font-size:14px}
.memo{display:flex;justify-content:space-between;align-items:center;gap:12px;
padding:9px 2px;border-top:1px solid var(--line)}
.memo button{padding:5px 14px;font-size:13px}
.md h2{font-size:19px;margin:22px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.md h3{font-size:16px;margin:18px 0 6px}
.md h4{font-size:14px;margin:15px 0 4px;color:var(--mut);
letter-spacing:.02em;text-transform:none}
.md h5,.md h6{font-size:13px;margin:12px 0 4px;color:var(--mut)}
.md ul{padding-left:22px;margin:6px 0}.md li{margin:3px 0}
.md li.task{list-style:none;margin-left:-22px;display:flex;align-items:flex-start;gap:8px}
.md li.task input{margin:0;flex:0 0 auto;position:relative;top:5px}
.md code{background:var(--line);padding:1px 5px;border-radius:5px;font-size:13px}
.tr{max-height:46vh;overflow-y:auto}
.row{display:flex;justify-content:space-between;align-items:center;gap:12px}
.pipeline{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:14px 0}
.stage{border:1px solid var(--line);border-radius:8px;padding:9px 10px;font-size:13px;
cursor:pointer;user-select:none}
.stage:hover{border-color:var(--acc);background:var(--line)}
.stage b{display:block;font-size:12px;color:var(--mut);font-weight:500}
.stage.on{border-color:var(--acc)}
.prog{height:7px;border-radius:99px;background:var(--line);overflow:hidden;margin-top:8px}
.prog>i{display:block;height:100%;width:0;background:var(--acc)}
.progline{margin-top:8px}
.item{display:block;padding:13px 16px;margin:8px 0;border:1px solid var(--line);
border-radius:10px;background:var(--card);color:var(--fg)}
.item:hover{border-color:var(--acc)}
.del{cursor:pointer;color:var(--mut);font-size:14px;padding:3px 9px;border-radius:7px;
user-select:none;white-space:nowrap;border:1px solid transparent}
.del:hover{color:#fff;background:var(--rec)}
.del.arm{color:#fff;background:var(--rec);font-size:12px;font-weight:600}
.files a{margin-right:14px;font-size:13px}
"""

_MD_JS = """
function md(src){
  const esc=t=>t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const inline=t=>esc(t)
    .replace(/\\*\\*(.+?)\\*\\*/g,'<b>$1</b>')
    .replace(/(^|\\s)_([^_]+)_(?=\\s|$)/g,'$1<i>$2</i>')
    .replace(/`(.+?)`/g,'<code>$1</code>')
    .replace(/\\[(.+?)\\]\\((https?:[^)]+)\\)/g,'<a href="$2">$1</a>');
  let out=[],inlist=false;
  for(const l of src.split('\\n')){
    const h=l.match(/^(#{1,6})\\s*(\\S.*)$/);
    if(h){if(inlist){out.push('</ul>');inlist=false}
      const n=Math.min(6,h[1].length+1);
      const txt=h[2].replace(/^(?:#{1,6}\\s*)+/,'');  // drop duplicated ### markers
      out.push(`<h${n}>${inline(txt)}</h${n}>`);continue}
    let m=l.match(/^\\s*[-*] \\[( |x)\\] (.*)/);
    if(m){if(!inlist){out.push('<ul>');inlist=true}
      out.push(`<li class=task><input type=checkbox disabled ${m[1]=='x'?'checked':''}>${inline(m[2])}</li>`);continue}
    m=l.match(/^\\s*[-*] (.*)/);
    if(m){if(!inlist){out.push('<ul>');inlist=true}
      out.push('<li>'+inline(m[1])+'</li>');continue}
    if(inlist){out.push('</ul>');inlist=false}
    if(l.startsWith('> ')){out.push('<p class=mut>'+inline(l.slice(2))+'</p>');continue}
    if(l.trim())out.push('<p>'+inline(l)+'</p>');
  }
  if(inlist)out.push('</ul>');
  return out.join('\\n');
}
"""

_I18N_JS = """
const LANGS=[['zh','中文'],['ja','日本語'],['en','English']];
const I18N={
 zh:{tagline:'录音 / 导入 → 转写 → 全文总结 → 一页纸',graph:'🕸️ 知识图谱',
  start:'● 开始录音',stop:'■ 停止录音',addAudio:'添加音频',upload:'⬆ 上传音频',
  memos:'🎤 语音备忘录',langTitle:'转写语言；Auto 每段自动识别',
  delConfirm:'确认删除?',deleting:'删除中…',delTitle:'删除',imported:'已导入 · 打开',
  import:'导入',importing:'导入中…',noSessions:'还没有记录。开始录音、上传或导入语音备忘录。',
  loadingMemos:'正在加载语音备忘录…',noMemos:'没有找到语音备忘录。',uploading:'正在上传',
  uploadFail:'上传失败',couldNotStart:'无法开始',importFail:'导入失败',delFail:'删除失败',
  allSessions:'← 全部记录',sessionTitle:'会话',transcriptH:'转写',waiting:'等待语音…',
  stopBtn:'■ 停止',resume:'重新转换',couldNotResume:'无法重新转换',
  l1:'原始转写',l2:'全文总结',l3:'一页纸',
  st_recording:'录音中',st_transcribing:'转写中…',st_summarizing:'总结中…',st_done:'完成',st_error:'错误',
  src_uploaded:'上传',src_memo:'语音备忘录',src_micspk:'麦克风 + 扬声器',src_mic:'麦克风',preparing:'正在准备音频…'},
 ja:{tagline:'録音 / 取り込み → 文字起こし → 全文要約 → 1ページ',graph:'🕸️ ナレッジグラフ',
  start:'● 録音開始',stop:'■ 録音停止',addAudio:'音声を追加',upload:'⬆ アップロード',
  memos:'🎤 ボイスメモ',langTitle:'文字起こし言語；Auto は自動判定',
  delConfirm:'削除しますか?',deleting:'削除中…',delTitle:'削除',imported:'取り込み済み · 開く',
  import:'取り込む',importing:'取り込み中…',noSessions:'まだ記録がありません。録音・アップロード・ボイスメモから。',
  loadingMemos:'ボイスメモを読み込み中…',noMemos:'ボイスメモが見つかりません。',uploading:'アップロード中',
  uploadFail:'アップロード失敗',couldNotStart:'開始できません',importFail:'取り込み失敗',delFail:'削除に失敗',
  allSessions:'← すべての記録',sessionTitle:'セッション',transcriptH:'文字起こし',waiting:'音声を待機中…',
  stopBtn:'■ 停止',resume:'変換を再開',couldNotResume:'再開できません',
  l1:'生の文字起こし',l2:'全文要約',l3:'1ページ',
  st_recording:'録音中',st_transcribing:'文字起こし中…',st_summarizing:'要約中…',st_done:'完了',st_error:'エラー',
  src_uploaded:'アップロード',src_memo:'ボイスメモ',src_micspk:'マイク + スピーカー',src_mic:'マイク',preparing:'音声を準備中…'},
 en:{tagline:'Record / import → transcript → full summary → one page',graph:'🕸️ Knowledge graph',
  start:'● Start recording',stop:'■ Stop recording',addAudio:'Add audio',upload:'⬆ Upload audio',
  memos:'🎤 Voice Memos',langTitle:'Transcription language; Auto detects each chunk',
  delConfirm:'Delete?',deleting:'Deleting…',delTitle:'Delete',imported:'Imported · Open',
  import:'Import',importing:'Importing…',noSessions:'No sessions yet. Record, upload, or import a Voice Memo.',
  loadingMemos:'Loading Voice Memos…',noMemos:'No Voice Memos found.',uploading:'Uploading',
  uploadFail:'upload failed',couldNotStart:'could not start',importFail:'import failed',delFail:'delete failed',
  allSessions:'← all sessions',sessionTitle:'Session',transcriptH:'Transcript',waiting:'Waiting for speech…',
  stopBtn:'■ Stop',resume:'Resume conversion',couldNotResume:'could not resume',
  l1:'Raw transcript',l2:'Full summary',l3:'One page',
  st_recording:'Recording',st_transcribing:'Transcribing…',st_summarizing:'Summarizing…',st_done:'Done',st_error:'Error',
  src_uploaded:'uploaded',src_memo:'Voice Memo',src_micspk:'mic + speakers',src_mic:'mic',preparing:'Preparing audio…'}
};
function curLang(){let l=localStorage.getItem('vn_lang');if(!l){const n=(navigator.language||'zh').toLowerCase();l=n.startsWith('ja')?'ja':n.startsWith('en')?'en':'zh';}return I18N[l]?l:'zh';}
function setLang(l){localStorage.setItem('vn_lang',l);if(!localStorage.getItem('vn_transcription_explicit'))localStorage.setItem('vn_transcription_lang',l);location.reload();}
function t(k){const d=I18N[curLang()]||I18N.zh;return (k in d)?d[k]:(k in I18N.zh?I18N.zh[k]:k);}
function transcriptionLang(){return localStorage.getItem('vn_transcription_lang')||curLang();}
function setTranscriptionLang(l){localStorage.setItem('vn_transcription_lang',l);localStorage.setItem('vn_transcription_explicit','1');}
function langSelect(){const c=curLang();return '<select onchange="setLang(this.value)" title="UI language" style="padding:6px 8px">'+LANGS.map(function(p){return '<option value="'+p[0]+'"'+(p[0]===c?' selected':'')+'>'+p[1]+'</option>';}).join('')+'</select>';}
"""

_STATUS_JS = """
function stat(s){const m={recording:'rec',transcribing:'sum',summarizing:'sum',done:'done',error:'err'};
  return [m[s]||'done', t('st_'+s)];}
function esc(t){return (t||'').replace(/&/g,'&amp;').replace(/</g,'&lt;')}
function lang(){return encodeURIComponent(document.getElementById('lang')?.value||transcriptionLang())}
function progress(d){
  const pct=Math.max(0,Math.min(100,Number(d.progress_percent||0)));
  const label=esc(d.status?t('st_'+d.status):t('preparing'))+(d.duration?' · '+esc(d.duration):'');
  return `<div class=progline><div class=mut>${label} · ${pct}%</div><div class=prog><i style="width:${pct}%"></i></div></div>`;
}
"""

_INDEX_HTML = """<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>voice-notes</title><style>__CSS__</style>
<div class=wrap>
 <div class=row><div><h1>🎙️ voice-notes</h1>
  <div class=mut id=tagline></div></div>
  <div style="display:flex;gap:10px;align-items:center">
   <span id=uilang></span>
   <a class=b2 id=graphlink href="/graph" target="_blank"
     style="padding:9px 16px;border-radius:9px;border:1px solid var(--line)"></a>
   <button id=btn onclick=toggle()></button></div></div>
 <div class=card><div class=row>
   <b id=addAudioLabel></b>
   <div>
    <select id=lang onchange="setTranscriptionLang(this.value)">
      <option value="zh" selected>中文</option>
      <option value="auto">Auto</option>
      <option value="en">English</option>
      <option value="ja">日本語</option>
    </select>
    <input type=file id=file accept="audio/*,.m4a,.mp3,.wav,.aac,.flac,.ogg,.mp4"
      style=display:none onchange=upload(this)>
    <button class=b2 id=uploadBtn onclick="document.getElementById('file').click()"></button>
    <button class=b2 id=memosBtn onclick=loadMemos()></button>
   </div></div>
  <div id=upmsg class=mut></div>
  <div id=memos></div>
 </div>
 <div id=list></div>
</div>
<script>__MDJS__ __I18NJS__ __STATUSJS__
function applyStatic(){
  document.getElementById('tagline').textContent=t('tagline');
  document.getElementById('graphlink').textContent=t('graph');
  document.getElementById('addAudioLabel').textContent=t('addAudio');
  document.getElementById('uploadBtn').textContent=t('upload');
  document.getElementById('memosBtn').textContent=t('memos');
  document.getElementById('lang').title=t('langTitle');
  document.getElementById('lang').value=transcriptionLang();
  document.getElementById('uilang').innerHTML=langSelect();
}
let live=null;
async function refresh(){
  const r=await fetch('/api/sessions');const d=await r.json();
  live=d.find(s=>s.status==='recording')||null;
  const btn=document.getElementById('btn');
  btn.textContent=live?t('stop'):t('start');
  btn.className=live?'stop':'';
  document.getElementById('list').innerHTML=d.map(s=>{
    const [cls,lab]=stat(s.status);
    const icon=s.source==='upload'?'📁 ':s.source==='memo'?'🎤 ':'';
    return `<a class=item href="/s/${s.id}"><div class=row>
      <div><b>${icon}${esc(s.title)}</b><div class=mut>${s.started.replace('T',' ')} · ${s.duration}</div></div>
      <div style="display:flex;align-items:center;gap:4px">
        <span class="pill ${cls}"><span class=dot></span>${lab}</span>
        <span class=del onclick="del(event,'${s.id}')" title="${t('delTitle')}">🗑</span>
      </div></div>${progress(s)}</a>`;
  }).join('')||`<div class="card mut">${t('noSessions')}</div>`;
}
async function toggle(){
  if(live){await fetch('/api/stop',{method:'POST'});refresh();}
  else{const r=await fetch('/api/start?language='+lang(),{method:'POST'});const d=await r.json();
    if(d.id)location='/s/'+d.id; else alert(d.error||t('couldNotStart'));}
}
async function del(ev,id){
  ev.preventDefault();ev.stopPropagation();
  const el=ev.currentTarget;
  if(el.dataset.armed!=='1'){                 // first click: arm, auto-disarm in 3s
    el.dataset.armed='1';el.textContent=t('delConfirm');el.classList.add('arm');
    clearTimeout(el._t);
    el._t=setTimeout(()=>{el.dataset.armed='';el.textContent='🗑';el.classList.remove('arm');},3000);
    return;
  }
  clearTimeout(el._t);el.textContent=t('deleting');
  const r=await fetch('/api/delete',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({id})});
  const j=await r.json();
  if(j.ok){refresh();} else {el.textContent='🗑';el.dataset.armed='';el.classList.remove('arm');alert(j.error||t('delFail'));}
}
async function upload(inp){
  const f=inp.files[0];if(!f)return;
  document.getElementById('upmsg').textContent=t('uploading')+' '+f.name+'…';
  const r=await fetch('/api/upload?name='+encodeURIComponent(f.name)+'&language='+lang(),
    {method:'POST',body:f});
  const d=await r.json();
  if(d.id)location='/s/'+d.id;
  else document.getElementById('upmsg').textContent=d.error||t('uploadFail');
  inp.value='';
}
async function loadMemos(){
  const el=document.getElementById('memos');
  el.innerHTML='<p class=mut>'+t('loadingMemos')+'</p>';
  const r=await fetch('/api/voicememos');const d=await r.json();
  if(d.error){el.innerHTML='<p class=mut>'+esc(d.error)+'</p>';return}
  el.innerHTML=d.items.map((m,i)=>{
    const meta=`<span class=mut>${m.date}${m.duration?' · '+m.duration:''}</span>`;
    const right=m.sid
      ? `<a class="pill done" href="/s/${m.sid}"><span class=dot></span>${t('imported')}</a>`
      : `<button onclick=imp(${i},this)>${t('import')}</button>`;
    return `<div class=memo><div><b>${esc(m.title)}</b> ${meta}</div>${right}</div>`;
  }).join('')||`<p class=mut>${t('noMemos')}</p>`;
}
async function imp(i,btn){
  btn.disabled=true;btn.textContent=t('importing');
  const r=await fetch('/api/voicememos/import',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({i,language:transcriptionLang()})});
  const d=await r.json();
  if(d.id)location='/s/'+d.id;
  else{alert(d.error||t('importFail'));btn.disabled=false;btn.textContent=t('import');}
}
applyStatic();refresh();setInterval(refresh,2000);
</script>"""

_SESSION_HTML = """<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>notes __SID__</title><style>__CSS__</style>
<div class=wrap>
 <div class=row><div>
   <div class=mut><a id=backlink href=/></a></div>
   <h1 id=title></h1><div class=mut id=sub></div></div>
  <div style="text-align:right">
   <div id=uilang style="margin-bottom:8px"></div>
   <div id=pill class=pill><span class=dot></span><span id=plabel>…</span></div><br>
   <button id=stop class=stop style="margin-top:8px;display:none"
     onclick="fetch('/api/stop',{method:'POST'})"></button>
   <button id=resume class=b2 style="margin-top:8px;display:none"
     onclick="resumeImport()"></button>
  </div></div>
 <div id=progress></div>
 <div class=card id=errcard style="display:none;color:var(--rec)"></div>
 <div class=pipeline>
   <div id=st1 class=stage onclick="jump(1)"><b>Layer 1</b><span id=l1lab></span></div>
   <div id=st2 class=stage onclick="jump(2)"><b>Layer 2</b><span id=l2lab></span></div>
   <div id=st3 class=stage onclick="jump(3)"><b>Layer 3</b><span id=l3lab></span></div>
 </div>
 <div class=card id=notescard style=display:none><div id=notes class=md></div>
  <div class=files style="margin-top:10px">
   <a href="/f/__SID__/notes.md" download>⬇ notes.md</a></div></div>
 <div class=card id=sumcard style=display:none><div id=summary class=md></div>
  <div class=files style="margin-top:10px">
   <a href="/f/__SID__/summary.md" download>⬇ summary.md</a>
   <a href="/f/__SID__/transcript.md" download>⬇ transcript.md</a>
   <a href="/f/__SID__/audio.wav" download>⬇ audio.wav</a></div></div>
 <h2 id=transcriptH></h2>
 <div class="card tr"><div id=transcript class=md>
   <p class=mut id=waitmsg></p></div></div>
</div>
<script>__MDJS__ __I18NJS__ __STATUSJS__
const sid=__SIDJS__;let last='',lastNotes='';
function applyStatic(){
  document.getElementById('uilang').innerHTML=langSelect();
  document.getElementById('backlink').textContent=t('allSessions');
  document.getElementById('stop').textContent=t('stopBtn');
  document.getElementById('resume').textContent=t('resume');
  document.getElementById('l1lab').textContent=t('l1');
  document.getElementById('l2lab').textContent=t('l2');
  document.getElementById('l3lab').textContent=t('l3');
  document.getElementById('transcriptH').textContent=t('transcriptH');
  document.getElementById('waitmsg').textContent=t('waiting');
}
function stages(s,hasTranscript,hasNotes){
  document.getElementById('st1').className='stage '+(hasTranscript?'on':'');
  document.getElementById('st2').className='stage '+(hasNotes?'on':'');
  document.getElementById('st3').className='stage '+(s==='done'?'on':'');
}
function scrollToEl(el){
  // Instant scroll (smooth is unreliable over the long notes card) with a
  // small top margin so the heading isn't flush against the window edge.
  const y=el.getBoundingClientRect().top+window.scrollY-16;
  window.scrollTo(0,Math.max(0,y));
}
function jump(n){
  // Scroll to the "## Layer n" heading inside the rendered notes.
  const card=document.getElementById('notescard');
  if(card&&card.style.display!=='none'){
    for(const h of card.querySelectorAll('h3')){
      if(h.textContent.trim().indexOf('Layer '+n)===0){scrollToEl(h);return;}
    }
    scrollToEl(card);return;
  }
  // Layer 1 with no notes yet -> the live transcript card.
  const tr=document.getElementById('transcript');
  if(tr)scrollToEl(tr.closest('.card'));
}
async function refresh(){
  const r=await fetch('/api/s/'+sid);if(!r.ok)return;const d=await r.json();
  document.getElementById('title').textContent=d.title||t('sessionTitle');
  document.getElementById('sub').textContent=d.started.replace('T',' ')+' · '+d.duration+
    ' · '+(d.source==='upload'?t('src_uploaded'):d.source==='memo'?t('src_memo'):
     d.system_audio?t('src_micspk'):t('src_mic'));
  document.getElementById('progress').innerHTML=progress(d);
  const [cls,lab]=stat(d.status);
  document.getElementById('pill').className='pill '+cls;
  document.getElementById('plabel').textContent=lab;
  document.getElementById('stop').style.display=d.status==='recording'?'':'none';
  document.getElementById('resume').style.display=
    (d.source!=='live'&&d.status!=='done')?'':'none';
  const ec=document.getElementById('errcard');
  if(d.status==='error'){ec.style.display='';ec.textContent=d.error||'failed';}
  stages(d.status,!!d.transcript,!!d.notes);
  if(d.notes&&d.notes!==lastNotes){lastNotes=d.notes;
    document.getElementById('notescard').style.display='';
    document.getElementById('notes').innerHTML=md(d.notes);}
  if(d.transcript!==last){last=d.transcript;
    const el=document.getElementById('transcript');
    const box=el.parentElement;
    const stick=box.scrollTop+box.clientHeight>=box.scrollHeight-40;
    el.innerHTML=md(d.transcript)||('<p class=mut>'+t('waiting')+'</p>');
    if(stick)box.scrollTop=box.scrollHeight;}
  if(d.summary){document.getElementById('sumcard').style.display='';
    document.getElementById('summary').innerHTML=md(d.summary);}
  setTimeout(refresh,(d.status==='done'||d.status==='error')?10000:1500);
}
async function resumeImport(){
  const r=await fetch('/api/resume',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({sid})});
  const d=await r.json();
  if(!d.id)alert(d.error||t('couldNotResume'));
  refresh();
}
applyStatic();refresh();
</script>"""


def _render(template: str, **tokens) -> str:
    out = template.replace("__CSS__", _PAGE_CSS) \
                  .replace("__MDJS__", _MD_JS) \
                  .replace("__I18NJS__", _I18N_JS) \
                  .replace("__STATUSJS__", _STATUS_JS)
    for k, v in tokens.items():
        out = out.replace(f"__{k.upper()}__", v)
    return out


def _graph_viewer_html() -> str:
    """The shared standalone graph viewer, wired to this server's /api/graph.

    Reads graph.html bundled next to this module (a copy of the canonical
    graph-viewer/graph.html). Defaults the viewer's data source to /api/graph
    so /graph works with no query string."""
    try:
        html = (Path(__file__).with_name("graph.html")).read_text()
    except OSError:
        return ""
    # Point the viewer at our knowledge graph unless ?src= overrides it.
    return html.replace(
        'const SRC = qs.get("src") || "graph.json";',
        'const SRC = qs.get("src") || "/api/graph";').replace(
        'const TITLE = qs.get("title") || "";',
        'const TITLE = qs.get("title") || "";')


_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}0*$")
_FILES = {"summary.md": "text/markdown; charset=utf-8",
          "notes.md": "text/markdown; charset=utf-8",
          "transcript.md": "text/markdown; charset=utf-8",
          "chunks.json": "application/json; charset=utf-8",
          "audio.wav": "audio/wav"}
_MAX_UPLOAD = 2 * 1024 ** 3   # 2 GB


class NotesServer:
    """Tiny localhost UI: sessions, live view, start/stop, upload, Voice Memos.

    `controller` bridges back to the app:
        controller.start() -> session id | None
        controller.stop()  -> None
        controller.active() -> NotesSession | None
        controller.import_file(path, title, source) -> session id | None
        controller.resume_import(session_id) -> session id | None
    """

    def __init__(self, base_dir: Path, controller, host="127.0.0.1", port=8765):
        self.base_dir = base_dir
        self.controller = controller
        self.host = host
        self.port = port
        self.httpd = None
        self.url = None
        self._memos: list[dict] = []   # last listing; imports pick by index

    def _session_meta(self, sid: str) -> dict | None:
        active = self.controller.active()
        if active and active.id == sid:
            active._write_meta()
        try:
            meta = json.loads((self.base_dir / sid / "meta.json").read_text())
            return _meta_with_progress_defaults(meta)
        except Exception:
            return None

    def _api_session(self, sid: str) -> dict | None:
        meta = self._session_meta(sid)
        if meta is None:
            return None
        d = self.base_dir / sid
        def read(name):
            try:
                return (d / name).read_text()
            except OSError:
                return ""
        meta["transcript"] = read("transcript.md")
        meta["notes"] = read("notes.md")
        meta["summary"] = read("summary.md")
        return meta

    def _api_sessions(self) -> list[dict]:
        out = []
        if self.base_dir.exists():
            for p in sorted(self.base_dir.iterdir(), reverse=True):
                if p.is_dir() and _ID_RE.match(p.name):
                    meta = self._session_meta(p.name)
                    if meta:
                        out.append(meta)
        return out

    def _imported_index(self) -> dict[str, str]:
        """basename(source audio) -> session id, for imported sessions.

        Lets the Voice Memos list mark memos that are already imported and
        link straight to the existing session instead of importing again."""
        index: dict[str, str] = {}
        if not self.base_dir.exists():
            return index
        for p in sorted(self.base_dir.iterdir()):
            if not (p.is_dir() and _ID_RE.match(p.name)):
                continue
            try:
                meta = json.loads((p / "meta.json").read_text())
            except Exception:
                continue
            src = meta.get("source_path")
            if src:
                index.setdefault(Path(src).name, p.name)
        return index

    def _save_upload(self, handler, name: str) -> Path:
        safe = re.sub(r"[^\w.\- ]+", "_", Path(unquote(name)).name) or "upload"
        updir = self.base_dir / "_uploads"
        updir.mkdir(parents=True, exist_ok=True)
        dest = updir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{safe}"
        length = int(handler.headers.get("Content-Length") or 0)
        if not 0 < length <= _MAX_UPLOAD:
            raise ValueError("missing or oversized upload")
        with dest.open("wb") as f:
            remaining = length
            while remaining > 0:
                chunk = handler.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)
        return dest

    def _set_language(self, language: str | None):
        if language and hasattr(self.controller, "set_language"):
            self.controller.set_language(language)

    def _delete_session(self, sid: str) -> dict:
        """Permanently remove a session folder. Refuses a live recording."""
        if not isinstance(sid, str) or not _ID_RE.match(sid):
            return {"error": "bad session id"}
        active = self.controller.active()
        if active and getattr(active, "id", None) == sid \
                and getattr(active, "status", "") == "recording":
            return {"error": "正在录音，请先停止再删除"}
        target = (self.base_dir / sid).resolve()
        if target.parent != self.base_dir.resolve() or not target.is_dir():
            return {"error": "session not found"}
        try:
            shutil.rmtree(target)
        except OSError as e:
            return {"error": f"删除失败: {e}"}
        return {"ok": True}

    def start(self) -> str | None:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep the terminal clean
                pass

            def _send(self, code, body, ctype="text/html; charset=utf-8"):
                data = body if isinstance(body, bytes) else body.encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _json(self, obj, code=200):
                self._send(code, json.dumps(obj, ensure_ascii=False),
                           "application/json; charset=utf-8")

            def _body_json(self) -> dict:
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    return json.loads(self.rfile.read(min(length, 1 << 16)) or b"{}")
                except Exception:
                    return {}

            def do_GET(self):
                url = urlparse(self.path)
                parts = [p for p in url.path.split("/") if p]
                if not parts:
                    return self._send(200, _render(_INDEX_HTML))
                if parts == ["graph"]:
                    html = _graph_viewer_html()
                    if not html:
                        return self._send(404, "graph viewer not found", "text/plain")
                    return self._send(200, html)
                if parts[0] == "s" and len(parts) == 2 and _ID_RE.match(parts[1]):
                    return self._send(200, _render(
                        _SESSION_HTML, sid=parts[1], sidjs=json.dumps(parts[1])))
                if parts[0] == "api":
                    if parts[1:] == ["sessions"]:
                        return self._json(server._api_sessions())
                    if parts[1:] == ["graph"]:
                        self.send_response(200)
                        self.send_header("Content-Type",
                                         "application/json; charset=utf-8")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        body = json.dumps(aggregate_graph(server.base_dir),
                                          ensure_ascii=False).encode("utf-8")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        return self.wfile.write(body)
                    if parts[1:] == ["voicememos"]:
                        items, err = list_voice_memos()
                        if err:
                            return self._json({"error": err})
                        server._memos = items
                        imported = server._imported_index()
                        return self._json({"items": [
                            {**{k: m[k] for k in ("title", "date", "duration")},
                             "sid": imported.get(Path(m["path"]).name, "")}
                            for m in items]})
                    if len(parts) == 3 and parts[1] == "s" and _ID_RE.match(parts[2]):
                        data = server._api_session(parts[2])
                        return self._json(data or {"error": "not found"},
                                          200 if data else 404)
                if (parts[0] == "f" and len(parts) == 3
                        and _ID_RE.match(parts[1]) and parts[2] in _FILES):
                    path = server.base_dir / parts[1] / parts[2]
                    if path.exists():
                        return self._send(200, path.read_bytes(), _FILES[parts[2]])
                self._send(404, "not found", "text/plain")

            def do_POST(self):
                url = urlparse(self.path)
                parts = [p for p in url.path.split("/") if p]
                if parts == ["api", "start"]:
                    language = (parse_qs(url.query).get("language") or [""])[0]
                    server._set_language(language)
                    sid = server.controller.start()
                    return self._json({"id": sid} if sid else
                                      {"error": "busy or model still loading"})
                if parts == ["api", "stop"]:
                    server.controller.stop()
                    return self._json({"ok": True})
                if parts == ["api", "delete"]:
                    return self._json(
                        server._delete_session(self._body_json().get("id")))
                if parts == ["api", "upload"]:
                    qs = parse_qs(url.query)
                    name = (qs.get("name") or ["upload"])[0]
                    language = (qs.get("language") or [""])[0]
                    server._set_language(language)
                    try:
                        dest = server._save_upload(self, name)
                    except Exception as e:
                        return self._json({"error": f"upload failed: {e}"})
                    sid = server.controller.import_file(
                        dest, Path(name).stem, "upload")
                    return self._json({"id": sid} if sid else
                                      {"error": "model still loading — retry shortly"})
                if parts == ["api", "voicememos", "import"]:
                    body = self._body_json()
                    i = body.get("i")
                    server._set_language(body.get("language"))
                    if not isinstance(i, int) or not 0 <= i < len(server._memos):
                        return self._json({"error": "bad selection — reload the list"})
                    memo = server._memos[i]
                    sid = server.controller.import_file(
                        Path(memo["path"]), memo["title"], "memo")
                    return self._json({"id": sid} if sid else
                                      {"error": "model still loading — retry shortly"})
                if parts == ["api", "resume"]:
                    sid = self._body_json().get("sid")
                    if not isinstance(sid, str) or not _ID_RE.match(sid):
                        return self._json({"error": "bad session id"})
                    if not hasattr(server.controller, "resume_import"):
                        return self._json({"error": "resume not available"})
                    sid = server.controller.resume_import(sid)
                    return self._json({"id": sid} if sid else
                                      {"error": "missing source file or model still loading"})
                self._send(404, "not found", "text/plain")

        for port in range(self.port, self.port + 10):
            try:
                self.httpd = ThreadingHTTPServer((self.host, port), Handler)
                self.port = port
                break
            except OSError:
                continue
        if not self.httpd:
            print("[notes] Could not bind the notes web UI port.")
            return None
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://{self.host}:{self.port}"
        print(f"[notes] Web UI: {self.url}")
        return self.url

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd = None
        self.url = None
