#!/usr/bin/env python3
"""voice-notes MCP server.

Exposes the notes that voice-notes has already generated to any MCP client
(Claude Code, Claude Desktop, …) as read-only tools. It reads the same session
folders the app writes (see README.md), so nothing needs to be running — the
files on disk are the source of truth.

Data dir resolution (first that exists / is set):
  1. --data-dir CLI flag
  2. $VOICE_NOTES_DATA_DIR
  3. [recording] dir in ~/.config/voice-notes/config.toml
  4. ~/.local/share/voice-notes/sessions   (the app default)

Run (stdio transport):
    ./voice-notes-mcp
    # or: .venv/bin/python voice_notes_mcp.py

Tools:
  list_notes(limit=50)                 -> newest sessions with metadata
  read_note(session_id, part="summary")-> summary | one_page | full | notes
                                          | transcript markdown of one session
  search_notes(query, limit=20)        -> sessions whose notes mention query,
                                          with matching snippets
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from mcp.server.fastmcp import FastMCP

from graph_build import aggregate_graph

CONFIG_PATH = Path.home() / ".config" / "voice-notes" / "config.toml"
DEFAULT_DIR = Path.home() / ".local" / "share" / "voice-notes" / "sessions"
# Session folder names look like 20260713-165802 (optionally trailing zeros).
_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}0*$")


def resolve_data_dir(cli_dir: str | None = None) -> Path:
    if cli_dir:
        return Path(cli_dir).expanduser()
    env = os.environ.get("VOICE_NOTES_DATA_DIR")
    if env:
        return Path(env).expanduser()
    try:
        cfg = tomllib.loads(CONFIG_PATH.read_text())
        d = cfg.get("recording", {}).get("dir")
        if d:
            return Path(str(d)).expanduser()
    except Exception:
        pass
    return DEFAULT_DIR


DATA_DIR = resolve_data_dir()

mcp = FastMCP("voice-notes")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _session_dirs() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(
        (p for p in DATA_DIR.iterdir() if p.is_dir() and _ID_RE.match(p.name)),
        reverse=True,  # newest id first
    )


def _read(session: Path, name: str) -> str:
    try:
        return (session / name).read_text()
    except OSError:
        return ""


def _meta(session: Path) -> dict:
    try:
        return json.loads((session / "meta.json").read_text())
    except Exception:
        return {}


def _summary_card(session: Path) -> tuple[str, str]:
    """Split summary.md into (one_page, full_summary).

    Layout written by the app: the Layer 3 one-pager, a '---' rule, a
    '# 详细总结（Layer 2）' heading, then the Layer 2 full summary."""
    text = _read(session, "summary.md")
    if not text:
        return "", ""
    marker = "\n# 详细总结（Layer 2）"
    if marker in text:
        head, tail = text.split(marker, 1)
        one = head.rstrip().removesuffix("---").rstrip()
        full = tail.lstrip("\n")
        return one, full
    return text, ""


def _brief(session: Path) -> dict:
    m = _meta(session)
    return {
        "id": m.get("id", session.name),
        "title": m.get("title", session.name),
        "date": (m.get("started") or "").replace("T", " "),
        "duration": m.get("duration", ""),
        "status": m.get("status", ""),
        "source": m.get("source", ""),
    }


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@mcp.tool()
def list_notes(limit: int = 50) -> str:
    """List generated voice-notes sessions, newest first.

    Returns a JSON array of {id, title, date, duration, status, source}.
    Use the id with read_note to fetch a session's actual notes.
    """
    sessions = _session_dirs()[: max(1, limit)]
    return json.dumps([_brief(p) for p in sessions],
                      ensure_ascii=False, indent=1)


@mcp.tool()
def read_note(session_id: str, part: str = "summary") -> str:
    """Read one session's generated notes.

    session_id: the folder id from list_notes (e.g. "20260713-165802").
    part:
      "summary"    (default) the one-page note + the full summary
      "one_page"   just the Layer 3 one-page note
      "full"       just the Layer 2 detailed full summary
      "notes"      notes.md — all three layers incl. raw transcript chunks
      "transcript" the raw timestamped transcript
    """
    if not _ID_RE.match(session_id):
        return f"error: invalid session id {session_id!r}"
    session = DATA_DIR / session_id
    if not session.is_dir():
        return f"error: no session {session_id!r} in {DATA_DIR}"

    part = part.lower()
    if part == "transcript":
        return _read(session, "transcript.md") or "(no transcript)"
    if part == "notes":
        return _read(session, "notes.md") or "(no notes yet)"

    one, full = _summary_card(session)
    if part == "one_page":
        return one or "(no one-page summary yet)"
    if part == "full":
        return full or one or "(no full summary yet)"
    # default: summary (whole card as written)
    return _read(session, "summary.md") or "(no summary yet)"


@mcp.tool()
def search_notes(query: str, limit: int = 20) -> str:
    """Search across all sessions' summaries and notes for a keyword/phrase.

    Case-insensitive substring match over title, summary.md and notes.md.
    Returns a JSON array of {id, title, date, where, snippet} for the newest
    `limit` matching sessions. Follow up with read_note for the full text.
    """
    q = query.strip()
    if not q:
        return "error: empty query"
    ql = q.lower()
    out = []
    for session in _session_dirs():
        if len(out) >= max(1, limit):
            break
        brief = _brief(session)
        summary = _read(session, "summary.md")
        notes = _read(session, "notes.md")
        where, hay = None, None
        if ql in brief["title"].lower():
            where, hay = "title", brief["title"]
        elif ql in summary.lower():
            where, hay = "summary", summary
        elif ql in notes.lower():
            where, hay = "notes", notes
        if not where:
            continue
        idx = hay.lower().find(ql)
        start = max(0, idx - 60)
        snippet = hay[start: idx + len(q) + 60].replace("\n", " ").strip()
        if start > 0:
            snippet = "…" + snippet
        out.append({**brief, "where": where, "snippet": snippet})
    return json.dumps(out, ensure_ascii=False, indent=1)


# --------------------------------------------------------------------------- #
# Knowledge graph
# --------------------------------------------------------------------------- #
def _graph():
    return aggregate_graph(DATA_DIR)


def _entity_meetings(g: dict, node_id: str) -> list[dict]:
    """Session nodes that mention the entity node_id."""
    idx = {n["id"]: n for n in g["nodes"]}
    out = []
    for e in g["edges"]:
        if e.get("to") == node_id and not e.get("rel"):
            s = idx.get(e["from"])
            if s and s["type"] == "session":
                out.append({"title": s["label"],
                            "url": s.get("meta", {}).get("url", "")})
    return out


@mcp.tool()
def knowledge_graph(top: int = 20) -> str:
    """Overview of the cross-meeting knowledge graph built from all notes.

    Entities (person/project/org/concept/decision/todo) are merged across
    meetings by name/alias; recordings that are parts of one meeting are merged
    too. Returns JSON: {meetings, entities, edges, by_type, top_entities:
    [{name, type, mentions, desc, meetings:[titles]}]} — top_entities are the
    most-mentioned hubs. Use find_entity / list_notes for detail."""
    g = _graph()
    ents = [n for n in g["nodes"] if n["type"] != "session"]
    sess = [n for n in g["nodes"] if n["type"] == "session"]
    by_type: dict[str, int] = {}
    for n in ents:
        by_type[n["type"]] = by_type.get(n["type"], 0) + 1
    ents.sort(key=lambda n: -n["meta"].get("mentions", 0))
    top_entities = [{
        "name": n["label"], "type": n["type"],
        "mentions": n["meta"].get("mentions", 0),
        "desc": n["meta"].get("desc", ""),
        "meetings": [m["title"] for m in _entity_meetings(g, n["id"])],
    } for n in ents[: max(1, top)]]
    return json.dumps({
        "meetings": len(sess), "entities": len(ents), "edges": len(g["edges"]),
        "by_type": by_type, "top_entities": top_entities,
    }, ensure_ascii=False, indent=1)


@mcp.tool()
def find_entity(name: str) -> str:
    """Look up an entity (person / project / concept / …) in the knowledge
    graph by name (case-insensitive substring). Returns, for each match:
    type, description, how many meetings mention it, those meetings (title +
    session url), and related entities (from extracted relations)."""
    q = name.strip().lower()
    if not q:
        return "error: empty name"
    g = _graph()
    idx = {n["id"]: n for n in g["nodes"]}
    matches = [n for n in g["nodes"]
               if n["type"] != "session" and q in n["label"].lower()]
    if not matches:
        return json.dumps({"matches": []}, ensure_ascii=False)
    out = []
    for n in matches[:10]:
        related = []
        for e in g["edges"]:
            if not e.get("rel"):
                continue
            if e["from"] == n["id"] and idx.get(e["to"]):
                related.append({"name": idx[e["to"]]["label"],
                                "label": e.get("label", "")})
            elif e["to"] == n["id"] and idx.get(e["from"]):
                related.append({"name": idx[e["from"]]["label"],
                                "label": e.get("label", "")})
        out.append({
            "name": n["label"], "type": n["type"],
            "desc": n["meta"].get("desc", ""),
            "mentions": n["meta"].get("mentions", 0),
            "meetings": _entity_meetings(g, n["id"]),
            "related": related[:20],
        })
    return json.dumps({"matches": out}, ensure_ascii=False, indent=1)


def main() -> None:
    global DATA_DIR
    ap = argparse.ArgumentParser(description="voice-notes MCP server (stdio)")
    ap.add_argument("--data-dir", help="override the session data directory")
    args = ap.parse_args()
    if args.data_dir:
        DATA_DIR = resolve_data_dir(args.data_dir)
    print(f"[voice-notes-mcp] serving notes from {DATA_DIR}", file=sys.stderr)
    mcp.run()


if __name__ == "__main__":
    main()
