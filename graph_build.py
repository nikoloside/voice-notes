"""Knowledge-graph aggregation for voice-notes.

Lightweight (stdlib only, no numpy/whisper) so both the app (voice_notes.py)
and the MCP server (voice_notes_mcp.py) can build the same graph. Reads each
session's meta.json + entities.json off disk and merges them into the shared
{nodes, edges} contract (see graph-viewer/README.md).

Two kinds of merging make the graph less fragmented:
  1. Meeting grouping — recordings that are parts of one meeting ("Meeting - 1",
     "Meeting - 2", …) collapse into a single meeting node.
  2. Entity aliasing — an optional graph_aliases.json ({alias: canonical})
     merges different names for the same entity/topic. It is produced by an
     LLM pass in voice_notes.py; if absent, entities merge by exact name only.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ENTITY_TYPES = ("person", "project", "org", "concept", "decision", "todo")
_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}0*$")

# Trailing "- 3" / " 2" / "(1)" / "part 2" / "第3部分" / "#2" part markers.
_PART_SUFFIX = re.compile(
    r"[\s\-–—_·#]*(?:part|pt\.?|第)?\s*[\(（]?\s*\d{1,3}\s*[\)）]?\s*(?:部分|集|段|回)?\s*$",
    re.IGNORECASE,
)


def _norm_key(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip()).lower()


def meeting_key(title: str) -> str:
    """Base name a recording belongs to, with a trailing part marker removed."""
    base = _PART_SUFFIX.sub("", str(title)).strip(" -–—_·#")
    return base or str(title).strip()


def load_aliases(base_dir: Path) -> dict:
    """{normalized alias -> canonical name} from graph_aliases.json, or {}."""
    try:
        raw = json.loads((base_dir / "graph_aliases.json").read_text())
        if isinstance(raw, dict):
            return {_norm_key(k): str(v) for k, v in raw.items()}
    except Exception:
        pass
    return {}


def _load_sessions(base_dir: Path):
    out = []
    if not base_dir.exists():
        return out
    for p in sorted(base_dir.iterdir()):
        if not (p.is_dir() and _ID_RE.match(p.name)):
            continue
        try:
            meta = json.loads((p / "meta.json").read_text())
        except Exception:
            continue
        try:
            graph = json.loads((p / "entities.json").read_text())
        except Exception:
            graph = {"entities": [], "relations": []}
        out.append((p.name, meta, graph))
    return out


def aggregate_graph(base_dir) -> dict:
    """Merge every session into one knowledge graph ({nodes, edges})."""
    base_dir = Path(base_dir)
    aliases = load_aliases(base_dir)
    sessions = _load_sessions(base_dir)

    # Group recordings into meetings by base title (only merge 2+; a lone
    # session keeps its own full title).
    groups: dict[str, list] = defaultdict(list)
    for sid, meta, graph in sessions:
        groups[meeting_key(meta.get("title", sid))].append((sid, meta, graph))

    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    entity_id: dict[str, str] = {}
    seen_edges: set = set()

    def canon(name: str) -> str:
        return aliases.get(_norm_key(name), str(name).strip())

    def entity_node(name: str, etype: str, desc: str, count: bool) -> str:
        cname = canon(name)
        if not cname:
            return ""
        key = _norm_key(cname)
        nid = entity_id.get(key)
        if nid is None:
            nid = "e:" + key
            entity_id[key] = nid
            etype = etype if etype in ENTITY_TYPES else "concept"
            nodes[nid] = {"id": nid, "label": cname, "type": etype,
                          "group": etype, "meta": {"desc": desc, "mentions": 0}}
        n = nodes[nid]
        if count:
            n["meta"]["mentions"] += 1     # counts distinct meetings
        if desc and len(desc) > len(n["meta"].get("desc", "")):
            n["meta"]["desc"] = desc
        return nid

    def add_edge(e: dict):
        k = (e["from"], e["to"], e.get("label", ""), bool(e.get("rel")))
        if k not in seen_edges:
            seen_edges.add(k)
            edges.append(e)

    for _, members in groups.items():
        first_sid, first_meta = members[0][0], members[0][1]
        mid = "s:" + first_sid
        if len(members) == 1:
            label = first_meta.get("title", first_sid)
            extra = {}
        else:
            label = meeting_key(first_meta.get("title", first_sid))
            extra = {"parts": " · ".join(m[1].get("title", m[0]) for m in members)}
        nodes[mid] = {
            "id": mid, "label": label, "type": "session", "group": "session",
            "meta": {"date": (first_meta.get("started") or "").replace("T", " "),
                     "duration": first_meta.get("duration", ""),
                     "url": f"/s/{first_sid}", **extra},
        }
        counted: set = set()   # entity keys already counted for this meeting
        local: dict = {}
        for sid, meta, graph in members:
            for e in graph.get("entities", []):
                cname = canon(e.get("name", ""))
                key = _norm_key(cname)
                nid = entity_node(e.get("name", ""), e.get("type", "concept"),
                                  e.get("desc", ""), count=key not in counted)
                if not nid:
                    continue
                counted.add(key)
                local[key] = nid
                add_edge({"from": mid, "to": nid, "label": "提到"})
            for r in graph.get("relations", []):
                a = local.get(_norm_key(canon(r.get("from", ""))))
                b = local.get(_norm_key(canon(r.get("to", ""))))
                if a and b and a != b:
                    add_edge({"from": a, "to": b,
                              "label": r.get("label", ""), "rel": True})
    return {"nodes": list(nodes.values()), "edges": edges}


def entity_names_by_type(base_dir) -> list[tuple[str, str]]:
    """Distinct (name, type) across all sessions, for alias generation."""
    seen: dict[str, tuple[str, str]] = {}
    for _, _, graph in _load_sessions(Path(base_dir)):
        for e in graph.get("entities", []):
            name = str(e.get("name", "")).strip()
            k = _norm_key(name)
            if name and k not in seen:
                etype = str(e.get("type", "concept")).strip().lower()
                seen[k] = (name, etype if etype in ENTITY_TYPES else "concept")
    return list(seen.values())
