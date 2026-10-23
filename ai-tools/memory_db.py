#!/usr/bin/env python3
"""SQLite-backed conversational memory for the AI assistant.

Stores chat turns, compresses them, and distills them into tagged, scored
"memory" rows the model can retrieve later. Stdlib only (sqlite3 + zlib), so it
imports cleanly from the bootstrapped venv or a bare interpreter.

Layout:
  conversations  one row per chat thread, with a rolling compressed summary
  messages       raw turns (zlib BLOB above a size threshold)
  memories       distilled atomic facts with importance + supersede chain
  tags/memory_tags  normalized tag vocabulary
  *_fts          FTS5 mirrors, maintained explicitly (not via triggers, since
                 message bodies may be compressed blobs and memory tags live in
                 a side table)
"""
import re
import math
import zlib
import sqlite3
from datetime import datetime, timezone

SCHEMA_VERSION = 1

KINDS = ("fact", "preference", "project", "event", "summary")
SOURCES = ("chat", "screenshot", "manual")

DEFAULT_COMPRESS_THRESHOLD = 2048
DEFAULT_HALF_LIFE_DAYS = 30.0
DEFAULT_TOP_K = 6

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS conversations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    title         TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    summary       TEXT,
    message_count INTEGER NOT NULL DEFAULT 0,
    archived      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         BLOB NOT NULL,
    compressed      INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    token_estimate  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);

CREATE TABLE IF NOT EXISTS memories (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
    kind             TEXT NOT NULL DEFAULT 'fact',
    content          TEXT NOT NULL,
    importance       REAL NOT NULL DEFAULT 0.5,
    created_at       TEXT NOT NULL,
    last_accessed_at TEXT,
    access_count     INTEGER NOT NULL DEFAULT 0,
    source           TEXT NOT NULL DEFAULT 'chat',
    superseded_by    INTEGER REFERENCES memories(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_live ON memories(superseded_by, importance);
CREATE INDEX IF NOT EXISTS idx_memories_conv ON memories(conversation_id);

CREATE TABLE IF NOT EXISTS tags (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS memory_tags (
    memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    tag_id    INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (memory_id, tag_id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content, tags);
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content);
"""

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "doing", "of", "to", "in", "on", "at", "by", "for",
    "with", "and", "or", "but", "if", "then", "than", "so", "as", "it", "its",
    "this", "that", "these", "those", "i", "me", "my", "you", "your", "we",
    "our", "he", "she", "they", "them", "what", "which", "who", "how", "when",
    "where", "why", "can", "could", "would", "should", "will", "shall", "may",
    "have", "has", "had", "not", "no", "yes", "there", "here", "about", "from",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_ts(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _age_days(value):
    dt = _parse_ts(value)
    if dt is None:
        return 0.0
    delta = datetime.now(timezone.utc) - dt
    return max(0.0, delta.total_seconds() / 86400.0)


def _clamp(value, low=0.0, high=1.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, value))


def _norm_tag(name):
    name = re.sub(r"[^a-z0-9_\-/ ]+", "", str(name).strip().lower())
    name = re.sub(r"\s+", "-", name).strip("-")
    return name[:48]


def _normalize_text(text):
    """Lowercased, punctuation-free form used for duplicate detection."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())).strip()


def _tokens(text):
    return {t for t in _normalize_text(text).split() if len(t) > 1 and t not in _STOPWORDS}


def fts_query(text, limit=32):
    """Turn free-form text into a safe FTS5 OR-query (operators stripped)."""
    seen, out = set(), []
    for tok in re.findall(r"[A-Za-z0-9_]+", (text or "").lower()):
        if len(tok) < 2 or tok in _STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        out.append('"%s"' % tok)
        if len(out) >= limit:
            break
    return " OR ".join(out)


# ---------------------------------------------------------------------------
# connection
# ---------------------------------------------------------------------------
def open_db(path, check_same_thread=True):
    """Open (creating if needed) the memory database in multi-process WAL mode.

    Pass check_same_thread=False when a single connection is shared between
    threads behind a caller-held lock (the daemon does this).
    """
    import os
    directory = os.path.dirname(os.path.abspath(os.path.expanduser(path)))
    if directory:
        os.makedirs(directory, exist_ok=True)

    conn = sqlite3.connect(
        os.path.expanduser(path), timeout=10.0, check_same_thread=check_same_thread
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# conversations & messages
# ---------------------------------------------------------------------------
def start_conversation(conn, title=None):
    now = _now()
    cur = conn.execute(
        "INSERT INTO conversations(title, created_at, updated_at) VALUES (?,?,?)",
        (title, now, now),
    )
    conn.commit()
    return cur.lastrowid


def rename_conversation(conn, conversation_id, title):
    conn.execute(
        "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
        (title[:120], _now(), conversation_id),
    )
    conn.commit()


def set_summary(conn, conversation_id, summary):
    conn.execute(
        "UPDATE conversations SET summary=?, updated_at=? WHERE id=?",
        (summary, _now(), conversation_id),
    )
    conn.commit()


def add_message(conn, conversation_id, role, content,
                compress_threshold=DEFAULT_COMPRESS_THRESHOLD):
    """Append a turn; bodies above the threshold are stored zlib-compressed."""
    content = content or ""
    raw = content.encode("utf-8")
    if len(raw) > compress_threshold:
        stored, flag = sqlite3.Binary(zlib.compress(raw, 6)), 1
    else:
        stored, flag = content, 0

    now = _now()
    cur = conn.execute(
        "INSERT INTO messages(conversation_id, role, content, compressed, created_at, token_estimate) "
        "VALUES (?,?,?,?,?,?)",
        (conversation_id, role, stored, flag, now, max(1, len(raw) // 4)),
    )
    message_id = cur.lastrowid
    # FTS always indexes the plain text, never the compressed blob.
    conn.execute("INSERT INTO messages_fts(rowid, content) VALUES (?,?)", (message_id, content))
    conn.execute(
        "UPDATE conversations SET message_count = message_count + 1, updated_at = ? WHERE id = ?",
        (now, conversation_id),
    )
    conn.commit()
    return message_id


def _decode(row):
    if row["compressed"]:
        return zlib.decompress(bytes(row["content"])).decode("utf-8", "replace")
    value = row["content"]
    return value if isinstance(value, str) else bytes(value).decode("utf-8", "replace")


def get_messages(conn, conversation_id, limit=None):
    sql = "SELECT * FROM messages WHERE conversation_id=? ORDER BY id"
    rows = conn.execute(sql, (conversation_id,)).fetchall()
    if limit:
        rows = rows[-limit:]
    return [
        {"id": r["id"], "role": r["role"], "content": _decode(r), "created_at": r["created_at"]}
        for r in rows
    ]


def get_conversation(conn, conversation_id):
    row = conn.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone()
    return dict(row) if row else None


def list_conversations(conn, limit=50, include_archived=False):
    sql = "SELECT * FROM conversations"
    if not include_archived:
        sql += " WHERE archived=0"
    sql += " ORDER BY updated_at DESC LIMIT ?"
    return [dict(r) for r in conn.execute(sql, (limit,)).fetchall()]


def purge_empty_conversations(conn, keep=None):
    """Drop conversation rows that never received a message.

    Threads are created lazily, but a crash (or an older build) can still leave
    empty shells behind; they would otherwise clutter the history sidebar.
    """
    rows = conn.execute(
        "SELECT c.id FROM conversations c "
        "LEFT JOIN messages m ON m.conversation_id = c.id "
        "WHERE m.id IS NULL GROUP BY c.id"
    ).fetchall()
    removed = 0
    for row in rows:
        if keep is not None and row["id"] == keep:
            continue
        conn.execute("DELETE FROM conversations WHERE id=?", (row["id"],))
        removed += 1
    if removed:
        conn.commit()
    return removed


def purge_conversation(conn, conversation_id):
    ids = [r["id"] for r in conn.execute(
        "SELECT id FROM messages WHERE conversation_id=?", (conversation_id,))]
    for mid in ids:
        conn.execute("DELETE FROM messages_fts WHERE rowid=?", (mid,))
    conn.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
    conn.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
    conn.commit()


# ---------------------------------------------------------------------------
# memories
# ---------------------------------------------------------------------------
def _tag_ids(conn, tags):
    ids = []
    for raw in tags or []:
        name = _norm_tag(raw)
        if not name:
            continue
        conn.execute("INSERT OR IGNORE INTO tags(name) VALUES (?)", (name,))
        row = conn.execute("SELECT id FROM tags WHERE name=?", (name,)).fetchone()
        if row:
            ids.append((row["id"], name))
    return ids


def add_memory(conn, content, kind="fact", importance=0.5, tags=None,
               conversation_id=None, source="chat", commit=True):
    content = (content or "").strip()
    if not content:
        return None
    if kind not in KINDS:
        kind = "fact"
    if source not in SOURCES:
        source = "chat"

    cur = conn.execute(
        "INSERT INTO memories(conversation_id, kind, content, importance, created_at, source) "
        "VALUES (?,?,?,?,?,?)",
        (conversation_id, kind, content, _clamp(importance), _now(), source),
    )
    memory_id = cur.lastrowid

    pairs = _tag_ids(conn, tags)
    for tag_id, _ in pairs:
        conn.execute(
            "INSERT OR IGNORE INTO memory_tags(memory_id, tag_id) VALUES (?,?)",
            (memory_id, tag_id),
        )
    conn.execute(
        "INSERT INTO memories_fts(rowid, content, tags) VALUES (?,?,?)",
        (memory_id, content, " ".join(name for _, name in pairs)),
    )
    if commit:
        conn.commit()
    return memory_id


def add_memories_bulk(conn, items, conversation_id=None, source="chat"):
    """items: iterable of dicts with content/kind/importance/tags."""
    ids = []
    for item in items or []:
        if isinstance(item, str):
            item = {"content": item}
        memory_id = add_memory(
            conn,
            item.get("content", ""),
            kind=item.get("kind", "fact"),
            importance=item.get("importance", 0.5),
            tags=item.get("tags"),
            conversation_id=conversation_id,
            source=source,
            commit=False,
        )
        if memory_id:
            ids.append(memory_id)
    conn.commit()
    return ids


def update_memory(conn, memory_id, content=None, kind=None, importance=None, tags=None):
    """Edit one memory in place, keeping the FTS mirror and tags in sync."""
    row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
    if row is None:
        return False

    new_content = (content if content is not None else row["content"]).strip()
    if not new_content:
        return False
    new_kind = kind if kind in KINDS else row["kind"]
    new_importance = _clamp(importance if importance is not None else row["importance"])

    conn.execute(
        "UPDATE memories SET content=?, kind=?, importance=? WHERE id=?",
        (new_content, new_kind, new_importance, memory_id),
    )

    if tags is not None:
        conn.execute("DELETE FROM memory_tags WHERE memory_id=?", (memory_id,))
        pairs = _tag_ids(conn, tags)
        for tag_id, _ in pairs:
            conn.execute(
                "INSERT OR IGNORE INTO memory_tags(memory_id, tag_id) VALUES (?,?)",
                (memory_id, tag_id),
            )
        tag_text = " ".join(name for _, name in pairs)
    else:
        tag_text = " ".join(get_tags(conn, memory_id))

    conn.execute("DELETE FROM memories_fts WHERE rowid=?", (memory_id,))
    conn.execute(
        "INSERT INTO memories_fts(rowid, content, tags) VALUES (?,?,?)",
        (memory_id, new_content, tag_text),
    )
    conn.commit()
    return True


def delete_memory(conn, memory_id):
    """Remove a memory permanently, along with its tags and FTS row."""
    conn.execute("UPDATE memories SET superseded_by=NULL WHERE superseded_by=?", (memory_id,))
    conn.execute("DELETE FROM memory_tags WHERE memory_id=?", (memory_id,))
    conn.execute("DELETE FROM memories_fts WHERE rowid=?", (memory_id,))
    cur = conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
    conn.execute("DELETE FROM tags WHERE id NOT IN (SELECT tag_id FROM memory_tags)")
    conn.commit()
    return cur.rowcount > 0


def get_memory(conn, memory_id):
    row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
    if row is None:
        return None
    item = dict(row)
    item["tags"] = get_tags(conn, memory_id)
    return item


def search_conversations(conn, query, k=25, half_life_days=DEFAULT_HALF_LIFE_DAYS):
    """Rank conversations by their best-matching message, title/summary overlap
    and recency. Returns conversation dicts with `score` and a `snippet`."""
    match = fts_query(query)
    best = {}

    if match:
        try:
            rows = conn.execute(
                "SELECT m.conversation_id AS cid, m.id AS mid, bm25(messages_fts) AS bm "
                "FROM messages_fts JOIN messages m ON m.id = messages_fts.rowid "
                "WHERE messages_fts MATCH ? ORDER BY bm LIMIT 800",
                (match,),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        for r in rows:
            relevance = max(0.0, -float(r["bm"]))
            relevance = relevance / (1.0 + relevance)
            current = best.get(r["cid"])
            if current is None or relevance > current[0]:
                best[r["cid"]] = (relevance, r["mid"])

    query_tokens = _tokens(query)
    half_life = max(0.5, float(half_life_days))
    results = []

    for convo in conn.execute("SELECT * FROM conversations WHERE archived=0").fetchall():
        relevance, message_id = best.get(convo["id"], (0.0, None))
        if query_tokens:
            meta = _tokens((convo["title"] or "") + " " + (convo["summary"] or ""))
            if meta:
                overlap = len(query_tokens & meta) / float(len(query_tokens))
                if overlap:
                    relevance = max(relevance, 0.35 + 0.65 * overlap)
        if relevance <= 0:
            continue

        recency = math.exp(-_age_days(convo["updated_at"]) / half_life)
        item = dict(convo)
        item["score"] = round(relevance * (0.6 + 0.4 * recency), 5)
        item["snippet"] = ""
        if message_id:
            row = conn.execute("SELECT * FROM messages WHERE id=?", (message_id,)).fetchone()
            if row is not None:
                item["snippet"] = re.sub(r"\s+", " ", _decode(row)).strip()[:120]
        results.append(item)

    results.sort(key=lambda entry: entry["score"], reverse=True)
    return results[:k]


def get_tags(conn, memory_id):
    return [
        r["name"] for r in conn.execute(
            "SELECT t.name FROM tags t JOIN memory_tags mt ON mt.tag_id=t.id "
            "WHERE mt.memory_id=? ORDER BY t.name",
            (memory_id,),
        )
    ]


def top_tags(conn, limit=20):
    return [
        (r["name"], r["n"]) for r in conn.execute(
            "SELECT t.name, COUNT(*) AS n FROM tags t JOIN memory_tags mt ON mt.tag_id=t.id "
            "JOIN memories m ON m.id=mt.memory_id WHERE m.superseded_by IS NULL "
            "GROUP BY t.name ORDER BY n DESC, t.name LIMIT ?",
            (limit,),
        )
    ]


def _row_to_memory(conn, row, score=None):
    item = {
        "id": row["id"],
        "kind": row["kind"],
        "content": row["content"],
        "importance": row["importance"],
        "created_at": row["created_at"],
        "access_count": row["access_count"],
        "source": row["source"],
        "conversation_id": row["conversation_id"],
        "tags": get_tags(conn, row["id"]),
    }
    if score is not None:
        item["score"] = round(score, 5)
    return item


def search(conn, query, k=DEFAULT_TOP_K, min_importance=0.0,
           half_life_days=DEFAULT_HALF_LIFE_DAYS, tags=None, touch=True):
    """Rank live memories by relevance x importance x recency x familiarity.

    Falls back to importance/recency ordering when the query yields no usable
    search tokens (e.g. "hi" or pure punctuation).
    """
    k = max(1, int(k))
    pool = max(40, k * 8)
    match = fts_query(query)
    tag_filter = [t for t in (_norm_tag(x) for x in (tags or [])) if t]

    if match:
        sql = (
            "SELECT m.*, bm25(memories_fts) AS bm "
            "FROM memories_fts JOIN memories m ON m.id = memories_fts.rowid "
            "WHERE memories_fts MATCH ? AND m.superseded_by IS NULL AND m.importance >= ? "
            "ORDER BY bm LIMIT ?"
        )
        try:
            rows = conn.execute(sql, (match, min_importance, pool)).fetchall()
        except sqlite3.OperationalError:
            rows = []
    else:
        rows = []

    if not rows:
        rows = conn.execute(
            "SELECT *, 0.0 AS bm FROM memories WHERE superseded_by IS NULL AND importance >= ? "
            "ORDER BY importance DESC, created_at DESC LIMIT ?",
            (min_importance, pool),
        ).fetchall()

    half_life = max(0.5, float(half_life_days))
    scored = []
    for row in rows:
        # bm25() is negative, more negative = better; map into (0, 1).
        relevance = max(0.0, -float(row["bm"]))
        rel = relevance / (1.0 + relevance) if relevance else 0.15
        weight = 0.5 + 0.5 * _clamp(row["importance"])
        recency = math.exp(-_age_days(row["created_at"]) / half_life)
        familiarity = 1.0 + 0.1 * math.log1p(max(0, row["access_count"]))
        scored.append((rel * weight * recency * familiarity, row))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    results = []
    for score, row in scored:
        item = _row_to_memory(conn, row, score)
        if tag_filter and not set(tag_filter) & set(item["tags"]):
            continue
        results.append(item)
        if len(results) >= k:
            break

    if touch and results:
        now = _now()
        conn.executemany(
            "UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? WHERE id = ?",
            [(now, item["id"]) for item in results],
        )
        conn.commit()
    return results


def format_for_prompt(memories):
    """Render retrieved rows for injection into a system message or tool reply."""
    if not memories:
        return ""
    lines = []
    for item in memories:
        tags = (" [%s]" % ", ".join(item["tags"])) if item["tags"] else ""
        when = (item["created_at"] or "")[:10]
        lines.append(
            "- (%s, importance %.2f, %s)%s %s"
            % (item["kind"], item["importance"], when, tags, item["content"])
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# maintenance
# ---------------------------------------------------------------------------
def compact(conn, similarity=0.85, min_tokens=4):
    """Collapse duplicate/near-duplicate memories onto the newest survivor.

    Exact normalized matches always merge. Near matches need a Jaccard token
    overlap of `similarity` across at least `min_tokens` tokens, so unrelated
    rows sharing a couple of common words are left alone.
    """
    rows = conn.execute(
        "SELECT id, content, importance FROM memories WHERE superseded_by IS NULL ORDER BY id DESC"
    ).fetchall()

    survivors = []          # (id, normalized, tokenset)
    merged = 0
    by_exact = {}

    for row in rows:
        norm = _normalize_text(row["content"])
        if not norm:
            continue
        toks = _tokens(row["content"])

        winner = by_exact.get(norm)
        if winner is None:
            # Short memories only merge on an exact normalized match -- a
            # Jaccard overlap over 3 tokens is noise, not a duplicate.
            if len(toks) >= min_tokens:
                for sid, _, stoks in survivors:
                    if len(stoks) < min_tokens:
                        continue
                    overlap = len(toks & stoks) / float(len(toks | stoks))
                    if overlap >= similarity:
                        winner = sid
                        break

        if winner is not None:
            conn.execute("UPDATE memories SET superseded_by=? WHERE id=?", (winner, row["id"]))
            conn.execute("DELETE FROM memories_fts WHERE rowid=?", (row["id"],))
            # The survivor inherits the stronger importance score.
            conn.execute(
                "UPDATE memories SET importance = MAX(importance, ?) WHERE id = ?",
                (row["importance"], winner),
            )
            merged += 1
        else:
            by_exact[norm] = row["id"]
            survivors.append((row["id"], norm, toks))

    conn.execute(
        "DELETE FROM tags WHERE id NOT IN (SELECT tag_id FROM memory_tags)"
    )
    conn.commit()
    conn.execute("VACUUM")
    return {"merged": merged, "remaining": len(survivors)}


def stats(conn):
    def scalar(sql, args=()):
        row = conn.execute(sql, args).fetchone()
        return row[0] if row else 0

    return {
        "conversations": scalar("SELECT COUNT(*) FROM conversations"),
        "messages": scalar("SELECT COUNT(*) FROM messages"),
        "compressed_messages": scalar("SELECT COUNT(*) FROM messages WHERE compressed=1"),
        "memories": scalar("SELECT COUNT(*) FROM memories WHERE superseded_by IS NULL"),
        "superseded": scalar("SELECT COUNT(*) FROM memories WHERE superseded_by IS NOT NULL"),
        "tags": scalar("SELECT COUNT(*) FROM tags"),
        "avg_importance": round(
            scalar("SELECT COALESCE(AVG(importance),0) FROM memories WHERE superseded_by IS NULL"), 3
        ),
        "top_tags": top_tags(conn, 10),
    }
