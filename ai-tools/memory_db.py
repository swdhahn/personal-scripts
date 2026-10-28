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

SCHEMA_VERSION = 2

KINDS = ("fact", "preference", "project", "event", "summary")
SOURCES = ("chat", "screenshot", "manual")

DEFAULT_COMPRESS_THRESHOLD = 2048
DEFAULT_HALF_LIFE_DAYS = 30.0
DEFAULT_TOP_K = 6
# Fraction of the query's content words a memory must share to be considered
# relevant at all. One word out of three ("write") is noise, not a match.
DEFAULT_MIN_COVERAGE = 0.5
DEFAULT_MIN_SCORE = 0.05

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
    archived      INTEGER NOT NULL DEFAULT 0,
    -- Highest messages.id already folded into `summary`. Turns at or below it
    -- are represented by the summary and are not replayed verbatim, so the
    -- cost of compacting a long thread is paid once instead of every turn.
    summary_through_id INTEGER NOT NULL DEFAULT 0
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
    # The distiller writes every memory in the third person ("The user prefers
    # ..."), so "user" and "assistant" appear in almost all of them and carry
    # no discriminating signal -- they only let unrelated rows clear the
    # coverage gate.
    "user", "assistant", "please", "tell", "say", "write", "give", "make",
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


def _stem(token):
    """Deliberately minimal: fold plural / third-person -s only.

    Enough to make "live"/"lives" and "own"/"owns" compare equal, which is what
    coverage needs, without the false conflations a full Porter stemmer brings
    to a corpus this small. Applied to both sides, so it only has to be
    consistent, not linguistically correct.
    """
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(text):
    return {_stem(t) for t in _normalize_text(text).split()
            if len(t) > 1 and t not in _STOPWORDS}


def fts_query(text, limit=32):
    """Turn free-form text into a safe FTS5 OR-query (operators stripped).

    Terms of four characters or more are emitted as stemmed prefix matches, so
    "live" also finds "lives" and "lived". The FTS index itself has no stemmer,
    and without this a query could miss the one memory that answers it purely
    on a plural. Short tokens stay exact, where a prefix would over-match.
    """
    seen, out = set(), []
    for tok in re.findall(r"[A-Za-z0-9_]+", (text or "").lower()):
        if len(tok) < 2 or tok in _STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        stem = _stem(tok)
        out.append('"%s"*' % stem if len(stem) >= 4 else '"%s"' % stem)
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
    _migrate(conn)
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()
    return conn


def _migrate(conn):
    """Bring an existing database up to SCHEMA_VERSION.

    _SCHEMA only uses CREATE TABLE IF NOT EXISTS, so columns added to a table
    that already exists need an explicit ALTER. Each step is idempotent and
    guarded by the column actually being absent, so running against a fresh
    database (where _SCHEMA already created it) is a no-op.
    """
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(conversations)")}
    if "summary_through_id" not in existing:
        conn.execute("ALTER TABLE conversations "
                     "ADD COLUMN summary_through_id INTEGER NOT NULL DEFAULT 0")
        conn.commit()


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


def set_summary(conn, conversation_id, summary, through_id=None):
    """Store a conversation summary.

    `through_id` records the highest message id the summary covers; pass it
    when the summary was produced by folding specific turns away, so they are
    not replayed alongside it. Omit it to leave the watermark untouched.
    """
    if through_id is not None:
        conn.execute(
            "UPDATE conversations SET summary=?, summary_through_id=?, updated_at=? WHERE id=?",
            (summary, int(through_id), _now(), conversation_id),
        )
        conn.commit()
        return
    return _set_summary_only(conn, conversation_id, summary)


def _set_summary_only(conn, conversation_id, summary):
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


def get_messages(conn, conversation_id, limit=None, after_id=0):
    """Turns of a conversation, oldest first.

    `after_id` skips everything already folded into the summary; `limit` then
    keeps the most recent N of what remains.
    """
    sql = "SELECT * FROM messages WHERE conversation_id=? AND id>? ORDER BY id"
    rows = conn.execute(sql, (conversation_id, int(after_id or 0))).fetchall()
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


def _coverage(query_terms, content):
    """Fraction of the query's distinct content words present in `content`.

    Both sides are stemmed by _tokens(), so "where does the user live" covers
    "The user lives in Orange, CA" fully rather than at one half.
    """
    if not query_terms:
        return 1.0
    body = _tokens(content)
    if not body:
        return 0.0
    return len(set(query_terms) & body) / len(set(query_terms))


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
           half_life_days=DEFAULT_HALF_LIFE_DAYS, tags=None, touch=True,
           min_coverage=DEFAULT_MIN_COVERAGE, min_score=DEFAULT_MIN_SCORE,
           allow_fallback=False):
    """Rank live memories by relevance x coverage x importance x recency.

    The FTS query is an OR of the query's terms, so a single shared common word
    is enough to produce a match: asking "can you write me a dangerous story"
    retrieved a note beginning "Please write a narrative..." purely on "write",
    and the answer was then built around that unrelated note. `min_coverage`
    is the guard -- a candidate must share a real fraction of the query's
    terms, not just one of them -- and `min_score` drops weak matches outright.

    `allow_fallback` re-enables the old behaviour of returning the most
    important recent memories when nothing matches at all. That is useful when
    the caller already knows the user is asking about the past ("what did we
    talk about yesterday"), and actively harmful otherwise, so it is off by
    default rather than on.
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

    if not rows and (allow_fallback or not match):
        # Nothing matched. Either the query had no usable terms at all ("hi"),
        # or the caller explicitly wants the most salient recent memories.
        rows = conn.execute(
            "SELECT *, 0.0 AS bm FROM memories WHERE superseded_by IS NULL AND importance >= ? "
            "ORDER BY importance DESC, created_at DESC LIMIT ?",
            (min_importance, pool),
        ).fetchall()
        # These rows were chosen by salience, not by matching, so a term
        # coverage test on them would reject every one and leave the caller
        # with nothing at all -- which is the opposite of what asking for a
        # fallback meant.
        min_coverage, min_score = 0.0, 0.0

    half_life = max(0.5, float(half_life_days))
    query_terms = _tokens(query)
    scored = []
    for row in rows:
        # bm25() is negative, more negative = better; map into (0, 1).
        relevance = max(0.0, -float(row["bm"]))
        # bm25() collapses to ~0 for a term that appears in nearly every row
        # ("user" here), which drove the score of perfectly good memories to
        # zero. A match is always worth at least the baseline; letting
        # coverage do the discriminating is far more reliable than bm25 on a
        # corpus of a few dozen short rows.
        rel = max(0.15, relevance / (1.0 + relevance))
        weight = 0.5 + 0.5 * _clamp(row["importance"])
        recency = math.exp(-_age_days(row["created_at"]) / half_life)
        familiarity = 1.0 + 0.1 * math.log1p(max(0, row["access_count"]))
        coverage = _coverage(query_terms, row["content"])
        # Coverage multiplies rather than only gating, so a memory that matches
        # every term still outranks one that clips a single word.
        scored.append((rel * weight * recency * familiarity * (0.25 + 0.75 * coverage),
                       coverage, row))

    scored.sort(key=lambda triple: triple[0], reverse=True)

    results = []
    for score, coverage, row in scored:
        if query_terms and coverage < min_coverage:
            continue
        if score < min_score:
            continue
        item = _row_to_memory(conn, row, score)
        item["coverage"] = round(coverage, 3)
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
# A memory is worth keeping only if it is about this person. The distiller
# writes in the third person ("The user ..."), so a reference to the user or to
# something of theirs is the signal; general knowledge has none.
_PERSONAL_RE = re.compile(
    r"\b(user|users|user's|their|theirs|they|them|he|she|his|her|your|yours|you|"
    r"my|mine|i|we|our|system|machine|laptop|desktop|workstation|host|rig|"
    r"repo|repository|setup|config|configuration)\b", re.I)

# Local-setup wording counts as personal even with no pronoun in the sentence:
# "The waybar signal number is configured to 8" is a fact about this machine,
# not about the world.
_SETUP_RE = re.compile(
    r"\b(configured|configure|installed|enabled|disabled|set to|default[s]? to|"
    r"port|path|directory|folder|keybind|shortcut|alias|script|daemon|service|"
    r"server|venv|dotfile[s]?|theme|font|wallpaper|hotkey)\b", re.I)

# Things that are never durable regardless of phrasing: the clock, the calendar
# and anything explicitly framed as current.
_EPHEMERAL_RE = re.compile(
    r"\b(today is|tomorrow is|yesterday was|current (date|time|year|version|price)|"
    r"it is \d{1,2}:\d{2}|this (morning|afternoon|evening)|right now,|"
    r"as of (today|now)|national .{3,30} day)\b", re.I)


def is_personal(content, kind=None):
    """True when a memory is about the user rather than about the world.

    The distillation prompt asks for this, but a prompt is a request, not a
    guarantee -- the store had filled up with rifle calibres, the mercury
    content of dolphin meat and what day it was. This is the part that holds.
    """
    text = (content or "").strip()
    if not text:
        return False
    if _EPHEMERAL_RE.search(text):
        return False
    return bool(_PERSONAL_RE.search(text) or _SETUP_RE.search(text))


def find_impersonal(conn):
    """Stored memories that are general knowledge rather than about the user."""
    rows = conn.execute(
        "SELECT * FROM memories WHERE superseded_by IS NULL ORDER BY id").fetchall()
    return [_row_to_memory(conn, r) for r in rows
            if not is_personal(r["content"], r["kind"])]


def purge_impersonal(conn, dry_run=True, keep=()):
    """`keep` spares specific ids.

    The rule cannot tell "The coolant tank on the Boxster has been replaced"
    (the user's car, no pronoun in the sentence) from a specification anyone
    could look up, so the decision on the edge cases stays with the user.
    """
    keep = {int(k) for k in (keep or ())}
    found = [i for i in find_impersonal(conn) if i["id"] not in keep]
    if not dry_run:
        for item in found:
            delete_memory(conn, item["id"])
    return {
        "found": len(found), "deleted": 0 if dry_run else len(found),
        "dry_run": bool(dry_run),
        "items": [{"id": i["id"], "kind": i["kind"], "preview": i["content"][:100]}
                  for i in found],
    }


def find_raw_dumps(conn, min_chars=400):
    """Undistilled transcript rows written by the old failed-distillation path.

    These are verbatim copies of turns that are still in the messages table, so
    they add no durability -- but they are 20x the size of a real memory and
    match on any common word, which is how an unrelated note ended up steering
    a story prompt. Identified by the tag the fallback stamped on them.
    """
    rows = conn.execute(
        "SELECT m.* FROM memories m "
        "JOIN memory_tags mt ON mt.memory_id = m.id "
        "JOIN tags t ON t.id = mt.tag_id "
        "WHERE t.name = 'unparsed' AND m.superseded_by IS NULL "
        "AND length(m.content) >= ? ORDER BY m.id",
        (int(min_chars),),
    ).fetchall()
    return [_row_to_memory(conn, row) for row in rows]


def purge_raw_dumps(conn, min_chars=400, dry_run=True):
    """Report (and optionally delete) the rows find_raw_dumps() identifies."""
    found = find_raw_dumps(conn, min_chars=min_chars)
    if not dry_run:
        for item in found:
            delete_memory(conn, item["id"])
    return {
        "found": len(found),
        "deleted": 0 if dry_run else len(found),
        "dry_run": bool(dry_run),
        "items": [{"id": i["id"], "chars": len(i["content"]),
                   "preview": i["content"][:90]} for i in found],
    }


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
