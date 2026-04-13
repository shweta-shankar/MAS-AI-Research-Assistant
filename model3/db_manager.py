"""
db_manager.py — SQLite layer for gitSearch_v7

Handles all database operations:
  - Verified answers (the semantic cache)
  - Pending questions (the review queue)
  - FAQ export

All vector operations (serialisation of numpy arrays) are handled here.
"""

import sqlite3
import json
import numpy as np
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "gitSearch_v7.db"


# ── INIT ──────────────────────────────────────────────────────────────────────

def init_db():
    """Create tables if they don't already exist."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS verified_answers (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            question         TEXT    NOT NULL,
            answer           TEXT    NOT NULL,
            category         TEXT    DEFAULT 'General',
            question_embedding BLOB  NOT NULL,
            approved_by      TEXT    DEFAULT 'reviewer',
            created_at       TEXT    DEFAULT (datetime('now')),
            updated_at       TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS pending_questions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            question    TEXT NOT NULL,
            rag_answer  TEXT,
            category    TEXT DEFAULT 'General',
            sources     TEXT,           -- JSON-serialised list of citation strings
            status      TEXT DEFAULT 'pending',   -- pending | approved | rejected
            asked_at    TEXT DEFAULT (datetime('now'))
        );
    """)

    con.commit()
    con.close()


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _encode_embedding(vec: np.ndarray) -> bytes:
    """Serialise a numpy float32 array to bytes for BLOB storage."""
    return vec.astype(np.float32).tobytes()


def _decode_embedding(blob: bytes) -> np.ndarray:
    """Deserialise bytes from BLOB storage back to a numpy float32 array."""
    return np.frombuffer(blob, dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ── VERIFIED ANSWERS (cache) ───────────────────────────────────────────────────

def add_verified_answer(question: str, answer: str, category: str,
                        question_embedding: np.ndarray,
                        approved_by: str = "reviewer") -> int:
    """
    Insert a new verified answer into the cache.
    Returns the new row id.
    """
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """INSERT INTO verified_answers
               (question, answer, category, question_embedding, approved_by)
           VALUES (?, ?, ?, ?, ?)""",
        (question, answer, category,
         _encode_embedding(question_embedding), approved_by)
    )
    row_id = cur.lastrowid
    con.commit()
    con.close()
    return row_id


def update_verified_answer(verified_id: int, answer: str):
    """Update the answer text of an existing verified entry."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "UPDATE verified_answers SET answer=?, updated_at=? WHERE id=?",
        (answer, datetime.utcnow().isoformat(), verified_id)
    )
    con.commit()
    con.close()


def get_all_verified():
    """
    Return all verified answers as a list of dicts.
    Each dict has: id, question, answer, category, question_embedding (np.ndarray).
    """
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "SELECT id, question, answer, category, question_embedding FROM verified_answers"
    )
    rows = cur.fetchall()
    con.close()
    return [
        {
            "id": r[0],
            "question": r[1],
            "answer": r[2],
            "category": r[3],
            "question_embedding": _decode_embedding(r[4]),
        }
        for r in rows
    ]


def find_semantic_match(query_embedding: np.ndarray,
                        high_threshold: float = 0.85,
                        mid_threshold: float = 0.70):
    """
    Compare query_embedding against all verified questions.

    Returns a tuple (tier, score, row) where tier is:
      'high'  — score >= high_threshold  (cache hit, skip LLM)
      'mid'   — score >= mid_threshold   (use as extra context hint)
      'low'   — no strong match found

    row is the matching verified_answers dict (or None for 'low').
    """
    verified = get_all_verified()
    if not verified:
        return ("low", 0.0, None)

    best_score = -1.0
    best_row = None

    for row in verified:
        score = _cosine(query_embedding, row["question_embedding"])
        if score > best_score:
            best_score = score
            best_row = row

    if best_score >= high_threshold:
        return ("high", best_score, best_row)
    elif best_score >= mid_threshold:
        return ("mid", best_score, best_row)
    else:
        return ("low", best_score, None)


# ── PENDING QUESTIONS (review queue) ─────────────────────────────────────────

def add_pending(question: str, rag_answer: str, category: str,
                sources: list[str]) -> int:
    """Log a new question to the review queue. Returns new row id."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """INSERT INTO pending_questions (question, rag_answer, category, sources)
           VALUES (?, ?, ?, ?)""",
        (question, rag_answer, category, json.dumps(sources))
    )
    row_id = cur.lastrowid
    con.commit()
    con.close()
    return row_id


def get_pending_questions(status: str = "pending") -> list[dict]:
    """Return all pending questions with the given status."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """SELECT id, question, rag_answer, category, sources, asked_at
           FROM pending_questions WHERE status=? ORDER BY asked_at ASC""",
        (status,)
    )
    rows = cur.fetchall()
    con.close()
    return [
        {
            "id": r[0],
            "question": r[1],
            "rag_answer": r[2],
            "category": r[3],
            "sources": json.loads(r[4]) if r[4] else [],
            "asked_at": r[5],
        }
        for r in rows
    ]


def approve_pending(pending_id: int, final_answer: str,
                    question_embedding: np.ndarray,
                    approved_by: str = "reviewer") -> int:
    """
    Move a pending question to verified_answers.
    Marks the pending entry as 'approved'.
    Returns the new verified_answer row id.
    """
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # fetch the pending row
    cur.execute(
        "SELECT question, category FROM pending_questions WHERE id=?",
        (pending_id,)
    )
    row = cur.fetchone()
    if not row:
        con.close()
        raise ValueError(f"No pending question with id={pending_id}")

    question, category = row

    # mark pending as approved
    cur.execute(
        "UPDATE pending_questions SET status='approved' WHERE id=?",
        (pending_id,)
    )
    con.commit()
    con.close()

    # insert into verified
    return add_verified_answer(question, final_answer, category,
                               question_embedding, approved_by)


def reject_pending(pending_id: int):
    """Mark a pending question as rejected (won't go into cache)."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "UPDATE pending_questions SET status='rejected' WHERE id=?",
        (pending_id,)
    )
    con.commit()
    con.close()


# ── FAQ EXPORT ────────────────────────────────────────────────────────────────

def export_faq(faq_path: str = None) -> str:
    """
    Generate a Markdown FAQ from all verified answers, grouped by category.
    Writes to faq_path (defaults to FAQ.md beside the DB).
    Returns the path written.
    """
    if faq_path is None:
        faq_path = str(DB_PATH.parent / "FAQ.md")

    verified = get_all_verified()
    if not verified:
        return None

    # group by category
    by_category: dict[str, list] = {}
    for row in verified:
        cat = row["category"] or "General"
        by_category.setdefault(cat, []).append(row)

    lines = [
        "# digiBONE Knowledge Base — FAQ",
        f"\n_Last generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_",
        "\n---\n",
    ]

    for category, entries in sorted(by_category.items()):
        lines.append(f"## {category}\n")
        for entry in entries:
            lines.append(f"**Q: {entry['question']}**\n")
            lines.append(f"{entry['answer']}\n")

    content = "\n".join(lines)
    with open(faq_path, "w", encoding="utf-8") as f:
        f.write(content)

    return faq_path
