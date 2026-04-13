import sqlite3
import json
import numpy as np
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "gitSearch_v8.db"

# ── INIT ──────────────────────────────────────────────────────────────────────

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS verified_answers (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            question              TEXT    NOT NULL,
            original_ai_answer    TEXT    NOT NULL,
            verified_answer       TEXT,
            reviewer_suggestions  TEXT,
            category              TEXT    DEFAULT 'General',
            question_embedding    BLOB    NOT NULL,
            approved_by           TEXT    DEFAULT 'reviewer',
            created_at            TEXT    DEFAULT (datetime('now')),
            updated_at            TEXT    DEFAULT (datetime('now'))
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
    return vec.astype(np.float32).tobytes()

def _decode_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ── VERIFIED ANSWERS (cache) ───────────────────────────────────────────────────

def add_verified_answer(question: str, original_ai_answer: str,
                        verified_answer: str, reviewer_suggestions: str,
                        category: str, question_embedding: np.ndarray,
                        approved_by: str = "reviewer") -> int:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """INSERT INTO verified_answers
               (question, original_ai_answer, verified_answer, reviewer_suggestions, category, question_embedding, approved_by)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (question, original_ai_answer, verified_answer, reviewer_suggestions, category,
         _encode_embedding(question_embedding), approved_by)
    )
    row_id = cur.lastrowid
    con.commit()
    con.close()
    return row_id

def get_all_verified():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "SELECT id, question, original_ai_answer, verified_answer, reviewer_suggestions, category, question_embedding FROM verified_answers"
    )
    rows = cur.fetchall()
    con.close()
    return [
        {
            "id": r[0],
            "question": r[1],
            "original_ai_answer": r[2],
            "verified_answer": r[3],
            "reviewer_suggestions": r[4],
            "category": r[5],
            "question_embedding": _decode_embedding(r[6]),
        }
        for r in rows
    ]

def find_semantic_match(query_embedding: np.ndarray,
                        high_threshold: float = 0.85,
                        mid_threshold: float = 0.70):
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

def approve_pending(pending_id: int, verified_answer: str, reviewer_suggestions: str,
                    question_embedding: np.ndarray,
                    approved_by: str = "reviewer") -> int:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute(
        "SELECT question, rag_answer, category FROM pending_questions WHERE id=?",
        (pending_id,)
    )
    row = cur.fetchone()
    if not row:
        con.close()
        raise ValueError(f"No pending question with id={pending_id}")

    question, rag_answer, category = row

    cur.execute(
        "UPDATE pending_questions SET status='approved' WHERE id=?",
        (pending_id,)
    )
    con.commit()
    con.close()

    return add_verified_answer(question, rag_answer, verified_answer, reviewer_suggestions, category,
                               question_embedding, approved_by)

def reject_pending(pending_id: int):
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
    if faq_path is None:
        faq_path = str(DB_PATH.parent / "FAQ.md")

    verified = get_all_verified()
    if not verified:
        return None

    by_category: dict[str, list] = {}
    for row in verified:
        cat = row["category"] or "General"
        by_category.setdefault(cat, []).append(row)

    lines = [
        "# digiBONE Knowledge Base — FAQ (v8)",
        f"\n_Last generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_",
        "\n---\n",
    ]

    for category, entries in sorted(by_category.items()):
        lines.append(f"## {category}\n")
        for entry in entries:
            # Prefer rewritten answer, otherwise original answer
            answer_to_show = entry['verified_answer'] if entry['verified_answer'] else entry['original_ai_answer']
            lines.append(f"**Q: {entry['question']}**\n")
            lines.append(f"{answer_to_show}\n")

    content = "\n".join(lines)
    with open(faq_path, "w", encoding="utf-8") as f:
        f.write(content)

    return faq_path
