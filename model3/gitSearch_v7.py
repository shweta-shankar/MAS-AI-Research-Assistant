#!/usr/bin/python
"""
gitSearch_v7.py — digiBONE Knowledge Base with Reviewer-Verifier System

Builds on v6 (chunking + Sentence Transformer RAG + Ollama LLM) and adds:
  - SQLite-backed verified Q&A cache with semantic matching
  - Confidence-based routing (cache hit / hybrid / full RAG)
  - Auto-categorisation of new questions
  - Naive user feedback logged to feedback.jsonl
  - Reviewer CLI for approving/editing answers and exporting FAQ.md

Usage:
    python gitSearch_v7.py --mode user                   # interactive user mode
    python gitSearch_v7.py --mode user "What is MAD?"    # single question
    python gitSearch_v7.py --mode reviewer               # reviewer console
"""

import re
import sys
import json
import argparse
import requests
import numpy as np
from datetime import datetime
from pathlib import Path

import db_manager as db

# ── CONFIG ────────────────────────────────────────────────────────────────────
GITINGEST_PATH = "/home/shwetashankar/Shweta/Projects/Sem6_Project/gitSearch/model1/goellab-digibone-8a5edab282632443.txt"
OLLAMA_MODEL   = "qwen2.5:7b"
OLLAMA_BASE_URL = "http://localhost:11434"

FEEDBACK_LOG   = Path(__file__).parent / "feedback.jsonl"

# Confidence routing thresholds
HIGH_THRESHOLD = 0.85   # cache hit — skip LLM entirely
MID_THRESHOLD  = 0.70   # hybrid  — verified answer used as context hint

# Categories available for auto-classification
CATEGORIES = [
    "Methodology",
    "Results",
    "Dataset",
    "Preprocessing",
    "Model Architecture",
    "General",
    "Out of Scope",
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LOAD & CHUNK KNOWLEDGE BASE  (identical to v6)
# ─────────────────────────────────────────────────────────────────────────────

with open(GITINGEST_PATH, "r", encoding="utf-8") as f:
    raw = f.read()

def split_into_files(raw):
    files = {}
    sections = re.split(r"={10,}\n", raw)
    i = 0
    while i < len(sections):
        section = sections[i].strip()
        if section.startswith("FILE:"):
            filename = section.replace("FILE:", "").strip()
            if i + 1 < len(sections):
                files[filename] = sections[i + 1]
                i += 2
                continue
        i += 1
    return files

files = split_into_files(raw)


def chunk_by_paragraph(filename, content):
    chunks = []
    paragraphs = re.split(r"\n\s*\n", content)
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i].strip()
        if not para:
            i += 1
            continue
        if len(para) < 50 and i + 1 < len(paragraphs):
            next_para = paragraphs[i + 1].strip()
            if next_para:
                para = para + "\n\n" + next_para
                i += 2
                chunks.append({"file": filename, "chunk_id": i, "text": para, "type": "paragraph"})
                continue
        chunks.append({"file": filename, "chunk_id": i, "text": para, "type": "paragraph"})
        i += 1
    return chunks


def chunk_by_function(filename, content):
    chunks = []
    lines = content.splitlines()
    current_chunk_lines = []
    current_start = 0
    for i, line in enumerate(lines):
        if (line.startswith("def ") or line.startswith("class ")) and current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines).strip()
            if len(chunk_text) > 30:
                chunks.append({"file": filename, "chunk_id": current_start, "text": chunk_text, "type": "function"})
            current_chunk_lines = [line]
            current_start = i
        else:
            current_chunk_lines.append(line)
    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines).strip()
        if len(chunk_text) > 30:
            chunks.append({"file": filename, "chunk_id": current_start, "text": chunk_text, "type": "function"})
    return chunks


def chunk_file(filename, content):
    if filename.endswith(".md"):
        return chunk_by_paragraph(filename, content)
    elif filename.endswith(".py"):
        return chunk_by_function(filename, content)
    else:
        return [{"file": filename, "chunk_id": 0, "text": content.strip(), "type": "other"}]


all_chunks = []
for filename, content in files.items():
    all_chunks += chunk_file(filename, content)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — SENTENCE TRANSFORMER  (identical to v6)
# ─────────────────────────────────────────────────────────────────────────────

from sentence_transformers import SentenceTransformer

print("Loading sentence transformer model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding knowledge base chunks...")
chunk_texts = [c["text"] for c in all_chunks]
chunk_embeddings = embedder.encode(chunk_texts, show_progress_bar=True)
print(f"Encoded {len(chunk_embeddings)} chunks ({chunk_embeddings[0].shape[0]} dims each)\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SEARCH & LLM  (identical to v6)
# ─────────────────────────────────────────────────────────────────────────────

def search(query: str, top_k: int = 8) -> list[dict]:
    query_embedding = embedder.encode([query])[0]
    sims = []
    for i, ce in enumerate(chunk_embeddings):
        score = np.dot(query_embedding, ce) / (np.linalg.norm(query_embedding) * np.linalg.norm(ce))
        sims.append((float(score), i))
    sims.sort(reverse=True)
    return [
        {
            "file": all_chunks[idx]["file"],
            "chunk_id": all_chunks[idx]["chunk_id"],
            "text": all_chunks[idx]["text"],
            "score": round(score, 3),
            "type": all_chunks[idx]["type"],
        }
        for score, idx in sims[:top_k]
    ]


def call_llm(prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=300,
    )
    return response.json()["response"]


def ask_llm(question: str, search_results: list[dict]) -> str:
    context = ""
    for r in search_results:
        context += f"\nSource [{r['file']}:{r['chunk_id']}]\n{r['text']}\n---"

    prompt = f"""Answer the question using only the sources below.
Cite every fact like [filename:chunk_id].
If the answer is not in the sources, say "Not found in knowledge base."

Sources:
{context}

Question: {question}
Answer:"""
    return call_llm(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CATEGORISATION
# ─────────────────────────────────────────────────────────────────────────────

def categorise_question(question: str) -> str:
    """
    Ask the LLM to assign one of the predefined categories to the question.
    Falls back to 'General' if the LLM returns something unexpected.
    """
    cats_str = " | ".join(CATEGORIES)
    prompt = f"""You are a classifier. Assign exactly ONE category to the question below.

Available categories: {cats_str}

Rules:
- Reply with ONLY the category name, nothing else.
- If the question is completely unrelated to bone age assessment or the knowledge base, reply: Out of Scope

Question: {question}
Category:"""
    try:
        raw_cat = call_llm(prompt).strip().strip(".")
        # find the best matching category (case-insensitive)
        for cat in CATEGORIES:
            if cat.lower() in raw_cat.lower():
                return cat
    except Exception:
        pass
    return "General"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — CORE QUERY PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_query(question: str, verbose: bool = True) -> dict:
    """
    Full query pipeline with confidence-based routing.

    Returns a dict with:
        answer      : str   — the final answer shown to the user
        source      : str   — 'cache' | 'hybrid' | 'rag'
        score       : float — best semantic match score against verified cache
        pending_id  : int | None — DB id if question was logged for review
        rag_sources : list[str]  — citation strings (empty for cache hits)
    """
    import time
    t0 = time.time()

    # Encode the incoming question
    query_vec = embedder.encode([question])[0]

    # ── Step 1: Check verified cache ──────────────────────────────────────────
    tier, match_score, matched_row = db.find_semantic_match(
        query_vec, HIGH_THRESHOLD, MID_THRESHOLD
    )

    if verbose:
        print(f"\n[Cache] Best match score: {match_score:.3f} → tier: {tier}")

    pending_id = None
    rag_sources = []

    # ── Tier: HIGH — return verified answer instantly ─────────────────────────
    if tier == "high":
        answer = matched_row["answer"]
        source = "cache"
        if verbose:
            _print_separator()
            print(f"[CACHE HIT] Verified Answer (cached -- {match_score:.2f} match)")
            if matched_row.get("category"):
                print(f"   Category: {matched_row['category']}")
            print(f"\n{answer}")
            _print_separator()
            print(f"Time: {time.time()-t0:.1f}s")

    # ── Tier: MID — run RAG and prepend verified answer as context hint ────────
    elif tier == "mid":
        rag_results = _run_rag(question, verbose)
        if rag_results is None:                       # out of scope
            return {"answer": "Out of scope.", "source": "rag",
                    "score": match_score, "pending_id": None, "rag_sources": []}

        # inject the verified answer as an extra high-quality source
        hint = (
            f"\n[VERIFIED ANSWER — similar question '{matched_row['question']}']\n"
            f"{matched_row['answer']}\n---"
        )
        # build context with hint first
        context = hint
        for r in rag_results:
            context += f"\nSource [{r['file']}:{r['chunk_id']}]\n{r['text']}\n---"
            rag_sources.append(f"[{r['file']}:{r['chunk_id']}]")

        prompt = f"""Answer the question using only the sources below.
Cite every fact like [filename:chunk_id].
If the answer is not in the sources, say "Not found in knowledge base."

Sources:
{context}

Question: {question}
Answer:"""
        answer = call_llm(prompt)
        source = "hybrid"

        # log for review (reviewer can see these hybrid answers too)
        category = categorise_question(question)
        pending_id = db.add_pending(question, answer, category, rag_sources)

        if verbose:
            _print_separator()
            print(f"[HYBRID] Answer (similar verified answer used as context hint -- {match_score:.2f} match)")
            print(f"\n{answer}")
            _print_separator()
            print(f"Time: {time.time()-t0:.1f}s")

    # ── Tier: LOW — full RAG, log for review ──────────────────────────────────
    else:
        rag_results = _run_rag(question, verbose)
        if rag_results is None:
            return {"answer": "Out of scope.", "source": "rag",
                    "score": match_score, "pending_id": None, "rag_sources": []}

        answer = ask_llm(question, rag_results)
        source = "rag"
        rag_sources = [f"[{r['file']}:{r['chunk_id']}]" for r in rag_results]

        category = categorise_question(question)
        pending_id = db.add_pending(question, answer, category, rag_sources)

        if verbose:
            _print_separator()
            print(f"[RAG] Answer (logged for review -- id: {pending_id}, category: {category})")
            print(f"\n{answer}")
            _print_separator()
            print(f"Time: {time.time()-t0:.1f}s")

    return {
        "answer": answer,
        "source": source,
        "score": match_score,
        "pending_id": pending_id,
        "rag_sources": rag_sources,
    }


def _run_rag(question: str, verbose: bool) -> list[dict] | None:
    """Run the RAG retrieval step. Returns None if out of scope."""
    results = search(question)
    if not results or results[0]["score"] < 0.2:
        if verbose:
            print("[WARNING] Out of scope -- best RAG score too low to answer reliably.")
        return None

    top_score = results[0]["score"]
    filtered = [r for r in results if r["score"] >= top_score * 0.6]
    return filtered


def _print_separator():
    print("\n" + "─" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — FEEDBACK LOGGER
# ─────────────────────────────────────────────────────────────────────────────

def collect_feedback(question: str, answer: str, pending_id: int | None):
    """
    Prompt the naive user for optional feedback.
    Result is appended to feedback.jsonl.
    """
    print("\n─── Feedback (optional) ─────────────────────────────────────")
    choice = input("Was this answer helpful? [y/n/skip]: ").strip().lower()
    if choice == "skip" or choice == "":
        return

    rating = "good" if choice == "y" else "bad"
    comment = ""
    if rating == "bad":
        comment = input("What was wrong or missing? (press Enter to skip): ").strip()

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "answer_shown": answer,
        "pending_id": pending_id,
        "feedback": rating,
        "comment": comment,
    }

    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    print("Feedback saved. Thank you!")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — REVIEWER CLI
# ─────────────────────────────────────────────────────────────────────────────

def reviewer_console():
    """Interactive reviewer console."""
    print("\n" + "═" * 70)
    print("  digiBONE Reviewer Console")
    print("═" * 70)

    while True:
        print("\n[1] List pending questions")
        print("[2] Review a question")
        print("[3] Export FAQ.md")
        print("[4] Ask a question (as reviewer)")
        print("[5] View feedback log")
        print("[6] Exit")
        choice = input("\n> ").strip()

        if choice == "1":
            _reviewer_list_pending()
        elif choice == "2":
            _reviewer_review_question()
        elif choice == "3":
            _reviewer_export_faq()
        elif choice == "4":
            _reviewer_ask_question()
        elif choice == "5":
            _reviewer_view_feedback()
        elif choice == "6":
            print("Exiting reviewer console.")
            break
        else:
            print("Invalid option.")


def _reviewer_list_pending():
    pending = db.get_pending_questions(status="pending")
    if not pending:
        print("\nNo pending questions.")
        return
    print(f"\n{'─'*70}")
    print(f"  {len(pending)} pending question(s):")
    print(f"{'─'*70}")
    for p in pending:
        print(f"\n  ID {p['id']} | {p['asked_at'][:16]} | [{p['category']}]")
        print(f"  Q: {p['question']}")
        preview = (p['rag_answer'] or "")[:120]
        print(f"  RAG: {preview}{'…' if len(p['rag_answer'] or '') > 120 else ''}")
    print()


def _reviewer_review_question():
    pending = db.get_pending_questions(status="pending")
    if not pending:
        print("\nNo pending questions to review.")
        return

    try:
        pid = int(input("Enter question ID to review (or 0 to cancel): ").strip())
    except ValueError:
        print("Invalid ID.")
        return
    if pid == 0:
        return

    # find the selected pending question
    target = next((p for p in pending if p["id"] == pid), None)
    if not target:
        print(f"No pending question with ID {pid}.")
        return

    print(f"\n{'═'*70}")
    print(f"Question:  {target['question']}")
    print(f"Category:  {target['category']}")
    print(f"Sources:   {', '.join(target['sources']) if target['sources'] else 'none'}")
    print(f"\nRAG Answer:\n{target['rag_answer']}")
    print(f"{'═'*70}")
    print("\n[a] Approve RAG answer as-is")
    print("[e] Edit/replace the answer")
    print("[r] Reject (won't be cached)")
    print("[s] Skip")
    action = input("> ").strip().lower()

    if action == "a":
        final_answer = target["rag_answer"]
    elif action == "e":
        print("Enter your revised answer (type END on a new line when done):")
        lines = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        final_answer = "\n".join(lines).strip()
        if not final_answer:
            print("Empty answer — skipping.")
            return
    elif action == "r":
        db.reject_pending(pid)
        print(f"Question {pid} rejected.")
        return
    else:
        print("Skipped.")
        return

    # re-encode the original question for the verified_answers embedding
    q_vec = embedder.encode([target["question"]])[0]
    new_id = db.approve_pending(pid, final_answer, q_vec)
    print(f"Answer approved and cached (verified_answers id={new_id}).")


def _reviewer_export_faq():
    path = db.export_faq()
    if path:
        print(f"FAQ exported -> {path}")
    else:
        print("No verified answers yet — FAQ is empty.")


def _reviewer_ask_question():
    question = input("\nQuestion: ").strip()
    if not question:
        return
    result = run_query(question, verbose=True)
    # reviewer can immediately approve the answer they just got
    if result["pending_id"] is not None:
        approve = input("\nApprove this answer to cache it now? [y/n]: ").strip().lower()
        if approve == "y":
            q_vec = embedder.encode([question])[0]
            new_id = db.approve_pending(result["pending_id"], result["answer"], q_vec)
            print(f"Answer approved and cached (verified_answers id={new_id}).")


def _reviewer_view_feedback():
    if not FEEDBACK_LOG.exists():
        print("\nNo feedback logged yet.")
        return
    with open(FEEDBACK_LOG, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        print("\nNo feedback logged yet.")
        return
    print(f"\n{'─'*70}")
    print(f"  {len(lines)} feedback entries (most recent 10):")
    print(f"{'─'*70}")
    for line in lines[-10:]:
        entry = json.loads(line)
        icon = "[+]" if entry["feedback"] == "good" else "[-]"
        print(f"\n  {icon}  {entry['timestamp'][:16]}")
        print(f"  Q: {entry['question']}")
        if entry.get("comment"):
            print(f"  Comment: {entry['comment']}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — USER CLI
# ─────────────────────────────────────────────────────────────────────────────

def user_session(single_question: str = None):
    """Interactive session for naive users."""
    print("\n" + "═" * 70)
    print("  digiBONE Knowledge Base  (type 'exit' to quit)")
    print("═" * 70 + "\n")

    def ask_one(question: str):
        result = run_query(question, verbose=True)
        # user feedback — only prompt for RAG / hybrid answers (cache hits are self-evident)
        if result["source"] in ("rag", "hybrid"):
            collect_feedback(question, result["answer"], result["pending_id"])

    if single_question:
        ask_one(single_question)
        return

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break
        if not question:
            continue
        ask_one(question)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Initialise the SQLite database (creates tables if they don't exist)
    db.init_db()

    parser = argparse.ArgumentParser(
        description="digiBONE Knowledge Base — v7 with Reviewer-Verifier System"
    )
    parser.add_argument(
        "--mode",
        choices=["user", "reviewer"],
        default="user",
        help="Run as 'user' (ask questions) or 'reviewer' (manage Q&A cache)",
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Optional: pass a question directly (user mode only)",
    )
    args = parser.parse_args()

    if args.mode == "reviewer":
        reviewer_console()
    else:
        single_q = " ".join(args.question).strip() if args.question else None
        user_session(single_q)
