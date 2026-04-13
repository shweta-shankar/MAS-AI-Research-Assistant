#!/usr/bin/python
"""
gitSearch_v8.py — digiBONE Knowledge Base with Reviewer-Verifier System (v8)

New features:
  - Groq API (`llama-3.3-70b-versatile`)
  - AST Syntax-Aware Chunking strictly for Python via LangChain
  - Hybrid Search: SentenceTransformers + BM25 okapi keyword search
  - Cross-Encoder Re-Ranking
  - Advanced Reviewer Options (Suggest vs Overwrite)

Usage:
    python gitSearch_v8.py --mode user                   # interactive user mode
    python gitSearch_v8.py --mode user "What is MAD?"    # single question
    python gitSearch_v8.py --mode reviewer               # reviewer console
"""

import os
import re
import sys
import json
import argparse
import numpy as np
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

import db_manager_v8 as db

# ── CONFIG ────────────────────────────────────────────────────────────────────
load_dotenv()
load_dotenv("groq_api_key.env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GITINGEST_PATH = "/home/shwetashankar/Shweta/Projects/Sem6_Project/gitSearch/model1/goellab-digibone-8a5edab282632443.txt"
GROQ_MODEL   = "llama-3.3-70b-versatile"

FEEDBACK_LOG   = Path(__file__).parent / "feedback.jsonl"

# Confidence routing thresholds
HIGH_THRESHOLD = 0.85   # cache hit — skip full RAG
MID_THRESHOLD  = 0.70   # hybrid  — verified answer used as context hint

CATEGORIES = [
    "Methodology",
    "Results",
    "Dataset",
    "Preprocessing",
    "Model Architecture",
    "General",
    "Out of Scope",
]

# Install client early to catch missing key
if not GROQ_API_KEY:
    print("[ERROR] GROQ_API_KEY not found in .env file. Please add it.")
    sys.exit(1)
groq_client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LOAD & CHUNK AST KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────

try:
    with open(GITINGEST_PATH, "r", encoding="utf-8") as f:
        raw = f.read()
except FileNotFoundError:
    print(f"[ERROR] Could not find ingest file at {GITINGEST_PATH}")
    sys.exit(1)

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
print(f"[DEBUG] Parsed {len(files)} files from ingest. Keys: {list(files.keys())[:5]}")

# LangChain Context Aware Splitters
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=800, chunk_overlap=150
)
md_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN, chunk_size=800, chunk_overlap=150
)
generic_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=150
)

def chunk_file(filename, content):
    chunks = []
    if filename.endswith(".py"):
        split_texts = python_splitter.split_text(content)
    elif filename.endswith(".md"):
        split_texts = md_splitter.split_text(content)
    else:
        split_texts = generic_splitter.split_text(content)
        
    for i, text in enumerate(split_texts):
        chunks.append({"file": filename, "chunk_id": i, "text": text})
    return chunks

all_chunks = []
for filename, content in files.items():
    all_chunks += chunk_file(filename, content)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — HYBRID INDEXING
# ─────────────────────────────────────────────────────────────────────────────

print("Loading Embedding Models...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Loading Cross-Encoder Reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Encoding knowledge base for Vector Search...")
chunk_texts = [c["text"] for c in all_chunks]
chunk_embeddings = embedder.encode(chunk_texts, show_progress_bar=True)

print("Indexing knowledge base for BM25 Keyword Search...")
tokenized_corpus = [doc.lower().split(" ") for doc in chunk_texts]
bm25 = BM25Okapi(tokenized_corpus)

print(f"Index Ready. {len(all_chunks)} chunks processed.\n")
if len(all_chunks) == 0:
    print("[ERROR] No chunks were created! Check the ingest file separator format.")
    print("[DEBUG] First 300 chars of raw file:")
    print(repr(raw[:300]))
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SEARCH & LLM
# ─────────────────────────────────────────────────────────────────────────────

def expand_query_for_code(query: str) -> str:
    """
    Uses the LLM to translate human questions into codebase vocabulary
    (e.g. variable names, array names, function definitions).
    """
    prompt = f"""You are a search query optimizer for a Python codebase.
The user is asking: "{query}"

Rewrite this query by extracting the core concepts and appending potential Python variable names, array names, or raw syntax terms that might contain the answer in the code.
For example, if they ask for "male Greulich-Pyle mapping classes", you should append terms like: male_classes, m_classes, gp_classes, mapping_array, etc.
Output ONLY the expanded query string, nothing else.

Expanded query:"""
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",  # Use the small model for blazing fast expansion
            temperature=0.2
        )
        expanded = response.choices[0].message.content.strip()
        print(f"\n[DEBUG] Expanded Query: {expanded}")
        return expanded
    except Exception:
        return query


def search(query: str, top_k_vector=20, top_k_bm25=20, top_k_final=6) -> list[dict]:
    # 0. Query Expansion (HyDE) — Bridging vocabulary gap
    expanded_query = expand_query_for_code(query)

    # 1. Vector Search (using expanded query)
    query_embedding = embedder.encode([expanded_query])[0]
    sims = []
    for i, ce in enumerate(chunk_embeddings):
        score = np.dot(query_embedding, ce) / (np.linalg.norm(query_embedding) * np.linalg.norm(ce))
        sims.append((float(score), i))
    sims.sort(reverse=True)
    vector_results = [all_chunks[idx] for _, idx in sims[:top_k_vector]]
    
    # 2. Sparse BM25 Search (using expanded query)
    tokenized_query = expanded_query.lower().split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k_bm25]
    bm25_results = [all_chunks[idx] for idx in bm25_top_indices if bm25_scores[idx] > 0]
    
    # 3. Combine unique candidates
    unique_candidates = { (c["file"], c["chunk_id"]): c for c in vector_results + bm25_results }
    candidates = list(unique_candidates.values())
    if not candidates:
        return []
    
    # 4. Cross-Encoder Re-Ranking
    cross_input = [[query, c["text"]] for c in candidates]
    rerank_scores = reranker.predict(cross_input)
    
    for idx, c in enumerate(candidates):
        c["cross_score"] = float(rerank_scores[idx])
        
    candidates.sort(key=lambda x: x["cross_score"], reverse=True)
    top = candidates[:top_k_final]
    print(f"[DEBUG] Top {len(top)} reranked chunks:")
    for c in top:
        print(f"  cross_score={c['cross_score']:.2f} | {c['file']} | preview: {c['text'][:80].strip()!r}")
    return top


def call_llm(prompt: str) -> str:
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\n[ERROR] Groq API Call failed: {e}")
        return "Failed to generate answer."


def ask_llm(question: str, search_results: list[dict]) -> str:
    context = ""
    print(f"\n[DEBUG] Passing {len(search_results)} chunks to LLM:")
    for r in search_results:
        print(f"  cross_score={r.get('cross_score', 'N/A'):.2f} | {r['file']}:{r['chunk_id']} | {r['text'][:100].strip()!r}")
        context += f"\nSource [{r['file']}:{r['chunk_id']}]\n{r['text']}\n---"

    prompt = f"""You are an expert assistant for the digiBONE project. Answer the question using the sources below.
Cite every fact like [filename:chunk_id].
If relevant data is mentioned in the source code (e.g. array lengths, variable values, constants), use that to infer the answer.
Only say "Not found in knowledge base" if you have genuinely checked all sources and found nothing even indirectly related.

Sources:
{context}

Question: {question}
Answer:"""
    return call_llm(prompt)


def apply_reviewer_suggestions(question: str, original_ai_answer: str, reviewer_suggestions: str) -> str:
    prompt = f"""You are answering a question based on a previously verified AI answer, but you must apply the new improvements suggested by an expert reviewer.
    
Original Question: {question}
Original Answer: {original_ai_answer}

Expert Reviewer Suggestions: {reviewer_suggestions}

Please write a new, improved final answer that fully incorporates the reviewer's suggestions. Ensure it perfectly answers the question.
Answer:"""
    return call_llm(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CATEGORISATION
# ─────────────────────────────────────────────────────────────────────────────

def categorise_question(question: str) -> str:
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
    t0 = time.time()
    query_vec = embedder.encode([question])[0]

    tier, match_score, matched_row = db.find_semantic_match(
        query_vec, HIGH_THRESHOLD, MID_THRESHOLD
    )

    if verbose:
        print(f"\n[Cache] Best match score: {match_score:.3f} → tier: {tier}")

    pending_id = None
    rag_sources = []

    if tier == "high":
        # Check if reviewer overwrote it completely
        if matched_row.get("verified_answer"):
            answer = matched_row["verified_answer"]
            source = "cache_verified_override"
        elif matched_row.get("reviewer_suggestions"):
            # Dynamically regenerate using suggestions
            answer = apply_reviewer_suggestions(question, matched_row["original_ai_answer"], matched_row["reviewer_suggestions"])
            source = "cache_dynamic_suggestions"
        else:
            answer = matched_row["original_ai_answer"]
            source = "cache_original"
            
        if verbose:
            _print_separator()
            print(f"[{source.upper()}] (score -- {match_score:.2f})")
            print(f"\n{answer}")
            _print_separator()
            print(f"Time: {time.time()-t0:.1f}s")

    elif tier == "mid":
        rag_results = search(question)
        if not rag_results:
            return {"answer": "Out of scope.", "source": "rag", "score": match_score, "pending_id": None, "rag_sources": []}

        # Inject verified answer as a hint
        hint_ans = matched_row.get("verified_answer") or matched_row["original_ai_answer"]
        hint = f"\n[VERIFIED ANSWER — similar question '{matched_row['question']}']\n{hint_ans}\n---"
        
        context = hint
        for r in rag_results:
            context += f"\nSource [{r['file']}:{r['chunk_id']}]\n{r['text']}\n---"
            rag_sources.append(f"[{r['file']}:{r['chunk_id']}]")

        prompt = f"""Answer the question using only the sources below. Cite every fact.
Sources:
{context}

Question: {question}
Answer:"""
        answer = call_llm(prompt)
        source = "hybrid"

        category = categorise_question(question)
        pending_id = db.add_pending(question, answer, category, rag_sources)

        if verbose:
            _print_separator()
            print(f"[HYBRID] Answer (similar verified answer used as context hint -- {match_score:.2f} match)")
            print(f"\n{answer}")
            _print_separator()
            print(f"Time: {time.time()-t0:.1f}s")

    else:
        rag_results = search(question)
        if not rag_results:
            if verbose:
                print("[WARNING] Out of scope -- no chunks retrieved.")
            return {"answer": "Out of scope.", "source": "rag", "score": match_score, "pending_id": None, "rag_sources": []}

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


def _print_separator():
    print("\n" + "─" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — REVIEWER CLI
# ─────────────────────────────────────────────────────────────────────────────

def reviewer_console():
    db.init_db()
    print("\n" + "═" * 70)
    print("  digiBONE Reviewer Console (v8 API Mode)")
    print("═" * 70)

    while True:
        print("\n[1] List pending questions")
        print("[2] Review a question")
        print("[3] Export FAQ.md")
        print("[4] Exit")
        choice = input("\n> ").strip()

        if choice == "1":
            _reviewer_list_pending()
        elif choice == "2":
            _reviewer_review_question()
        elif choice == "3":
            db.export_faq()
            print("Exported FAQ.md to disk.")
        elif choice == "4":
            break


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
        print("\nNo pending questions.")
        return

    try:
        pid = int(input("Enter ID (0 cancel): ").strip())
    except:
        return
    if pid == 0: return

    target = next((p for p in pending if p["id"] == pid), None)
    if not target: return

    print(f"\n{'═'*70}")
    print(f"Question:  {target['question']}")
    print(f"\nRAG Answer:\n{target['rag_answer']}")
    print(f"{'═'*70}")
    print("\n[a] Approve original AI answer")
    print("[e] Overwrite answer entirely (provide verified_answer)")
    print("[s] Suggest improvements (dynamically applied via LLM next time)")
    print("[r] Reject")
    print("[q] Skip")
    action = input("> ").strip().lower()

    verified_answer = None
    reviewer_suggestions = None

    if action == "a":
        pass
    elif action == "e":
        print("Enter verified answer (type END on new line):")
        lines = []
        while True:
            line = input()
            if line == "END": break
            lines.append(line)
        verified_answer = "\n".join(lines).strip()
    elif action == "s":
        print("Enter suggestions for the LLM to follow (type END on new line):")
        lines = []
        while True:
            line = input()
            if line == "END": break
            lines.append(line)
        reviewer_suggestions = "\n".join(lines).strip()
    elif action == "r":
        db.reject_pending(pid)
        print("Rejected.")
        return
    else:
        return

    q_vec = embedder.encode([target["question"]])[0]
    new_id = db.approve_pending(pid, verified_answer, reviewer_suggestions, q_vec)
    print(f"Approved (id={new_id}).")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — USER CLI
# ─────────────────────────────────────────────────────────────────────────────

def user_session(single_question: str = None):
    db.init_db()
    print("\n" + "═" * 70)
    print("  digiBONE Knowledge Base (v8 Hybrid LLM) ")
    print("═" * 70 + "\n")

    if single_question:
        run_query(single_question, verbose=True)
        return

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except:
            break
        if question.lower() in ("exit", "quit", "q"):
            break
        if question:
            run_query(question, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["user", "reviewer"], default="user")
    parser.add_argument("question", nargs="*")
    args = parser.parse_args()

    if args.mode == "reviewer":
        reviewer_console()
    else:
        single_q = " ".join(args.question).strip() if args.question else None
        user_session(single_q)
