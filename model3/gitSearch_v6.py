#!/usr/bin/python

import re
import requests

# ── CONFIG ────────────────────────────────────────────────────────────────────
GITINGEST_PATH = "/home/shwetashankar/Shweta/Projects/Sem6_Project/gitSearch/model1/goellab-digibone-8a5edab282632443.txt"
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_BASE_URL = "http://localhost:11434"

# ── LOAD RAW FILE ─────────────────────────────────────────────────────────────
with open(GITINGEST_PATH, "r", encoding="utf-8") as f:
    raw = f.read()

print(f"Loaded {len(raw)} characters")
print(f"First 200 chars: {raw[:200]}")

# ── SPLIT INTO INDIVIDUAL FILES ───────────────────────────────────────────────
def split_into_files(raw):
    files = {}
    sections = re.split(r"={10,}\n", raw)
    
    i = 0
    while i < len(sections):
        section = sections[i].strip()
        if section.startswith("FILE:"):
            filename = section.replace("FILE:", "").strip()
            if i + 1 < len(sections):
                content = sections[i + 1]
                files[filename] = content
                i += 2
                continue
        i += 1
    
    return files

files = split_into_files(raw)

print(f"Files found: {list(files.keys())}")
for filename, content in files.items():
    print(f"{filename}: {len(content)} characters, {len(content.splitlines())} lines")


# ── CHUNKERS ──────────────────────────────────────────────────────────────────

def chunk_by_paragraph(filename, content):
    chunks = []
    paragraphs = re.split(r"\n\s*\n", files["README.md"])
    
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i].strip()
        
        if not para:  # skip only truly empty
            i += 1
            continue
        
        # if this paragraph is very short (likely a header),
        # merge it with the next paragraph
        if len(para) < 50 and i + 1 < len(paragraphs):
            next_para = paragraphs[i + 1].strip()
            if next_para:
                para = para + "\n\n" + next_para
                i += 2  # skip next since we merged it
                chunks.append({
                    "file": filename,
                    "chunk_id": i,
                    "text": para,
                    "type": "paragraph"
                })
                continue
        
        chunks.append({
            "file": filename,
            "chunk_id": i,
            "text": para,
            "type": "paragraph"
        })
        i += 1
    
    return chunks

def chunk_by_function(filename, content):
    """For .py files — chunk by function/class"""
    chunks = []
    lines = content.splitlines()
    
    current_chunk_lines = []
    current_start = 0
    
    for i, line in enumerate(lines):
        # detect start of new function or class
        if (line.startswith("def ") or line.startswith("class ")) and current_chunk_lines:
            # save previous chunk
            chunk_text = "\n".join(current_chunk_lines).strip()
            if len(chunk_text) > 30:
                chunks.append({
                    "file": filename,
                    "chunk_id": current_start,
                    "text": chunk_text,
                    "type": "function"
                })
            current_chunk_lines = [line]
            current_start = i
        else:
            current_chunk_lines.append(line)
    
    # save last chunk
    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines).strip()
        if len(chunk_text) > 30:
            chunks.append({
                "file": filename,
                "chunk_id": current_start,
                "text": chunk_text,
                "type": "function"
            })
    
    return chunks


def chunk_file(filename, content):
    """Route each file to the right chunker"""
    if filename.endswith(".md"):
        return chunk_by_paragraph(filename, content)
    elif filename.endswith(".py"):
        return chunk_by_function(filename, content)
    else:
        # for LICENSE, requirements.txt etc — treat as one chunk
        return [{
            "file": filename,
            "chunk_id": 0,
            "text": content.strip(),
            "type": "other"
        }]


# ── BUILD ALL CHUNKS ──────────────────────────────────────────────────────────
all_chunks = []
for filename, content in files.items():
    file_chunks = chunk_file(filename, content)
    all_chunks += file_chunks
    print(f"{filename}: {len(file_chunks)} chunks")

print(f"\nTotal chunks: {len(all_chunks)}")


# verify chunks look right
print("\nSample chunks:")
for chunk in all_chunks[:3]:
    print(f"\nFile: {chunk['file']} | Type: {chunk['type']} | chunk_id: {chunk['chunk_id']}")
    print(f"Text preview: {chunk['text'][:200]}")
    print("---")

# also check python function chunks
print("\nPython function chunks:")
for chunk in all_chunks:
    if chunk['file'] == 'segment_crops.py':
        print(f"chunk_id: {chunk['chunk_id']} | preview: {chunk['text'][:80]}")
        print("---")


#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------

from sentence_transformers import SentenceTransformer
import numpy as np

# ── ENCODE CHUNKS ─────────────────────────────────────────────────────────────
print("Loading sentence transformer model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Encoding chunks...")
chunk_texts = [c["text"] for c in all_chunks]
chunk_embeddings = embedder.encode(chunk_texts, show_progress_bar=True)

print(f"Encoded {len(chunk_embeddings)} chunks")
print(f"Each embedding has {len(chunk_embeddings[0])} dimensions")


# ── SEARCH ────────────────────────────────────────────────────────────────────
def search(query, top_k=8):
    query_embedding = embedder.encode([query])[0]
    
    similarities = []
    for i, chunk_embedding in enumerate(chunk_embeddings):
        similarity = np.dot(query_embedding, chunk_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
        )
        similarities.append((similarity, i))
    
    similarities.sort(reverse=True)
    
    results = []
    for similarity_score, idx in similarities[:top_k]:
        chunk = all_chunks[idx]
        results.append({
            "file": chunk["file"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "score": round(float(similarity_score), 3),
            "type": chunk["type"]
        })
    
    return results

# ── TEST SEARCH ───────────────────────────────────────────────────────────────
print("\nTest search: 'MAD values'")
results = search("MAD values")
for r in results:
    print(f"{r['score']} | {r['file']} | {r['type']}")
    print(f"  {r['text'][:100]}")
    print("---")




# ── LLM ───────────────────────────────────────────────────────────────────────
def call_llm(prompt):
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1  # makes output deterministic
            }
        },
        timeout=300
    )
    return response.json()["response"]

'''
def ask_llm(question, search_results):
    context = ""
    for r in search_results:
        context += f"\nSource [{r['file']}:{r['chunk_id']}]\n{r['text']}\n---"


    print("CONTEXT BEING SENT:")
    print(context)
    print("END CONTEXT")

    prompt = f"""You are a strict retrieval-based assistant for a pediatric bone age assessment system.

STRICT RULES — follow without exception:
1. Every factual claim MUST end with a citation in the format [filename:chunk_id]
2. ONLY use information explicitly stated in the context below
3. NEVER use your own knowledge or outside information
4. If a claim cannot be directly supported by the context, do NOT make that claim
5. If the context does not contain enough information, respond with exactly:
   "The knowledge base does not contain sufficient information to answer this question."
6. Do NOT infer, extrapolate, or reason beyond what is explicitly stated

Context:
{context}

Question: {question}

Remember: If you cannot cite it from the context above, do not say it.
"""
    return call_llm(prompt)
'''

def ask_llm(question, search_results):
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


# ── RUN QUERY ─────────────────────────────────────────────────────────────────

def run_query(question):
    import time
    start = time.time()
    
    results = search(question)
    
    print(f"Top results:")
    for r in results[:3]:
        print(f"  {r['score']} | {r['file']} | {r['text'][:80]}")
    
    # out of scope if best score too low
    if results[0]["score"] < 0.2:
        print(f"Out of scope: best score was {results[0]['score']}")
        return
    
    # only keep results above 60% of top score
    top_score = results[0]["score"]
    filtered = [r for r in results if r["score"] >= top_score * 0.6]
    
    answer = ask_llm(question, filtered)
    
    end = time.time()
    print(f"\nAnswer:\n{answer}")
    print(f"\nTime: {(end-start)/60:.2f} mins")

'''
# ── TEST ──────────────────────────────────────────────────────────────────────
run_query("How is the error of the model calculated?")
run_query("What is MAD?")
run_query("What MAD values did the transfer-learned full-hand model achieve on the HCJBA dataset?")
run_query("How are short bone masks preprocessed to generate short-bone segments?")
run_query("How were the segmental GP labels assigned to each image?")
run_query("can this system be used to estimate the bone age of humans below the age of 17 years")
run_query("what is the basketball match score")
run_query("Is the bone-age assessment associated with diabetes")
run_query("Why is the bone-age assessment for boys and girls different?")
run_query("can this system be used to estimate the bone age of mummies?")
'''

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # question passed as command line argument
        question = " ".join(sys.argv[1:])
        run_query(question)
    else:
        # interactive mode
        print("\ndigiBONE Knowledge Base — type 'exit' to quit\n")
        while True:
            question = input("Question: ").strip()
            if question.lower() in ("exit", "quit", "q"):
                print("Goodbye.")
                break
            if not question:
                continue
            run_query(question)
