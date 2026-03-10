#!/usr/bin/python

from config import file_path, llm_model
import requests
import time

start=time.time()

with open(file_path, "r", encoding ="utf-8") as f:
    lines= f.readlines()
    

def search(query, lines):
    results=[]

    for i, line in enumerate(lines):
        if query.lower().replace(" ", "").replace("-", "") in line.lower().replace(" ", "").replace("-", ""):

            #scan backwards to find which file the line belongs to
            current_file = "unknown"
            for j in range(i,0,-1):
                if lines[j].startswith("FILE:"):
                    current_file = lines[j].replace("FILE:", "").strip()
                    break


            #get surrounding lines as snippet
            start=max(0,i-2)
            end=min(len(lines),i+20)
            snippet="".join(lines[start:end])

            results.append({
                "file":current_file,
                "line_number": i+1,
                "snippet":snippet
            })

    return results

def call_ollama(prompt):
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": llm_model,
        "prompt": prompt,
        "stream": False
    })
    return response.json()["response"]

def extract_words(question):
    prompt = f"""Extract 3-5 keywords strictly from the question below.
Only use words that actually appear in the question.
Do NOT add related terms or your own knowledge.
Return ONLY a comma separated list, nothing else.

Question: {question}
"""
    return call_ollama(prompt).split(",")

def filter_results(results, question):
    question_words = set(question.lower().replace("-","").replace(" ","").split())
    scored = []
    for r in results:
        snippet_words = set(r["snippet"].lower().replace("-","").replace(" ","").split())
        score = len(question_words & snippet_words)  # count matching words
        scored.append((score, r))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for score, r in scored[:8]]  # only top 8


def ask_llm(question, search_results):
    context = ""
    for r in search_results:
        context += f"\nFile: {r['file']} | Line: {r['line_number']}\n{r['snippet']}\n---"

    print("CONTEXT LENGTH:", len(context))  
    print("CONTEXT PREVIEW:", context[:200])

    prompt = f"""You are a helpful assistant answering questions about a codebase.
Using ONLY the context below, write a clear and well structured answer.
- Start with the plain English definition or explanation first
- Then add implementation details and how it works in this specific project
- End every claim with a citation like [README.md:42]
- Do NOT start with code or file references
- If the context doesn't contain the answer, say so explicitly

Context:
{context}

Question: {question}
"""
    
    return call_ollama(prompt)

    
question = "I am a computer vision expert. Explain what segmentation models are."
keywords = extract_words(question)
print("Keywords:", keywords)

results=[]

for keyword in keywords:
    results += search(keyword, lines)

#for r in results:
#    print(r["file"], "| line:", r["line_number"])
#    print(r["snippet"][:100])
#    print("---")
    
# deduplicate
seen = set()
unique_results = []
for r in results:
    key = (r["file"], r["line_number"])
    if key not in seen:
        seen.add(key)
        unique_results.append(r)

print("Unique results:", len(unique_results))

filtered_results = filter_results(unique_results, question)
answer = ask_llm(question, filtered_results)
print(answer)

end = time.time()
print(f"Time taken: {(end - start)/60:.2f} mins")
