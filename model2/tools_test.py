#!/usr/bin/python

from config import file_path, llm_model, pdf1, pdf2
import requests
import time
import pdfplumber

#-----------------------------------------------------------------------------------

start=time.time()

#-----------------------------------------------------------------------------------
# For loading the pdf

def load_pdf(pdf_path):
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                # Add a FILE: header so our search function knows the source
                lines.append(f"================================================\n")
                lines.append(f"FILE: {pdf_path} (page {i+1})\n")
                lines.append(f"================================================\n")
                for line in text.splitlines(keepends=True):
                    lines.append(line)
    return lines


#----------------------------------------------------------------------------------
# Parsing the files

with open(file_path, "r", encoding ="utf-8") as f:
    lines= f.readlines()

lines += load_pdf(pdf1)
lines += load_pdf(pdf2)

#-----------------------------------------------------------------------------------
# Search tool for llm  

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

#-----------------------------------------------------------------------------------
# Setting the parameters of LLM

def call_ollama(prompt):
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": llm_model,
        "prompt": prompt,
        "stream": False
    })
    return response.json()["response"]

#-----------------------------------------------------------------------------------
# Extracting keywords from query using LLM

def extract_words(question):
    prompt = f"""Extract 3-5 keywords strictly from the question below.
Only use words that actually appear in the question.
Do NOT add related terms or your own knowledge.
Return ONLY a comma separated list, nothing else.

Question: {question}
"""
    return call_ollama(prompt).split(",")

#-----------------------------------------------------------------------------------
# This is to check if result context generated is relevant to question or not

def filter_results(results, question):
    question_words = set(question.lower().replace("-","").replace(" ","").split())
    scored = []
    for r in results:
        snippet_words = set(r["snippet"].lower().replace("-","").replace(" ","").split())
        score = len(question_words & snippet_words)  # count matching words
        scored.append((score, r))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for score, r in scored[:8]]  # only top 8

#-------------------------------------------------------------------------------------
# Setting up the query answering LLM

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

#-------------------------------------------------------------------------------------
######################################################################################
#Querying

question = "How are short bone masks preprossed to generate short-bone segments?"
keywords = extract_words(question)
print("Keywords:", keywords)

results=[]

for keyword in keywords:
    results += search(keyword, lines)
    
#-----------------------------------------------------------------------------------
#for r in results:
#    print(r["file"], "| line:", r["line_number"])
#    print(r["snippet"][:100])
#    print("---")
#------------------------------------------------------------------------------------
    
# Removing deduplicates
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

#-----------------------------------------------------------------------------------

end = time.time()
print(f"Time taken: {(end - start)/60:.2f} mins")

#-----------------------------------------------------------------------------------
####################################################################################
