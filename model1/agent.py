# agent.py — Core agent loop and Ollama interface

import re
import json
import requests
from config import OLLAMA_MODEL, OLLAMA_BASE_URL, MAX_ITERATIONS, MAX_TOKENS, REQUIRE_CITATIONS
from tools import TOOLS
from prompts import SYSTEM_PROMPT


# ── OLLAMA INTERFACE ──────────────────────────────────────────────────────────

def call_ollama(messages: list) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": MAX_TOKENS,
        }
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]
    except requests.RequestException as e:
        return f"Error calling Ollama: {str(e)}"


# ── RESPONSE PARSER ───────────────────────────────────────────────────────────

def parse_action(response_text: str):
    lines = response_text.strip().splitlines()
    action = None
    params = {}

    for line in lines:
        line = line.strip()
        if line.startswith("ACTION:"):
            action = line.split("ACTION:", 1)[1].strip().lower()
        elif line.startswith("QUERY:"):
            params["query"] = line.split("QUERY:", 1)[1].strip()
        elif line.startswith("FILE:"):
            params["filepath"] = line.split("FILE:", 1)[1].strip()
        elif line.startswith("DIRECTORY:"):
            params["directory"] = line.split("DIRECTORY:", 1)[1].strip()
        elif line.startswith("ANSWER:"):
            params["answer"] = line.split("ANSWER:", 1)[1].strip()

    if action:
        return action, params
    return None


# ── CITATION VERIFIER ─────────────────────────────────────────────────────────

def verify_citations(answer: str, retrieved_files: set) -> dict:
    citation_pattern = r'\[([^\]]+\.[\w]+:\d+-\d+)\]'
    citations = re.findall(citation_pattern, answer)
    unverified = []
    for citation in citations:
        filepath = citation.split(":")[0]
        if filepath not in retrieved_files:
            unverified.append(citation)
    return {
        "valid": len(unverified) == 0,
        "citations_found": citations,
        "unverified": unverified
    }


# ── MAIN AGENT LOOP ───────────────────────────────────────────────────────────

def run_agent(question: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"}
    ]

    retrieved_files = set()
    tool_calls_log = []
    final_answer = None

    for iteration in range(MAX_ITERATIONS):
        print(f"\n[Agent] Iteration {iteration + 1}/{MAX_ITERATIONS}")

        response_text = call_ollama(messages)
        print(f"[LLM Response]\n{response_text}\n")

        # Append LLM response to history
        messages.append({"role": "assistant", "content": response_text})

        parsed = parse_action(response_text)

        if not parsed:
            messages.append({
                "role": "user",
                "content": "Please continue. Use one of the available actions: search_repo, open_file, list_files, or answer."
            })
            continue

        action, params = parsed

        # ── Final answer ──────────────────────────────────────────
        if action == "answer":
            final_answer = params.get("answer", response_text)
            break

        # ── Execute tool ──────────────────────────────────────────
        tool_fn = TOOLS.get(action)
        if not tool_fn:
            tool_result = f"Unknown tool: {action}"
        else:
            try:
                tool_result = tool_fn(**params)
                tool_calls_log.append({"action": action, "params": params})

                # Track files for citation verification
                if action == "open_file" and "filepath" in params:
                    retrieved_files.add(params["filepath"])
                elif action == "search_repo" and isinstance(tool_result, list):
                    for r in tool_result:
                        if isinstance(r, dict) and "file" in r:
                            retrieved_files.add(r["file"])

            except Exception as e:
                tool_result = f"Tool error: {str(e)}"

        # ── Feed result back — THIS IS THE KEY FIX ───────────────
        # Serialize result clearly so the LLM can read it
        if isinstance(tool_result, (dict, list)):
            result_str = json.dumps(tool_result, indent=2)
        else:
            result_str = str(tool_result)

        # Append as a user message with clear labeling
        messages.append({
            "role": "user",
            "content": (
                f"Tool '{action}' returned the following results. "
                f"Use this information to answer the question or call another tool:\n\n"
                f"{result_str}\n\n"
                f"Now continue. Either call another tool or provide your final answer using ACTION: answer"
            )
        })

        print(f"[Tool Result for '{action}']\n{result_str[:500]}...\n")

    if not final_answer:
        final_answer = "Agent reached maximum iterations without producing a final answer."

    verification = verify_citations(final_answer, retrieved_files) if REQUIRE_CITATIONS else {
        "valid": None, "citations_found": [], "unverified": []
    }

    return {
        "answer": final_answer,
        "citations": verification["citations_found"],
        "verified": verification["valid"],
        "unverified_citations": verification["unverified"],
        "iterations": iteration + 1,
        "tool_calls": tool_calls_log
    }
