# tools.py — Repository interaction tools for the agent

import re
from config import SEARCH_TOP_K, CHUNK_SIZE

# In-memory store: { "filename.py": ["line1\n", "line2\n", ...] }
_repo_store = {}


# ── 1. FETCH REPO ─────────────────────────────────────────────────────────────

def fetch_repo(file_path: str) -> str:
    """Loads a gitingest .txt dump from a local file into memory."""
    global _repo_store

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            dump = f.read()

        _repo_store = _parse_gitingest_dump(dump)
        return f"Repo loaded successfully. {len(_repo_store)} files indexed."

    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error loading repo dump: {str(e)}"


def _parse_gitingest_dump(dump: str) -> dict:
    """
    Parses gitingest output into {filepath: [lines]}.

    gitingest format:
        ================================================
        FILE: filename.py
        ================================================
        <file content>
        ================================================
        FILE: next_file.py
        ...

    After splitting on '====\\n', sections alternate:
        even index = FILE: name
        odd index  = file content
    """
    file_store = {}
    sections = re.split(r"={10,}\n", dump)

    i = 0
    while i < len(sections):
        section = sections[i].strip()
        if section.startswith("FILE:"):
            filename = section.replace("FILE:", "").strip()
            # Next section is the file content
            if i + 1 < len(sections):
                content = sections[i + 1]
                file_store[filename] = content.splitlines(keepends=True)
                i += 2  # skip the content section
                continue
        i += 1

    return file_store


# ── 2. SEARCH REPO ────────────────────────────────────────────────────────────

def search_repo(query: str) -> list:
    """
    Searches all indexed files for lines matching the query.
    Returns up to SEARCH_TOP_K results with file, line numbers, and snippet.
    """
    if not _repo_store:
        return [{"error": "No repo loaded. Call fetch_repo() first."}]

    results = []
    query_lower = query.lower()

    for filepath, lines in _repo_store.items():
        for i, line in enumerate(lines):
            if query_lower in line.lower():
                start = max(0, i - 2)
                end = min(len(lines), i + CHUNK_SIZE)
                snippet = "".join(lines[start:end])

                results.append({
                    "file": filepath,
                    "line_start": start + 1,
                    "line_end": end,
                    "snippet": snippet.strip()
                })

                if len(results) >= SEARCH_TOP_K:
                    return results

    return results if results else [{"message": f"No results found for: '{query}'"}]


# ── 3. OPEN FILE ──────────────────────────────────────────────────────────────

def open_file(filepath: str) -> dict:
    """Returns the full content of a specific file with line numbers."""
    if not _repo_store:
        return {"error": "No repo loaded. Call fetch_repo() first."}

    if filepath not in _repo_store:
        matches = [f for f in _repo_store if filepath in f]
        if not matches:
            return {"error": f"File not found: {filepath}. Available files: {list(_repo_store.keys())}"}
        filepath = matches[0]

    lines = _repo_store[filepath]
    numbered = "".join(f"{i+1:4} | {line}" for i, line in enumerate(lines))

    return {
        "file": filepath,
        "total_lines": len(lines),
        "content": numbered
    }


# ── 4. LIST FILES ─────────────────────────────────────────────────────────────

def list_files(directory: str = "") -> dict:
    """Lists all indexed files, optionally filtered by directory prefix."""
    if not _repo_store:
        return {"error": "No repo loaded. Call fetch_repo() first."}

    all_files = list(_repo_store.keys())
    filtered = [f for f in all_files if f.startswith(directory)] if directory else all_files

    return {
        "directory": directory or "(root)",
        "files": sorted(filtered)
    }


# ── TOOL REGISTRY ─────────────────────────────────────────────────────────────

TOOLS = {
    "search_repo": search_repo,
    "open_file": open_file,
    "list_files": list_files,
}
