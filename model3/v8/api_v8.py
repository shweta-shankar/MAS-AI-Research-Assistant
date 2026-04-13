from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import db_manager_v8 as db
from gitSearch_v8 import run_query, embedder

app = FastAPI(title="digiBONE API")

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    source_tier: str
    score: float
    sources: List[str]

class ReviewUpdate(BaseModel):
    id: int
    action: str  # "approve" or "reject"
    verified_answer: Optional[str] = None
    reviewer_suggestions: Optional[str] = None

@app.post("/api/chat", response_model=ChatResponse)
def handle_chat(request: ChatRequest):
    try:
        result = run_query(request.query)
        sources_list = []
        if isinstance(result.get("rag_sources"), list):
            for s in result["rag_sources"]:
                # s is a formatted string natively from gitSearch_v8
                sources_list.append(str(s))

        return ChatResponse(
            answer=result.get("answer", "No answer provided."),
            source_tier=result.get("source", "unknown"),
            score=float(result.get("score", 0.0)),
            sources=sources_list
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pending")
def get_pending():
    return db.get_pending_questions()

@app.post("/api/review")
def post_review(update: ReviewUpdate):
    if update.action == "approve":
        # Fetch question text to generate embedding
        pending_list = db.get_pending_questions()
        target = next((p for p in pending_list if p["id"] == update.id), None)
        if not target:
            raise HTTPException(status_code=404, detail="Pending ID not found")
            
        q_vec = embedder.encode([target["question"]])[0]
        
        db.approve_pending(
            update.id, 
            verified_answer=update.verified_answer, 
            reviewer_suggestions=update.reviewer_suggestions,
            question_embedding=q_vec
        )
        return {"status": "approved"}
    elif update.action == "reject":
        db.reject_pending(update.id)
        return {"status": "rejected"}
    else:
        raise HTTPException(status_code=400, detail="Invalid action")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
