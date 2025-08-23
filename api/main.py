# api/main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from components.graph import build_assistant_graph

app = FastAPI(title="DirectEd Assistant API")

# Request/response models
class ChatRequest(BaseModel):
    user_id: str
    user_type: str
    request_type: str
    subject: str | None = None
    difficulty: str | None = "intermediate"
    query: str
    quick_action: str | None = None

class ChatResponse(BaseModel):
    conversation_response: str | None = None
    generated_content: dict | None = None
    analysis: dict | None = None
    logs: list | None = None

# build graph once (cached)
graph_app = build_assistant_graph()

@app.post("/api/assistant/chat", response_model=ChatResponse)
def assistant_chat(req: ChatRequest):
    state = {
        "user_id": req.user_id,
        "user_type": req.user_type,
        "request_type": req.request_type,
        "subject": req.subject,
        "difficulty": req.difficulty,
        "level": req.difficulty,
        "query": req.query,
        "quick_action": req.quick_action,
        "logs": [],
    }

    try:
        result_state = graph_app.invoke(state)
        return ChatResponse(
            conversation_response=result_state.get("conversation_response"),
            generated_content=result_state.get("generated_content"),
            analysis=result_state.get("analysis"),
            logs=result_state.get("logs"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
