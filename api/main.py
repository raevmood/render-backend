# api/main.py
import os
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langserve import add_routes
from langchain_core.runnables import RunnableLambda

from components.graph import build_assistant_graph
from components.base import make_groq_llm

# ---------------------------
# Request / Response Models
# ---------------------------
class ChatRequest(BaseModel):
    user_id: str
    user_type: str  # "student" or "instructor"
    request_type: str  # tutoring, quiz_generation, content_creation
    subject: Optional[str] = None
    auto_detect_topic: Optional[bool] = False
    difficulty_level: Optional[str] = "intermediate"
    query: str

class ChatResponse(BaseModel):
    conversation_response: Optional[str] = None
    logs: Optional[list] = None

# If using pydantic v2, ensure models are rebuilt for runtime introspection
try:
    # Pydantic v2 method
    ChatRequest.model_rebuild()
    ChatResponse.model_rebuild()
except Exception:
    # pydantic v1 has no model_rebuild — ignore
    pass

# ---------------------------
# Normalize / helper
# ---------------------------
def normalize_request(req: ChatRequest) -> Dict[str, Any]:
    state = {
        "user_id": req.user_id,
        "user_type": req.user_type,
        "request_type": req.request_type,
        "subject": req.subject,
        "difficulty": req.difficulty_level,
        "query": req.query,
        "logs": [],
    }

    # Auto-detect subject if enabled
    if not req.subject and req.auto_detect_topic:
        llm = make_groq_llm()
        detect_prompt = f"""
You are a curriculum assistant.
Identify the most relevant DirectEd subject/topic for this query:
"{req.query}"
Return ONLY the subject name, nothing else.
"""
        detection = llm.invoke([{"role": "user", "content": detect_prompt}])
        subject_name = getattr(detection, "content", str(detection)).strip()
        state["subject"] = subject_name

    return state

# ---------------------------
# App and Graph
# ---------------------------
app = FastAPI(title="DirectEd Assistant API")

# Allow CORS for local testing / frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

graph_app = build_assistant_graph()

# Root check route for quick verification
@app.get("/")
def root():
    return {"status": "DirectEd Assistant API running", "routes": ["/api/assistant/chat (POST)", "/api/assistant/chat/playground/"]}

# ---------------------------
# Runnable handler (returns plain dict)
# ---------------------------
def chat_handler(req: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts a dict (compatible with ChatRequest) to be robust for RunnableLambda.
    Returns a JSON-serializable dict.
    """
    try:
        # If a Pydantic model was passed, convert to dict
        if not isinstance(req, dict):
            req = dict(req)
        # Build minimal ChatRequest-like object
        # (normalize_request expects ChatRequest, but we can adapt: create a small wrapper)
        fake = ChatRequest(**req)  # this validates incoming payload
        state = normalize_request(fake)
        result_state = graph_app.invoke(state)

        final = {
            "conversation_response": result_state.get("conversation_response"),
            "logs": result_state.get("logs", []),
        }
        return final
    except Exception as e:
        # return error in JSON to avoid 500 stack pages in playground
        return {"error": str(e)}

# Wrap into a RunnableLambda and provide type info for the playground UI
chat_runnable = RunnableLambda(chat_handler).with_types(
    input_type=ChatRequest,
    output_type=Dict[str, Any]
)

# Explicitly register the invoke AND playground endpoints (and stream endpoints if you want streaming)
add_routes(
    app,
    chat_runnable,
    path="/api/assistant/chat",
)

# Optional: debug helper - print that app is set up
print("LangServe routes registered: /api/assistant/chat (invoke) and /api/assistant/chat/playground/")
