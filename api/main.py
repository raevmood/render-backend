# api/main.py
import os
from typing import Optional, Dict, Any, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain.chains import RetrievalQA

from components.graph import build_assistant_graph
from components.base import get_vectorstore, make_groq_llm

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
    logs: Optional[List[str]] = None


# ---------------------------
# Allowed subjects
# ---------------------------
ALLOWED_SUBJECTS = [
    "artificial intelligence", "programming", "python", "javascript", "html", "css",
    "react", "nodejs", "data structures", "algorithms", "tailwind CSS",
    "Typescript", "machine learning"
]


# ---------------------------
# RAG Handler for General Chat
# ---------------------------
def handle_general_chat(query: str) -> Dict[str, Any]:
    logs = []

    try:
        llm = make_groq_llm()
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
        logs.append(f"Retrieved top documents for query: '{query}'")

        friendly_prompt = f"""
        You are a friendly and helpful assistant.
        Answer the following question clearly and politely:
        "{query}"
        Use the retrieved documents for accuracy.
        """

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False
        )

        answer = qa_chain.invoke({"query": friendly_prompt})
        logs.append("Generated answer using RAG.")

        return {"conversation_response": answer["result"], "logs": logs}

    except Exception as e:
        logs.append(f"Error in RAG: {str(e)}")
        return {"conversation_response": "Sorry, I couldn't process your query.", "logs": logs}


# ---------------------------
# Normalize Request
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

    # Auto-detect subject if needed
    if not req.subject and req.auto_detect_topic and llm_checker is not None:
        detect_prompt = f"""
        You are a curriculum assistant.
        Identify the most relevant subject for this query:
        "{req.query}"
        Return ONLY the subject name.
        """
        detection = llm_checker.invoke([{"role": "user", "content": detect_prompt}])
        subject_name = getattr(detection, "content", str(detection)).strip()
        state["subject"] = subject_name

    return state


# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI(title="DirectEd Assistant API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://raevmood.github.io/frontend/"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy initialization
graph_app = None
llm_checker = None

@app.on_event("startup")
async def startup_event():
    global graph_app, llm_checker
    graph_app = build_assistant_graph()
    llm_checker = make_groq_llm()
    print("Graph + LLM checker initialized.")


# ---------------------------
# Root route
# ---------------------------
@app.get("/")
def root():
    return {
        "status": "DirectEd Assistant API running",
        "routes": [
            "/api/assistant/chat/invoke",
            "/api/assistant/chat/playground/"
        ]
    }


# ---------------------------
# Chat handler
# ---------------------------
def chat_handler(req: dict) -> ChatResponse:
    # Convert dict → ChatRequest
    if isinstance(req, dict) and "input" in req:
        req_obj = ChatRequest(**req["input"])
    elif isinstance(req, dict):
        req_obj = ChatRequest(**req)
    else:
        req_obj = req

    # If LLM/graph not initialized yet (Playground/schema inspection), return placeholder
    if graph_app is None or llm_checker is None:
        return ChatResponse(conversation_response="placeholder", logs=[])

    try:
        state = normalize_request(req_obj)

        # Educational requests
        if req_obj.request_type in ["tutoring", "quiz_generation", "flashcard_creation", "content_creation"]:
            subject_text = (state["subject"] or "").lower()

            if subject_text and subject_text not in ALLOWED_SUBJECTS:
                return ChatResponse(
                    conversation_response=f"Sorry, we do not offer courses on '{subject_text}'.",
                    logs=["Subject not allowed."]
                )

            if not subject_text:
                prompt = f"Is the following query related to programming? Answer Yes or No:\n'{req_obj.query}'"
                result = llm_checker.invoke([{"role": "user", "content": prompt}])
                content = getattr(result, "content", str(result))
                if "yes" not in content.lower():
                    return ChatResponse(
                        conversation_response="Sorry, only programming topics are supported.",
                        logs=["Query not related to programming."]
                    )

            # Graph execution
            result_state = graph_app.invoke(state)
            return ChatResponse(
                conversation_response=result_state.get("conversation_response"),
                logs=result_state.get("logs", [])
            )

        # General chat
        rag_result = handle_general_chat(req_obj.query)
        return ChatResponse(
            conversation_response=rag_result["conversation_response"],
            logs=rag_result["logs"]
        )

    except Exception as e:
        return ChatResponse(conversation_response=None, logs=[f"Error: {str(e)}"])


# ---------------------------
# FastAPI Routes
# ---------------------------
@app.post("/api/assistant/chat", response_model=ChatResponse)
def invoke_chat(req: ChatRequest):
    return chat_handler(req)


print("✅ FastAPI routes ready: /api/assistant/chat/invoke and /api/assistant/chat/playground/")
