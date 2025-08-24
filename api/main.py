# api/main.py
import os
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chains import RetrievalQA

from langserve import add_routes
from langchain_core.runnables import RunnableLambda

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
    logs: Optional[list] = None

# ---------------------------
# Allowed subjects
# ---------------------------
ALLOWED_SUBJECTS = ["artificial intelligence","programming","python", "javascript", "html", "css", "react", "nodejs", "data structures", "algorithms", "tailwind CSS", "Typescript", "machine learning"]


# ---------------------------
# RAG Handler for General Chat
# ---------------------------
def handle_general_chat(query: str) -> dict:
    """
    Handle general chat using RAG (retrieval + LLM generation).
    Returns a dict with 'conversation_response' and 'logs'.
    """

    llm= make_groq_llm()

    logs = []
    try:
        # Load vectorstore and setup retriever
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
        logs.append(f"Retrieved top 3 documents for query: '{query}'")

        # Friendly prompt template
        friendly_prompt = f"""
        You are a friendly and helpful assistant.
        Answer the following question in a clear and approachable manner:
        "{query}"
        Use the retrieved documents to provide accurate context.
        """

        # Setup RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False
        )

        answer = qa_chain.invoke({"query": friendly_prompt})
        logs.append("Generated friendly answer using RAG.")

        return {"conversation_response": answer["result"], "logs": logs}

    except Exception as e:
        logs.append(f"Error in RAG processing: {str(e)}")
        return {"conversation_response": "Sorry, I couldn't process your query.", "logs": logs}





# ---------------------------
# Normalize / helper
# ---------------------------
def normalize_request(req: ChatRequest) -> Dict[str, Any]:
    if isinstance(req, dict):
        req = ChatRequest(**req)

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
llm_checker = make_groq_llm()

# ---------------------------
# Root check route
# ---------------------------
@app.get("/")
def root():
    return {"status": "DirectEd Assistant API running", "routes": ["/api/assistant/chat (POST)", "/api/assistant/chat/playground/"]}

# ---------------------------
# Runnable handler
# ---------------------------
def chat_handler(req: ChatRequest) -> ChatResponse:
    try:
        if isinstance(req, dict):
            req = ChatRequest(**req)

        state = normalize_request(req)

        # Educational requests
        if req.request_type in ["tutoring", "quiz_generation", "flashcard_creation", "content_creation"]:
            subject_text = (state["subject"] or "").lower()
            if subject_text and subject_text not in ALLOWED_SUBJECTS:
                return ChatResponse(
                    conversation_response=f"Sorry, we do not offer courses on '{subject_text}'. Please choose a computing/programming topic.",
                    logs=["Subject not allowed."]
                )
            if not subject_text:
                prompt = f"Is the following query related to programming or computing? Answer Yes or No:\n'{req.query}'"
                result = llm_checker.invoke([{"role": "user", "content": prompt}])
                content = getattr(result, "content", str(result))
                if "yes" not in content.lower():
                    return ChatResponse(
                        conversation_response="Sorry, we currently only support programming/computing topics for tutoring. For general questions, you can chat normally.",
                        logs=["Query not related to computing/programming."]
                    )

            # Invoke graph
            result_state = graph_app.invoke(state)
            return ChatResponse(
                conversation_response=result_state.get("conversation_response"),
                logs=result_state.get("logs", [])
            )

        # General chat via RAG
        rag_result = handle_general_chat(req.query)
        return ChatResponse(
            conversation_response=rag_result["conversation_response"],
            logs=rag_result["logs"]
        )

    except Exception as e:
        return ChatResponse(
            conversation_response=None,
            logs=[f"Error: {str(e)}"]
        )

# ---------------------------
# LangServe Runnable
# ---------------------------
chat_runnable = RunnableLambda(chat_handler).with_types(
    input_type=ChatRequest,
    output_type=ChatResponse
)

# Explicitly register the invoke AND playground endpoints
add_routes(
    app,
    chat_runnable,
    path="/api/assistant/chat",
)

print("LangServe routes registered: /api/assistant/chat (invoke) and /api/assistant/chat/playground/")
