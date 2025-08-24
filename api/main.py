# api/main.py
import os
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from datetime import datetime
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from langchain.chains import RetrievalQA
    from components.graph import build_assistant_graph
    from components.base import get_vectorstore, make_groq_llm
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Import error: {e}")
    DEPENDENCIES_AVAILABLE = False

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

# Global variables for lazy initialization
graph_app = None
llm_checker = None


# ---------------------------
# Startup/Shutdown Context Manager
# ---------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global graph_app, llm_checker
    try:
        if DEPENDENCIES_AVAILABLE:
            graph_app = build_assistant_graph()
            llm_checker = make_groq_llm()
            logger.info("Graph + LLM checker initialized.")
        else:
            logger.warning("Dependencies not available, running in limited mode.")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
    
    yield
    
    # Shutdown (cleanup if needed)
    logger.info("Shutting down...")


# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI(
    title="DirectEd Assistant API",
    lifespan=lifespan
)

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Exception Handler
# ---------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return {"error": f"Internal server error: {str(exc)}"}


# ---------------------------
# RAG Handler for General Chat
# ---------------------------
def handle_general_chat(query: str) -> Dict[str, Any]:
    logs = []

    if not DEPENDENCIES_AVAILABLE:
        return {
            "conversation_response": "Service temporarily unavailable - dependencies not loaded",
            "logs": ["Dependencies not available"]
        }

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
        logger.error(f"RAG error: {e}")
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
        try:
            detect_prompt = f"""
            You are a curriculum assistant.
            Identify the most relevant subject for this query:
            "{req.query}"
            Return ONLY the subject name.
            """
            detection = llm_checker.invoke([{"role": "user", "content": detect_prompt}])
            subject_name = getattr(detection, "content", str(detection)).strip()
            state["subject"] = subject_name
        except Exception as e:
            logger.error(f"Subject detection error: {e}")
            state["logs"].append(f"Subject detection failed: {str(e)}")

    return state


# ---------------------------
# Root routes
# ---------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "dependencies": DEPENDENCIES_AVAILABLE,
        "graph_initialized": graph_app is not None,
        "llm_initialized": llm_checker is not None
    }

@app.get("/")
def root():
    return {
        "status": "DirectEd Assistant API running",
        "routes": [
            "/api/assistant/chat",
            "/health"
        ],
        "dependencies": DEPENDENCIES_AVAILABLE
    }


# ---------------------------
# Chat handler
# ---------------------------
def chat_handler(req: ChatRequest) -> ChatResponse:
    try:
        # If dependencies not available, return error
        if not DEPENDENCIES_AVAILABLE:
            return ChatResponse(
                conversation_response="Service temporarily unavailable",
                logs=["Dependencies not loaded"]
            )

        # If LLM/graph not initialized yet, try basic response
        if graph_app is None or llm_checker is None:
            return ChatResponse(
                conversation_response="System initializing, please try again shortly",
                logs=["System not fully initialized"]
            )

        state = normalize_request(req)

        # Educational requests
        if req.request_type in ["tutoring", "quiz_generation", "flashcard_creation", "content_creation"]:
            subject_text = (state["subject"] or "").lower()

            if subject_text and subject_text not in ALLOWED_SUBJECTS:
                return ChatResponse(
                    conversation_response=f"Sorry, we do not offer courses on '{subject_text}'.",
                    logs=["Subject not allowed."]
                )

            if not subject_text:
                try:
                    prompt = f"Is the following query related to programming? Answer Yes or No:\n'{req.query}'"
                    result = llm_checker.invoke([{"role": "user", "content": prompt}])
                    content = getattr(result, "content", str(result))
                    if "yes" not in content.lower():
                        return ChatResponse(
                            conversation_response="Sorry, only programming topics are supported.",
                            logs=["Query not related to programming."]
                        )
                except Exception as e:
                    logger.error(f"Programming check error: {e}")
                    return ChatResponse(
                        conversation_response="Error processing request",
                        logs=[f"Programming check failed: {str(e)}"]
                    )

            # Graph execution
            try:
                result_state = graph_app.invoke(state)
                return ChatResponse(
                    conversation_response=result_state.get("conversation_response"),
                    logs=result_state.get("logs", [])
                )
            except Exception as e:
                logger.error(f"Graph execution error: {e}")
                return ChatResponse(
                    conversation_response="Error processing educational request",
                    logs=[f"Graph error: {str(e)}"]
                )

        # General chat
        rag_result = handle_general_chat(req.query)
        return ChatResponse(
            conversation_response=rag_result["conversation_response"],
            logs=rag_result["logs"]
        )

    except Exception as e:
        logger.error(f"Chat handler error: {e}")
        return ChatResponse(
            conversation_response="Sorry, an error occurred processing your request",
            logs=[f"Error: {str(e)}"]
        )

# Add these endpoints to your main.py

@app.post("/debug/simple")
def debug_simple():
    """Simple test endpoint"""
    return {"status": "debug endpoint works", "timestamp": str(datetime.now())}

@app.post("/debug/chat")
def debug_chat(req: ChatRequest):
    """Debug version of chat endpoint"""
    try:
        return {
            "status": "received_request",
            "received_data": {
                "user_id": req.user_id,
                "user_type": req.user_type,
                "request_type": req.request_type,
                "subject": req.subject,
                "query": req.query[:50] + "..." if len(req.query) > 50 else req.query
            },
            "dependencies_available": DEPENDENCIES_AVAILABLE,
            "graph_initialized": graph_app is not None,
            "llm_initialized": llm_checker is not None
        }
    except Exception as e:
        return {"error": f"Debug endpoint error: {str(e)}"}

@app.post("/debug/components")
def debug_components():
    """Test if components can be imported"""
    try:
        from components.graph import build_assistant_graph
        from components.base import get_vectorstore, make_groq_llm
        return {"status": "components imported successfully"}
    except Exception as e:
        return {"error": f"Component import error: {str(e)}"}

# Don't forget to add this import at the top

# Add this debug chat handler to your main.py

@app.post("/debug/full-chat")
def debug_full_chat(req: ChatRequest):
    """Debug the full chat processing pipeline"""
    debug_info = []
    
    try:
        debug_info.append("1. Received request successfully")
        
        # Test normalize_request
        try:
            state = normalize_request(req)
            debug_info.append("2. normalize_request() - SUCCESS")
            debug_info.append(f"   - State: {state}")
        except Exception as e:
            debug_info.append(f"2. normalize_request() - FAILED: {str(e)}")
            return {"debug_info": debug_info, "error": "normalize_request failed"}
        
        # Test request type logic
        debug_info.append(f"3. Request type: {req.request_type}")
        
        if req.request_type in ["tutoring", "quiz_generation", "flashcard_creation", "content_creation"]:
            debug_info.append("4. Educational request path")
            
            subject_text = (state["subject"] or "").lower()
            debug_info.append(f"5. Subject: '{subject_text}'")
            
            if subject_text and subject_text not in ALLOWED_SUBJECTS:
                debug_info.append("6. Subject not allowed - early return")
                return {
                    "debug_info": debug_info,
                    "result": f"Sorry, we do not offer courses on '{subject_text}'."
                }
            
            if not subject_text:
                debug_info.append("7. Testing programming check...")
                try:
                    prompt = f"Is the following query related to programming? Answer Yes or No:\n'{req.query}'"
                    result = llm_checker.invoke([{"role": "user", "content": prompt}])
                    content = getattr(result, "content", str(result))
                    debug_info.append(f"8. Programming check result: {content}")
                    
                    if "yes" not in content.lower():
                        debug_info.append("9. Not programming related - early return")
                        return {
                            "debug_info": debug_info,
                            "result": "Sorry, only programming topics are supported."
                        }
                except Exception as e:
                    debug_info.append(f"8. Programming check FAILED: {str(e)}")
                    return {"debug_info": debug_info, "error": "Programming check failed"}
            
            # Test graph execution
            debug_info.append("10. About to call graph_app.invoke()...")
            try:
                result_state = graph_app.invoke(state)
                debug_info.append("11. Graph execution - SUCCESS")
                return {
                    "debug_info": debug_info,
                    "result": result_state.get("conversation_response"),
                    "logs": result_state.get("logs", [])
                }
            except Exception as e:
                debug_info.append(f"11. Graph execution - FAILED: {str(e)}")
                return {"debug_info": debug_info, "error": f"Graph execution failed: {str(e)}"}
        
        else:
            debug_info.append("4. General chat path")
            debug_info.append("5. About to call handle_general_chat()...")
            
            try:
                rag_result = handle_general_chat(req.query)
                debug_info.append("6. handle_general_chat() - SUCCESS")
                return {
                    "debug_info": debug_info,
                    "result": rag_result["conversation_response"],
                    "logs": rag_result["logs"]
                }
            except Exception as e:
                debug_info.append(f"6. handle_general_chat() - FAILED: {str(e)}")
                return {"debug_info": debug_info, "error": f"RAG failed: {str(e)}"}
                
    except Exception as e:
        debug_info.append(f"OUTER EXCEPTION: {str(e)}")
        return {"debug_info": debug_info, "error": f"Outer exception: {str(e)}"}
# ---------------------------
# FastAPI Routes
# ---------------------------
@app.post("/api/assistant/chat", response_model=ChatResponse)
def invoke_chat(req: ChatRequest):
    try:
        return chat_handler(req)
    except Exception as e:
        logger.error(f"Route error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
