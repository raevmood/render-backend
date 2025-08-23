# components/state.py
from typing import TypedDict, List, Optional, Dict, Any

class AssistantState(TypedDict, total=False):
    user_id: str
    user_type: str
    request_type: str
    subject: Optional[str]
    difficulty: Optional[str]
    level: Optional[str]  
    query: str
    quick_action: Optional[str]

    retrieved_chunks: Optional[List[Dict[str, Any]]]
    conversation_response: Optional[str]
    generated_content: Optional[Dict[str, Any]]
    analysis: Optional[Dict[str, Any]]

    logs: Optional[List[Dict[str, Any]]]
