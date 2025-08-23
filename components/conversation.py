# components/conversation.py
import json
from .state import AssistantState
from .base import make_groq_llm
from .prompts import load_prompt

def node_conversation(state: AssistantState) -> dict:
    llm = make_groq_llm()
    template = load_prompt("conversation_prompt.txt")
    context = "\n\n".join([f"[{c.get('source')}] {c.get('text')}" for c in (state.get("retrieved_chunks") or [])])

    difficulty = state.get("difficulty") or state.get("level") or "intermediate"


    prompt = template.format(
        context=context,
        query=state.get("query", ""),
        difficulty=difficulty,
        user_type=state.get("user_type", "student")
    )

    # ChatGroq LLM: using .invoke([...]) as per langchain-groq wrapper.
    response = llm.invoke([{"role": "user", "content": prompt}])
    # Normalize
    response_text = ""
    try:
        # ChatGroq may return dict-like. Try common attrs.
        if hasattr(response, "content"):
            response_text = response.content
        else:
            # fallback to str(response)
            response_text = str(response)
    except Exception:
        response_text = str(response)

    logs = state.get("logs", []) + [{"step": "conversation", "llm": llm.model_name}]
    print(response_text)
    return {"conversation_response": response_text, "logs": logs}
