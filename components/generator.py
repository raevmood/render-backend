# components/generator.py
import json
from .state import AssistantState
from .base import make_groq_llm
from .prompts import load_prompt

def node_generator(state: AssistantState) -> dict:
    llm = make_groq_llm()
    template = load_prompt("generator_prompt.txt")
    context = state["conversation_response"] or []

    difficulty = state.get("difficulty") or state.get("level") or "intermediate"
    

    prompt = template.format(
        topic=state.get("subject") or state.get("query", "General"),
        query=state.get("query", ""),
        difficulty=difficulty,
        request_type=state.get("request_type", "content_creation"),
        context=context
    )
    
    response = llm.invoke([{"role": "user", "content": prompt}])
    response_text = response.content if hasattr(response, "content") else str(response)
    # print(f"generetor resopnse : {response}")
    try:
        generated = json.loads(response_text)
    except Exception:
        # If model returned plain text, keep response_text under 'response_text' key
        generated = {"response_text": response_text}

    logs = state.get("logs", []) + [{"step": "generator", "llm": llm.model_name}]
    return {"generated_content": generated, "logs": logs}
