# components/analyzer.py
import json
from .state import AssistantState
from .base import make_groq_llm
from .prompts import load_prompt

def node_analyzer(state: AssistantState) -> dict:
    llm = make_groq_llm()
    template = load_prompt("analyzer_prompt.txt")

    # student answer: prefer a passed-in answer in generated_content else fallback to query
    student_answer = (state.get("generated_content") or {}).get("student_answer") or state.get("query", "")
    reference = (state.get("retrieved_chunks") or [{}])[0].get("text", "No reference found.")

    difficulty = state.get("difficulty") or state.get("level") or "intermediate"


    prompt = template.format(
        answer=student_answer, reference=reference, difficulty=difficulty
    )

    response = llm.invoke([{"role": "user", "content": prompt}])
    # print(f"Analyzer response : {response}")
    response_text = response.content if hasattr(response, "content") else str(response)
    response_text = response_text.strip()

    try:
        analysis = json.loads(response_text)
    except Exception:
        analysis = {"response_text": response_text}

    logs = state.get("logs", []) + [{"step": "analyzer", "llm": llm.model_name}]
    return {"analysis": analysis, "logs": logs}
