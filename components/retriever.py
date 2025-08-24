# components/retriever.py
from .state import AssistantState
from .base import get_vectorstore
# from .prompts import load_prompt

def node_retriever(state: AssistantState) -> dict:
    docsearch = get_vectorstore()
    query = state.get("query", "")
    docs = docsearch.similarity_search(query, search_kwargs={"k": 1})

    retrieved = []
    for i, d in enumerate(docs):
        meta = d.metadata if isinstance(d.metadata, dict) else {}
        retrieved.append({
            "id": getattr(d, "id", f"chunk_{i}"),
            "source": meta.get("source", "unknown"),
            "text": d.page_content,
        })

    logs = state.get("logs", []) + [{"step": "retriever", "query": query, "retrieved": len(retrieved)}]
    # print(f"retrieved:  {retrieved}")
    return {"retrieved_chunks": retrieved, "logs": logs}
