# components/base.py
import os
from typing import Optional
from langchain_groq import ChatGroq
from utils.helper import load_chroma_db, download_embbeding_model
from dotenv import load_dotenv


load_dotenv()


class ChatGroqWrapper(ChatGroq):
    @property
    def model(self):
        return self.model_name  
    
api_key = os.getenv("GROQ_API_KEY")

# LLM factory (Groq)
def make_groq_llm(model: Optional[str] = None, temperature: float = 0.15, max_tokens: int = 512) -> ChatGroq:
    model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in environment.")
    return ChatGroqWrapper(
        groq_api_key=api_key,
        model_name=model, temperature=temperature, max_tokens=max_tokens
    )

# Vectorstore factory
def get_vectorstore(persist_dir: str = "data/chroma"):
    embeddings = download_embbeding_model()
    vectordb = load_chroma_db(embeddings, persist_dir)
    return vectordb
