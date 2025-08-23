# components/prompts.py
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

def load_prompt(filename: str) -> str:
    """
    Load a prompt file by name (filename with extension or without).
    Example: load_prompt("conversation_prompt.txt") or load_prompt("conversation_prompt")
    """
    if not filename.endswith(".txt") and not filename.endswith(".md"):
        filename = filename + ".txt"
    path = PROMPT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")
