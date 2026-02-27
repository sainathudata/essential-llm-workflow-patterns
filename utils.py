"""
utils.py — Shared helpers for all LLM pattern examples.

All patterns import the Anthropic client from here.
Set your API key via:  export ANTHROPIC_API_KEY="sk-ant-..."
"""

import os
import textwrap
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Client ────────────────────────────────────────────────────────────────────
client = OpenAI(
    api_key=os.environ.get("OLLAMA_API_KEY"),
    base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434/v1"),
)

MODEL = "llama3.2"

# ── Helpers ───────────────────────────────────────────────────────────────────
def chat(
    user: str, 
    system: str = "You are a helpful assistant.", 
    *, 
    max_tokens: int = 1024, 
    temperature: float = 1.0
) -> str:
    """Single-turn chat helper for OpenAI Chat Completions."""
    
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    return response.choices[0].message.content



def section(title: str) -> None:
    """Print a bold section header to stdout."""
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def show(label: str, text: str, width: int = 80) -> None:
    """Pretty-print a labelled block of text."""
    print(f"\n[{label}]")
    print(textwrap.fill(text.strip(), width=width))