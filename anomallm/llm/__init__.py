"""
anomallm/llm/__init__.py
LLM provider factory — reads LLM_PROVIDER from env.
Falls back to Groq if the requested provider is unavailable.
"""
import os

from anomallm.llm.base   import LLMProvider, LLMResponse
from anomallm.llm.groq   import GroqProvider
from anomallm.llm.ollama import OllamaProvider
from anomallm.llm.openai import OpenAIProvider


def get_llm_provider() -> LLMProvider:
    """
    Factory — reads LLM_PROVIDER from env.
    Falls back to Groq if provider unavailable.

    NOTE: Named get_llm_provider() (not get_llm) to avoid collision with
    the ChatGroq-based get_llm() helper that already exists in graph.py.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "ollama":
        if not OllamaProvider.is_available():
            print(
                "[OmniML] Ollama not running at "
                f"{OllamaProvider.BASE_URL}. Falling back to Groq.",
                flush=True,
            )
            return GroqProvider()
        return OllamaProvider()

    elif provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            print("[OmniML] OPENAI_API_KEY not set. "
                  "Falling back to Groq.", flush=True)
            return GroqProvider()
        return OpenAIProvider()

    return GroqProvider()


__all__ = [
    "LLMProvider",
    "LLMResponse",
    "GroqProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "get_llm_provider",
]
