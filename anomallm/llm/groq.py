"""
anomallm/llm/groq.py
Groq LLM provider — thin wrapper around the existing Groq SDK calls.
Logic is unchanged from graph.py; only encapsulated into the provider interface.
"""
import os

from groq import Groq, AsyncGroq

from anomallm.llm.base import LLMProvider, LLMResponse


class GroqProvider(LLMProvider):
    """
    Default provider — uses Groq's API (requires internet access).
    Env vars:
        GROQ_API_KEY  — required
        GROQ_MODEL    — model name (default: openai/gpt-oss-120b)
    """

    MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    def __init__(self) -> None:
        key = os.environ["GROQ_API_KEY"]
        self._client  = Groq(api_key=key)
        self._aclient = AsyncGroq(api_key=key)

    def complete(
        self,
        messages:    list,
        temperature: float = 0.3,
        max_tokens:  int   = 4096,
    ) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return LLMResponse(
            content=resp.choices[0].message.content,
            model=self.MODEL,
            provider="groq",
            tokens_used=resp.usage.total_tokens if resp.usage else 0,
        )

    async def acomplete(
        self,
        messages:    list,
        temperature: float = 0.3,
        max_tokens:  int   = 4096,
    ) -> LLMResponse:
        resp = await self._aclient.chat.completions.create(
            model=self.MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return LLMResponse(
            content=resp.choices[0].message.content,
            model=self.MODEL,
            provider="groq",
            tokens_used=resp.usage.total_tokens if resp.usage else 0,
        )
