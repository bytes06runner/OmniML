"""
anomallm/llm/openai.py
OpenAI LLM provider -- thin wrapper around the OpenAI SDK.
Requires:
    pip install openai
Env vars:
    OPENAI_API_KEY
    OPENAI_MODEL (default: gpt-4o)
"""
import os
from anomallm.llm.base import LLMProvider, LLMResponse

class OpenAIProvider(LLMProvider):
    MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

    def __init__(self) -> None:
        try:
            from openai import OpenAI, AsyncOpenAI
            self._client  = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            self._aclient = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    def complete(self, messages: list, temperature: float = 0.3, max_tokens: int = 4096) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return LLMResponse(
            content=resp.choices[0].message.content,
            model=self.MODEL,
            provider="openai",
            tokens_used=resp.usage.total_tokens if resp.usage else 0,
        )

    async def acomplete(self, messages: list, temperature: float = 0.3, max_tokens: int = 4096) -> LLMResponse:
        resp = await self._aclient.chat.completions.create(
            model=self.MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return LLMResponse(
            content=resp.choices[0].message.content,
            model=self.MODEL,
            provider="openai",
            tokens_used=resp.usage.total_tokens if resp.usage else 0,
        )
