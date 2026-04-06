"""
anomallm/llm/ollama.py
100% offline LLM provider via Ollama.
No data leaves the machine. Zero external network calls.

Setup:
    1. Install Ollama: https://ollama.ai  (or: brew install ollama)
    2. Pull a model:   ollama pull llama3:70b
    3. Set in .env:    LLM_PROVIDER=ollama
                       OLLAMA_MODEL=llama3:70b

Recommended models by capability tier:
    llama3:70b      -> closest to gpt-oss-120b quality (requires ~48 GB RAM)
    llama3:8b       -> faster, lower RAM (16 GB minimum)
    mistral:7b      -> good for code generation tasks
    deepseek-coder  -> specialized for engineer_node code output
    phi3:mini       -> ultra-fast, CPU-only capable
"""
import os
import httpx
from anomallm.llm.base import LLMProvider, LLMResponse

class OllamaProvider(LLMProvider):
    """
    Offline LLM provider using a locally-running Ollama instance.
    Communicates only with localhost -- safe for air-gapped environments.
    """
    BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    MODEL    = os.getenv("OLLAMA_MODEL",    "llama3:70b")

    @classmethod
    def is_available(cls) -> bool:
        try:
            resp = httpx.get(f"{cls.BASE_URL}/api/tags", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def complete(self, messages: list, temperature: float = 0.3, max_tokens: int = 4096) -> LLMResponse:
        prompt = self._messages_to_prompt(messages)
        resp = httpx.post(
            f"{self.BASE_URL}/api/generate",
            json={
                "model":   self.MODEL,
                "prompt":  prompt,
                "stream":  False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx":     8192,
                },
            },
            timeout=300.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return LLMResponse(
            content=data["response"],
            model=self.MODEL,
            provider="ollama",
            tokens_used=data.get("eval_count", 0),
        )

    async def acomplete(self, messages: list, temperature: float = 0.3, max_tokens: int = 4096) -> LLMResponse:
        prompt = self._messages_to_prompt(messages)
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{self.BASE_URL}/api/generate",
                json={
                    "model":   self.MODEL,
                    "prompt":  prompt,
                    "stream":  False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "num_ctx":     8192,
                    },
                },
            )
        resp.raise_for_status()
        data = resp.json()
        return LLMResponse(
            content=data["response"],
            model=self.MODEL,
            provider="ollama",
            tokens_used=data.get("eval_count", 0),
        )

    def _messages_to_prompt(self, messages: list[dict]) -> str:
        """
        Convert OpenAI-style messages to a single prompt string.
        Ollama's /api/generate takes a flat prompt, not messages.
        """
        parts = []
        for m in messages:
            role    = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "user":
                parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}")
        parts.append("<|assistant|>")
        return "\n".join(parts)
