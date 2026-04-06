"""
anomallm/llm/base.py
Abstract base class + response dataclass for all LLM providers.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    content:     str
    model:       str
    provider:    str
    tokens_used: int = field(default=0)


class LLMProvider(ABC):
    """
    Common interface for all LLM backends.
    Implement both sync (complete) and async (acomplete) variants.
    """

    @abstractmethod
    def complete(
        self,
        messages:    list,
        temperature: float = 0.3,
        max_tokens:  int   = 4096,
    ) -> LLMResponse:
        """Synchronous completion."""
        ...

    @abstractmethod
    async def acomplete(
        self,
        messages:    list,
        temperature: float = 0.3,
        max_tokens:  int   = 4096,
    ) -> LLMResponse:
        """Asynchronous completion."""
        ...

    @classmethod
    def is_available(cls) -> bool:
        """Return True if this provider can be used in the current environment."""
        return True
