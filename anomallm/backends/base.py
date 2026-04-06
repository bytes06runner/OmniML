"""
anomallm/backends/base.py
Abstract base class for all execution backends.
"""
from abc import ABC, abstractmethod
from typing import AsyncGenerator


class ExecutionBackend(ABC):

    @abstractmethod
    async def run(
        self,
        script: str,
        working_dir: str,
        timeout_seconds: int = 600,
    ) -> AsyncGenerator[str, None]:
        """
        Execute a Python training script.
        Yields JSON-formatted log lines matching the OmniML stdout protocol:
          {"type":"epoch_metric","epoch":N,"loss":X,...}
          {"type":"hpt_trial","trial":N,...}
          {"type":"hpt_complete","best_params":{...},"best_value":X,...}
          {"type":"log","message":"..."}
          {"type":"error","message":"..."}
          {"type":"complete","success":true}
        """
        ...
        # Required for abstract async generator — yield never actually reached.
        yield  # type: ignore[misc]

    @classmethod
    def is_available(cls) -> bool:
        """Return True if this backend can be used in the current environment."""
        return True
