"""
anomallm/backends/__init__.py
Execution backend factory — reads EXECUTION_BACKEND from env.
Falls back gracefully if the requested backend is unavailable.
"""
import os

from anomallm.backends.base       import ExecutionBackend
from anomallm.backends.subprocess import SubprocessBackend
from anomallm.backends.docker     import DockerBackend
from anomallm.backends.modal      import ModalBackend
from anomallm.backends.runpod     import RunPodBackend


def get_backend() -> ExecutionBackend:
    """
    Factory — reads EXECUTION_BACKEND from env.
    Falls back gracefully if backend unavailable.
    """
    backend = os.getenv("EXECUTION_BACKEND", "subprocess").lower()

    if backend == "docker":
        if not DockerBackend.is_available():
            print("[OmniML] Docker not found, falling back to subprocess",
                  flush=True)
            return SubprocessBackend()
        return DockerBackend()

    elif backend == "modal":
        if not os.getenv("MODAL_TOKEN_ID"):
            print("[OmniML] MODAL_TOKEN_ID not set, "
                  "falling back to subprocess", flush=True)
            return SubprocessBackend()
        return ModalBackend()

    elif backend == "runpod":
        if not os.getenv("RUNPOD_API_KEY"):
            print("[OmniML] RUNPOD_API_KEY not set, "
                  "falling back to subprocess", flush=True)
            return SubprocessBackend()
        return RunPodBackend()

    return SubprocessBackend()


__all__ = [
    "ExecutionBackend",
    "SubprocessBackend",
    "DockerBackend",
    "ModalBackend",
    "RunPodBackend",
    "get_backend",
]
