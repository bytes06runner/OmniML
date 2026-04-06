"""
anomallm/backends/runpod.py
RunPod serverless GPU backend for OmniML.

Requires:
    pip install runpod
Env vars:
    RUNPOD_API_KEY      — RunPod API key
    RUNPOD_ENDPOINT_ID  — Serverless endpoint ID
"""
import asyncio
import json
import os
from typing import AsyncGenerator

from anomallm.backends.base import ExecutionBackend


class RunPodBackend(ExecutionBackend):
    """
    Executes training on RunPod serverless GPU endpoints.
    Requires: pip install runpod
    """

    API_KEY     = os.getenv("RUNPOD_API_KEY",     "")
    ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")

    async def run(
        self,
        script: str,
        working_dir: str,
        timeout_seconds: int = 600,
    ) -> AsyncGenerator[str, None]:
        try:
            import runpod
            runpod.api_key = self.API_KEY
        except ImportError:
            yield json.dumps({
                "type":    "error",
                "message": "runpod package not installed. Run: pip install runpod",
            })
            return

        yield json.dumps({
            "type":    "log",
            "message": "Submitting job to RunPod serverless GPU...",
        })

        endpoint    = runpod.Endpoint(self.ENDPOINT_ID)
        run_request = endpoint.run({
            "script":  script,
            "timeout": timeout_seconds,
        })

        # Poll until completion or timeout
        poll_interval = 5
        elapsed       = 0

        while elapsed < timeout_seconds:
            status = run_request.status()

            if status == "COMPLETED":
                output = run_request.output()
                for line in output.get("logs", []):
                    if line:
                        yield line
                yield json.dumps({"type": "complete", "success": True})
                return

            elif status == "FAILED":
                yield json.dumps({
                    "type":    "error",
                    "message": "RunPod job failed",
                })
                return

            else:
                yield json.dumps({
                    "type":    "log",
                    "message": f"RunPod status: {status} ({elapsed}s elapsed)",
                })

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        yield json.dumps({
            "type":    "error",
            "message": f"RunPod job timed out after {timeout_seconds}s",
        })
