"""
anomallm/backends/modal.py
Modal serverless GPU backend for OmniML.

Requires:
    pip install modal
Env vars:
    MODAL_TOKEN_ID      — Modal token ID
    MODAL_TOKEN_SECRET  — Modal token secret
    MODAL_GPU_TYPE      — GPU type (default: T4)
    MODAL_IMAGE         — Base Docker image for Modal function
"""
import json
import os
from typing import AsyncGenerator

from anomallm.backends.base import ExecutionBackend


class ModalBackend(ExecutionBackend):
    """
    Executes training on Modal serverless GPU infrastructure.
    Requires: pip install modal
    """

    GPU_TYPE   = os.getenv("MODAL_GPU_TYPE", "T4")
    IMAGE_NAME = os.getenv(
        "MODAL_IMAGE",
        "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime",
    )

    async def run(
        self,
        script: str,
        working_dir: str,
        timeout_seconds: int = 600,
    ) -> AsyncGenerator[str, None]:
        try:
            import modal
        except ImportError:
            yield json.dumps({
                "type":    "error",
                "message": "modal package not installed. Run: pip install modal",
            })
            return

        yield json.dumps({
            "type":    "log",
            "message": f"Submitting to Modal ({self.GPU_TYPE} GPU)...",
        })

        # Define Modal function dynamically
        stub  = modal.Stub("omniml-training")
        image = modal.Image.from_registry(self.IMAGE_NAME).pip_install(
            "optuna", "scikit-learn", "pandas", "numpy", "xgboost",
        )

        @stub.function(
            image=image,
            gpu=self.GPU_TYPE,
            timeout=timeout_seconds,
        )
        def run_training(script_content: str) -> list:
            import subprocess
            import sys
            import tempfile
            import os as _os

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8",
            ) as f:
                f.write(script_content)
                path = f.name

            result = subprocess.run(
                [sys.executable, path],
                capture_output=True, text=True,
            )
            _os.unlink(path)
            return result.stdout.splitlines()

        with stub.run():
            lines = run_training.call(script)
            for line in lines:
                if line:
                    yield line

        yield json.dumps({"type": "complete", "success": True})
