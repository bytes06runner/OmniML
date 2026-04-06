"""
anomallm/backends/subprocess.py
Default execution backend — runs training scripts in a local subprocess.
This is the original execution logic extracted from graph.py, unchanged.
"""
import asyncio
import json
import os
import sys
import tempfile
from typing import AsyncGenerator

from anomallm.backends.base import ExecutionBackend


class SubprocessBackend(ExecutionBackend):
    """
    Runs training scripts as a local subprocess using the current Python
    interpreter. This is the default backend and requires no extra
    infrastructure.
    """

    async def run(
        self,
        script: str,
        working_dir: str,
        timeout_seconds: int = 600,
    ) -> AsyncGenerator[str, None]:

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            dir=working_dir,
            encoding="utf-8",
        ) as f:
            f.write(script)
            script_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=working_dir,
            )

            async for line in proc.stdout:
                decoded = line.decode("utf-8", errors="replace").strip()
                if decoded:
                    yield decoded

            await proc.wait()

            if proc.returncode == 0:
                yield json.dumps({"type": "complete", "success": True})
            else:
                yield json.dumps({
                    "type": "error",
                    "message": f"Process exited with code {proc.returncode}",
                })
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass
