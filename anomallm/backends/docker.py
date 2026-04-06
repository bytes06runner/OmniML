"""
anomallm/backends/docker.py
Docker-based execution sandbox for enterprise-grade isolation.

Security properties:
  - No sensitive host directory mounts (script injected via a temp volume)
  - Network disabled (--network none) unless DOCKER_NETWORK_MODE is overridden
  - Read-only root filesystem with /tmp writeable via --tmpfs
  - Container auto-removed on exit (--rm)
  - Resource limits: CPU and memory capped via env vars
  - Non-root user inside container (USER 1000:1000)
  - No privilege escalation (--security-opt no-new-privileges)
"""
import asyncio
import json
import os
import shutil
import tempfile
import uuid
from typing import AsyncGenerator

from anomallm.backends.base import ExecutionBackend

DOCKER_IMAGE = os.getenv("OMNIML_DOCKER_IMAGE", "omniml-sandbox:latest")


class DockerBackend(ExecutionBackend):
    """
    Runs training scripts in an ephemeral Docker container.
    Build the image first:
        docker build -t omniml-sandbox:latest ./docker/sandbox/
    """

    CPU_LIMIT    = os.getenv("DOCKER_CPU_LIMIT",       "2.0")
    MEMORY_LIMIT = os.getenv("DOCKER_MEMORY_LIMIT",    "4g")
    NETWORK      = os.getenv("DOCKER_NETWORK_MODE",    "none")

    @classmethod
    def is_available(cls) -> bool:
        return shutil.which("docker") is not None

    async def run(
        self,
        script: str,
        working_dir: str,
        timeout_seconds: int = 600,
    ) -> AsyncGenerator[str, None]:

        container_name = f"omniml-{uuid.uuid4().hex[:12]}"

        # Write script + data into a dedicated temp dir that we mount read-only
        tmp_dir     = tempfile.mkdtemp(prefix="omniml_docker_")
        script_path = os.path.join(tmp_dir, "train.py")

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)

        # Copy dataset files so the container can read them without a
        # wide host mount of sensitive directories
        if os.path.exists(working_dir):
            data_dst = os.path.join(tmp_dir, "data")
            shutil.copytree(working_dir, data_dst, dirs_exist_ok=True)

        cmd = [
            "docker", "run",
            "--name",    container_name,
            "--rm",                              # auto-remove on exit
            "--network", self.NETWORK,           # isolated network
            "--cpus",    self.CPU_LIMIT,
            "--memory",  self.MEMORY_LIMIT,
            "--read-only",                       # read-only root filesystem
            "--tmpfs",   "/tmp:size=512m",       # writable /tmp only
            "--security-opt", "no-new-privileges",
            "--user",    "1000:1000",            # non-root
            "-v", f"{tmp_dir}:/workspace:ro",   # read-only script mount
            "-w", "/workspace",
            DOCKER_IMAGE,
            "python", "train.py",
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            try:
                async for line in proc.stdout:
                    decoded = line.decode("utf-8", errors="replace").strip()
                    if decoded:
                        yield decoded

                await asyncio.wait_for(proc.wait(), timeout=timeout_seconds)

            except asyncio.TimeoutError:
                # Kill container gracefully on timeout
                kill_proc = await asyncio.create_subprocess_exec(
                    "docker", "kill", container_name,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await kill_proc.wait()
                yield json.dumps({
                    "type":    "error",
                    "message": f"Container timed out after {timeout_seconds}s",
                })
                return

            if proc.returncode == 0:
                yield json.dumps({"type": "complete", "success": True})
            else:
                yield json.dumps({
                    "type":    "error",
                    "message": f"Container exited with code {proc.returncode}",
                })

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
