"""Runs code on untrusted files (uploaded documents, scans) in a bubblewrap sandbox.

Parsers for PDF, EPUB and DOCX are big attack surfaces, and an uploaded file is fully under the sender's
control. So parsing never happens in the bot's process: a child process gets its own namespaces, no network,
no environment, and only what is mounted here. Notably absent: the repository (so no `.env`), the bot's data
directory and every login. Only the `summarizer` package is mounted, read-only, so `python -m
summarizer.<module>` works inside.
"""
import os
import sys
from pathlib import Path

from . import config

APP = "/app"  # where the summarizer package appears inside the sandbox
MEMORY_LIMIT = 6 * 1024 ** 3  # address-space cap; generous because onnxruntime reserves big arenas


def inside(path: Path, job: Path) -> str:
    """A path in the job's directory as the sandbox sees it (/job/...)."""
    return f"/job/{Path(path).relative_to(job)}"


def command(job: Path, args: list[str], *, ro_binds: dict[Path, str] | None = None,
            memory: int | None = MEMORY_LIMIT, env: dict[str, str] | None = None, gpu: bool = False) -> list[str]:
    """Builds the full command that runs `args` sandboxed, with `job` as the only writable directory.

    Args:
        job: The job's work directory, mounted read-write at /job (also the working directory).
        args: The command to run inside, e.g. ["python", "-m", "summarizer.docparse", ...]. A first
            element "python" is replaced by the bot's interpreter.
        ro_binds: Extra host paths to mount read-only, mapped to their path inside (e.g. OCR models).
        memory: Address-space limit in bytes (prlimit), so a decompression bomb can't take the machine; None
            for no limit.
        env: Extra environment variables inside (the environment is otherwise empty).
        gpu: Give the process the NVIDIA GPU (its /dev/nvidia* devices). No address-space limit then: CUDA
            reserves far more address space than it uses, and any cap breaks it.

    Returns:
        The command for proc.run.
    """
    # Normalised but not resolved: resolving would follow the venv's python symlink to /usr/bin/python3,
    # which then wouldn't find the venv's packages.
    if args and args[0] == "python":
        args = [os.path.normpath(sys.executable), *args[1:]]
    venv = Path(os.path.normpath(sys.prefix))  # the interpreter finds its packages via this path
    cmd = [
        # Every namespace unshared, the network included; die with the bot so a killed job leaves no orphan.
        "bwrap", "--unshare-all", "--die-with-parent", "--new-session",
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
        "--ro-bind-try", "/etc/alternatives", "/etc/alternatives",  # some /usr/bin tools are symlinks here
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--ro-bind", str(venv), str(venv),
        "--ro-bind", str(config.ROOT / "summarizer"), f"{APP}/summarizer",
        "--bind", str(job), "/job",
    ]
    for host, inside in (ro_binds or {}).items():
        cmd += ["--ro-bind", str(host), inside]
    if gpu:
        for device in sorted(Path("/dev").glob("nvidia*")):
            cmd += ["--dev-bind", str(device), str(device)]
        cmd += ["--ro-bind-try", "/sys", "/sys"]  # CUDA reads the PCI topology from here
        memory = None
    cmd += [
        "--chdir", "/job",
        # No inherited environment: no tokens or keys from the bot's process.
        "--clearenv", "--setenv", "PATH", "/usr/bin", "--setenv", "HOME", "/tmp",
        "--setenv", "LANG", "C.UTF-8", "--setenv", "PYTHONPATH", APP,
        "--setenv", "PYTHONDONTWRITEBYTECODE", "1",  # the package is read-only
        "--setenv", "OMP_THREAD_LIMIT", "1",  # Tesseract: one thread per process; we choose the parallelism
    ]
    for key, value in (env or {}).items():
        cmd += ["--setenv", key, value]
    if memory:
        cmd += ["prlimit", f"--as={memory}", "--"]
    return cmd + args
