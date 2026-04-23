"""Helpers for invoking the frozen legacy/ scripts under Python 2 via Docker.

Python 2 reached EOL in 2020 and should not be installed on modern hosts. We run
the legacy scripts inside an ephemeral `python:2.7` Docker container mounting
`legacy/` read-only. The host needs docker + the `python:2.7` image; the
`legacy_runtime_available()` check covers both.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

LEGACY_DIR = Path(__file__).resolve().parent.parent / "legacy"
PYTHON2_IMAGE = "python:2.7"


def legacy_runtime_available() -> bool:
    """True iff docker is installed AND the python:2.7 image is pulled."""
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", PYTHON2_IMAGE],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return proc.returncode == 0


def run_legacy_script(script_name: str) -> str:
    """Run a legacy script under `python:2.7` in docker and return its stdout.

    Mounts `legacy/` read-only into the container at /work. Any file the script
    writes lands in /work (shared with the host) — tests should use a temp dir
    if they need to capture generated files.
    """
    path = LEGACY_DIR / script_name
    if not path.exists():
        raise FileNotFoundError(f"legacy script not found: {path}")
    proc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{LEGACY_DIR}:/work:ro",
            "-w",
            "/work",
            PYTHON2_IMAGE,
            "python",
            script_name,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def run_legacy_script_in_cwd(script_name: str, cwd: Path) -> str:
    """Run a legacy script with `cwd` mounted read-write at /work.

    Use this when the script writes output files the test needs to read back
    (e.g., sage-equations.py writes anf.txt to its working directory). The
    script file itself is copied into `cwd` first.
    """
    src = LEGACY_DIR / script_name
    if not src.exists():
        raise FileNotFoundError(f"legacy script not found: {src}")
    dest = cwd / script_name
    dest.write_bytes(src.read_bytes())
    proc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{cwd}:/work",
            "-w",
            "/work",
            PYTHON2_IMAGE,
            "python",
            script_name,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def read_legacy_output_field(stdout: str, prefix: str) -> str:
    """Extract the value following a 'Prefix: ' line in legacy stdout."""
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    raise ValueError(f"no line starting with {prefix!r} in:\n{stdout}")
