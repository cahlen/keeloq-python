"""Helpers for invoking the frozen legacy/ scripts under python2."""
from __future__ import annotations

import subprocess
from pathlib import Path

LEGACY_DIR = Path(__file__).resolve().parent.parent / "legacy"


def run_legacy_script(script_name: str) -> str:
    """Run a legacy script under python2 and return its stdout."""
    path = LEGACY_DIR / script_name
    if not path.exists():
        raise FileNotFoundError(f"legacy script not found: {path}")
    proc = subprocess.run(
        ["python2", str(path)],
        capture_output=True, text=True, check=True,
        cwd=LEGACY_DIR,
    )
    return proc.stdout


def read_legacy_output_field(stdout: str, prefix: str) -> str:
    """Extract the value following a 'Prefix: ' line in legacy stdout."""
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    raise ValueError(f"no line starting with {prefix!r} in:\n{stdout}")
