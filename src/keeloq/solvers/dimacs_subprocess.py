"""Shell-out wrapper for external DIMACS-speaking SAT solvers (kissat, minisat).

Only accepts CNFInstance — external solvers don't understand our HybridInstance
XOR clauses. If you want XOR, use solvers.cryptominisat.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from keeloq.encoders import CNFInstance, SolverInstance
from keeloq.encoders.cnf import to_dimacs
from keeloq.errors import SolverError
from keeloq.solvers import SolveResult, SolverStats


def solve(instance: SolverInstance, solver_binary: str, timeout_s: float) -> SolveResult:
    if not isinstance(instance, CNFInstance):
        raise SolverError(
            "DIMACS subprocess solvers only accept CNFInstance "
            "(HybridInstance requires a native XOR-capable solver)"
        )

    binary_path = shutil.which(solver_binary) or solver_binary
    if not Path(binary_path).exists():
        raise SolverError(f"solver binary not found: {solver_binary!r}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".cnf", delete=False) as f:
        f.write(to_dimacs(instance))
        cnf_path = f.name

    cmd = [binary_path, cnf_path]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        return SolveResult(
            status="TIMEOUT",
            assignment=None,
            stats=SolverStats(
                wall_time_s=elapsed,
                num_vars=instance.num_vars,
                num_clauses=len(instance.clauses),
                num_xors=0,
                solver_name=Path(binary_path).name,
            ),
        )
    finally:
        Path(cnf_path).unlink(missing_ok=True)
    elapsed = time.perf_counter() - t0

    output = proc.stdout
    status, assignment_lits = _parse_dimacs_output(output)

    stats = SolverStats(
        wall_time_s=elapsed,
        num_vars=instance.num_vars,
        num_clauses=len(instance.clauses),
        num_xors=0,
        solver_name=Path(binary_path).name,
    )

    if status == "UNSAT":
        return SolveResult(status="UNSAT", assignment=None, stats=stats)
    if status == "UNKNOWN":
        # treat as timeout; solver ran but didn't decide in time
        return SolveResult(status="TIMEOUT", assignment=None, stats=stats)

    if assignment_lits is None:
        raise SolverError(
            f"solver {solver_binary!r} reported SAT but emitted "
            f"no v-line. stdout:\n{output}\nstderr:\n{proc.stderr}"
        )

    assignment: dict[str, int] = {}
    lit_set = set(assignment_lits)
    for i, name in enumerate(instance.var_names):
        vid = i + 1
        if vid in lit_set:
            assignment[name] = 1
        elif -vid in lit_set:
            assignment[name] = 0
        else:
            assignment[name] = 0  # unconstrained; default 0
    return SolveResult(status="SAT", assignment=assignment, stats=stats)


def _parse_dimacs_output(text: str) -> tuple[str, list[int] | None]:
    """Parse DIMACS-style solver output.

    Returns (status, assignment_literals).
    status in {"SAT", "UNSAT", "UNKNOWN"}.
    """
    status = "UNKNOWN"
    v_lits: list[int] = []
    saw_v = False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("s "):
            tok = line.split()[1] if len(line.split()) > 1 else ""
            if tok == "SATISFIABLE":
                status = "SAT"
            elif tok == "UNSATISFIABLE":
                status = "UNSAT"
            elif tok == "UNKNOWN":
                status = "UNKNOWN"
        elif line.startswith("v "):
            saw_v = True
            for tok in line.split()[1:]:
                if tok == "0":
                    continue
                v_lits.append(int(tok))
    return status, v_lits if saw_v else None
