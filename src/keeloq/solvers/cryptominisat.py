"""CryptoMiniSat wrapper using pycryptosat's native API.

Accepts both CNFInstance and HybridInstance. For hybrid, routes XOR clauses
to add_xor_clause() — the whole reason we picked this solver.
"""
from __future__ import annotations

import time

from pycryptosat import Solver as _CMSSolver

from keeloq.encoders import CNFInstance, HybridInstance, SolverInstance
from keeloq.errors import SolverError
from keeloq.solvers import SolveResult, SolverStats


def solve(instance: SolverInstance, timeout_s: float) -> SolveResult:
    s = _CMSSolver()

    if isinstance(instance, CNFInstance):
        for clause in instance.clauses:
            s.add_clause(list(clause))
        num_clauses = len(instance.clauses)
        num_xors = 0
    elif isinstance(instance, HybridInstance):
        for clause in instance.cnf_clauses:
            s.add_clause(list(clause))
        for lits, rhs in instance.xor_clauses:
            s.add_xor_clause(list(lits), bool(rhs))
        num_clauses = len(instance.cnf_clauses)
        num_xors = len(instance.xor_clauses)
    else:
        raise SolverError(f"unknown SolverInstance type: {type(instance).__name__}")

    t0 = time.perf_counter()
    sat, raw_assignment = s.solve(time_limit=timeout_s)
    elapsed = time.perf_counter() - t0

    stats = SolverStats(
        wall_time_s=elapsed,
        num_vars=instance.num_vars,
        num_clauses=num_clauses,
        num_xors=num_xors,
        solver_name="cryptominisat",
    )

    if sat is None:
        return SolveResult(status="TIMEOUT", assignment=None, stats=stats)
    if sat is False:
        return SolveResult(status="UNSAT", assignment=None, stats=stats)

    # raw_assignment is indexed from 1; index 0 is None.
    assignment: dict[str, int] = {}
    for i, name in enumerate(instance.var_names):
        v = raw_assignment[i + 1]
        assignment[name] = 1 if v else 0
    return SolveResult(status="SAT", assignment=assignment, stats=stats)
