"""Protocol for SAT solvers and shared result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from keeloq.encoders import SolverInstance


@dataclass(frozen=True)
class SolverStats:
    wall_time_s: float
    num_vars: int
    num_clauses: int
    num_xors: int
    restarts: int | None = None
    conflicts: int | None = None
    propagations: int | None = None
    solver_name: str = "unknown"


@dataclass(frozen=True)
class SolveResult:
    status: Literal["SAT", "UNSAT", "TIMEOUT"]
    assignment: dict[str, int] | None
    stats: SolverStats


class Solver(Protocol):
    def solve(self, instance: SolverInstance, timeout_s: float) -> SolveResult: ...
