"""Protocol for ANF -> SAT instance encoders.

Two concrete encoders live alongside this module:
  - encoders.cnf: pure DIMACS CNF, PolyBoRi-equivalent. Works with any SAT solver.
  - encoders.xor_aware: CryptoMiniSat-native hybrid (CNF + XOR clauses). Dramatic
    speedup on crypto problems because linear equations become one XOR constraint
    each instead of 2^(n-1) CNF clauses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from keeloq.anf import BoolPoly


@dataclass(frozen=True)
class CNFInstance:
    num_vars: int
    clauses: tuple[tuple[int, ...], ...]
    var_names: tuple[str, ...]  # index i -> variable name; DIMACS var ids are 1-indexed


@dataclass(frozen=True)
class HybridInstance:
    """CNF clauses plus native XOR constraints."""

    num_vars: int
    cnf_clauses: tuple[tuple[int, ...], ...]
    xor_clauses: tuple[tuple[tuple[int, ...], int], ...]  # (vars_1indexed, rhs)
    var_names: tuple[str, ...]


type SolverInstance = CNFInstance | HybridInstance


class Encoder(Protocol):
    def encode(self, system: list[BoolPoly]) -> SolverInstance: ...
