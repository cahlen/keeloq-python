"""End-to-end key-recovery pipeline: anf -> encode -> solve -> verify."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from keeloq.anf import BoolPoly, system
from keeloq.cipher import encrypt
from keeloq.encoders import SolverInstance
from keeloq.errors import VerificationError
from keeloq.solvers import SolveResult


@dataclass(frozen=True)
class AttackResult:
    recovered_key: int | None
    status: Literal["SUCCESS", "WRONG_KEY", "UNSAT", "TIMEOUT", "CRASH"]
    solve_result: SolveResult
    verify_result: bool
    encoder_used: str
    solver_used: str


EncodeFn = Callable[[list[BoolPoly]], SolverInstance]
SolveFn = Callable[[SolverInstance, float], SolveResult]


def attack(
    rounds: int,
    pairs: list[tuple[int, int]],
    key_hints: dict[int, int] | None,
    encoder: EncodeFn,
    solver_fn: SolveFn,
    timeout_s: float,
) -> AttackResult:
    sys = system(rounds=rounds, pairs=pairs, key_hints=key_hints)
    inst = encoder(sys)
    result = solver_fn(inst, timeout_s)

    encoder_name = getattr(encoder, "__module__", "unknown").rsplit(".", 1)[-1]
    solver_name = result.stats.solver_name

    if result.status == "UNSAT":
        return AttackResult(None, "UNSAT", result, False, encoder_name, solver_name)
    if result.status == "TIMEOUT":
        return AttackResult(None, "TIMEOUT", result, False, encoder_name, solver_name)

    assert result.status == "SAT" and result.assignment is not None
    recovered = _extract_key(result.assignment)

    # Verify by re-encrypting every pair.
    all_ok = True
    for pt, ct in pairs:
        if encrypt(pt, recovered, rounds) != ct:
            all_ok = False
            break

    status: Literal["SUCCESS", "WRONG_KEY"] = "SUCCESS" if all_ok else "WRONG_KEY"
    return AttackResult(recovered, status, result, all_ok, encoder_name, solver_name)


def _extract_key(assignment: dict[str, int]) -> int:
    """Extract the 64-bit key from the solver assignment.

    Key bits not present in the assignment are unconstrained free variables;
    the solver may omit them.  We arbitrarily fix them to 0, which is a valid
    choice in the SAT solution.  The caller's verify step confirms that the
    resulting key is consistent with every plaintext/ciphertext pair.
    """
    key = 0
    for i in range(64):
        bit = assignment.get(f"K{i}", 0)  # 0 is a valid value for a free variable
        if bit not in (0, 1):
            raise VerificationError(f"K{i}={bit} is not a bit")
        key = (key << 1) | bit
    return key
