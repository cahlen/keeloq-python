"""Hybrid neural-prefix + SAT-suffix attack orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from keeloq.attack import attack as sat_attack
from keeloq.cipher import encrypt
from keeloq.encoders.xor_aware import encode as encode_xor
from keeloq.neural.distinguisher import Distinguisher
from keeloq.neural.key_recovery import recover_prefix
from keeloq.solvers.cryptominisat import solve as solve_cms


@dataclass(frozen=True)
class HybridAttackResult:
    recovered_key: int | None
    status: Literal[
        "SUCCESS",
        "WRONG_KEY",
        "UNSAT",
        "TIMEOUT",
        "NEURAL_FAIL",
        "BACKTRACK_EXHAUSTED",
        "CRASH",
    ]
    bits_recovered_neurally: int
    neural_wall_time_s: float
    sat_wall_time_s: float
    verify_result: bool


def hybrid_attack(
    rounds: int,
    pairs: list[tuple[int, int]],
    distinguisher: Distinguisher,
    beam_width: int = 8,
    neural_target_bits: int | None = None,
    sat_timeout_s: float = 60.0,
    max_backtracks: int = 8,
    extra_key_hints: dict[int, int] | None = None,
    sat_pairs: list[tuple[int, int]] | None = None,
) -> HybridAttackResult:
    """Recover a key via neural prefix peeling + SAT suffix completion.

    Args:
        rounds: KeeLoq round count of the target cipher.
        pairs: list of (ciphertext_0, ciphertext_1) differential pairs for the
            neural distinguisher, where ciphertext_0 = encrypt(pt, key, rounds)
            and ciphertext_1 = encrypt(pt ^ delta, key, rounds). These are also
            used as SAT constraints (interpreted as plaintext, ciphertext) when
            ``sat_pairs`` is None.
        distinguisher: a trained Distinguisher (from Task 5's train()).
        beam_width: initial beam width for recover_prefix.
        neural_target_bits: how many rounds to peel neurally. Default =
            min(64, max(0, rounds - 32)).
        sat_timeout_s: SAT solver wall-clock timeout.
        max_backtracks: neural-prefix backtracks if SAT rejects the prefix.
        extra_key_hints: additional K-bit hints to pass to SAT (e.g., for
            bits outside the neurally-recoverable range at shallow attack
            depths where the cyclic key schedule leaves bits unconstrained).
        sat_pairs: genuine (plaintext, ciphertext) pairs for the SAT attack.
            When provided, SAT uses these instead of ``pairs``. Useful when the
            neural step needs differential ciphertext pairs but SAT needs real
            plaintext/ciphertext constraints.
    """
    if neural_target_bits is None:
        neural_target_bits = min(64, max(0, rounds - 32))

    # The SAT constraint set: use explicit sat_pairs if given, else fall back
    # to pairs (which must then be genuine (plaintext, ciphertext) pairs).
    constraint_pairs = sat_pairs if sat_pairs is not None else pairs

    t0 = time.perf_counter()
    rec = recover_prefix(
        pairs=pairs,
        distinguisher=distinguisher,
        starting_rounds=rounds,
        max_bits_to_recover=neural_target_bits,
        beam_width=beam_width,
    )
    neural_wall = time.perf_counter() - t0

    hints = dict(rec.recovered_bits)
    if extra_key_hints:
        hints.update(extra_key_hints)
    neural_bit_order = sorted(rec.recovered_bits.items(), key=lambda kv: kv[0])
    flips_used = 0
    sat_wall_total = 0.0

    def _run_sat(
        hint_map: dict[int, int],
    ) -> tuple[str, int | None, float]:
        t_sat = time.perf_counter()
        r = sat_attack(
            rounds=rounds,
            pairs=constraint_pairs,
            key_hints=hint_map or None,
            encoder=encode_xor,
            solver_fn=solve_cms,
            timeout_s=sat_timeout_s,
        )
        return r.status, r.recovered_key, time.perf_counter() - t_sat

    status, recovered_key, sat_wall = _run_sat(hints)
    sat_wall_total += sat_wall

    while (
        status in ("UNSAT", "WRONG_KEY")
        and flips_used < max_backtracks
        and flips_used < len(neural_bit_order)
    ):
        # Flip the most-recently-peeled neural bit.
        flip_target = neural_bit_order[-(flips_used + 1)]
        idx, val = flip_target
        flipped_hints = dict(hints)
        flipped_hints[idx] = 1 - val
        flips_used += 1
        status, recovered_key, sat_wall = _run_sat(flipped_hints)
        sat_wall_total += sat_wall
        if status == "SUCCESS":
            # Commit the flip to the working hints so further backtracks
            # don't undo it.
            hints = flipped_hints
            break
        # Conservative fallback: drop the bit entirely and retry without it.
        del flipped_hints[idx]
        status, recovered_key, sat_wall = _run_sat(flipped_hints)
        sat_wall_total += sat_wall
        if status == "SUCCESS":
            hints = flipped_hints
            break

    # Mandatory cipher-verify for any claimed SUCCESS.
    verify_ok = False
    if status == "SUCCESS" and recovered_key is not None:
        verify_ok = all(encrypt(p, recovered_key, rounds) == c for p, c in constraint_pairs)
        if not verify_ok:
            status = "WRONG_KEY"

    final_status: Literal[
        "SUCCESS",
        "WRONG_KEY",
        "UNSAT",
        "TIMEOUT",
        "NEURAL_FAIL",
        "BACKTRACK_EXHAUSTED",
        "CRASH",
    ]
    if status in ("UNSAT", "WRONG_KEY") and flips_used >= max_backtracks:
        final_status = "BACKTRACK_EXHAUSTED"
    else:
        final_status = status  # type: ignore[assignment]

    return HybridAttackResult(
        recovered_key=recovered_key if final_status == "SUCCESS" else None,
        status=final_status,
        bits_recovered_neurally=len(rec.recovered_bits),
        neural_wall_time_s=neural_wall,
        sat_wall_time_s=sat_wall_total,
        verify_result=verify_ok,
    )
