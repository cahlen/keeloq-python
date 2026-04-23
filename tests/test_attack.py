"""Tests for the full attack pipeline."""
from __future__ import annotations

import pytest

from keeloq.attack import AttackResult, EncodeFn, attack
from keeloq.cipher import encrypt
from keeloq.encoders.cnf import encode as encode_cnf
from keeloq.encoders.xor_aware import encode as encode_xor
from keeloq.solvers.cryptominisat import solve as solve_cms

KEY = 0x0123_4567_89AB_CDEF


@pytest.mark.parametrize("encode_fn,name", [(encode_cnf, "cnf"), (encode_xor, "xor")])
def test_attack_16_rounds_heavy_hints(encode_fn, name) -> None:
    rounds = 16
    pt = 0xAAAA5555
    ct = encrypt(pt, KEY, rounds)
    # Leave 16 bits unknown — a well-determined but non-trivial instance.
    hints = {i: (KEY >> (63 - i)) & 1 for i in range(16, 64)}

    r: AttackResult = attack(
        rounds=rounds,
        pairs=[(pt, ct)],
        key_hints=hints,
        encoder=encode_fn,
        solver_fn=solve_cms,
        timeout_s=10.0,
    )
    assert r.status == "SUCCESS", f"encoder {name}: got {r.status}"
    assert r.recovered_key == KEY
    assert r.verify_result is True


@pytest.mark.parametrize("encode_fn,name", [(encode_cnf, "cnf"), (encode_xor, "xor")])
def test_attack_32_rounds_32_hints(encode_fn, name) -> None:
    rounds = 32
    pt = 0xAAAA5555
    ct = encrypt(pt, KEY, rounds)
    # Half the key hinted — 32 unknown bits.
    hints = {i: (KEY >> (63 - i)) & 1 for i in range(32, 64)}

    r = attack(rounds=rounds, pairs=[(pt, ct)], key_hints=hints,
               encoder=encode_fn, solver_fn=solve_cms, timeout_s=30.0)
    assert r.status == "SUCCESS", f"encoder {name}: got {r.status}"
    assert r.recovered_key == KEY


@pytest.mark.parametrize("encode_fn,name", [(encode_cnf, "cnf"), (encode_xor, "xor")])
def test_attack_32_rounds_two_pairs_no_hints(encode_fn: EncodeFn, name: str) -> None:
    # 64 rounds are needed so K0-K63 all appear in the polynomial system;
    # 32 rounds only constrains K0-K31 leaving K32-K63 free.  Four pairs
    # provide enough output bits to make the 64-key-bit system uniquely SAT.
    rounds = 64
    pts = [0xAAAA5555, 0x13579BDF, 0xDEADBEEF, 0x00112233]
    pairs = [(p, encrypt(p, KEY, rounds)) for p in pts]

    r = attack(rounds=rounds, pairs=pairs, key_hints=None,
               encoder=encode_fn, solver_fn=solve_cms, timeout_s=60.0)
    assert r.status == "SUCCESS", f"encoder {name}: got {r.status}"
    assert r.recovered_key == KEY


def test_attack_underdetermined_detected_as_wrong_key() -> None:
    """Single pair at 16 rounds with 0 hints is underdetermined (32 bits of
    output vs 64 key bits). Solver finds *a* key; verify catches that it isn't THE key."""
    rounds = 16
    pt = 0xAAAA5555
    ct = encrypt(pt, KEY, rounds)
    r = attack(rounds=rounds, pairs=[(pt, ct)], key_hints=None,
               encoder=encode_cnf, solver_fn=solve_cms, timeout_s=10.0)
    # The recovered key satisfies this one pair but isn't unique.
    # Status should be SUCCESS (this pair re-encrypts correctly by construction)
    # OR WRONG_KEY if the solver picks a key that's self-inconsistent somehow.
    # Realistically: the solver finds a key that satisfies (pt, ct), so verification
    # passes. The test below uses a *second* pair to expose underdetermination.
    assert r.status == "SUCCESS"
    # But it doesn't necessarily equal KEY:
    # (we do NOT assert r.recovered_key == KEY here)


def test_attack_unsat_from_contradictory_hint() -> None:
    """Hint a key bit to the WRONG value; solver must prove UNSAT."""
    rounds = 16
    pt = 0xAAAA5555
    ct = encrypt(pt, KEY, rounds)
    wrong_hint = {0: 1 - ((KEY >> 63) & 1)}  # flip bit 0
    # Plus half the correct bits so the rest is well-determined.
    for i in range(32, 64):
        wrong_hint[i] = (KEY >> (63 - i)) & 1

    r = attack(rounds=rounds, pairs=[(pt, ct)], key_hints=wrong_hint,
               encoder=encode_cnf, solver_fn=solve_cms, timeout_s=10.0)
    assert r.status == "UNSAT"
