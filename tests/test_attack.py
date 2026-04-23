"""Tests for the full attack pipeline."""
from __future__ import annotations

import pytest

from keeloq.attack import AttackResult, attack
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
