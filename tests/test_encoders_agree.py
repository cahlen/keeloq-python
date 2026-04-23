"""The TDD crown jewel: CNF and XOR-aware encoders must recover the same key."""

from __future__ import annotations

import pytest

from keeloq.attack import attack
from keeloq.cipher import encrypt
from keeloq.encoders.cnf import encode as encode_cnf
from keeloq.encoders.xor_aware import encode as encode_xor
from keeloq.solvers.cryptominisat import solve as solve_cms

KEY = 0x0123_4567_89AB_CDEF


@pytest.mark.parametrize(
    "rounds,hint_bits,num_pairs",
    [
        (16, 48, 1),
        (32, 32, 1),
        (
            64,
            0,
            4,
        ),  # corrected from (32, 0, 2): 32-round cycles only hit K0-K31; 64r+4pairs is well-determined
    ],
)
def test_cnf_and_xor_encoders_agree(rounds: int, hint_bits: int, num_pairs: int) -> None:
    pts = [0xAAAA5555, 0x13579BDF, 0xCAFEBABE, 0xFEEDFACE][:num_pairs]
    pairs = [(p, encrypt(p, KEY, rounds)) for p in pts]
    hints = {i: (KEY >> (63 - i)) & 1 for i in range(64 - hint_bits, 64)} or None

    r_cnf = attack(
        rounds=rounds,
        pairs=pairs,
        key_hints=hints,
        encoder=encode_cnf,
        solver_fn=solve_cms,
        timeout_s=120.0,
    )
    r_xor = attack(
        rounds=rounds,
        pairs=pairs,
        key_hints=hints,
        encoder=encode_xor,
        solver_fn=solve_cms,
        timeout_s=120.0,
    )

    assert r_cnf.status == "SUCCESS"
    assert r_xor.status == "SUCCESS"
    assert r_cnf.recovered_key == r_xor.recovered_key == KEY
