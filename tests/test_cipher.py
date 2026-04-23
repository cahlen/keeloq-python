"""Unit tests for keeloq.cipher."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from keeloq._types import bits_to_int
from keeloq.cipher import core, decrypt, encrypt


def _reference_core(a: int, b: int, c: int, d: int, e: int) -> int:
    """Spec definition of the KeeLoq NLF in ANF, verbatim from legacy code."""
    return (
        d
        + e
        + a * c
        + a * e
        + b * c
        + b * e
        + c * d
        + d * e
        + a * d * e
        + a * c * e
        + a * b * d
        + a * b * c
    ) % 2


@pytest.mark.parametrize("a", [0, 1])
@pytest.mark.parametrize("b", [0, 1])
@pytest.mark.parametrize("c", [0, 1])
@pytest.mark.parametrize("d", [0, 1])
@pytest.mark.parametrize("e", [0, 1])
def test_core_truth_table(a: int, b: int, c: int, d: int, e: int) -> None:
    assert core(a, b, c, d, e) == _reference_core(a, b, c, d, e)


def test_core_rejects_non_bit_inputs() -> None:
    with pytest.raises(ValueError):
        core(2, 0, 0, 0, 0)


# README 160-round KAT from legacy/keeloq160-python.py
PT_160 = bits_to_int("01100010100101110000101011100011")
KEY_160 = bits_to_int("0011010011011111100101100001110000011101100111001000001101110100")
ROUNDS_160 = 160


# 528-round KAT from legacy/keeloq-python.py
PT_528 = bits_to_int("01010101010101010101010101010101")
KEY_528 = bits_to_int("0000010000100010100011100000000010000110000011001001111000010001")
ROUNDS_528 = 528


@pytest.mark.parametrize(
    "pt,key,rounds",
    [
        (PT_160, KEY_160, ROUNDS_160),
        (PT_528, KEY_528, ROUNDS_528),
        (0, 0, 0),
        (0xDEADBEEF, 0x0123_4567_89AB_CDEF, 1),
        (0xDEADBEEF, 0x0123_4567_89AB_CDEF, 2),
        (0xDEADBEEF, 0x0123_4567_89AB_CDEF, 32),
    ],
)
def test_encrypt_decrypt_roundtrip(pt: int, key: int, rounds: int) -> None:
    ct = encrypt(pt, key, rounds)
    assert decrypt(ct, key, rounds) == pt


def test_zero_rounds_is_identity() -> None:
    assert encrypt(0xDEADBEEF, 0x0123_4567_89AB_CDEF, 0) == 0xDEADBEEF
    assert decrypt(0xDEADBEEF, 0x0123_4567_89AB_CDEF, 0) == 0xDEADBEEF


def test_encrypt_is_deterministic() -> None:
    a = encrypt(PT_160, KEY_160, ROUNDS_160)
    b = encrypt(PT_160, KEY_160, ROUNDS_160)
    assert a == b


def test_encrypt_rejects_oversized_inputs() -> None:
    with pytest.raises(ValueError):
        encrypt(1 << 32, 0, 10)
    with pytest.raises(ValueError):
        encrypt(0, 1 << 64, 10)
    with pytest.raises(ValueError):
        encrypt(0, 0, -1)


@settings(max_examples=500)
@given(
    plaintext=st.integers(min_value=0, max_value=(1 << 32) - 1),
    key=st.integers(min_value=0, max_value=(1 << 64) - 1),
    rounds=st.integers(min_value=0, max_value=600),
)
def test_encrypt_decrypt_roundtrip_property(plaintext: int, key: int, rounds: int) -> None:
    assert decrypt(encrypt(plaintext, key, rounds), key, rounds) == plaintext
