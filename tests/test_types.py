"""Unit tests for keeloq._types bit-string <-> int conversions."""

from __future__ import annotations

import pytest

from keeloq._types import bits_to_int, int_to_bits


@pytest.mark.parametrize(
    "bits,expected",
    [
        ("00000000000000000000000000000000", 0),
        ("00000000000000000000000000000001", 1),
        ("10000000000000000000000000000000", 0x80000000),
        ("01010101010101010101010101010101", 0x55555555),
        ("11111111111111111111111111111111", 0xFFFFFFFF),
    ],
)
def test_bits_to_int_32(bits: str, expected: int) -> None:
    assert bits_to_int(bits) == expected


def test_bits_to_int_64() -> None:
    key = "0000010000100010100011100000000010000110000011001001111000010001"
    assert bits_to_int(key) == 0x0422_8E00_860C_9E11


def test_int_to_bits_32() -> None:
    assert int_to_bits(0x55555555, 32) == "01010101010101010101010101010101"


def test_int_to_bits_64() -> None:
    assert (
        int_to_bits(0x0422_8E00_860C_9E11, 64)
        == "0000010000100010100011100000000010000110000011001001111000010001"
    )


def test_roundtrip() -> None:
    for n in (0, 1, 42, 0xDEADBEEF, 0xFFFFFFFF):
        assert bits_to_int(int_to_bits(n, 32)) == n


def test_bits_to_int_rejects_non_binary() -> None:
    with pytest.raises(ValueError):
        bits_to_int("01020")


def test_int_to_bits_rejects_overflow() -> None:
    with pytest.raises(ValueError):
        int_to_bits(1 << 33, 32)
