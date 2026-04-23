"""Unit tests for keeloq.cipher."""

from __future__ import annotations

import pytest

from keeloq.cipher import core


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
