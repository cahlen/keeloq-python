"""Tests for keeloq.anf polynomial arithmetic over GF(2)."""

from __future__ import annotations

import pytest

from keeloq.anf import BoolPoly, one, var, variables, zero


def test_zero_is_empty() -> None:
    assert zero() == BoolPoly(frozenset())
    assert zero().monomials == frozenset()


def test_one_is_empty_monomial() -> None:
    assert one() == BoolPoly(frozenset([frozenset()]))


def test_var_is_single_monomial() -> None:
    assert var("x") == BoolPoly(frozenset([frozenset({"x"})]))


def test_addition_is_symmetric_difference() -> None:
    assert var("x") + var("y") == BoolPoly(frozenset([frozenset({"x"}), frozenset({"y"})]))
    assert var("x") + var("x") == zero()
    assert var("x") + one() + one() == var("x")


def test_multiplication_distributes() -> None:
    # (x + y) * (a + b) = xa + xb + ya + yb
    assert (var("x") + var("y")) * (var("a") + var("b")) == (
        var("x") * var("a") + var("x") * var("b") + var("y") * var("a") + var("y") * var("b")
    )


def test_multiplication_idempotent_over_gf2() -> None:
    # x * x = x in GF(2)
    assert var("x") * var("x") == var("x")


def test_substitute_total_assignment() -> None:
    # (x + y*z + 1) at x=1, y=1, z=0 -> 1 + 0 + 1 = 0
    poly = var("x") + var("y") * var("z") + one()
    assert poly.substitute({"x": 1, "y": 1, "z": 0}) == 0
    # At x=0, y=1, z=1 -> 0 + 1 + 1 = 0
    assert poly.substitute({"x": 0, "y": 1, "z": 1}) == 0
    # At x=1, y=1, z=1 -> 1 + 1 + 1 = 1
    assert poly.substitute({"x": 1, "y": 1, "z": 1}) == 1


def test_substitute_partial_raises() -> None:
    poly = var("x") + var("y")
    with pytest.raises(ValueError, match="missing"):
        poly.substitute({"x": 1})


def test_variables_of_polynomial() -> None:
    poly = var("x") + var("y") * var("z") + one()
    assert poly.variables() == {"x", "y", "z"}


def test_boolpoly_is_hashable() -> None:
    d = {var("x") + var("y"): "hello"}
    assert d[var("x") + var("y")] == "hello"


def test_variable_count_single_pair() -> None:
    # 64 K's + 1 pair * (3*rounds + 32 L's) = 64 + 3*rounds + 32 for num_pairs=1
    for rounds in (1, 16, 32, 160):
        vs = variables(rounds=rounds, num_pairs=1)
        assert len(vs) == 64 + 1*(3*rounds + 32), f"rounds={rounds}"
        assert len(vs) == len(set(vs)), "no duplicates"


def test_variable_count_multi_pair() -> None:
    vs = variables(rounds=32, num_pairs=2)
    assert len(vs) == 64 + 2*(3*32 + 32)  # 64 + 2*128 = 320
    assert len(vs) == len(set(vs))


def test_variable_naming_single_pair_includes_p0_suffix() -> None:
    vs = variables(rounds=4, num_pairs=1)
    assert "K0" in vs and "K63" in vs
    assert "L0_p0" in vs and "L35_p0" in vs  # L0..L{rounds+31}, so L0..L35 for rounds=4
    assert "A0_p0" in vs and "A3_p0" in vs
    assert "B0_p0" in vs and "B3_p0" in vs


def test_variable_naming_multi_pair() -> None:
    vs = variables(rounds=2, num_pairs=2)
    assert "L0_p0" in vs
    assert "L0_p1" in vs
    assert "A0_p0" in vs
    assert "A0_p1" in vs
    # K's are shared
    assert vs.count("K0") == 1


def test_variable_ordering() -> None:
    vs = variables(rounds=2, num_pairs=1)
    # K's come first, then per-pair A, B, L
    k_idx = vs.index("K0")
    a_idx = vs.index("A0_p0")
    b_idx = vs.index("B0_p0")
    l_idx = vs.index("L0_p0")
    assert k_idx < a_idx < b_idx < l_idx
