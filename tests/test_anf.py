"""Tests for keeloq.anf polynomial arithmetic over GF(2)."""

from __future__ import annotations

import pytest

from keeloq.anf import BoolPoly, one, var, zero


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
