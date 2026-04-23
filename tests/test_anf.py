"""Tests for keeloq.anf polynomial arithmetic over GF(2)."""

from __future__ import annotations

import pytest

from keeloq.anf import BoolPoly, one, round_equations, system, var, variables, zero
from keeloq.cipher import encrypt


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
        assert len(vs) == 64 + 1 * (3 * rounds + 32), f"rounds={rounds}"
        assert len(vs) == len(set(vs)), "no duplicates"


def test_variable_count_multi_pair() -> None:
    vs = variables(rounds=32, num_pairs=2)
    assert len(vs) == 64 + 2 * (3 * 32 + 32)  # 64 + 2*128 = 320
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


def test_round_equations_returns_three_polys() -> None:
    eqs = round_equations(round_idx=0, pair_idx=0)
    assert len(eqs) == 3


def test_round_equation_eq2_is_a_equals_l31_l26() -> None:
    """From legacy/sage-equations.py:32:  A{i} + L{i+31}*L{i+26}.
    In our naming: A0_p0 + L31_p0*L26_p0."""
    _, eq2, _ = round_equations(round_idx=0, pair_idx=0)
    expected = var("A0_p0") + var("L31_p0") * var("L26_p0")
    assert eq2 == expected


def test_round_equation_eq3_is_b_equals_l31_l1() -> None:
    _, _, eq3 = round_equations(round_idx=0, pair_idx=0)
    expected = var("B0_p0") + var("L31_p0") * var("L1_p0")
    assert eq3 == expected


def test_round_equation_eq1_structure_at_round_0() -> None:
    """Match legacy/sage-equations.py:31 exactly, modulo pair suffix."""
    eq1, _, _ = round_equations(round_idx=0, pair_idx=0)
    # eq1 = L32 + K0 + L0 + L16 + L9 + L1 + L31*L20 + B0 + L26*L20
    #       + L26*L1 + L20*L9 + L9*L1 + B0*L9 + B0*L20 + A0*L9 + A0*L20
    expected = (
        var("L32_p0")
        + var("K0")
        + var("L0_p0")
        + var("L16_p0")
        + var("L9_p0")
        + var("L1_p0")
        + var("L31_p0") * var("L20_p0")
        + var("B0_p0")
        + var("L26_p0") * var("L20_p0")
        + var("L26_p0") * var("L1_p0")
        + var("L20_p0") * var("L9_p0")
        + var("L9_p0") * var("L1_p0")
        + var("B0_p0") * var("L9_p0")
        + var("B0_p0") * var("L20_p0")
        + var("A0_p0") * var("L9_p0")
        + var("A0_p0") * var("L20_p0")
    )
    assert eq1 == expected


def test_round_equation_indices_shift_with_round() -> None:
    """At round i, eq1 involves L{32+i}, K{i%64}, L{i}, L{i+16}, ..."""
    eq1, eq2, eq3 = round_equations(round_idx=5, pair_idx=0)
    assert "L37_p0" in eq1.variables()  # L{32+5}
    assert "K5" in eq1.variables()
    assert "A5_p0" in eq2.variables()
    assert "B5_p0" in eq3.variables()


def test_round_equation_key_index_wraps_at_64() -> None:
    eq1_round0, _, _ = round_equations(round_idx=0, pair_idx=0)
    eq1_round64, _, _ = round_equations(round_idx=64, pair_idx=0)
    # Both should reference K0 (0 % 64 == 0 and 64 % 64 == 0)
    assert "K0" in eq1_round0.variables()
    assert "K0" in eq1_round64.variables()


def test_round_equation_pair_index_changes_names() -> None:
    eq1_p0, _, _ = round_equations(round_idx=0, pair_idx=0)
    eq1_p1, _, _ = round_equations(round_idx=0, pair_idx=1)
    assert "L32_p0" in eq1_p0.variables()
    assert "L32_p1" in eq1_p1.variables()
    # K variables still shared
    assert "K0" in eq1_p0.variables()
    assert "K0" in eq1_p1.variables()


def test_system_size_single_pair_no_hints() -> None:
    rounds = 4
    pt, ct = 0xDEADBEEF, 0
    sys = system(rounds=rounds, pairs=[(pt, ct)], key_hints=None)
    # Expected: 32 plaintext bindings + 32 ciphertext bindings + 3*rounds round equations
    assert len(sys) == 32 + 32 + 3 * rounds


def test_system_size_with_hints() -> None:
    rounds = 4
    sys = system(rounds=rounds, pairs=[(0, 0)], key_hints={0: 1, 5: 0, 10: 1})
    assert len(sys) == 32 + 32 + 3 * rounds + 3


def test_system_size_multi_pair() -> None:
    rounds = 4
    sys = system(rounds=rounds, pairs=[(0, 0), (1, 1)], key_hints=None)
    # 2 pairs each contribute 32+32+3*rounds; K-hints are shared (0 here)
    assert len(sys) == 2 * (32 + 32 + 3 * rounds)


def test_true_solution_satisfies_every_equation_single_pair() -> None:
    """The cornerstone test: generated equations are correct iff the true
    (key, L-values, A-values, B-values) zero every polynomial in the system."""
    rounds = 16
    pt = 0xCAFEBABE
    key = 0x0123_4567_89AB_CDEF
    ct = encrypt(pt, key, rounds)

    sys = system(rounds=rounds, pairs=[(pt, ct)], key_hints=None)
    assignment = _derive_true_assignment(pt, ct, key, rounds, pair_idx=0)
    for idx, poly in enumerate(sys):
        assert poly.substitute(assignment) == 0, (
            f"equation {idx} not satisfied: vars={poly.variables()}"
        )


def test_true_solution_satisfies_multi_pair() -> None:
    rounds = 16
    key = 0x0123_4567_89AB_CDEF
    pairs_plain = [0xCAFEBABE, 0xDEADBEEF, 0x13579BDF]
    pairs = [(p, encrypt(p, key, rounds)) for p in pairs_plain]

    sys = system(rounds=rounds, pairs=pairs, key_hints=None)
    assignment: dict[str, int] = {}
    for p_idx, (pt, ct) in enumerate(pairs):
        assignment.update(_derive_true_assignment(pt, ct, key, rounds, pair_idx=p_idx))
    # K variables only need to be added once (shared across pairs)
    for bit_idx in range(64):
        assignment.setdefault(f"K{bit_idx}", (key >> (63 - bit_idx)) & 1)

    for idx, poly in enumerate(sys):
        assert poly.substitute(assignment) == 0, f"equation {idx} unsatisfied"


def _derive_true_assignment(
    pt: int, ct: int, key: int, rounds: int, pair_idx: int
) -> dict[str, int]:
    """Run the cipher and record every L/A/B intermediate value + K bits."""
    from keeloq.cipher import _key_bit, _state_bit, core

    assignment: dict[str, int] = {}
    for bit_idx in range(64):
        assignment[f"K{bit_idx}"] = _key_bit(key, bit_idx)

    state = pt
    # L0..L31 are plaintext bits (MSB-first)
    for bit_idx in range(32):
        assignment[f"L{bit_idx}_p{pair_idx}"] = _state_bit(pt, bit_idx)

    for i in range(rounds):
        # Capture A_i = L{i+31}*L{i+26} BEFORE this round's shift
        # and B_i = L{i+31}*L{i+1}.
        # Current `state` corresponds to L{i}..L{i+31} mapped MSB-first.
        l31 = _state_bit(state, 31)
        l26 = _state_bit(state, 26)
        l20 = _state_bit(state, 20)
        l9 = _state_bit(state, 9)
        l1 = _state_bit(state, 1)
        l16 = _state_bit(state, 16)
        l0 = _state_bit(state, 0)

        assignment[f"A{i}_p{pair_idx}"] = (l31 * l26) % 2
        assignment[f"B{i}_p{pair_idx}"] = (l31 * l1) % 2

        kbit = _key_bit(key, i % 64)
        newb = (kbit + l0 + l16 + core(l31, l26, l20, l9, l1)) % 2
        state = ((state << 1) & 0xFFFFFFFF) | newb
        assignment[f"L{i + 32}_p{pair_idx}"] = newb

    # Sanity: post-rounds state matches ciphertext
    assert state == ct, f"cipher reference disagreement: got {state:08x} want {ct:08x}"
    return assignment
