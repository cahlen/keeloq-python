"""Tests for the XOR-aware hybrid encoder."""
from __future__ import annotations

from keeloq.anf import one, system, var
from keeloq.cipher import encrypt
from keeloq.encoders.xor_aware import encode


def test_encode_pure_xor_becomes_single_xor_clause() -> None:
    # x + y + z + 1 = 0  ->  XOR(x, y, z) = 1
    inst = encode([var("x") + var("y") + var("z") + one()])
    assert len(inst.xor_clauses) == 1
    lits, rhs = inst.xor_clauses[0]
    assert sorted(lits) == [1, 2, 3]
    assert rhs == 1
    assert inst.cnf_clauses == ()


def test_encode_nonlinear_monomial_gets_tseitin_cnf() -> None:
    # x*y + 1 = 0  ->  x AND y = 1
    inst = encode([var("x") * var("y") + one()])
    # Should emit CNF clauses for the AND gadget + an XOR clause linking aux to rhs=1
    assert len(inst.cnf_clauses) >= 3
    assert len(inst.xor_clauses) == 1


def test_keeloq_round_equation_emits_one_xor_per_round() -> None:
    """Every round equation has a linear XOR chain — one XOR clause per round."""
    rounds = 8
    pt = 0xAAAA5555
    key = 0x0123_4567_89AB_CDEF
    ct = encrypt(pt, key, rounds)
    sys = system(rounds=rounds, pairs=[(pt, ct)], key_hints=None)
    inst = encode(sys)
    # 64 pt/ct bindings (32 each) each become 1 xor clause.
    # 3 round equations per round; eq1 has linear part (XOR), eq2/eq3 don't have
    # a linear free part without constant (they're just "A + stuff"), so still emit XOR.
    # Every non-empty polynomial emits exactly one XOR clause in our encoder,
    # so total xor clauses == total input polynomials.
    assert len(inst.xor_clauses) == len(sys)


def test_solves_tiny_instance() -> None:
    """Sanity: the hybrid encoding is satisfiability-preserving."""
    rounds = 2
    pt = 0xAAAA5555
    key = 0x0123_4567_89AB_CDEF
    ct = encrypt(pt, key, rounds)
    hints = {i: (key >> (63 - i)) & 1 for i in range(64) if i != 0}
    sys = system(rounds=rounds, pairs=[(pt, ct)], key_hints=hints)
    inst = encode(sys)

    from pycryptosat import Solver
    s = Solver()
    for c in inst.cnf_clauses:
        s.add_clause(list(c))
    for lits, rhs in inst.xor_clauses:
        # pycryptosat: add_xor_clause(vars, rhs as bool)
        s.add_xor_clause(list(lits), bool(rhs))
    sat, a = s.solve()
    assert sat is True
    k0_idx = inst.var_names.index("K0") + 1
    recovered_k0 = 1 if a[k0_idx] else 0
    assert recovered_k0 == ((key >> 63) & 1)
