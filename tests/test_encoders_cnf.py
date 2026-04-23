"""Tests for the pure-CNF encoder."""
from __future__ import annotations

from keeloq.anf import one, system, var
from keeloq.encoders.cnf import encode, from_dimacs, to_dimacs


def test_encode_empty_system() -> None:
    inst = encode([])
    assert inst.num_vars == 0
    assert inst.clauses == ()


def test_encode_unsatisfiable_zero_equals_one() -> None:
    # The polynomial "1" (constant, non-zero) represents an UNSAT equation 1 = 0
    inst = encode([one()])
    to_dimacs(inst)
    # Must contain at least the empty clause (pure UNSAT) or an equivalent contradiction.
    # We assert by solving.
    from pycryptosat import Solver
    s = Solver()
    for clause in inst.clauses:
        s.add_clause(list(clause))
    sat, _ = s.solve()
    assert sat is False


def test_encode_single_variable_equation() -> None:
    # x + 1 = 0  ->  x = 1  ->  unit clause [x]
    poly = var("x") + one()
    inst = encode([poly])
    assert inst.num_vars == 1
    assert inst.var_names == ("x",)
    # Should be satisfiable with x=1
    from pycryptosat import Solver
    s = Solver()
    for clause in inst.clauses:
        s.add_clause(list(clause))
    sat, assignment = s.solve()
    assert sat is True
    assert assignment[1] is True  # pycryptosat indexes from 1; index 0 is None


def test_encode_xor_equation() -> None:
    # x + y + 1 = 0  ->  x XOR y = 1  ->  two clauses: [x, y] and [-x, -y]
    poly = var("x") + var("y") + one()
    inst = encode([poly])
    from pycryptosat import Solver
    s = Solver()
    for clause in inst.clauses:
        s.add_clause(list(clause))
    # Must be SAT with exactly x != y
    sat, a = s.solve()
    assert sat is True
    assert a[1] != a[2]


def test_encode_simple_and_equation() -> None:
    # x*y + 1 = 0  ->  x AND y = 1  ->  unit clauses [x], [y]
    poly = var("x") * var("y") + one()
    inst = encode([poly])
    from pycryptosat import Solver
    s = Solver()
    for clause in inst.clauses:
        s.add_clause(list(clause))
    sat, a = s.solve()
    assert sat is True and a[1] is True and a[2] is True


def test_to_dimacs_roundtrip() -> None:
    inst = encode([var("x") + var("y") + one(), var("z") + one()])
    text = to_dimacs(inst)
    recovered = from_dimacs(text, var_names=inst.var_names)
    assert recovered.num_vars == inst.num_vars
    assert set(recovered.clauses) == set(inst.clauses)


def test_encode_tiny_keeloq_instance_is_solvable_with_heavy_hints() -> None:
    """2-round attack with 63 of 64 key bits hinted should solve trivially."""
    from keeloq.cipher import encrypt
    rounds = 2
    pt = 0xAAAA5555
    key = 0x0123_4567_89AB_CDEF
    ct = encrypt(pt, key, rounds)
    hints = {i: (key >> (63 - i)) & 1 for i in range(64) if i != 0}
    sys = system(rounds=rounds, pairs=[(pt, ct)], key_hints=hints)
    inst = encode(sys)

    from pycryptosat import Solver
    s = Solver()
    for clause in inst.clauses:
        s.add_clause(list(clause))
    sat, a = s.solve()
    assert sat is True
    # Recover K0 from the assignment and confirm it matches the true key bit
    k0_idx = inst.var_names.index("K0") + 1
    recovered_k0 = 1 if a[k0_idx] else 0
    assert recovered_k0 == ((key >> 63) & 1)
