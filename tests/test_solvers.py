"""Tests for solver wrappers."""

from __future__ import annotations

import pytest

from keeloq.anf import one, system, var
from keeloq.cipher import encrypt
from keeloq.encoders.cnf import encode as encode_cnf
from keeloq.encoders.xor_aware import encode as encode_xor
from keeloq.solvers.cryptominisat import solve as solve_cms
from keeloq.solvers.dimacs_subprocess import solve as solve_subprocess


def test_trivial_sat_cnf() -> None:
    inst = encode_cnf([var("x") + one()])
    r = solve_cms(inst, timeout_s=5.0)
    assert r.status == "SAT"
    assert r.assignment is not None
    assert r.assignment["x"] == 1


def test_trivial_unsat_cnf() -> None:
    inst = encode_cnf([one()])  # 1 = 0, UNSAT
    r = solve_cms(inst, timeout_s=5.0)
    assert r.status == "UNSAT"
    assert r.assignment is None


def test_hybrid_xor_path_sat() -> None:
    inst = encode_xor([var("x") + var("y") + one()])
    r = solve_cms(inst, timeout_s=5.0)
    assert r.status == "SAT"
    assert r.assignment is not None
    assert r.assignment["x"] + r.assignment["y"] == 1


def test_tiny_keeloq_instance_solves_and_key_matches() -> None:
    rounds = 2
    pt = 0xAAAA5555
    key = 0x0123_4567_89AB_CDEF
    ct = encrypt(pt, key, rounds)
    hints = {i: (key >> (63 - i)) & 1 for i in range(64) if i != 0}
    sys = system(rounds=rounds, pairs=[(pt, ct)], key_hints=hints)

    for encode_fn, name in [(encode_cnf, "cnf"), (encode_xor, "xor")]:
        inst = encode_fn(sys)
        r = solve_cms(inst, timeout_s=10.0)
        assert r.status == "SAT", f"encoder {name} failed"
        assert r.assignment is not None
        assert r.assignment["K0"] == ((key >> 63) & 1), f"encoder {name} wrong K0"
        assert r.stats.wall_time_s > 0


def test_stats_are_populated() -> None:
    inst = encode_cnf([var("x") + var("y") + one()])
    r = solve_cms(inst, timeout_s=5.0)
    assert r.stats.num_vars >= 2
    assert r.stats.num_clauses >= 1
    assert r.stats.solver_name == "cryptominisat"


@pytest.mark.solver_kissat
def test_kissat_trivial_sat() -> None:
    inst = encode_cnf([var("x") + one()])
    r = solve_subprocess(inst, solver_binary="kissat", timeout_s=5.0)
    assert r.status == "SAT"
    assert r.assignment is not None and r.assignment["x"] == 1


@pytest.mark.solver_kissat
def test_kissat_trivial_unsat() -> None:
    inst = encode_cnf([one()])
    r = solve_subprocess(inst, solver_binary="kissat", timeout_s=5.0)
    assert r.status == "UNSAT"


def test_subprocess_rejects_hybrid_instance() -> None:
    inst = encode_xor([var("x") + one()])
    with pytest.raises(Exception, match="CNFInstance"):
        solve_subprocess(inst, solver_binary="kissat", timeout_s=5.0)  # type: ignore[arg-type]
