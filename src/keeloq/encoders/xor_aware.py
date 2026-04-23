"""XOR-aware hybrid encoder for CryptoMiniSat.

Each polynomial becomes exactly one XOR clause (linearized: Tseitin auxes
standing in for nonlinear monomials). The CNF side carries only the Tseitin
AND-gadgets linking aux vars to their underlying products.
"""
from __future__ import annotations

from keeloq.anf import BoolPoly
from keeloq.encoders import HybridInstance
from keeloq.encoders.cnf import _tseitin_and  # reuse


def encode(system: list[BoolPoly]) -> HybridInstance:
    var_names: list[str] = []
    seen: set[str] = set()
    for poly in system:
        for mono in poly.monomials:
            for v in sorted(mono):
                if v not in seen:
                    seen.add(v)
                    var_names.append(v)

    var_id = {name: i + 1 for i, name in enumerate(var_names)}
    next_aux = len(var_names)
    cnf: list[tuple[int, ...]] = []
    xors: list[tuple[tuple[int, ...], int]] = []

    for poly in system:
        xor_vars: list[int] = []
        rhs = 0
        for mono in poly.monomials:
            if len(mono) == 0:
                rhs ^= 1
            elif len(mono) == 1:
                (v,) = tuple(mono)
                xor_vars.append(var_id[v])
            else:
                next_aux += 1
                aux = next_aux
                ops = [var_id[v] for v in sorted(mono)]
                cnf.extend(_tseitin_and(aux, ops))
                xor_vars.append(aux)
        # The polynomial equals 0 iff XOR(xor_vars) == rhs.
        # Note: a poly == 0 means equation holds; so rhs here is the *constant term*.
        # XOR of monomial-values = 0  iff XOR of (linear literals + aux_for_nonlinear) = constant
        # So the XOR clause enforces: XOR(xor_vars) == rhs.
        xors.append((tuple(xor_vars), rhs))

    return HybridInstance(
        num_vars=next_aux,
        cnf_clauses=tuple(cnf),
        xor_clauses=tuple(xors),
        var_names=tuple(var_names),
    )
