"""Pure-CNF encoder: BoolPoly polynomial system -> DIMACS CNF.

Strategy:
  - Collect all variables from the system in a stable order.
  - For each polynomial `p = m_1 + m_2 + ... + m_k`:
      - Build a Tseitin-style tree: introduce a fresh aux variable `t_j` for
        each nonlinear monomial `m_j` (product of >= 2 variables), asserting
        `t_j = m_j` via AND-gadget clauses.
      - The polynomial's satisfaction condition becomes an XOR over the linear
        variables and the aux vars, with constant adjustment. Encode that XOR
        as a disjunction of parity clauses (2^(n-1) clauses for n operands).
  - Emit CNF in DIMACS form.
"""
from __future__ import annotations

from itertools import combinations

from keeloq.anf import BoolPoly
from keeloq.encoders import CNFInstance


def _tseitin_and(aux: int, operands: list[int]) -> list[tuple[int, ...]]:
    """Clauses asserting aux <-> AND(operands), for positive-literal operands.

    aux and operands are 1-indexed DIMACS variable ids (positive).
      aux -> each operand:   [-aux, op_i]
      operands together -> aux:  [-op_1, -op_2, ..., aux]
    """
    clauses: list[tuple[int, ...]] = []
    for op in operands:
        clauses.append((-aux, op))
    clauses.append(tuple([-op for op in operands] + [aux]))
    return clauses


def _xor_clauses(literals: list[int], rhs: int) -> list[tuple[int, ...]]:
    """Encode XOR(literals) = rhs as CNF — 2^(n-1) clauses, one per parity.

    For n literals, enumerate subsets S with |S| of a fixed parity; negate
    literals in S, keep the rest; each such disjunction is a clause.
    """
    n = len(literals)
    if n == 0:
        if rhs == 1:
            return [()]  # empty clause -> UNSAT
        return []
    clauses: list[tuple[int, ...]] = []
    # We want: XOR = rhs  iff  parity of (number of 1's) == rhs.
    # Clause form: for each assignment that makes XOR != rhs, rule it out.
    # Equivalent: for each subset S of literals with |S| having parity (n - rhs) % 2,
    # the clause is the literals negated at positions in S and positive elsewhere.
    # Forbidden assignments: those where parity(a) != rhs.
    # forbidden_parity = parity of the number of True values in a forbidden assignment.
    # If rhs=1, forbidden have even parity (0,2,4,...). If rhs=0, odd parity.
    forbidden_parity = (rhs + 1) % 2
    # A clause forbids one assignment. For assignment a, the clause negates True literals
    # and keeps False literals positive, so it is false exactly under a.
    # flip_size = number of positions that are True in the forbidden assignment
    # (those positions get negated in the clause).
    for flip_size in range(n + 1):
        if flip_size % 2 != forbidden_parity:
            continue
        for flips in combinations(range(n), flip_size):
            clause = tuple(
                -literals[i] if i in flips else literals[i] for i in range(n)
            )
            clauses.append(clause)
    return clauses


def encode(system: list[BoolPoly]) -> CNFInstance:
    # Collect variables in order of first appearance for stable ids.
    var_names: list[str] = []
    seen: set[str] = set()
    for poly in system:
        for mono in poly.monomials:
            for v in sorted(mono):
                if v not in seen:
                    seen.add(v)
                    var_names.append(v)

    var_id = {name: i + 1 for i, name in enumerate(var_names)}
    next_aux = len(var_names)  # will increment to get fresh ids (1-indexed)
    all_clauses: list[tuple[int, ...]] = []

    for poly in system:
        xor_literals: list[int] = []
        rhs = 0
        for mono in poly.monomials:
            if len(mono) == 0:
                rhs ^= 1
            elif len(mono) == 1:
                (v,) = tuple(mono)
                xor_literals.append(var_id[v])
            else:
                # Tseitin aux for the AND of these vars
                next_aux += 1
                aux = next_aux
                ops = [var_id[v] for v in sorted(mono)]
                all_clauses.extend(_tseitin_and(aux, ops))
                xor_literals.append(aux)

        all_clauses.extend(_xor_clauses(xor_literals, rhs))

    return CNFInstance(
        num_vars=next_aux,
        clauses=tuple(all_clauses),
        var_names=tuple(var_names),
    )


def to_dimacs(instance: CNFInstance) -> str:
    lines = [f"p cnf {instance.num_vars} {len(instance.clauses)}"]
    for clause in instance.clauses:
        lines.append(" ".join(str(lit) for lit in clause) + " 0")
    return "\n".join(lines) + "\n"


def from_dimacs(text: str, var_names: tuple[str, ...]) -> CNFInstance:
    num_vars = 0
    clauses: list[tuple[int, ...]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("p"):
            parts = line.split()
            num_vars = int(parts[2])
            continue
        lits = [int(x) for x in line.split()]
        if lits and lits[-1] == 0:
            lits = lits[:-1]
        clauses.append(tuple(lits))
    return CNFInstance(num_vars=num_vars, clauses=tuple(clauses),
                       var_names=var_names)
