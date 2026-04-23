"""Boolean polynomial arithmetic over GF(2), and the KeeLoq ANF system generator.

Polynomials are represented as a frozenset of monomials, where each monomial is
a frozenset of variable names. The empty frozenset represents the constant 1;
the empty outer frozenset represents the constant 0. Arithmetic is:
  - Addition: symmetric difference of monomial sets (x + x = 0)
  - Multiplication: cross-product of monomials, then union within each, then sum
  - Variable power: x * x = x (GF(2) idempotence handled implicitly by using sets)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoolPoly:
    """Polynomial in GF(2)[x_1, ..., x_n]/(x_i^2 - x_i).

    `monomials` is a frozenset of frozensets-of-variable-names.
    The empty outer frozenset is the zero polynomial.
    A monomial equal to the empty frozenset represents the constant 1.
    """

    monomials: frozenset[frozenset[str]]

    def __add__(self, other: BoolPoly) -> BoolPoly:
        return BoolPoly(self.monomials.symmetric_difference(other.monomials))

    def __mul__(self, other: BoolPoly) -> BoolPoly:
        result: set[frozenset[str]] = set()
        for m1 in self.monomials:
            for m2 in other.monomials:
                product = m1 | m2  # GF(2) idempotence: x*x = x
                if product in result:
                    result.remove(product)
                else:
                    result.add(product)
        return BoolPoly(frozenset(result))

    def substitute(self, assignment: dict[str, int]) -> int:
        """Evaluate under a total assignment of all variables to 0/1."""
        needed = self.variables()
        missing = needed - assignment.keys()
        if missing:
            raise ValueError(f"substitute: missing values for {sorted(missing)}")
        total = 0
        for m in self.monomials:
            product = 1
            for v in m:
                product *= assignment[v]
                if product == 0:
                    break
            total = (total + product) & 1
        return total

    def variables(self) -> set[str]:
        out: set[str] = set()
        for m in self.monomials:
            out.update(m)
        return out


def zero() -> BoolPoly:
    return BoolPoly(frozenset())


def one() -> BoolPoly:
    return BoolPoly(frozenset([frozenset()]))


def var(name: str) -> BoolPoly:
    return BoolPoly(frozenset([frozenset([name])]))


def variables(rounds: int, num_pairs: int = 1) -> list[str]:
    """Return the ordered list of all ANF variables.

    Layout: K0..K63, then for each pair p in 0..num_pairs-1:
        A0_p, A1_p, ..., A{rounds-1}_p,
        B0_p, B1_p, ..., B{rounds-1}_p,
        L0_p, L1_p, ..., L{rounds+31}_p.

    Keys are shared across pairs (single underlying key). L/A/B are per-pair
    because the state evolution depends on which plaintext was encrypted.
    """
    if rounds < 0:
        raise ValueError(f"rounds={rounds} is negative")
    if num_pairs < 1:
        raise ValueError(f"num_pairs={num_pairs} must be >= 1")

    out = [f"K{i}" for i in range(64)]
    for p in range(num_pairs):
        out += [f"A{i}_p{p}" for i in range(rounds)]
        out += [f"B{i}_p{p}" for i in range(rounds)]
        out += [f"L{i}_p{p}" for i in range(rounds + 32)]
    return out


def round_equations(round_idx: int, pair_idx: int = 0) -> tuple[BoolPoly, BoolPoly, BoolPoly]:
    """Return the three ANF equations for a single round, for a specific pair.

    Equation shapes (from legacy/sage-equations.py:31-33):

        eq1 = L{i+32} + K{i%64} + L{i} + L{i+16} + L{i+9} + L{i+1}
              + L{i+31}*L{i+20} + B{i}
              + L{i+26}*L{i+20} + L{i+26}*L{i+1}
              + L{i+20}*L{i+9} + L{i+9}*L{i+1}
              + B{i}*L{i+9} + B{i}*L{i+20}
              + A{i}*L{i+9} + A{i}*L{i+20}
        eq2 = A{i} + L{i+31}*L{i+26}
        eq3 = B{i} + L{i+31}*L{i+1}
    """
    if round_idx < 0 or pair_idx < 0:
        raise ValueError("round_idx and pair_idx must be non-negative")

    i = round_idx
    p = pair_idx

    def lv(offset: int) -> BoolPoly:
        return var(f"L{offset}_p{p}")

    kv = var(f"K{i % 64}")
    av = var(f"A{i}_p{p}")
    bv = var(f"B{i}_p{p}")

    eq1 = (
        lv(i + 32)
        + kv
        + lv(i)
        + lv(i + 16)
        + lv(i + 9)
        + lv(i + 1)
        + lv(i + 31) * lv(i + 20)
        + bv
        + lv(i + 26) * lv(i + 20)
        + lv(i + 26) * lv(i + 1)
        + lv(i + 20) * lv(i + 9)
        + lv(i + 9) * lv(i + 1)
        + bv * lv(i + 9)
        + bv * lv(i + 20)
        + av * lv(i + 9)
        + av * lv(i + 20)
    )
    eq2 = av + lv(i + 31) * lv(i + 26)
    eq3 = bv + lv(i + 31) * lv(i + 1)
    return eq1, eq2, eq3


def system(
    rounds: int,
    pairs: list[tuple[int, int]],
    key_hints: dict[int, int] | None = None,
) -> list[BoolPoly]:
    """Generate the full ANF polynomial system.

    For each (plaintext, ciphertext) pair, emits:
      - 32 plaintext bit bindings: L{j}_p{p} + plaintext_bit_j  (so the equation
        equals zero iff L{j}_p{p} equals the plaintext bit)
      - 32 ciphertext bit bindings on the final-round state (L{rounds+j}_p{p})
      - 3 * rounds round equations

    Additionally emits key-hint bindings K{i} + value for each entry in key_hints.

    Args:
        rounds: number of KeeLoq rounds (>= 1).
        pairs: list of (plaintext_int, ciphertext_int) pairs; must have length >= 1.
        key_hints: optional mapping from key bit index (0..63, MSB-first) to bit value.
    """
    if rounds < 1:
        raise ValueError(f"rounds={rounds} must be >= 1")
    if not pairs:
        raise ValueError("pairs must be non-empty")
    key_hints = key_hints or {}
    for bit_idx, v in key_hints.items():
        if not 0 <= bit_idx < 64:
            raise ValueError(f"key_hints bit index {bit_idx} out of range")
        if v not in (0, 1):
            raise ValueError(f"key_hints value {v} for bit {bit_idx} is not a bit")

    out: list[BoolPoly] = []

    for p_idx, (pt, ct) in enumerate(pairs):
        if not 0 <= pt < (1 << 32):
            raise ValueError(f"plaintext pair {p_idx}={pt} does not fit in 32 bits")
        if not 0 <= ct < (1 << 32):
            raise ValueError(f"ciphertext pair {p_idx}={ct} does not fit in 32 bits")

        # Plaintext bindings: L{j}_p{p_idx} + pt_bit_j = 0  ->  L{j}_p{p_idx} = pt_bit_j
        for j in range(32):
            pt_bit = (pt >> (31 - j)) & 1
            binding = var(f"L{j}_p{p_idx}")
            if pt_bit:
                binding = binding + one()
            out.append(binding)

        # Ciphertext bindings: L{rounds+j}_p{p_idx} + ct_bit_j = 0
        for j in range(32):
            ct_bit = (ct >> (31 - j)) & 1
            binding = var(f"L{rounds + j}_p{p_idx}")
            if ct_bit:
                binding = binding + one()
            out.append(binding)

        # Round equations
        for i in range(rounds):
            eq1, eq2, eq3 = round_equations(round_idx=i, pair_idx=p_idx)
            out.extend([eq1, eq2, eq3])

    # Key hints (shared across pairs)
    for bit_idx, v in key_hints.items():
        binding = var(f"K{bit_idx}")
        if v:
            binding = binding + one()
        out.append(binding)

    return out
